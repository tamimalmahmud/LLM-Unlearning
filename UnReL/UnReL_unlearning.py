from data_module import TextDatasetQA, custom_data_collator
from dataloader import CustomTrainer

import shutil
import os
import re
import json
import time
from pathlib import Path
from typing import Dict, List, Tuple, Optional

import hydra
from hydra.utils import get_original_cwd
import torch
from torch import nn
import transformers
from transformers import AutoTokenizer, AutoModelForCausalLM, set_seed
from transformers.trainer_callback import TrainerCallback, TrainerControl, TrainerState

from utils import get_model_identifiers_from_yaml
from peft import LoraConfig, get_peft_model  # optional LoRA


# ===== Utility functions =====

def extract_shard_part(split: str):
    match = re.search(r"retain_(shard\d+_\d+)_forget(?:01DP|03DP|05DP|10DP|20DP)", split)
    if match:
        return match.group(1)
    match = re.search(r"retain_(shard\d+_\d+)_forget(?:01|02|03|04|05|10|20)", split)
    if match:
        return match.group(1)
    elif re.match(r"shard\d+_\d+", split):
        return split
    return None


def save_milestone(model, tokenizer, base_dir: str, idx: int, dpath: str, split, meta: Dict,
                   start_time: float, shard_start_persisted: str,
                   is_last: bool = False, forget_tag: Optional[str] = None):
    """Save checkpoint for each split. Rename final one as Final_Model_<forget>."""
    shard_part = extract_shard_part(split)
    if shard_part is None:
        shard_part = split

    if is_last and forget_tag:
        ckpt_dir = os.path.join(base_dir, f"Final_Model_{forget_tag}")
    else:
        ckpt_dir = os.path.join(base_dir, split)

    if os.path.exists(ckpt_dir):
        print(f"[INFO] Replacing existing checkpoint: {ckpt_dir}")
        shutil.rmtree(ckpt_dir)
    Path(ckpt_dir).mkdir(parents=True, exist_ok=True)

    # Save model and tokenizer
    model.save_pretrained(ckpt_dir)
    tokenizer.save_pretrained(ckpt_dir)

    # Metadata
    meta_out = dict(meta)
    meta_out.update({
        "dataset_index": idx,
        "dataset_path": dpath,
        "split": split,
        "checkpoint_dir": ckpt_dir,
        "time_iso": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())
    })
    with open(os.path.join(ckpt_dir, "metadata.json"), "w", encoding="utf-8") as f:
        json.dump(meta_out, f, indent=2)

    elapsed = time.time() - start_time
    with open(os.path.join(ckpt_dir, "running_time.txt"), "w") as f:
        f.write(f"Total running time: {elapsed:.2f} seconds\n")

    if is_last and forget_tag:
        print(f"[INFO] Final model saved as: {ckpt_dir}")
    else:
        print(f"[Milestone] Saved checkpoint: {ckpt_dir}")
    return ckpt_dir


# ===== Heartbeat logging =====

class HeartbeatCallback(transformers.TrainerCallback):
    def __init__(self, every: int = 50):
        self.every = max(1, int(every))

    def on_step_end(self, args, state: TrainerState, control: TrainerControl, **kwargs):
        if state.global_step and state.max_steps and (state.global_step % self.every == 0):
            loss = None
            if state.log_history:
                last = state.log_history[-1]
                loss = last.get("loss", last.get("train_loss", None))
            msg = f"[step {state.global_step}/{state.max_steps}]"
            if loss is not None:
                try:
                    msg += f" loss={float(loss):.4f}"
                except Exception:
                    msg += f" loss={loss}"
            print(msg, flush=True)
        return control


# ===== OSL classes =====

class ActivationTap:
    def __init__(self, model: nn.Module, target_linear_only: bool = True):
        self.inputs, self.outputs, self.hooks = {}, {}, []
        self.target_linear_only = target_linear_only
        for name, m in model.named_modules():
            if target_linear_only and not isinstance(m, nn.Linear):
                continue
            if isinstance(m, nn.Linear):
                self.hooks.append(m.register_forward_pre_hook(
                    lambda mod, inp, n=name: self.inputs.setdefault(n, []).append(inp[0].detach())))
                self.hooks.append(m.register_forward_hook(
                    lambda mod, inp, out, n=name: self.outputs.setdefault(n, []).append(out.detach())))

    def remove(self):
        [h.remove() for h in self.hooks]
        self.hooks = []

    def cosine_importance(self) -> Dict[str, float]:
        scores = {}
        for n in set(self.inputs) & set(self.outputs):
            Xs, Ys = self.inputs[n], self.outputs[n]
            nmin = min(len(Xs), len(Ys))
            sims = [
                (torch.nn.functional.normalize(x.reshape(x.shape[0], -1), dim=-1) *
                 torch.nn.functional.normalize(y.reshape(y.shape[0], -1), dim=-1)).sum(-1).mean().item()
                for x, y in zip(Xs[:nmin], Ys[:nmin])
            ]
            if sims:
                scores[n] = sum(sims) / len(sims)
        return scores


class SVDProjector:
    def __init__(self, keep: Dict[str, Tuple[torch.Tensor, torch.Tensor]]):
        self.keep = keep
        self._cached = {}
        self.hooks = []

    def _get_cached(self, name, grad):
        pair = self._cached.get(name, None)
        if pair is not None:
            Uc, Vc = pair
            if Uc is None or Vc is None:
                return (None, None)
            if Uc.device != grad.device or Uc.dtype != grad.dtype:
                Uc, Vc = Uc.to(grad.device, grad.dtype), Vc.to(grad.device, grad.dtype)
                self._cached[name] = (Uc, Vc)
            return (Uc, Vc)
        U, V = self.keep.get(name, (None, None))
        if U is None or V is None or U.numel() == 0 or V.numel() == 0:
            self._cached[name] = (None, None)
            return (None, None)
        Uc, Vc = U.to(grad.device, grad.dtype), V.to(grad.device, grad.dtype)
        self._cached[name] = (Uc, Vc)
        return (Uc, Vc)

    def _hook(self, name):
        def fn(grad):
            if grad is None:
                return None
            Uh, Vh = self._get_cached(name, grad)
            if Uh is None or Vh is None:
                return grad
            return grad - (Uh @ (Uh.T @ grad @ Vh) @ Vh.T)
        return fn

    def register(self, model):
        for n, p in model.named_parameters():
            if p.requires_grad and p.ndim == 2 and n.endswith(".weight") and n in self.keep:
                self.hooks.append(p.register_hook(self._hook(n)))
        print(f"[OSL] Registered projectors for {len(self.hooks)} params.")

    def remove(self):
        [h.remove() for h in self.hooks]
        self.hooks = []
        self._cached.clear()


def build_osl_subspaces_gpu(model, imp, mrr=0.10, trr=0.80, target_linear_only=True):
    dev = next(model.parameters()).device if any(True for _ in model.parameters()) else torch.device("cuda")
    keep = {}
    with torch.no_grad():
        for name, mod in model.named_modules():
            if target_linear_only and not isinstance(mod, nn.Linear):
                continue
            if isinstance(mod, nn.Linear):
                W = mod.weight.detach().to(torch.float32).to(dev)
                if W.numel() == 0:
                    continue
                U, S, Vh = torch.linalg.svd(W, full_matrices=False)
                I = float(max(0.0, min(1.0, imp.get(name, 0.5))))
                ratio = mrr + I * (trr - mrr)
                k = int(min(W.shape) * ratio)
                if k == 0:
                    keep[f"{name}.weight"] = (
                        torch.zeros((W.shape[0], 0), device=dev),
                        torch.zeros((W.shape[1], 0), device=dev),
                    )
                else:
                    keep[f"{name}.weight"] = (
                        U[:, :k].contiguous(),
                        Vh[:k, :].contiguous().T,
                    )
    return keep


# ===== Main =====

@hydra.main(version_base=None, config_path="config", config_name="UnReL_unlearning")
def main(cfg):
    start_time = time.time()
    set_seed(cfg.seed)

    # === Load model ===
    resume_path = getattr(cfg, "resume_from", None)
    model = None
    tokenizer = None

    if resume_path and resume_path.lower() != "none":
        if not os.path.isabs(resume_path):
            resume_path = os.path.join(get_original_cwd(), resume_path)
        resume_path = os.path.abspath(resume_path)
        print(f"[Resume] Loading model from {resume_path}")
        if not os.path.exists(resume_path):
            raise FileNotFoundError(f"Checkpoint not found: {resume_path}")
        model = AutoModelForCausalLM.from_pretrained(resume_path, torch_dtype=torch.bfloat16)
        tokenizer = AutoTokenizer.from_pretrained(resume_path)
    else:
        model_cfg = get_model_identifiers_from_yaml(cfg.model_family)
        model_id = model_cfg["hf_key"]
        print(f"[Init] Loading pretrained model family: {cfg.model_family} ({model_id})")
        model = AutoModelForCausalLM.from_pretrained(model_id, torch_dtype=torch.bfloat16, trust_remote_code=True)
        tokenizer = AutoTokenizer.from_pretrained(model_id)

    tokenizer.pad_token = tokenizer.eos_token

    data_paths = list(getattr(cfg, "data_paths", []))
    splits = list(getattr(cfg, "splits", []))
    if not data_paths:
        data_paths = [cfg.data_path] * len(splits)

    batch_size, gas = cfg.batch_size, cfg.gradient_accumulation_steps
    num_devices = int(os.environ.get("WORLD_SIZE", 1))
    max_length = 500

    cont = getattr(cfg, "continual", {})
    osl_cfg = cont.get("osl", {}) if cont else {}
    osl_mrr, osl_trr = float(osl_cfg.get("mrr", 0.1)), float(osl_cfg.get("trr", 0.8))
    osl_cali_batches = int(osl_cfg.get("cali_batches", 0))
    osl_target_linear_only = bool(osl_cfg.get("target_linear_only", True))

    shard_start_persisted = None
    for di, (dpath, dsplit) in enumerate(zip(data_paths, splits), start=1):
        print(f"\n=== Dataset {di}/{len(data_paths)} === {dpath} | Split={dsplit}")
        ds = TextDatasetQA(dpath, tokenizer=tokenizer, model_family=cfg.model_family, max_length=max_length, split=dsplit)
        max_steps = int(cfg.num_epochs * len(ds)) // (batch_size * gas * num_devices)
        print(f"max_steps: {max_steps}")

        training_args = transformers.TrainingArguments(
            per_device_train_batch_size=batch_size,
            gradient_accumulation_steps=gas,
            max_steps=max_steps,
            learning_rate=cfg.lr,
            bf16=True,
            logging_dir=f"{cfg.save_dir}/logs",
            output_dir=cfg.save_dir,
            save_steps=max_steps if max_steps > 0 else 1,
            save_only_model=True,
            eval_strategy="no",
            report_to="none",
            seed=cfg.seed,
        )

        imp = {}
        if osl_cali_batches > 0:
            tap = ActivationTap(model, osl_target_linear_only)
            with torch.no_grad():
                for idx in range(min(len(ds), batch_size * osl_cali_batches)):
                    try:
                        batch = custom_data_collator([ds[idx]])
                        _ = model(**{k: v.to(model.device) for k, v in batch.items()})
                    except Exception:
                        pass
            imp = tap.cosine_importance()
            tap.remove()

        keep = build_osl_subspaces_gpu(model, imp, mrr=osl_mrr, trr=osl_trr, target_linear_only=osl_target_linear_only)
        projector = SVDProjector(keep)
        projector.register(model)

        trainer = CustomTrainer(
            model=model,
            train_dataset=ds,
            args=training_args,
            data_collator=custom_data_collator,
            callbacks=[HeartbeatCallback(every=max(1, max_steps // 100) if max_steps > 0 else 10)],
        )
        model.config.use_cache = False
        trainer.train()
        projector.remove()

        milestone_meta = {
            "model_family": getattr(cfg, "model_family", "phi"),
            "osl": {"mrr": osl_mrr, "trr": osl_trr},
        }

        is_last = (di == len(data_paths))
        save_milestone(
            model,
            tokenizer,
            cfg.save_dir,
            di,
            dpath,
            dsplit,
            milestone_meta,
            start_time,
            shard_start_persisted,
            is_last=is_last,
            forget_tag=getattr(cfg, "forget", None),
        )


if __name__ == "__main__":
    main()
