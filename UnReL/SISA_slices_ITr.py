from data_module import TextDatasetQA, custom_data_collator
from dataloader import CustomTrainer

import os
import re
import json
import time
from pathlib import Path
from typing import Dict, List, Tuple, Optional

import hydra
import torch
from torch import nn
import transformers
from transformers import AutoTokenizer, AutoModelForCausalLM, set_seed
from transformers.trainer_callback import TrainerCallback, TrainerControl, TrainerState
from omegaconf import OmegaConf

from utils import get_model_identifiers_from_yaml


# =========================
# Utility helpers
# =========================

def find_all_linear_names(model):
    cls = torch.nn.Linear
    linear_module_names = set()
    for name, module in model.named_modules():
        if isinstance(module, cls):
            parts = name.split('.')
            linear_module_names.add(parts[0] if len(parts) == 1 else parts[-1])
    linear_module_names.discard('lm_head')
    return list(linear_module_names)


def print_trainable_parameters(model):
    trainable_params = 0
    all_param = 0
    for _, p in model.named_parameters():
        all_param += p.numel()
        if p.requires_grad:
            trainable_params += p.numel()
    pct = 100 * trainable_params / max(1, all_param)
    print(f"trainable params: {trainable_params} || all params: {all_param} || trainable%: {pct:.4f}")


def rename_final_checkpoint(ckpt_dir: str):
    final_model_name = "Final_Model"
    new_ckpt_dir = os.path.join(os.path.dirname(ckpt_dir), final_model_name)
    if os.path.exists(ckpt_dir):
        os.rename(ckpt_dir, new_ckpt_dir)
        print(f"Checkpoint renamed to: {new_ckpt_dir}")
    return new_ckpt_dir


def _slugify(name: str) -> str:
    name = Path(str(name)).name
    name = re.sub(r"[^a-zA-Z0-9._-]+", "_", name).strip("_")
    return name or "dataset"


def save_milestone(model, tokenizer, base_dir: str, idx: int, dpath: str, split, meta: Dict, start_time: float):
    tag_piece = split if (split is not None and str(split).strip() != "") else dpath
    tag = f"{_slugify(tag_piece)}"
    ckpt_dir = os.path.join(base_dir, tag)
    Path(ckpt_dir).mkdir(parents=True, exist_ok=True)

    model.save_pretrained(ckpt_dir)
    tokenizer.save_pretrained(ckpt_dir)

    meta_out = dict(meta)
    meta_out.update({
        "dataset_index": idx,
        "dataset_path": dpath,
        "split": split,
        "checkpoint_dir": ckpt_dir,
        "time_iso": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
    })
    with open(os.path.join(ckpt_dir, "metadata.json"), "w", encoding="utf-8") as f:
        json.dump(meta_out, f, indent=2)

    elapsed = time.time() - start_time
    with open(os.path.join(ckpt_dir, 'running_time.txt'), 'w') as f:
        f.write(f"Total running time: {elapsed:.2f} seconds\n")

    if split in ["shard8_8", "shard4_4", "shard2_2"]:
        new_ckpt_dir = rename_final_checkpoint(ckpt_dir)
        return new_ckpt_dir
    print(f"Saved per-dataset checkpoint to: {ckpt_dir}")
    return ckpt_dir


# =========================
# Heartbeat logging
# =========================

class HeartbeatCallback(TrainerCallback):
    def __init__(self, every:int=50):
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


# =========================
# OSL components
# =========================

class ActivationTap:
    """Capture input/output activations for Linear layers to compute per-layer importance."""
    def __init__(self, model: nn.Module, target_linear_only: bool = True):
        self.inputs: Dict[str, List[torch.Tensor]] = {}
        self.outputs: Dict[str, List[torch.Tensor]] = {}
        self.hooks: List[torch.utils.hooks.RemovableHandle] = []
        self.target_linear_only = target_linear_only

        for name, m in model.named_modules():
            if target_linear_only and not isinstance(m, nn.Linear):
                continue
            if isinstance(m, nn.Linear):
                self.hooks.append(m.register_forward_pre_hook(self._make_in_hook(name)))
                self.hooks.append(m.register_forward_hook(self._make_out_hook(name)))

    def _make_in_hook(self, name: str):
        def fn(mod, inp):
            try:
                x = inp[0].detach()
                self.inputs.setdefault(name, []).append(x)
            except Exception:
                pass
        return fn

    def _make_out_hook(self, name: str):
        def fn(mod, _, out):
            try:
                y = out.detach()
                self.outputs.setdefault(name, []).append(y)
            except Exception:
                pass
        return fn

    def remove(self):
        for h in self.hooks:
            try:
                h.remove()
            except Exception:
                pass
        self.hooks = []

    def cosine_importance(self) -> Dict[str, float]:
        scores: Dict[str, float] = {}
        for name in set(self.outputs):  # use outputs only
            Ys = self.outputs[name]
            n = len(Ys)
            if n == 0:
                continue
            cumval, cnt = 0.0, 0
            for y in Ys:
                y_f = y.reshape(y.shape[0], -1)
                val = torch.norm(y_f, dim=-1).mean().item()
                cumval += val
                cnt += 1
            scores[name] = cumval / max(1, cnt)
    # normalize to [0,1]
        if scores:
            maxval = max(scores.values())
            minval = min(scores.values())
            rng = maxval - minval if maxval > minval else 1.0
            for k in scores:
                scores[k] = (scores[k] - minval) / rng
        return scores



class SVDProjector:
    """Gradient projector that protects high-rank subspaces per Linear layer."""
    def __init__(self, keep: Dict[str, Tuple[torch.Tensor, torch.Tensor]]):
        # keep: param_name -> (U_h [out,k], V_h [in,k])
        self.keep = keep
        self._cached: Dict[str, Tuple[torch.Tensor, torch.Tensor]] = {}
        self.hooks: List[torch.utils.hooks.RemovableHandle] = []

    def _get_cached(self, name: str, grad: torch.Tensor):
        pair = self._cached.get(name, None)
        if pair is not None:
            Uc, Vc = pair
            if Uc is None or Vc is None:
                return (None, None)
            if Uc.device != grad.device or Uc.dtype != grad.dtype:
                Uc = Uc.to(grad.device, grad.dtype, non_blocking=True)
                Vc = Vc.to(grad.device, grad.dtype, non_blocking=True)
                self._cached[name] = (Uc, Vc)
            return (Uc, Vc)

        U, V = self.keep.get(name, (None, None))
        if U is None or V is None or U.numel() == 0 or V.numel() == 0:
            self._cached[name] = (None, None)
            return (None, None)

        Uc = U.to(grad.device, grad.dtype, non_blocking=True)
        Vc = V.to(grad.device, grad.dtype, non_blocking=True)
        self._cached[name] = (Uc, Vc)
        return (Uc, Vc)

    def _hook(self, name: str):
        def fn(grad):
            if grad is None:
                return None
            Uh, Vh = self._get_cached(name, grad)
            if Uh is None or Vh is None or Uh.numel() == 0 or Vh.numel() == 0:
                return grad
            # Project: G <- G - Uh (Uh^T G Vh) Vh^T
            kGk = Uh.transpose(0, 1) @ grad @ Vh
            return grad - (Uh @ kGk @ Vh.transpose(0, 1))
        return fn

    def register(self, model: nn.Module):
        attached = 0
        for n, p in model.named_parameters():
            if p.requires_grad and p.ndim == 2 and n.endswith(".weight") and n in self.keep:
                self.hooks.append(p.register_hook(self._hook(n)))
                attached += 1
        print(f"[OSL] Registered gradient projectors for {attached} parameters.")

    def remove(self):
        for h in self.hooks:
            try:
                h.remove()
            except Exception:
                pass
        self.hooks = []
        self._cached.clear()


def build_osl_subspaces_gpu(model: nn.Module,
                            imp: Dict[str, float],
                            mrr: float = 0.10,
                            trr: float = 0.80,
                            target_linear_only: bool = True) -> Dict[str, Tuple[torch.Tensor, torch.Tensor]]:
    """Build (U_h, V_h) per Linear layer via SVD on GPU (CUDA) in float32."""
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA required for OSL; no GPU detected.")

    keep: Dict[str, Tuple[torch.Tensor, torch.Tensor]] = {}
    with torch.no_grad():
        dev = next(model.parameters()).device
        if dev.type != "cuda":
            dev = torch.device("cuda")

        for name, mod in model.named_modules():
            if target_linear_only and not isinstance(mod, nn.Linear):
                continue
            if isinstance(mod, nn.Linear):
                for p_name, p in mod.named_parameters(recurse=False):
                    if p_name != "weight":
                        continue
                    full_name = f"{name}.weight"
                    if not p.requires_grad:
                        continue

                    W = p.detach().to(torch.float32).to(dev)
                    if W.numel() == 0:
                        continue
                    try:
                        U, S, Vh = torch.linalg.svd(W, full_matrices=False)
                    except RuntimeError:
                        U, S, Vh = torch.linalg.svd(W.cpu(), full_matrices=False)
                        U = U.to(dev)
                        Vh = Vh.to(dev)

                    # If importance missing, treat as 0.5 (flat prior)
                    I_l = float(max(0.0, min(1.0, imp.get(name, 0.5))))
                    ratio = mrr + I_l * (trr - mrr)
                    k_max = int(min(W.shape))
                    k = int(max(0, min(int(torch.ceil(torch.tensor(ratio * k_max)).item()), k_max)))
                    if k == 0:
                        Uh = torch.zeros((W.shape[0], 0), device=dev, dtype=torch.float32)
                        Vh_t = torch.zeros((0, W.shape[1]), device=dev, dtype=torch.float32)
                        keep[full_name] = (Uh, Vh_t.transpose(0, 1))
                    else:
                        Uh = U[:, :k].contiguous().to(device=dev, dtype=torch.float32)
                        Vh_k = Vh[:k, :].contiguous().to(device=dev, dtype=torch.float32)
                        keep[full_name] = (Uh, Vh_k.transpose(0, 1))
    return keep


def run_osl_setup(model: nn.Module,
                  ds,
                  batch_size: int,
                  mrr: float,
                  trr: float,
                  calib_steps: int,
                  device: torch.device,
                  target_linear_only: bool = True) -> Optional[SVDProjector]:
    """
    Run calibration + build OSL subspaces + attach gradient projectors.
    Returns projector (must call .remove() after training shard).
    """
    if not torch.cuda.is_available():
        print("[OSL] CUDA not available; skipping.")
        return None

    # --- Step 1: Collect activations for importance ---
    imp: Dict[str, float] = {}
    if calib_steps > 0:
        tap = ActivationTap(model, target_linear_only=target_linear_only)
        from torch.utils.data import DataLoader
        calib_loader = DataLoader(
            ds, batch_size=min(4, batch_size), shuffle=False,
            collate_fn=custom_data_collator
        )

        model.eval()
        with torch.no_grad():
            for i, batch in enumerate(calib_loader):
                # Normalize batch to dict
                if isinstance(batch, (list, tuple)):
                    # assume (input_ids, labels) or (input_ids, attention_mask, labels)
                    if len(batch) == 2:
                        batch = {"input_ids": batch[0], "labels": batch[1]}
                    elif len(batch) == 3:
                        batch = {"input_ids": batch[0], "attention_mask": batch[1], "labels": batch[2]}
                    else:
                        raise ValueError(f"Unexpected batch tuple length: {len(batch)}")
                elif not isinstance(batch, dict):
                    raise ValueError(f"Unexpected batch type: {type(batch)}")

                # Move tensors to device
                for k, v in list(batch.items()):
                    if isinstance(v, torch.Tensor):
                        batch[k] = v.to(device, non_blocking=True)

                try:
                    _ = model(**batch)
                except Exception as e:
                    print(f"[OSL] Warning: calibration forward failed with {e}")
                    pass

                if i + 1 >= calib_steps:
                    break

        imp = tap.cosine_importance()
        tap.remove()

    # --- Step 2: Build subspaces ---
    keep = build_osl_subspaces_gpu(
        model, imp, mrr=mrr, trr=trr,
        target_linear_only=target_linear_only
    )

    # --- Step 3: Attach projectors ---
    projector = SVDProjector(keep)
    projector.register(model)
    model.train()

    return projector


# =========================
# Main
# =========================

@hydra.main(version_base=None, config_path="config", config_name="SISA_slices_ITr")
def main(cfg):
    start_time = time.time()
    set_seed(cfg.seed)

    # Enable fast matmul paths if available
    try:
        torch.backends.cuda.matmul.allow_tf32 = True
    except Exception:
        pass
    try:
        torch.set_float32_matmul_precision("high")
    except Exception:
        pass

    # Model IDs
    model_cfg = get_model_identifiers_from_yaml(cfg.model_family)
    model_id = model_cfg["hf_key"]

    # Save config snapshot
    Path(cfg.save_dir).mkdir(parents=True, exist_ok=True)
    if os.environ.get('LOCAL_RANK') is None or int(os.environ.get('LOCAL_RANK', '0')) == 0:
        with open(f'{cfg.save_dir}/cfg.yaml', 'w') as f:
            OmegaConf.save(cfg, f)

    # Tokenizer
    tokenizer = AutoTokenizer.from_pretrained(model_id)
    tokenizer.pad_token = tokenizer.eos_token

    # Data paths & splits (from YAML)
    data_paths = list(cfg.data_paths)
    splits = list(cfg.splits)

    # Train settings
    max_length = 500
    batch_size = cfg.batch_size
    gradient_accumulation_steps = cfg.gradient_accumulation_steps
    num_devices = int(os.environ.get('WORLD_SIZE', 1))
    print(f"num_devices: {num_devices}")

    # Model (bf16)
    model = AutoModelForCausalLM.from_pretrained(
        model_id, use_flash_attention_2=False, torch_dtype=torch.bfloat16, trust_remote_code=True
    )
    model.generation_config.do_sample = True
    if model_cfg.get("gradient_checkpointing", "false") == "true":
        model.gradient_checkpointing_enable()

    # OSL settings (from cfg.continual.osl)
    osl_cfg = getattr(cfg, "continual", {}).get("osl", {})
    osl_mrr = float(osl_cfg.get("mrr", 0.10))
    osl_trr = float(osl_cfg.get("trr", 0.85))
    osl_calib_steps = int(osl_cfg.get("cali_batches", 0))
    osl_target_linear_only = bool(osl_cfg.get("target_linear_only", True))

    try:
        device = next(model.parameters()).device
    except StopIteration:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # ===== Train over shards =====
    for di, (dpath, dsplit) in enumerate(zip(data_paths, splits), start=1):
        print(f"\n========== Dataset {di}/{len(data_paths)} ==========")
        print(f"Path: {dpath} | Split: {dsplit}")

        ds = TextDatasetQA(
            dpath, tokenizer=tokenizer, model_family=cfg.model_family, max_length=max_length, split=dsplit
        )

        max_steps = int(cfg.num_epochs * len(ds)) // max(1, (batch_size * gradient_accumulation_steps * num_devices))
        print(f"max_steps (dataset {di}): {max_steps}")

        training_args = transformers.TrainingArguments(
            per_device_train_batch_size=batch_size,
            per_device_eval_batch_size=batch_size,
            gradient_accumulation_steps=gradient_accumulation_steps,
            warmup_steps=max(1, max_steps // max(1, int(getattr(cfg, 'num_epochs', 1)))),
            max_steps=max_steps,
            learning_rate=cfg.lr,
            bf16=True,
            bf16_full_eval=True,
            logging_strategy="steps",
            logging_steps=max(1, max_steps // 100) if max_steps > 0 else 1,
            output_dir=cfg.save_dir,
            save_strategy="steps",
            save_steps=max_steps if max_steps > 0 else 1,
            save_only_model=True,
            ddp_find_unused_parameters=False,
            eval_strategy="no",
            weight_decay=cfg.weight_decay,
            seed=cfg.seed,
        )

        # ---- OSL: per-shard setup (build subspaces & register hooks) ----
        print(f"[OSL] Building protected subspaces for shard {dsplit or dpath} "
              f"(mrr={osl_mrr}, trr={osl_trr}, cali_batches={osl_calib_steps}, "
              f"linear_only={osl_target_linear_only})")
        projector = run_osl_setup(
            model, ds, batch_size,
            mrr=osl_mrr, trr=osl_trr,
            calib_steps=osl_calib_steps,
            device=device,
            target_linear_only=osl_target_linear_only
        )

        # Train
        trainer = CustomTrainer(
            model=model,
            train_dataset=ds,
            eval_dataset=ds,
            args=training_args,
            data_collator=custom_data_collator,
            callbacks=[HeartbeatCallback(every=max(1, max_steps // 100))],
        )
        model.config.use_cache = False
        trainer.train()

        # Detach projectors (we rebuild for the next shard)
        if projector is not None:
            projector.remove()

        # Save milestone
        milestone_meta = {
            "model_family": cfg.model_family,
            "num_epochs": cfg.num_epochs,
            "batch_size": batch_size,
            "gradient_accumulation_steps": gradient_accumulation_steps,
            "learning_rate": cfg.lr,
            "weight_decay": cfg.weight_decay,
            "seed": cfg.seed,
            "osl_mrr": osl_mrr,
            "osl_trr": osl_trr,
            "osl_cali_batches": osl_calib_steps,
            "osl_target_linear_only": osl_target_linear_only,
        }
        save_milestone(model, tokenizer, cfg.save_dir, di, dpath, dsplit, milestone_meta, start_time)

        # Memory hygiene
        torch.cuda.empty_cache()
        del ds
        torch.cuda.empty_cache()

    print("\n[Done] Training across slices.")


if __name__ == "__main__":
    main()
