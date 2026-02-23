import os
import torch
import hydra
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import PeftModel, PeftConfig
from copy import deepcopy
from pathlib import Path
from omegaconf import OmegaConf, DictConfig
import shutil
from utils import get_model_identifiers_from_yaml  
import time


def load_lora_adapters(base_model_id, adapter_dirs):
    """
    Load all LoRA adapters independently to avoid duplicate PEFT warnings.
    """
    print("Loading base model tokenizer and structure...")
    tokenizer = AutoTokenizer.from_pretrained(base_model_id)

    adapters = []
    for i, adapter_dir in enumerate(adapter_dirs, start=1):
        print(f"[{i}/{len(adapter_dirs)}] Loading LoRA adapter from: {adapter_dir}")
        # Load a fresh base model for each adapter to avoid shared PEFT config
        base_tmp = AutoModelForCausalLM.from_pretrained(base_model_id)
        adapter = PeftModel.from_pretrained(base_tmp, adapter_dir)
        adapters.append(deepcopy(adapter))
        del base_tmp  # free memory between loads
    return adapters, tokenizer


def aggregate_lora_weights(adapters, weights=None):
    """
    Weighted averaging of LoRA adapter weights.
    """
    print("Aggregating LoRA weights across adapters...")

    if weights is None or len(weights) != len(adapters):
        print("[WARN] No valid adapter_weights found — using equal weighting.")
        weights = [1.0 / len(adapters)] * len(adapters)
    else:
        total = sum(weights)
        if total == 0:
            raise ValueError("Sum of adapter_weights cannot be zero.")
        weights = [w / total for w in weights]

    for i, w in enumerate(weights):
        print(f"  Adapter {i + 1} weight: {w:.4f}")

    ref_model = adapters[0]
    lora_keys = [
        name for name, _ in ref_model.named_parameters()
        if "lora_" in name and "weight" in name
    ]

    agg_state_dict = {}
    for key in lora_keys:
        weighted_sum = None
        for adapter, w in zip(adapters, weights):
            tensor = adapter.state_dict()[key].float()
            if weighted_sum is None:
                weighted_sum = tensor * w
            else:
                weighted_sum += tensor * w
        agg_state_dict[key] = weighted_sum
    return agg_state_dict


def save_aggregated_adapter(base_model_id, tokenizer, agg_weights, output_dir, config_template_dir):
    """
    Save merged LoRA adapter on top of base model.
    """
    print(f"Saving aggregated adapter to: {output_dir}")
    os.makedirs(output_dir, exist_ok=True)

    # Reload a clean base model to attach the aggregated LoRA weights
    base_model = AutoModelForCausalLM.from_pretrained(base_model_id)
    base_model.save_pretrained(output_dir)
    tokenizer.save_pretrained(output_dir)

    # Use the adapter config from first adapter as template
    adapter_config = PeftConfig.from_pretrained(config_template_dir)
    peft_model = PeftModel(base_model, adapter_config)

    missing, unexpected = peft_model.load_state_dict(agg_weights, strict=False)
    if missing:
        print(f"[INFO] Missing {len(missing)} non-LoRA keys (expected for delta-only merge).")
    if unexpected:
        print(f"[WARN] Unexpected keys: {unexpected}")

    # Save merged adapter
    peft_model.save_pretrained(output_dir)

    # Ensure adapter_config.json exists
    adapter_config_path = os.path.join(config_template_dir, "adapter_config.json")
    if os.path.exists(adapter_config_path):
        shutil.copyfile(adapter_config_path, os.path.join(output_dir, "adapter_config.json"))

    print("[INFO] Aggregated adapter saved successfully.")


@hydra.main(version_base=None, config_path="config", config_name="APA_aggregate.yaml")
def main(cfg: DictConfig):
    start_time = time.time()

    local_rank = 0
    if os.environ.get('LOCAL_RANK') is not None:
        local_rank = int(os.environ.get('LOCAL_RANK', '0'))
        device_map = {'': local_rank}

    os.environ["WANDB_DISABLED"] = "true"

    model_cfg = get_model_identifiers_from_yaml(cfg.model_family)
    model_id = model_cfg["hf_key"]
    base_model_id = model_id  
    save_dir = cfg.save_dir
    Path(save_dir).mkdir(parents=True, exist_ok=True)

    # Save config for reproducibility
    if local_rank == 0 or os.environ.get('LOCAL_RANK') is None:
        with open(f'{save_dir}/cfg.yaml', 'w') as f:
            OmegaConf.save(cfg, f)

    adapter_dirs = cfg.adapter_dirs
    adapter_weights = getattr(cfg, "adapter_weights", None)
    output_dir = cfg.save_dir

    adapters, tokenizer = load_lora_adapters(base_model_id, adapter_dirs)
    agg_weights = aggregate_lora_weights(adapters, adapter_weights)
    save_aggregated_adapter(base_model_id, tokenizer, agg_weights, output_dir, adapter_dirs[0])

    print("APA weighted merging completed successfully.")

    elapsed = time.time() - start_time
    print(f"Total running time: {elapsed:.2f} seconds")
    with open(os.path.join(save_dir, 'running_time.txt'), 'w') as f:
        f.write(f"Total running time: {elapsed:.2f} seconds\n")


if __name__ == "__main__":
    main()
