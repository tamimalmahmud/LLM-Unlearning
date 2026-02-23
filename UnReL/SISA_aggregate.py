import os
from pathlib import Path
import torch
from transformers import AutoModelForCausalLM
from omegaconf import OmegaConf
import time


def average_models(model_paths, save_path, dtype=torch.bfloat16):
    assert len(model_paths) >= 2, "Need at least two model paths to merge."

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Using device: {device}")

    # Load the first model to initialize structure
    print(f"Loading model 1 from {model_paths[0]}")
    model = AutoModelForCausalLM.from_pretrained(model_paths[0], torch_dtype=dtype).to(device)
    avg_state_dict = {k: v.to(torch.float32).clone() for k, v in model.state_dict().items()}

    # Load remaining models and sum their weights
    for idx, model_path in enumerate(model_paths[1:], start=2):
        print(f"Loading model {idx} from {model_path}")
        model_i = AutoModelForCausalLM.from_pretrained(model_path, torch_dtype=dtype).to(device)
        state_dict_i = model_i.state_dict()

        for k in avg_state_dict:
            avg_state_dict[k] += state_dict_i[k].to(torch.float32)

    #Equal-weight averaging
    num_models = len(model_paths)
    for k in avg_state_dict:
        avg_state_dict[k] /= num_models

    # Load averaged weights into a fresh model
    print("Loading averaged weights...")
    merged_model = AutoModelForCausalLM.from_pretrained(model_paths[0], torch_dtype=dtype).to(device)
    merged_model.load_state_dict(avg_state_dict)

    # Save merged model
    print(f"Saving merged model to {save_path}")
    Path(save_path).mkdir(parents=True, exist_ok=True)
    merged_model.save_pretrained(save_path)
    print("Merge complete.")


if __name__ == "__main__":
    start_time = time.time()
    config_path = "config/SISA_aggregate.yaml"
    cfg = OmegaConf.load(config_path)

    model_paths = cfg.sharded_model_paths
    save_path = cfg.merged_save_path

    average_models(model_paths, save_path)

    # Save the cfg.yaml file in the merged directory
    OmegaConf.save(config=cfg, f=os.path.join(save_path, "cfg.yaml"))

    end_time = time.time()
    elapsed_time = end_time - start_time
    print(f"Total running time: {elapsed_time:.2f} seconds")

    with open(os.path.join(save_path, 'running_time.txt'), 'w') as f:
        f.write(f"Total running time: {elapsed_time:.2f} seconds\n")
