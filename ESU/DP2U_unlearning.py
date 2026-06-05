from data_module import TextDatasetQA, custom_data_collator
from dataloader import CustomTrainer
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM, AutoConfig, set_seed
import gc
import hydra 
import transformers
import os
from pathlib import Path
from omegaconf import OmegaConf
from utils import get_model_identifiers_from_yaml
import time
def find_all_linear_names(model):
    cls = torch.nn.Linear
    lora_module_names = set()
    for name, module in model.named_modules():
        if isinstance(module, cls):
            names = name.split('.')
            lora_module_names.add(names[0] if len(names) == 1 else names[-1])
    if 'lm_head' in lora_module_names: 
        lora_module_names.remove('lm_head')
    return list(lora_module_names)


def print_trainable_parameters(model):
    """
    Prints the number of trainable parameters in the model.
    """
    trainable_params = 0
    all_param = 0
    for _, param in model.named_parameters():
        all_param += param.numel()
        if param.requires_grad:
            trainable_params += param.numel()
    print(
        f"trainable params: {trainable_params} || all params: {all_param} || trainable%: {100 * trainable_params / all_param}"
    )

@hydra.main(version_base=None, config_path="config", config_name="DP2U_unlearning")
def main(cfg):
    start_time = time.time()

    if os.environ.get('LOCAL_RANK') is not None:
        local_rank = int(os.environ.get('LOCAL_RANK', '0'))
        device_map = {'': local_rank}
    set_seed(cfg.seed)
    os.environ["WANDB_DISABLED"] = "true"
    model_cfg = get_model_identifiers_from_yaml(cfg.model_family)
    model_id = model_cfg["hf_key"]
    if cfg.resume_from_base_model is not None and len(cfg.resume_from_base_model) > 0:
        model_path = cfg.resume_from_base_model
        print(f"Resuming from base model checkpoint: {model_path}")
    else:
        model_path = model_id
        print(f"Training from HF base model: {model_path}")

    Path(cfg.save_dir).mkdir(parents=True, exist_ok=True)
    if os.environ.get('LOCAL_RANK') is None or local_rank == 0:
        with open(f'{cfg.save_dir}/cfg.yaml', 'w') as f:
            OmegaConf.save(cfg, f)

    tokenizer = AutoTokenizer.from_pretrained(
        model_path,
        trust_remote_code=True
    )
    tokenizer.pad_token = tokenizer.eos_token

    max_length = 350
    torch_format_dataset = TextDatasetQA(cfg.data_path, tokenizer=tokenizer, model_family = cfg.model_family, max_length=max_length, split=cfg.split)

    batch_size = cfg.batch_size
    gradient_accumulation_steps = cfg.gradient_accumulation_steps
    num_devices = int(os.environ.get('WORLD_SIZE', 1))
    print(f"num_devices: {num_devices}")

    
    max_steps = int(cfg.num_epochs*len(torch_format_dataset))//(batch_size*gradient_accumulation_steps*num_devices)
    print(f"max_steps: {max_steps}")
    training_args = transformers.TrainingArguments(
            per_device_train_batch_size=batch_size,
            per_device_eval_batch_size=batch_size,
            gradient_accumulation_steps=gradient_accumulation_steps,
            warmup_steps=max(1, max_steps//cfg.num_epochs),
            max_steps=max_steps,
            learning_rate=cfg.lr,
            bf16=True,
            bf16_full_eval=True,
            logging_steps=max(1,max_steps//20),
            logging_dir=f'{cfg.save_dir}/logs',
            output_dir=cfg.save_dir,
            optim="paged_adamw_32bit",
            save_steps=max_steps,
            save_only_model=True,
            ddp_find_unused_parameters= False,
            eval_strategy="no",
            weight_decay = cfg.weight_decay,
            seed = cfg.seed,
        )

    model = AutoModelForCausalLM.from_pretrained(
        model_path,
        torch_dtype=torch.bfloat16,
        trust_remote_code=True
    )

    model.generation_config.do_sample = True

    if model_cfg["gradient_checkpointing"] == "true":
        model.gradient_checkpointing_enable()

    trainer = CustomTrainer(
        model=model,
        train_dataset=torch_format_dataset,
        eval_dataset=torch_format_dataset,
        args=training_args,
        data_collator=custom_data_collator,
    )
    model.config.use_cache = False  
    trainer.train()

    model.save_pretrained(cfg.save_dir)
    tokenizer.save_pretrained(cfg.save_dir)

    end_time = time.time()
    elapsed_time = end_time - start_time
    print(f"Total running time: {elapsed_time:.2f} seconds")

    with open(os.path.join(cfg.save_dir, 'running_time.txt'), 'w') as f:
        f.write(f"Total running time: {elapsed_time:.2f} seconds\n")

if __name__ == "__main__":
    main()
