from data_module import TextDatasetQA, custom_data_collator
from dataloader import CustomTrainer
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM, set_seed
import hydra
import transformers
import os
from pathlib import Path
from omegaconf import OmegaConf
from utils import get_model_identifiers_from_yaml
import re
def find_all_linear_names(model):
    cls = torch.nn.Linear
    lora_module_names = set()
    for name, module in model.named_modules():
        if isinstance(module, cls):
            names = name.split('.')
            lora_module_names.add(names[0] if len(names) == 1 else names[-1])
    if 'lm_head' in lora_module_names:  # needed for 16-bit
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

@hydra.main(version_base=None, config_path="config", config_name="RT_DP_BaseModel.yaml")
def main(cfg):
    if os.environ.get('LOCAL_RANK') is not None:
        local_rank = int(os.environ.get('LOCAL_RANK', '0'))
        device_map = {'': local_rank}
    set_seed(cfg.seed)
    os.environ["WANDB_DISABLED"] = "true"

    # Extract base_model_name and eps from base_model_dir
    base_model_dir = cfg.base_model_dir
    match = re.search(r"checkpoints/(.*?)_Tr_.*_eps_(\d+\.\d+)", base_model_dir)  
    if match:
        base_model_name = match.group(1)  # Extract 'our_dp'
        eps = match.group(2)
    else:
        raise ValueError("Pattern not found in base_model_dir")

    # Dynamically update the save_dir with base_model_name and eps
    cfg.save_dir = f"checkpoints/UM_FT_{base_model_name}_on_{cfg.split}_epoch_{cfg.num_epochs}_lr_{cfg.lr}_{cfg.model_family}_wd_{cfg.weight_decay}_eps_{eps}"

    # Load the saved model from the directory (finetune_dp_opa.py output) instead of phi pretrained model
    save_model_dir = cfg.base_model_dir  # Directory where the model was saved from finetune_dp_opa.py
    
    tokenizer = AutoTokenizer.from_pretrained(save_model_dir)
    model = AutoModelForCausalLM.from_pretrained(save_model_dir).to("cuda" if torch.cuda.is_available() else "cpu")

    Path(cfg.save_dir).mkdir(parents=True, exist_ok=True)

    # Save the cfg file
    if os.environ.get('LOCAL_RANK') is None or local_rank == 0:
        with open(f'{cfg.save_dir}/cfg.yaml', 'w') as f:
            OmegaConf.save(cfg, f)

    # Prepare dataset
    max_length = 500
    torch_format_dataset = TextDatasetQA(cfg.data_path, tokenizer=tokenizer, model_family=cfg.model_family, max_length=max_length, split=cfg.split)

    # Training configurations
    batch_size = cfg.batch_size
    gradient_accumulation_steps = cfg.gradient_accumulation_steps
    num_devices = int(os.environ.get('WORLD_SIZE', 1))

    max_steps = int(cfg.num_epochs * len(torch_format_dataset)) // (batch_size * gradient_accumulation_steps * num_devices)
    print(f"max_steps: {max_steps}")

    training_args = transformers.TrainingArguments(
        per_device_train_batch_size=batch_size,
        per_device_eval_batch_size=batch_size,
        gradient_accumulation_steps=gradient_accumulation_steps,
        warmup_steps=max(1, max_steps // cfg.num_epochs),
        max_steps=max_steps,
        learning_rate=cfg.lr,
        bf16=True,
        bf16_full_eval=True,
        logging_steps=max(1, max_steps // 20),
        logging_dir=f'{cfg.save_dir}/logs',
        output_dir=cfg.save_dir,
        optim="paged_adamw_32bit",
        save_steps=max_steps,
        save_only_model=True,
        ddp_find_unused_parameters=False,
        evaluation_strategy="no",
        weight_decay=cfg.weight_decay,
        seed=cfg.seed,
    )

    # Hot fix for generation config
    model.generation_config.do_sample = True

    # LoRA fine-tuning (if applicable)
    if cfg.LoRA.r != 0:
        config = LoraConfig(
            r=cfg.LoRA.r,
            lora_alpha=cfg.LoRA.alpha,
            target_modules=find_all_linear_names(model),
            lora_dropout=cfg.LoRA.dropout,
            bias="none",
            task_type="CAUSAL_LM"
        )
        model = get_peft_model(model, config)
        model.enable_input_require_grads()

    # Prepare for training
    trainer = CustomTrainer(
        model=model,
        train_dataset=torch_format_dataset,
        eval_dataset=torch_format_dataset,
        args=training_args,
        data_collator=custom_data_collator,
    )

    model.config.use_cache = False  # Disable cache during training to avoid warnings

    # Train the model
    trainer.train()

    # Save the fine-tuned model
    if cfg.LoRA.r != 0:
        model = model.merge_and_unload()
    
    model.save_pretrained(cfg.save_dir)
    tokenizer.save_pretrained(cfg.save_dir)

if __name__ == "__main__":
    main()