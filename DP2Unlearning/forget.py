from data_module import TextForgetDatasetQA, TextForgetDatasetDPOQA
from dataloader import CustomTrainerForgetting, custom_data_collator_forget
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM, AutoConfig, set_seed

import hydra
import transformers
import os
from peft import LoraConfig, get_peft_model, PeftModel
from pathlib import Path
from utils import get_model_identifiers_from_yaml
from omegaconf import OmegaConf
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
    """Prints the number of trainable parameters in the model."""
    trainable_params = 0
    all_param = 0
    for _, param in model.named_parameters():
        all_param += param.numel()
        if param.requires_grad:
            trainable_params += param.numel()
    print(
        f"trainable params: {trainable_params} || all params: {all_param} || trainable%: {100 * trainable_params / all_param}"
    )

@hydra.main(version_base=None, config_path="config", config_name="forget")
def main(cfg):
    num_devices = int(os.environ.get('WORLD_SIZE', 1))
    print(f"num_devices: {num_devices}")

    local_rank = int(os.environ.get('LOCAL_RANK', '0'))  # Default to 0 if LOCAL_RANK is not set
    device_map = {'': local_rank}

    set_seed(cfg.seed)

    os.environ["WANDB_DISABLED"] = "true"
    model_cfg = get_model_identifiers_from_yaml(cfg.model_family)
    model_id = model_cfg["hf_key"]
    if cfg.model_path is None:
        cfg.model_path = model_cfg["ft_model_path"]

    print("######################")
    print("Saving to: ", cfg.save_dir)
    print("######################")
    
    # Save configuration in cfg.save_dir
    if local_rank == 0:
        if os.path.exists(cfg.save_dir):
            print("Directory already exists")
            if not cfg.overwrite_dir:
                exit()

        Path(cfg.save_dir).mkdir(parents=True, exist_ok=True)

        with open(f"{cfg.save_dir}/config.yaml", "w") as file:
            OmegaConf.save(cfg, file)

    tokenizer = AutoTokenizer.from_pretrained(model_id)
    tokenizer.pad_token = tokenizer.eos_token

    max_length = 500
    if cfg.forget_loss == "dpo":
        torch_format_dataset = TextForgetDatasetDPOQA(cfg.data_path, tokenizer=tokenizer, model_family=cfg.model_family, max_length=max_length, split=cfg.split)
    else:
        torch_format_dataset = TextForgetDatasetQA(cfg.data_path, tokenizer=tokenizer, model_family=cfg.model_family, max_length=max_length, split=cfg.split, loss_type=cfg.forget_loss)
    
    batch_size = cfg.batch_size
    gradient_accumulation_steps = cfg.gradient_accumulation_steps
    steps_per_epoch = len(torch_format_dataset) // (batch_size * gradient_accumulation_steps * num_devices)
    import math
    max_steps = math.ceil(cfg.num_epochs * len(torch_format_dataset) / (batch_size * gradient_accumulation_steps * num_devices))
    print(f"max_steps: {max_steps}")

    training_args = transformers.TrainingArguments(
            per_device_train_batch_size=batch_size,
            per_device_eval_batch_size=batch_size,
            gradient_accumulation_steps=gradient_accumulation_steps,
            warmup_steps=max(1, steps_per_epoch),
            max_steps=max_steps,
            learning_rate=cfg.lr,
            bf16=True,
            bf16_full_eval=True,
            logging_steps=max(1, max_steps // 20),
            logging_dir=f'{cfg.save_dir}/logs',
            output_dir=cfg.save_dir,
            optim="paged_adamw_32bit",
            save_strategy="steps" if cfg.save_model and (not cfg.eval_only) else "no",
            save_steps=steps_per_epoch,
            save_only_model=True,
            ddp_find_unused_parameters=False,
            weight_decay=cfg.weight_decay,
            eval_steps=steps_per_epoch,
            evaluation_strategy="steps" if cfg.eval_while_train else "no",
            seed=cfg.seed
    )

    # Checking model path for checkpoint files
    path_found = False
    for file in os.listdir(cfg.model_path):
        if re.search(r"pytorch.*\.bin", file):
            path_found = True
            break
        if re.search(r"model.*\.safetensors", file):
            path_found = True
            break

    oracle_model = None
    if path_found:
        config = AutoConfig.from_pretrained(model_id)
        print("Loading from checkpoint")
        model = AutoModelForCausalLM.from_pretrained(cfg.model_path, config=config, use_flash_attention_2=(model_cfg["flash_attention2"]=="true"), torch_dtype=torch.bfloat16, trust_remote_code=True)
        if cfg.forget_loss == "KL":
            oracle_model = AutoModelForCausalLM.from_pretrained(cfg.model_path, config=config, use_flash_attention_2=(model_cfg["flash_attention2"]=="true"), torch_dtype=torch.bfloat16, trust_remote_code=True)
    else:
        print("Loading after merge and unload")
        model = AutoModelForCausalLM.from_pretrained(model_id, use_flash_attention_2=(model_cfg["flash_attention2"]=="true"), torch_dtype=torch.bfloat16, device_map=device_map)
        model = PeftModel.from_pretrained(model, model_id=cfg.model_path)
        model = model.merge_and_unload()
        model.save_pretrained(cfg.model_path)
    
    model.generation_config.do_sample = True

    if model_cfg["gradient_checkpointing"] == "true":
        model.gradient_checkpointing_enable()
    config = LoraConfig(
        r=cfg.LoRA.r, 
        lora_alpha=cfg.LoRA.alpha, 
        target_modules=find_all_linear_names(model), 
        lora_dropout=cfg.LoRA.dropout,
        bias="none", 
        task_type="CAUSAL_LM"
    )
    if cfg.LoRA.r != 0:
        model = get_peft_model(model, config)
        print_trainable_parameters(model)

    # # trainer = CustomTrainerForgetting(
    # #     model=model,
    # #     tokenizer=tokenizer,
    # #     train_dataset=torch_format_dataset,
    # #     eval_dataset=torch_format_dataset,
    # #     compute_metrics=None,
    # #     args=training_args,
    # #     data_collator=custom_data_collator_forget,
    # #     oracle_model=oracle_model,
    # #     forget_loss=cfg.forget_loss,
    # #     eval_cfg=cfg.eval,
    # )
    trainer = CustomTrainerForgetting(
        model=model,
        tokenizer=tokenizer,
        train_dataset=torch_format_dataset,
        eval_dataset=torch_format_dataset,
        compute_metrics=None,
        args=training_args,
        data_collator=custom_data_collator_forget,
        forget_loss=cfg.forget_loss,
        eval_cfg=cfg.eval,
    )

    model.config.use_cache = False  # silence the warnings. Re-enable for inference if needed

    if cfg.eval_only:
        trainer.evaluate()
    else:
        trainer.train()

    if cfg.save_model and (not cfg.eval_only):
        model.save_pretrained(cfg.save_dir)
        tokenizer.save_pretrained(cfg.save_dir)

    if local_rank == 0:
        for file in Path(cfg.save_dir).glob("checkpoint-*"):
            for global_step_dir in file.glob("global_step*"):
                import shutil
                shutil.rmtree(global_step_dir)

if __name__ == "__main__":
    main()
