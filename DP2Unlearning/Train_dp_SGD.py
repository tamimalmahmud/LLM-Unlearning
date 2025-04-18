import torch
import hydra
from pathlib import Path
import torch.optim as optim
from torch.utils.data import DataLoader
from transformers import AutoTokenizer, AutoModelForCausalLM
from opacus import PrivacyEngine
from omegaconf import OmegaConf
from tqdm import tqdm
import yaml
import os
from opacus.utils.batch_memory_manager import BatchMemoryManager
from opacus.utils.uniform_sampler import UniformWithReplacementSampler  # For Poisson sampling
from data_module import TextDatasetQA, custom_data_collator

# Set seed function directly in this script
def set_seed(seed):
    import random
    import numpy as np
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

# Function to load model identifiers from model_config.yaml
def get_model_identifiers_from_yaml(model_family):
    """
    Load model identifiers from config/model_config.yaml based on the specified model family.
    """
    config_path = os.path.join(os.getcwd(), "config", "model_config.yaml")
    if not os.path.exists(config_path):
        raise FileNotFoundError(f"model_config.yaml not found at: {config_path}")
    
    with open(config_path, "r") as file:
        model_configs = yaml.safe_load(file)
    
    if model_family not in model_configs:
        raise ValueError(f"Model family '{model_family}' not found in model_config.yaml")
    
    return model_configs[model_family]

# Create Poisson Sampling DataLoader
def get_dataloader(dataset, batch_size, sample_rate):
    sampler = UniformWithReplacementSampler(
        num_samples=len(dataset), 
        sample_rate=sample_rate  # Determines the probability of selecting each sample
    )
    
    return DataLoader(
        dataset, 
        batch_sampler=sampler,  # Use Poisson sampling with the custom sampler
        collate_fn=custom_data_collator, 
    )

@hydra.main(version_base=None, config_path="config", config_name="finetune_dp_opa.yaml")
def main(cfg):
    # Device setup
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    
    # Set seed for reproducibility
    set_seed(cfg.seed)
    
    # Load model identifiers from model_config.yaml
    model_cfg = get_model_identifiers_from_yaml(cfg.model_family)
    model_family = cfg.model_family

    # Dynamically construct the save directory path
    save_dir = cfg.save_dir.format(
        num_epochs=cfg.num_epochs,
        lr=cfg.lr,
        model_family=model_family,  
        split=cfg.split,
        target_epsilon=cfg.target_epsilon,
        target_delta=cfg.target_delta
    )
    Path(save_dir).mkdir(parents=True, exist_ok=True)

    # Load tokenizer and model
    tokenizer = AutoTokenizer.from_pretrained(model_cfg["hf_key"])
    tokenizer.pad_token = tokenizer.eos_token
    model = AutoModelForCausalLM.from_pretrained(model_cfg["hf_key"]).to(device)
    model.train()
    max_length=150
    # Load dataset using TextDatasetQA and DataLoader
    dataset = TextDatasetQA(cfg.data_path, tokenizer, cfg.model_family, max_length=max_length, split=cfg.split)
    
    # Poisson Sampling: Create data loader using Poisson sampling method
    sample_rate = cfg.batch_size / len(dataset)  # Sample rate = batch size / dataset size
    data_loader = get_dataloader(dataset, cfg.batch_size, sample_rate)

    # Optimizer with weight decay applied from config
    optimizer = optim.SGD(model.parameters(), lr=cfg.lr, weight_decay=cfg.weight_decay)

    # Privacy Engine
    privacy_engine = PrivacyEngine()
    model, optimizer, data_loader = privacy_engine.make_private_with_epsilon(
        module=model,
        optimizer=optimizer,
        data_loader=data_loader,
        epochs=cfg.num_epochs,
        target_epsilon=cfg.target_epsilon,
        target_delta=cfg.target_delta,
        max_grad_norm=cfg.max_grad_norm,
    )

    # Training loop using BatchMemoryManager and clearing unused variables
    def train(model, data_loader, optimizer, epochs=cfg.num_epochs):
        model.train()
        for epoch in range(epochs):
            running_loss = 0.0
            optimizer.zero_grad()
            print(f"Epoch [{epoch+1}/{epochs}]")
            num_batches = 0
            with BatchMemoryManager(
                data_loader=data_loader, 
                max_physical_batch_size=cfg.batch_size,  # Adjust based on your memory capacity
                optimizer=optimizer
            ) as memory_safe_data_loader:
                for step, batch in tqdm(
                    enumerate(memory_safe_data_loader), 
                    total=len(memory_safe_data_loader), 
                    desc="Training Progress"
                ):
                    inputs, labels, masks = batch[:3]  # Safely unpack the first 3 elements                    
                    inputs = inputs.to(device)
                    labels = labels.to(device)
                    masks = masks.to(device)
                    if inputs.size(0) == 0 or masks.sum() == 0:
                        continue
                    outputs = model(input_ids=inputs, attention_mask=masks, labels=labels)
                    loss = outputs.loss
                    loss.backward()
                    # Optimizer step without gradient accumulation
                    optimizer.step()
                    optimizer.zero_grad()
                    running_loss += loss.item()
                    num_batches += 1  # Increment batch count
                    if step % 100 == 0:  # Log every 100 steps
                        print(f"Step {step}: Loss = {loss.item()}")

                    # Clear unused variables to free up memory
                    del inputs, labels, masks, outputs, loss
                    torch.cuda.empty_cache()
                
            # Calculate average loss for the epoch
            average_loss = running_loss / num_batches
            epsilon = privacy_engine.get_epsilon(delta=cfg.target_delta)
            print(f"Epoch {epoch+1} completed. Total Loss: {running_loss:.4f}, Average Loss: {average_loss:.4f}, Privacy budget (epsilon): {epsilon:.4f}")
    
    # Train the model
    train(model, data_loader, optimizer, epochs=cfg.num_epochs)
    
    # Unwrap the model before saving
    model_to_save = model._module if hasattr(model, '_module') else model
    model_to_save.save_pretrained(save_dir)
    tokenizer.save_pretrained(save_dir)

    # Save the configuration
    with open(f'{save_dir}/cfg.yaml', 'w') as f:
        OmegaConf.save(cfg, f)

if __name__ == "__main__":
    main()
