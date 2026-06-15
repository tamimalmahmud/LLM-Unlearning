# ESU: Efficient Selective Unlearning with Privacy Guarantees for LLMs

This repository contains the reproducibility code for **ESU**, an efficient selective unlearning framework for large language models. ESU is designed to reduce the cost of machine unlearning requests while preserving model utility and providing disclosure protection for removal-prone or sensitive data.

**Paper:** *ESU: Efficient Selective Unlearning with Privacy Guarantees for Large Language Models*  
**Authors:** Tamim Al Mahmud, David Sánchez, and Josep Domingo-Ferrer

---

## Overview

Large language models can memorize and reproduce parts of their training data. This creates privacy, copyright, and compliance concerns when a user or data owner requests that specific information be removed from a trained model. Full retraining on retained data is a strong reference baseline, but it is expensive for LLMs.

ESU addresses this problem through three connected stages:

1. **P1: Unlearning-aware pretraining**  
   Train a disclosure-protected base model using DP-protected non-public data.

2. **P2: Shard-based fine-tuning**  
   Fine-tune the base model over ordered shards and save cumulative checkpoints.

3. **P3: Selective unlearning**  
   Roll back to the checkpoint before the earliest affected shard, remove forgotten samples, and fine-tune only on retained data from the affected and later shards.

This repository also includes baseline implementations for DP2U, SISA, APA, and approximate unlearning methods such as GA, GD, and KL.

---

## Main Features

- DP-based disclosure protection using masked language model based rewriting
- Heuristic AutoScoring for estimating future unlearning likelihood
- Risk-aware shard assignment for ESU
- Checkpointed shard-wise fine-tuning
- Rollback-based selective unlearning
- Baselines for exact, approximate, and privacy-guaranteed unlearning
- TOFU-style model utility and forget quality evaluation
- Additional distributional metrics including Jensen-Shannon Divergence and entropy difference

---

## Project Structure

```text
ESU/
├── DP2UMLM.py                 # DP-MLM style text perturbation
├── ESU_P1.py                  # P1: disclosure-protected base model training
├── ESU_P2.py                  # P2: shard-wise full-data fine-tuning with checkpoints
├── ESU_P3.py                  # P3: rollback-based selective unlearning
├── DP2U_training.py           # DP2U protected training baseline
├── DP2U_unlearning.py         # DP2U unlearning baseline
├── autoscoring.py             # ESU likelihood scoring and shard assignment
├── finetune.py                # retraining or vanilla fine-tuning baseline
├── forget.py                  # approximate unlearning baselines: GA, GD, KL, etc.
├── evaluate_util.py           # evaluation for utility and forgetting metrics
├── aggregate_eval_stat.py     # result aggregation and distributional metrics
├── APA_training.py            # APA baseline training
├── APA_aggregate.py           # APA adapter aggregation
├── SISA_P50_training.py       # SISA public shard training
├── SISA_slices_ITr.py         # SISA shard and slice training
├── SISA_aggregate.py          # SISA model aggregation
├── SISA_unlearning.py         # SISA unlearning
├── data_module.py             # dataset processing utilities
├── dataloader.py              # custom trainer and training helpers
├── utils.py                   # shared utility functions
└── config/
    ├── ESU_P1.yaml
    ├── ESU_P2.yaml
    ├── ESU_P3.yaml
    ├── DP2U_training.yaml
    ├── DP2U_unlearning.yaml
    ├── finetune.yaml
    ├── forget.yaml
    ├── evaluate_util.yaml
    ├── aggregate_eval_stat.yaml
    ├── APA_training.yaml
    ├── APA_aggregate.yaml
    ├── SISA_P50_training.yaml
    ├── SISA_slices_ITr.yaml
    ├── SISA_unlearning.yaml
    ├── SISA_aggregate.yaml
    └── model_config.yaml
```

---

## Installation

Create a Conda environment with PyTorch and CUDA support:

```bash
conda create --name torch-env pytorch torchvision pytorch-cuda=12.1 -c pytorch -c nvidia
conda activate torch-env
```

Install the required packages:

```bash
pip install datasets accelerate evaluate matplotlib hydra-core omegaconf peft rouge_score tqdm einops packaging bitsandbytes scipy ninja
pip install pandas sentence-transformers nltk spacy natsort pyyaml
python -m spacy download en_core_web_sm
```

Depending on your CUDA version, GPU memory, and model size, you may need to update some libraries.

---

## Dataset and Models

The default configuration files use:

- **Dataset:** `talmahmud/tofu_custom_split_ESU`
- **Models:**
  - `phi`: `microsoft/phi-1_5`
  - `qwen3-4b`: `Qwen/Qwen3-4B-Instruct-2507`

Model identifiers are defined in:

```text
config/model_config.yaml
```

To change the model, update the YAML file or override the Hydra argument:

```bash
python ESU_P1.py model_family=phi
```

---

## Important Configuration Notes

Before running the project, update the paths in the YAML files according to your machine:

- `data_path`
- `split`
- `save_dir`
- `resume_from`
- `resume_from_base_model`
- `model_path`
- `retain_result`
- `ckpt_result`

Most current configs contain absolute paths from the original experiment environment, such as:

```text
/data/urveim/talmahmud/ESU/...
```

Replace these with your own local paths before running.

---

## Quick Start

Run commands from the repository root:

```bash
cd ESU
```

Hydra already loads configs from the `config/` directory. You can run scripts directly:

```bash
python ESU_P1.py
```

Or explicitly pass the config path and config name:

```bash
python ESU_P1.py --config-path config --config-name ESU_P1
```

You can also override config values from the command line:

```bash
python ESU_P1.py model_family=phi num_epochs=10 batch_size=4
```

---

## Step 1: Traditional Retraining or Vanilla Fine-Tuning Baseline

Use `finetune.py` to train the reference model. Depending on the selected split, this can be used for full-data fine-tuning or retain-only retraining.

```bash
python finetune.py --config-path config --config-name finetune
```

Default config:

```text
config/finetune.yaml
```

Update `split` depending on the baseline you want to train, for example:

```yaml
split: full
```

or for retain-only training:

```yaml
split: retain180
```

---

## Step 2: DP-MLM Perturbation

`DP2UMLM.py` applies masked language model based DP-style perturbation to sensitive tokens in selected datasets.

```bash
python DP2UMLM.py
```

Before running, check and update inside the script:

```python
dataset_config = "shard_R50"
EPSILON = 1.0
```

The script saves the perturbed data under:

```text
dp_data/
```

---

## Step 3: AutoScoring and Shard Assignment

Run the AutoScoring module to estimate the likelihood of future unlearning requests and assign data into ESU shards.

```bash
python autoscoring.py
```

By default, it reads local files from:

```text
./CalculateL_x/
```

The scorer combines:

- semantic sensitivity using `sentence-transformers/all-MiniLM-L6-v2`
- explicit PII-like signals such as emails, phone numbers, URLs, IBANs, IDs, and card-like numbers
- temporal exposure using an exponential decay half-life

It prints ESU shard assignments for two-shard and four-shard settings.

---

## Step 4: ESU P1 - Base Model Training

P1 trains the base model using the public and DP-protected data together.

```bash
python ESU_P1.py --config-path config --config-name ESU_P1
```

Default config:

```text
config/ESU_P1.yaml
```

Important fields:

```yaml
model_family: phi/qwen3-4b
data_path: talmahmud/tofu_custom_split_ESU
split: shard_P50R50DP_EP1
num_epochs: 10/8
lr: 5e-5
```

The output is a base model checkpoint.

---

## Step 5: ESU P2 - Shard-Based Full-Data Fine-Tuning

P2 fine-tunes the base model over ordered shards and stores cumulative checkpoints.

```bash
python ESU_P2.py --config-path config --config-name ESU_P2
```

Default config:

```text
config/ESU_P2.yaml
```

Important fields:

```yaml
num_shards: 2/4
resume_from: /path/to/P1_base_model
data_paths:
  - talmahmud/tofu_custom_split_ESU
splits:
  - shard_P50
  - shard2_1/shard4_1
  - shard2_2/shard4_2
  - etc.
num_epochs: 5/4
```

P2 saves intermediate checkpoints after each shard. The final checkpoint is the full-data model prepared for efficient future unlearning.

---

## Step 6: ESU P3 - Selective Unlearning

P3 performs rollback-based selective unlearning for a given forget request.

```bash
python ESU_P3.py --config-path config --config-name ESU_P3
```

Default config:

```text
config/ESU_P3.yaml
```

Important fields:

```yaml
num_shards: 2/4
resume_from: /path/to/checkpoint_before_earliest_affected_shard
splits:
  - retain_shard2_1_forget02
  - retain_shard2_1_forget05
  - retain_shard2_1_forget10
  - retain_shard2_2_forget05
  - retain_shard2_2_forget10  
  - retain_shard4_1_forget05
  - retain_shard4_2_forget05
  - retain_shard4_3_forget05
  - retain_shard4_4_forget05
  - etc.
forget: forget02/forget10/forget20
num_epochs: 5/4
```

After unlearning, the model is saved to the configured `save_dir`.

---

## DP2U Baseline

Train the DP2U disclosure-protected model:

```bash
python DP2U_training.py --config-path config --config-name DP2U_training
```

Run DP2U unlearning:

```bash
python DP2U_unlearning.py --config-path config --config-name DP2U_unlearning
```

Configs:

```text
config/DP2U_training.yaml
config/DP2U_unlearning.yaml
```

---

## SISA Baseline

Train the public shard model:

```bash
python SISA_P50_training.py --config-path config --config-name SISA_P50_training
```

Train SISA slices:

```bash
python SISA_slices_ITr.py --config-path config --config-name SISA_slices_ITr
```

Aggregate SISA models:

```bash
python SISA_aggregate.py --config-path config --config-name SISA_aggregate
```

Run SISA unlearning:

```bash
python SISA_unlearning.py --config-path config --config-name SISA_unlearning
```

---

## APA Baseline

Train APA adapters:

```bash
python APA_training.py --config-path config --config-name APA_training
```

Aggregate APA adapters:

```bash
python APA_aggregate.py --config-path config --config-name APA_aggregate.yaml
```

Update the adapter paths and adapter weights in:

```text
config/APA_aggregate.yaml
```

---

## Approximate Unlearning Baselines

Run approximate unlearning methods using:

```bash
python forget.py --config-path config --config-name forget
```

Default config:

```text
config/forget.yaml
```

Select the method with:

```yaml
forget_loss: GA
```

Depending on your implementation and config, this can be changed to methods such as `GA`, `GD`, or `KL`.

---

## Evaluation

Evaluate a trained or unlearned model with:

```bash
python evaluate_util.py --config-path config --config-name evaluate_util.yaml
```

Default config:

```text
config/evaluate_util.yaml
```

Important fields:

```yaml
model_path: /path/to/model
save_dir: ${model_path}/eval_results/ds_size${ds_size}
data_path:
  - talmahmud/tofu_custom_split_ESU
split_list:
  - retain_perturbed
  - real_authors_perturbed
  - world_facts_perturbed
  - ${split}
ds_size: 400
```

The evaluation computes metrics such as:

- ROUGE-L
- probability based metrics
- truth ratio
- retain performance
- forget-set behavior

---

## Aggregating Evaluation Results

Aggregate evaluation statistics with:

```bash
python aggregate_eval_stat.py --config-path config --config-name aggregate_eval_stat
```

Default config:

```text
config/aggregate_eval_stat.yaml
```

Set the following paths before running:

```yaml
retain_result: /path/to/retain_reference/eval_log_aggregated.json
ckpt_result: /path/to/unlearned_model/eval_log_aggregated.json
save_file: results/output.csv
```

The aggregation script reports:

- Model Utility
- Forget Quality using KS-test p-value
- KS statistic
- Jensen-Shannon Divergence
- Entropy Difference

---

## Recommended End-to-End Running Order

A typical ESU run follows this order:

```bash
# 1. Optional: create DP-perturbed data
python DP2UMLM.py

# 2. Train P1 base model
python ESU_P1.py --config-path config --config-name ESU_P1

# 3. Score data and prepare shard assignment
python autoscoring.py

# 4. Train P2 full-data model with checkpoints
python ESU_P2.py --config-path config --config-name ESU_P2

# 5. Perform P3 selective unlearning
python ESU_P3.py --config-path config --config-name ESU_P3

# 6. Evaluate the unlearned model
python evaluate_util.py --config-path config --config-name evaluate_util.yaml

# 7. Aggregate metrics
python aggregate_eval_stat.py --config-path config --config-name aggregate_eval_stat
```

---

## Hardware Notes

The experiments described in the paper were run with high-memory GPU hardware. For smaller GPUs, consider:

- reducing `batch_size`
- increasing `gradient_accumulation_steps`
- lowering `num_epochs` for debugging
- using LoRA or other parameter-efficient fine-tuning
- enabling gradient checkpointing
- using smaller model families first

The default configs use:

```yaml
batch_size: 4
gradient_accumulation_steps: 4
lr: 5e-5
weight_decay: 0.01
seed: 42
```

---

## Expected Results

In the paper, ESU is evaluated against retraining from scratch, DP2U, SISA, APA, GA, GD, and KL. ESU provides a strong balance between model utility, forget quality, and unlearning runtime.

Example unlearning runtimes reported in the paper:

| Model | ESU, 2 shards | ESU, 4 shards |
|---|---:|---:|
| Phi | 3m14s | 1m40s |
| Qwen3 | 5m41s | 3m05s |

These results show that ESU can reduce request-time unlearning cost while maintaining competitive utility and meaningful forgetting behavior.

---


## Citation

Please update the venue, volume, issue, pages, and DOI after publication metadata becomes available.

```bibtex
@article{almahmud2026esu,
  title={ESU: Efficient Selective Unlearning with Privacy Guarantees for Large Language Models},
  author={Al Mahmud, Tamim and S{\'a}nchez, David and Domingo-Ferrer, Josep},
  journal={Under Review},
  year={2026},
  note={Manuscript}
}
```

---

## Acknowledgment
This project builds on LLM unlearning evaluation practices used in the TOFU benchmark.
