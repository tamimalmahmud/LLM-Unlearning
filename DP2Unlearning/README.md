# DP2Unlearning

**Paper**: [DP2Unlearning: An Efficient and Guaranteed Unlearning Framework for Llms
](https://papers.ssrn.com/sol3/papers.cfm?abstract_id=5217160)

This section contains methods for training, fine-tuning, and evaluating LLMs in a privacy-preserving manner.

## Installation

To set up the environment for this project, follow these steps:

1. **Create a Conda Environment**:
    ```bash
    conda create --name torch-env pytorch torchvision pytorch-cuda=12.1 -c pytorch -c nvidia
    conda activate torch-env
    ```

2. **Install Required Libraries**:
    ```bash
    pip install datasets accelerate evaluate matplotlib hydra-core omegaconf peft rouge_score tqdm einops packaging bitsandbytes scipy ninja
    ```

## Traditional Retraining from Scratch (Benchmark retain model)

To perform traditional retraining from scratch, run the following command:

```bash
python finetune.py --config-path /home/user_name/project_name/config --config-name finetune.yaml
```
Do necessary modification in finetune.yaml file based on your hardware and GPU capacity.

## Unlearning Ready Training (Disclosure protected base model)

To train a disclosure-protected base model for unlearning, use one of the following commands:

```bash
python Train_dp_MLM.py --config-path /home/user_name/project_name/config --config-name Train_dp_MLM.yaml
```
or
```bash
python Train_dp_SGD.py --config-path /home/user_name/project_name/config --config-name Train_dp_SGD.yaml
```
Do necessary modification in Train_dp_MLM.yaml or Train_dp_SGD.yaml based on your hardware and GPU capacity. 

## DP2Unlearning Fine-Tuning

For DP2Unlearning fine-tuning, run:

```bash
python FT_BaseModel.py --config-path /home/user_name/project_name/config --config-name FT_BaseModel.yaml
```
Do necessary modification to FT_BaseModel.yaml based on forgetting percentage (1%:retain99, 5%:retain95, or 10%:retain90)

## Approximate Unlearning Baselines Fine-Tuning

To perform approximate unlearning fine-tuning, execute the following:

```bash
python forget.py --config-path /home/user_name/project_name/config --config-name forget.yaml
```

## Evaluation

To evaluate the models, use this command:

```bash
python evaluate_util.py --config-path /home/user_name/project_name/config --config-name eval_everything.yaml
```
You need to provide the specific model path that you wish to evaluate.

## Aggregation

To aggregate the evaluation statistics, use:

```bash
python aggregate_eval_stat.py --config-path /home/user_name/project_name/config --config-name aggregate_eval_stat.yaml
```

Ensure you have the paths to your results in aggregate_eval_stat.yaml:

```bash
retain_result=${path_to_traditional_retraining_from_scratch}
ckpt_result=${path_to_your_unlearned_method}
```

## Beyond KS Test

To run the Beyond KS Test, execute:

```bash
python Beyond_KS_test.py --config-path /home/user_name/project_name/config --config-name aggregate_eval_stat.yaml
```

-----------------------------------------------------------------------------------------------------------------------------

<small>The baseline methods are implemented from </small> [[1](https://locuslab.github.io/tofu/)].
