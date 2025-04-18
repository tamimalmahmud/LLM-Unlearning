# LLM-Unlearning: Privacy-Preserving Unlearning for Trustworthy AI

**LLM-Unlearning** is an open-source project focused on enabling **unlearning** techniques in **Large Language Models (LLMs)**. The project includes **exact** and **approximate unlearning methods** designed to help make models more **privacy-preserving**, **trustworthy**, and **ethical**. It aims to facilitate the development of **AI models** that can forget unwanted or sensitive data, ensuring **data privacy** and compliance with regulations.

## Key Features:
- **Exact and Approximate Unlearning**: Efficient methods for forgetting specific data while preserving model performance.
- **Privacy-Preserving AI**: Ensures AI models can securely forget sensitive data, protecting user privacy.
- **Trustworthy AI**: Build ethical models that ensure transparency and fairness in data processing.

## Project1: DP2Unlearning

**Paper**: [DP2Unlearning: Privacy-Preserving Data Forgetting in AI](https://papers.ssrn.com/sol3/papers.cfm?abstract_id=5217160)

This section contains methods for fine-tuning, training, and evaluating LLMs in a privacy-preserving manner.

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

# LLM-Unlearning
The project focuses on LLM unlearning, including efficient exact and approximate unlearning methods for privacy-preserving unlearning. Aiming to advance trustworthy AI through ethical unlearning techniques and data privacy solutions.

-----------------------------------------------------------------------------------------------------------------------------
# Project1: DP2Unlearning

## Paper Link: https://papers.ssrn.com/sol3/papers.cfm?abstract_id=5217160

## Installation

To set up the environment for the project, create a conda environment using the following command:

```bash
$ conda create --name torch-env pytorch torchvision pytorch-cuda=12.1 -c pytorch -c nvidia
$ conda activate torch-env
```

Then, install the following libraries:

```bash
pip install datasets
pip install accelerate
pip install evaluate
pip install matplotlib
pip install hydra-core
pip install omegaconf
pip install peft
pip install rouge_score
pip install tqdm
pip install einops
pip install packaging
pip install bitsandbytes
pip install scipy
pip install ninja
install additional libraries if required 
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

Ensure you have the paths to your results:

```bash
retain_result=${path_to_traditional_retraining_from_scratch}
ckpt_result=${path_to_your_unlearned_method}
```

## Beyond KS Test

To run the Beyond KS Test, execute:

```bash
python Beyond_KS_test.py --config-path /home/user_name/project_name/config --config-name aggregate_eval_stat.yaml
```

-------------------------------------------------------------------------------------------------------------------------------------------
