# LLM Unlearning: Privacy-Preserving Unlearning for Trustworthy AI

**LLM Unlearning** is a parent repository for multiple research projects on machine unlearning for **Large Language Models (LLMs)**. The repository collects methods, experiments, and evaluation workflows for helping LLMs reduce the influence of unwanted, sensitive, or legally protected training data while keeping useful model behavior.

Each subproject has its own detailed `README.md` with installation steps, scripts, configuration files, and reproduction instructions.

## Repository Projects

| Project | Focus | Directory |
|---|---|---|
| **DP2Unlearning** | Differential privacy based guaranteed unlearning for LLMs | [`DP2Unlearning`](https://github.com/tamimalmahmud/LLM-Unlearning/tree/main/DP2Unlearning) |
| **UnReL** | Efficient unlearning through risk-aware data scoring and targeted relearning | [`UnReL`](https://github.com/tamimalmahmud/LLM-Unlearning/tree/main/UnReL) |
| **ESU** | Selective and efficient unlearning with privacy-aware preparation, risk-based data organization, checkpoints, and rollback updates | [`ESU`](https://github.com/tamimalmahmud/LLM-Unlearning/tree/main/ESU) |

## Project 1: DP2Unlearning

**Paper:** [DP2Unlearning: An Efficient and Guaranteed Unlearning Framework for LLMs](https://doi.org/10.1016/j.neunet.2025.107879)

**DP2Unlearning** studies privacy-guaranteed unlearning for LLMs. It uses differential privacy based mechanisms to prepare disclosure-protected models and evaluate how well they preserve utility after targeted data removal.

To use it:

```bash
cd DP2Unlearning
```

Then follow the detailed README inside the project directory.

## Project 2: UnReL

**Paper:** [UnReL: Unlearning via ReLearning]

**UnReL** focuses on reducing unlearning cost through data scoring, shard assignment, and targeted relearning. It estimates which data may be more likely to require removal and uses that information to support efficient retraining strategies.

To use it:

```bash
cd UnReL
```

Then follow the detailed README inside the project directory.

## Project 3: ESU

**Paper:** [ESU: Efficient Selective Unlearning with Privacy Guarantees for Large Language Models]

**ESU** is a selective unlearning project for LLMs. It prepares the model for future deletion requests by protecting disclosure-prone information, organizing non-public data according to removal risk, and saving checkpoints during training. When an unlearning request arrives, ESU restores an appropriate checkpoint, removes the requested data from the remaining update path, and updates the model using retained data.

To use it:

```bash
cd ESU
```

Then follow the detailed README inside the project directory.

## Quick Start

Clone the repository:

```bash
git clone https://github.com/tamimalmahmud/LLM-Unlearning.git
cd LLM-Unlearning
```

Choose the project you want to reproduce or extend:

```bash
cd DP2Unlearning
# or
cd UnReL
# or
cd ESU
```

## Contributing

Contributions are welcome. You can help by improving existing methods, adding new baselines, testing more datasets or models, reporting bugs, improving documentation, or adding new evaluation metrics.

## Citation

If you use this repository, please cite the relevant project paper and the corresponding subproject. Citation details are provided in the individual project directories.
