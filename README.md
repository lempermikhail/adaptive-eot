# Adaptive EOT Smoothing

This repository implements an **Adaptive Expectation Over Transformation (EOT)** adversarial attack against **randomized smoothing** defenses. The attack adaptively selects the most effective input transformation at each step based on gradient signal strength, making it more effective than traditional PGD or fixed EOT attacks — especially in targeted settings.

---

## Key Features

- **Adaptive EOT Attack**: 
  - Dynamically selects the best transformation at each optimization step using gradient feedback.
- **Baseline Attacks**:
  - **Vanilla PGD** (targeted)
  - **PGD-EOT** (PGD over random transformations)
- **Certified Defense**: 
  - Randomized smoothing implementation (based on Cohen et al., 2019).
- **Evaluation Pipeline**:
  - Targeted attack success rate comparison across base and smoothed classifiers.
  - Visualization of loss and transformation behavior.

---

## Experimental Results (Targeted Attacks on CIFAR-10)

| Attack Method   | Base Classifier | Smoothed Classifier |
|----------------|-----------------|----------------------|
| PGD            | 4.00%           | 12.00%               |
| PGD-EOT        | 0.00%           | 13.00%               |
| **Adaptive EOT** | **52.00%**     | **32.00%**           |

> Adaptive EOT significantly improves success rate over baselines. Randomized smoothing still offers stronger defense than the base model, but it remains vulnerable to transformation-aware attacks.

---

## Setup

Install dependencies:

```bash
pip install -r requirements.txt
```

---

## Usage

To run all experiments:

```bash
python adaptive_eot.py
```

This script performs:
- PGD attack (targeted) on both base and smoothed classifiers
- PGD-EOT attack (targeted) on both base and smoothed classifiers
- Adaptive EOT attack (targeted) on both base and smoothed classifiers
- Print success rate comparison

---

## Components

- **Dataset**: CIFAR-10 from `torchvision.datasets.CIFAR10`
- **Model**: Pretrained ResNet-20 from [`chenyaofo/pytorch-cifar-models`](https://github.com/chenyaofo/pytorch-cifar-models)
- **Randomized Smoothing**: `Smooth` class adapted from [locuslab/smoothing](https://github.com/locuslab/smoothing)
- **Adversarial Attacks**: PGD and EOT-PGD from [`torchattacks`](https://github.com/Harry24k/adversarial-attacks-pytorch)

---

## Citation

- Cohen et al., *Certified Adversarial Robustness via Randomized Smoothing*, ICML 2019  
- Athalye et al., *Synthesizing Robust Adversarial Examples*, ICML 2018  
- Chen, Y., *PyTorch CIFAR Models*, 2020 ([GitHub](https://github.com/chenyaofo/pytorch-cifar-models))
