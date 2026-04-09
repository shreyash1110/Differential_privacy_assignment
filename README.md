# CS729 Assignment: Differential Privacy in Machine Learning

**Author:** Shreyash Dwivedi //
**Roll Number:** 221035 //
**Course:** CS729: Differential Privacy in Machine Learning //
**Semester:** 2025-26, II //
**Date:** April 10, 2026 //

## Project Overview
This repository contains the implementation, experiments, and analysis for the CS729 Differential Privacy in Machine Learning assignment. The overarching objective of this project is to explore and analyze the practical implications of training deep learning models with differential privacy (DP).

All experiments in this repository are conducted on the **MNIST handwritten digit dataset** for 10-class image classification, utilizing a standard feed-forward neural network architecture. The implementation heavily leverages the PyTorch framework and the **Opacus library** to enable differential privacy.

## Repository Structure
The project is divided into five distinct experiments, each corresponding to a question in the assignment. Each folder contains its own Python scripts, Jupyter notebooks (if applicable), and a specific `readme.md`.

* `Question1/`: Optimizer Comparison (SGD vs. Adam) under privacy constraints.
* `Question2/`: Evaluation of different Privacy Accountants.
* `Question3/`: Analysis of the Gradient Clipping Norm and Ghost Clipping.
* `Question4/`: Empirical evaluation of a Membership Inference Attack (MIA).
* `Question5/`: Comparison of standard SGD/DP-SGD against FTRL/DP-FTRL.
* `assignment_report.pdf`: The comprehensive project report detailing the theoretical methodology, plotted results, and final observations.

## Environment & Dependencies
To run the scripts in this repository, ensure you have the following installed:
* **Python** (3.8+)
* **PyTorch**
* **Opacus** (Used for DP-SGD training, ghost clipping, and privacy accounting)
* Standard data science libraries: `NumPy`, `Matplotlib`, `Torchvision`

## Summary of Experiments

### 1. Optimizer Comparison for Private and Non-Private Training
This module compares the convergence behavior and privacy-utility trade-offs of Stochastic Gradient Descent (SGD) and Adaptive Moment Estimation (Adam) against their private counterparts (DP-SGD and DP-Adam). Models are evaluated at target privacy budgets of $\epsilon \in \{1, 10\}$.

### 2. Comparison of Privacy Accountants
Different privacy accounting mechanisms track the cumulative privacy loss during training. This experiment compares the classical Advanced Composition Theorem baseline against modern Opacus accountants: RDP (Rényi DP), GDP (Gaussian DP), and PRV (Privacy Random Variable). All models are evaluated at a fixed target budget of $\epsilon=5$ to analyze how the choice of accountant impacts the required noise multiplier and model utility.

### 3. Effect of Gradient Clipping Norm in DP-SGD
In DP-SGD, the clipping norm $C$ controls the bias-variance trade-off. This experiment trains models using clipping norms $C \in \{0.1, 10.0\}$ while keeping the noise multiplier fixed. To manage GPU memory complexity efficiently, the implementation utilizes Opacus's "flat" clipping combined with "ghost" clipping, preventing the explicit instantiation of full per-sample gradients.

### 4. Membership Inference Attack (MIA)
To empirically verify the individual-level privacy guarantees of differential privacy, we implement a loss-based Membership Inference Attack (MIA). The dataset is split into member ($D_{in}$) and non-member ($D_{out}$) sets. The attack compares a non-private SGD model against a private DP-SGD model, demonstrating how DP mitigates data leakage and lowers the Area Under the ROC Curve (AUC) of the attacker.

### 5. Comparison of SGD, DP-SGD, FTRL, and DP-FTRL
This experiment explores alternative optimization paradigms by comparing standard gradient-based optimization against Follow-The-Regularized-Leader (FTRL). The private implementation (DP-FTRL) utilizes tree-aggregated Gaussian noise and a custom privacy computation rather than Opacus' standard stepwise accountant, allowing us to evaluate the efficiency of FTRL-style updates under privacy constraints of $\epsilon \in \{1, 10\}$.

## How to Run
Navigate to any question's directory to execute the corresponding script. For example, to run the experiment for Question 1:

```bash
cd Question1
python q1.py --accountant rdp --epochs 10 --batch-size 64 --target-epsilons 1 10
