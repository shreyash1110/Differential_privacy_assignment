# Question 4: Membership Inference Attack (MIA)

## Objective
This section empirically investigates how vulnerable machine learning models are to a simple loss-based Membership Inference Attack (MIA). It compares the attack's effectiveness against a non-private SGD model versus a differentially private model (DP-SGD).

## Methodology
The MNIST dataset is divided into a training member set ($D_{in}$) and a holdout non-member set ($D_{out}$). Both models are trained exclusively on $D_{in}$. The attack attempts to determine if a sample was part of the training data based on whether its cross-entropy loss falls below a calculated threshold. 

## How to Run
Execute the attack simulation using:
```bash
python q4.py
