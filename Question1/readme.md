# Question 1: Optimizer Comparison for Private and Non-Private Training

## Objective
This section compares the convergence behavior of two optimizers: Stochastic Gradient Descent (SGD) and Adaptive Moment Estimation (Adam), along with their differentially private counterparts (DP-SGD and DP-Adam) on the MNIST dataset.

## Methodology
We use a feed-forward neural network for a 10-class classification task. The private models are created using the Opacus library with target privacy budgets of $\epsilon \in \{1, 10\}$ using the RDP accountant. We evaluate the effect of privacy constraints on both optimization convergence speed and test accuracy.

## How to Run
The experiment can be reproduced by running the main Python script:
```bash
python q1.py --accountant rdp --epochs 10 --batch-size 64 --target-epsilons 1 10
