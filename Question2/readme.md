# Question 2: Comparison of Privacy Accountants

## Objective
This question evaluates and compares different privacy accounting methods for DP-SGD on the MNIST dataset at a fixed target privacy budget of $\epsilon=5$. The compared methods are the Advanced Composition Theorem, RDP accountant, GDP accountant, and PRV accountant.

## Methodology
For each accountant, we calculate the required noise multiplier to meet the fixed privacy target and train a DP-SGD model for 10 epochs. The experiment measures how different accountants influence the calibrated noise, which in turn impacts the training loss trajectory and final test accuracy. 

## How to Run
To run the evaluation, execute:
```bash
python q2.py
