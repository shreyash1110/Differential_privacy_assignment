# Question 5: Comparison of SGD, DP-SGD, FTRL, and DP-FTRL

## Objective
This question compares the performance of standard gradient-based optimization against Follow-The-Regularized-Leader (FTRL) optimization in both private and non-private settings.

## Methodology
The models are trained using target privacy budgets of $\epsilon \in \{1, 10\}$. Unlike DP-SGD, which adds noise stepwise, the DP-FTRL method combines clipped gradients with cumulative tree-aggregated Gaussian noise inside an FTRL-style update. 

## How to Run
The main script relies on auxiliary DP-FTRL modules (`dpftrl_noise.py`, `dpftrl_optimizers.py`, `dpftrl_privacy.py`). Run the complete comparison using:
```bash
python q5.py
