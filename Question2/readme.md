# CS729 Assignment - Question 2: Privacy Accountants

This directory contains the solution for **Question 2** of the CS729 Differential Privacy in Machine Learning assignment. 

The objective of this task is to compare the performance and privacy utility of four different privacy accountants for DP-SGD on the MNIST dataset using a feed-forward neural network. We evaluate a manual implementation of the **Advanced Composition Theorem** against three modern accountants provided by the Opacus library: **RDP**, **GDP**, and **PRV**.

## Prerequisites and Setup

Before running the script, ensure you have the necessary libraries installed. 

**Python Version:** Python 3.8+ (Recommended 3.10+)

**Required Libraries:**
`bash
pip install torch torchvision opacus matplotlib pandas numpy
`

## How to Run

To execute the experiment, simply run the standalone Python script from your terminal:

`bash
python question_2.py
`

**What the script does:**
1. Downloads the MNIST dataset.
2. Calibrates the exact noise multiplier ($\sigma$) required for each accountant to hit a target privacy budget of $\epsilon = 5.0$ and $\delta = 10^{-5}$ over 10 epochs.
3. Trains a model sequentially using each of the four accounting methods.
4. Outputs the progress to the console and generates summary tables.
5. Automatically saves two comparison plots (`epsilon_vs_epochs.png` and `iterations_vs_loss.png`) in the same directory.

## Methodology

A uniform target privacy budget ($\epsilon = 5.0$) was set for all methods to ensure a fair comparison. 

* **Advanced Composition (Classical Baseline):** Since Opacus does not support standard advanced composition natively, it was implemented manually. The per-step privacy guarantee was computed using the Gaussian mechanism bound, incorporating privacy amplification by subsampling, and then composed over the total number of steps. A binary search was used to find the required noise multiplier.
* **Modern Accountants (Opacus):** The `rdp`, `gdp`, and `prv` bounds were utilized via Opacus's `get_noise_multiplier` utility to calibrate the noise.

## Results and Comparisons

### 1. Calibrated Noise Multipliers
Different accountants require drastically different amounts of noise to satisfy the exact same privacy budget of $\epsilon=5$. 

| Accountant | Noise Multiplier |
| :--- | :--- |
| **prv** | ~0.44 |
| **gdp** | ~0.45 |
| **rdp** | ~0.55 |
| **advanced** | ~2.42 |

As seen above, the classical advanced composition theorem is highly loose, requiring significantly more noise than modern accountants.

### 2. Privacy Spent vs Epochs

This plot illustrates how the privacy budget ($\epsilon$) accumulates over the 10 training epochs for each method.

<img width="691" height="470" alt="image" src="https://github.com/user-attachments/assets/59f2c3ff-ebd4-451b-a870-a4f1607aca11" />

### 3. Training Loss vs Iterations

Due to the massive difference in injected noise, the convergence behaviors of the models vary wildly. The advanced composition method struggles to converge smoothly compared to the tighter modern bounds.

<img width="764" height="470" alt="image" src="https://github.com/user-attachments/assets/176db25a-b750-4fa4-bd64-2634b082ca23" />

### 4. Final Utility (Test Accuracy)

Ultimately, the tighter privacy bounds yield vastly superior machine learning utility.

| Accountant | Final Epsilon | Final Test Accuracy |
| :--- | :--- | :--- |
| **gdp** | ~4.99 | ~92.8% |
| **prv** | ~4.99 | ~92.5% |
| **rdp** | ~4.99 | ~90.0% |
| **advanced** | ~4.99 | ~73.5% |

**Conclusion:** Modern privacy accountants (GDP, PRV) provide significantly tighter bounds on privacy loss than the classical Advanced Composition Theorem, allowing for much less noise injection and resulting in far better model accuracy for the same privacy budget.
