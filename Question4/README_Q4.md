# CS729 Question 4: Membership Inference Attack (MIA)

## Overview

This experiment studies a simple **loss-based Membership Inference Attack (MIA)** on the MNIST dataset using a feed-forward neural network. The goal is to empirically compare the vulnerability of a **non-private SGD model** and a **private DP-SGD model** to this attack.

A membership inference attack attempts to determine whether a sample was used during training. In this question, the attacker uses the **per-sample cross-entropy loss** as the attack signal.

For a sample $x$, the attacker predicts:

$$
\hat{x} =
\begin{cases}
\text{Member}, & \text{if } L(x) < \tau, \\
\text{Not-Member}, & \text{otherwise},
\end{cases}
$$

where $L(x)$ is the cross-entropy loss of the model on sample $x$, and $\tau$ is a chosen threshold.

---

## Procedure

### 1. Dataset split

The original MNIST training set is split into two disjoint subsets:

- $D_{in}$: member set used for training
- $D_{out}$: holdout set not seen during training

This ensures that the attack is evaluated on true members and true non-members.

### 2. Model training

Two models are trained on $D_{in}$:

1. **Non-private model:** trained using standard SGD
2. **Private model:** trained using DP-SGD with Opacus

The DP-SGD model uses gradient clipping and Gaussian noise addition to provide differential privacy.

### 3. Per-sample loss computation

For both models, the per-sample cross-entropy loss is computed on:

- all samples in $D_{in}$
- all samples in $D_{out}$

These losses serve as the attacker's scores.

### 4. Threshold-based attack

A threshold $\tau$ is chosen for each model using a calibration-based approach. The attack then predicts membership using the rule:

- predict **Member** if loss $< \tau$
- predict **Not-Member** otherwise

Using this rule, we compute the confusion matrix:

| Truth / Prediction | Member | Not-Member |
|---|---:|---:|
| Member | TP | FN |
| Not-Member | FP | TN |

### 5. ROC curve and AUC

To evaluate the attack over all possible thresholds, we vary $\tau$ and compute:

$$
\text{TPR} = \frac{TP}{TP + FN} = \Pr(\hat{x}=\text{Member} \mid x \in D_{in})
$$

$$
\text{FPR} = \frac{FP}{FP + TN} = \Pr(\hat{x}=\text{Member} \mid x \in D_{out})
$$

The ROC curve plots **TPR vs FPR** for different thresholds.

The **Area Under the Curve (AUC)** is used as a threshold-independent measure of attack strength.

---

## Files and Outputs

The script `q4.py` generates the following outputs:

- `nonprivate_history.csv`
- `private_history.csv`
- `attack_dataframe.csv`
- `loss_summary.csv`
- `tau_summary.csv`
- `attack_summary.csv`
- `sample_report.csv`
- `nonprivate_confusion_matrix_full.csv`
- `private_confusion_matrix_full.csv`
- `nonprivate_roc_curve.csv`
- `private_roc_curve.csv`
- `loss_distributions.png`
- `roc_curve.png`
- `sample_images.png`
- `results_summary.json`

---

## How to Run

Run the script using:

```bash
python q4.py
```

If required, install Opacus using:

```bash
pip install opacus
```

---

## Key Results

### 1. Confusion matrices

The threshold-based attacker is applied separately to the non-private and DP-SGD models, and confusion matrices are reported for both.

**Observation:** The non-private model typically shows slightly stronger membership leakage than the private model, since member samples often have lower loss than non-members. However, if the loss distributions overlap heavily, the attack remains weak for both models.

### 2. Sample-level report

A report is generated for 10 randomly selected samples:

- 5 members from $D_{in}$
- 5 non-members from $D_{out}$

For each sample, the report includes:

- image index
- true label
- true status (member / non-member)
- loss under the non-private model
- membership prediction under the non-private model
- loss under the private model
- membership prediction under the private model

### 3. ROC and AUC

The ROC curve summarizes attack performance across all thresholds.

AUC is a good metric here because:

- it does not depend on a single threshold choice
- it captures the global separability of member and non-member losses
- it allows a fair comparison between the two models

Interpretation:

- AUC close to $1$ means the attacker can clearly separate members from non-members
- AUC close to $0.5$ means the attacker performs nearly like random guessing

The model with the **larger AUC** is considered **more vulnerable** to the membership inference attack.

---

## Outcome Summary

In this experiment, both the non-private SGD model and the DP-SGD model are attacked using per-sample loss values. The expected qualitative outcome is:

- the **non-private model** is more vulnerable to the attack
- the **DP-SGD model** is less vulnerable because differential privacy reduces the difference between member and non-member behavior
- therefore, the **AUC of the private model should be lower or closer to $0.5$** than that of the non-private model

Even when the attack is weak overall, comparing the ROC curves and AUC values still gives a meaningful empirical verification of the privacy benefit of DP-SGD.

---

## Space for Plots

### Plot 1: Loss Distribution for Non-private and Private Models

<img width="1189" height="490" alt="image" src="https://github.com/user-attachments/assets/4be3e08d-03ec-4437-80f3-ea02fe762828" />

---

### Plot 2: ROC Curve for the Membership Inference Attack

<img width="691" height="547" alt="image" src="https://github.com/user-attachments/assets/2106f1e5-2001-413d-9551-7f909aeb8487" />

---

### Plot 3: Randomly Selected Member and Non-member Samples

<img width="1389" height="727" alt="image" src="https://github.com/user-attachments/assets/786db0d7-0b0b-4cb1-8776-936f675aa8fe" />

---

## Conclusion

This question empirically studies a simple loss-based membership inference attack on MNIST. Two models are compared: a non-private SGD model and a private DP-SGD model.

The experiment shows how per-sample losses can be used to infer membership and how ROC/AUC can quantify attack success. Differential privacy is expected to reduce this leakage, making the private model less vulnerable than the non-private one.

Thus, the experiment provides an empirical demonstration of the privacy protection offered by DP-SGD against membership inference attacks.
