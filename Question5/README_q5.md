# CS729 Assignment 1 — Question 5
## Comparing SGD, DP-SGD, FTRL, and DP-FTRL on MNIST

This question studies the empirical performance of three optimization settings on the MNIST dataset using a feed-forward neural network:

- **SGD**
- **DP-SGD**
- **FTRL / DP-FTRL**

The goal is to compare their optimization behavior and final predictive performance under different privacy budgets.

---

## Objective

We compare:

- **Non-private baselines:** SGD and FTRL
- **Private methods:** DP-SGD and DP-FTRL

For the private methods, experiments are conducted for:

$$
\varepsilon \in \{1,\;10\}
$$

The following plots are reported:

1. **Number of Iterations (Epoch × Batches) vs Training Loss**
2. **Epsilon Spent vs Epochs**
3. **Target Epsilon vs Test Accuracy**

In the third plot, the private methods are compared with their corresponding non-private baselines.

---

## Dataset

We use the **MNIST** dataset of handwritten digits.

- Training set: 60,000 images
- Test set: 10,000 images
- Input: grayscale images of size $28 \times 28$
- Number of classes: 10

The images are normalized before training.

---

## Model

A feed-forward neural network is used for all experiments.

Typical pipeline:

- Flatten input image
- One or more fully connected hidden layers
- ReLU activations
- Final linear layer with 10 outputs

The final layer outputs logits, and the classification loss is the **cross-entropy loss**.

---

## Methods

### 1. SGD
Standard stochastic gradient descent is used as the non-private baseline.

### 2. DP-SGD
DP-SGD is implemented using **Opacus**.  
Per-sample gradients are clipped and Gaussian noise is added to ensure differential privacy.

The DP-SGD update can be summarized as:

$$
\bar{g}_i = \frac{g_i}{\max\left(1, \frac{\|g_i\|_2}{C}\right)}
$$

$$
\tilde{g} = \frac{1}{B}\left(\sum_{i=1}^{B} \bar{g}_i + \mathcal{N}(0, \sigma^2 C^2 I)\right)
$$

where:

- $g_i$ is the per-sample gradient
- $C$ is the clipping norm
- $\sigma$ is the noise multiplier
- $B$ is the batch size

The **RDP accountant** is used to track privacy expenditure.

### 3. FTRL
FTRL is used as a non-private baseline corresponding to the private FTRL-style setup.

### 4. DP-FTRL
DP-FTRL is implemented using the provided DP-FTRL codebase/design.  
To keep the setup uniform, the same accountant family (**RDP**) is used for DP-SGD as required.

For DP-FTRL, a **fixed data order across epochs** is used rather than reshuffling every epoch, since the privacy accounting in the reference setup assumes the same batch order is maintained across epochs.

---

## Experimental Setup

The experiments are run for a fixed number of epochs on MNIST under the same neural architecture.

Key settings include:

- Same dataset and preprocessing across all methods
- Same test set for evaluation
- Same target privacy budgets for the private methods:

$$
\varepsilon \in \{1,\;10\}
$$

- DP-SGD privacy tracked with the **RDP accountant**
- DP-FTRL evaluated under the corresponding private accounting setup

---

## Results

### 1. Iterations vs Training Loss

This plot compares optimization behavior over training.

**Placeholder for plot:**

```text
[Insert Plot Here: Iterations vs Training Loss]
```

**Observation:**  
The non-private methods converge faster and to lower training loss than the private methods.  
Among the private methods, the added noise and clipping make optimization noisier and slower.

---

### 2. Epsilon Spent vs Epochs

This plot shows how privacy budget is consumed during training for the private methods.

**Placeholder for plot:**

```text
[Insert Plot Here: Epsilon Spent vs Epochs]
```

**Observation:**  
The privacy expenditure increases monotonically with epochs.  
As expected, longer training consumes more privacy budget.

---

### 3. Target Epsilon vs Test Accuracy

This plot compares final test accuracy across privacy budgets and against non-private baselines.

**Placeholder for plot:**

```text
[Insert Plot Here: Target Epsilon vs Test Accuracy]
```

---

## Summary of Final Results

Fill the table below with your actual outputs:

| Method | Target Epsilon | Final Test Accuracy | Final Test Loss | Final Epsilon Spent |
|---|---:|---:|---:|---:|
| SGD | N/A | [fill] | [fill] | N/A |
| FTRL | N/A | [fill] | [fill] | N/A |
| DP-SGD | 1 | [fill] | [fill] | [fill] |
| DP-SGD | 10 | [fill] | [fill] | [fill] |
| DP-FTRL | 1 | [fill] | [fill] | [fill] |
| DP-FTRL | 10 | [fill] | [fill] | [fill] |

---

## Discussion

The experimental trends are consistent with the expected privacy-utility tradeoff:

1. **Non-private methods outperform private methods** in terms of both convergence speed and final accuracy.
2. **DP-SGD and DP-FTRL incur utility loss** because of clipping and additive noise.
3. A **larger privacy budget** (for example, $\varepsilon = 10$ instead of $\varepsilon = 1$) generally allows better utility, since less restrictive privacy constraints require less noise.
4. The comparison between **DP-SGD and DP-FTRL** highlights differences not only in privacy but also in the optimization dynamics induced by the two training procedures.

Overall, the experiments show that enforcing differential privacy leads to a measurable degradation in optimization and generalization performance, while still allowing the model to achieve reasonable test accuracy on MNIST.

---

## Reproducibility

To reproduce the results:

1. Install the required Python dependencies.
2. Ensure MNIST is downloaded automatically by the script/notebook.
3. Run the Question 5 script or notebook from start to finish.
4. Save the generated plots.
5. Copy the final numerical outputs into the summary table above.

Example command:

```bash
python q5.py
```

---

## Files Included

Example expected files:

- `q5.py` — Python script for Question 5
- `q5-final.ipynb` — notebook version
- `README.md` — this file
- plot images for the three required graphs

---

## Conclusion

This question demonstrates the tradeoff between optimization quality and privacy preservation.  
While **SGD** and **FTRL** achieve stronger non-private performance, **DP-SGD** and **DP-FTRL** provide privacy guarantees at the cost of slower convergence and somewhat lower final accuracy. The results illustrate the central theme of differential privacy in machine learning: improved privacy requires sacrificing some utility.
