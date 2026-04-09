# Question 3: Effect of Gradient Clipping Norm in DP-SGD

## Objective

In this question, we study the effect of the gradient clipping norm in Differentially Private Stochastic Gradient Descent (DP-SGD) on the MNIST dataset using a feed-forward neural network.

We compare two clipping norms:

$$
C \in \{0.1,\; 10.0\}
$$

while keeping the **noise multiplier** and **batch size** fixed.

The goal is to understand how the clipping norm affects optimization, convergence, and final model performance under differential privacy.

---

## Background

In DP-SGD, for a mini-batch of size $B$, the per-sample gradients are first clipped and then noise is added before the optimizer step.

For each sample gradient $g_i$, clipping is performed as:

$$
\bar{g}_i 
= g_i \cdot \min\left(1, \frac{C}{\|g_i\|_2}\right)
$$

where $C$ is the clipping norm.

The clipped gradients are aggregated and Gaussian noise is added:

$$
\tilde{g}
= \frac{1}{B} \left( \sum_{i=1}^{B} \bar{g}_i + \mathcal{N}(0, \sigma^2 C^2 I) \right)
$$

where:
- $\sigma$ is the noise multiplier,
- $C$ is the clipping norm,
- $I$ is the identity matrix.

Thus, the clipping norm affects both:
1. **Clipping bias**: smaller $C$ clips more aggressively,
2. **Noise scale**: for fixed $\sigma$, the noise magnitude is proportional to $\sigma C$.

Hence, changing $C$ induces a **bias-variance trade-off**.

---

## Clipping Implementation Used

This implementation uses the following Opacus configuration:

- **Clipping rule:** `flat`
- **Gradient sample mode:** `ghost`
- **Accountant:** `rdp`

### Memory Complexity

Naively, DP-SGD requires storing per-sample gradients for all trainable parameters. If the batch size is $B$ and the total number of trainable parameters is $P$, then the naive memory complexity is:

$$
O(BP)
$$

This can become prohibitively expensive on the GPU.

With **ghost clipping**, Opacus avoids explicitly materializing the full per-sample gradient tensor for supported layers. Instead, it computes the quantities needed for clipping from layer-wise intermediate activations and backpropagated activations. Therefore, the additional memory overhead is much smaller than naive per-sample gradient storage, and no longer scales like full $O(BP)$ per-sample gradient materialization.

---

## Experimental Setup

### Dataset
- **MNIST**
- Input size: $28 \times 28 = 784$
- Number of classes: $10$

### Model
A feed-forward neural network with fully connected layers and ReLU activations is used.

### Optimizer
- **DP-SGD**

### Fixed Hyperparameters
The following quantities are kept fixed across the two runs:
- batch size,
- learning rate,
- number of epochs,
- noise multiplier,
- random seed.

### Varied Hyperparameter
Only the clipping norm is changed:

$$
C \in \{0.1,\; 10.0\}
$$

---

## What the Script Produces

The Python script `q3.py` performs the following:

1. Loads and preprocesses the MNIST dataset,
2. Builds the feed-forward neural network,
3. Trains two DP-SGD models with clipping norms $C = 0.1$ and $C = 10.0$,
4. Records the training loss at every iteration,
5. Evaluates final test loss and test accuracy,
6. Saves:
   - the **training loss vs iterations** plot,
   - a **CSV summary** of results,
   - a **JSON file** containing the full experiment outputs.

---

## How to Run

Run the script using:

```bash
python q3.py
