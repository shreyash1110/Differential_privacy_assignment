# Question 1 — Optimizer Comparison under Differential Privacy

## Objective

In this question, we compare the convergence and utility of two optimizers:

- **SGD**
- **Adam**

along with their differentially private counterparts:

- **DP-SGD**
- **DP-Adam**

The experiments are performed on the **MNIST** dataset using a feed-forward neural network.  
For the private models, we consider target privacy budgets

$$
\varepsilon \in \{1, 10\}
$$

and track how privacy affects training dynamics and final performance.

---

## Experimental Setup

### Dataset
- **MNIST**
- Standard train/test split

### Model
A fully connected feed-forward neural network for multi-class digit classification.

### Optimizers Compared
- Non-private:
  - SGD
  - Adam
- Private:
  - DP-SGD
  - DP-Adam

### Privacy Mechanism
Differential privacy is enforced using the **Opacus** library.  
We use the **RDP accountant** to track the privacy budget.

The privacy guarantee is reported in the form:

$$
(\varepsilon, \delta)\text{-DP}
$$

where $\delta$ is chosen to be a small value based on the dataset size.

---

## Procedure

For each optimizer, the following procedure is used:

1. Initialize the model with the same architecture.
2. Train the **non-private** versions of SGD and Adam normally.
3. Train the **private** versions using Opacus by attaching a privacy engine.
4. For each private optimizer, run experiments for:

$$
\varepsilon = 1 \quad \text{and} \quad \varepsilon = 10
$$

5. During training, record:
   - training loss at each iteration
   - privacy spent after each epoch
   - final test accuracy

---

## Results

### 1. Number of Iterations vs Training Loss

This plot compares optimization behavior across the private methods by showing how the training loss evolves with the total number of iterations.

<img width="1590" height="490" alt="image" src="https://github.com/user-attachments/assets/fa47762b-b6b9-48ef-9327-bc36063de3c1" />

<!-- Example: ![Iterations vs Training Loss](path/to/plot1.png) -->

---

### 2. $\varepsilon$ Spent vs Epochs

This plot shows how the privacy budget is consumed over training epochs for the private optimizers.

<img width="1590" height="490" alt="image" src="https://github.com/user-attachments/assets/db2d991f-a3af-4b91-a8f9-7098536a09aa" />

<!-- Example: ![Epsilon vs Epochs](path/to/plot2.png) -->

---

### 3. Target $\varepsilon$ vs Test Accuracy

This plot compares final test accuracy for:
- DP-SGD
- DP-Adam
- non-private SGD
- non-private Adam

It highlights the privacy-utility tradeoff.

<img width="834" height="463" alt="image" src="https://github.com/user-attachments/assets/75a0914a-1ff0-4903-86ca-0ca44bcac675" />

<!-- Example: ![Target Epsilon vs Test Accuracy](path/to/plot3.png) -->

---

## Discussion

The experiments show the standard **privacy-utility tradeoff**:

- smaller $\varepsilon$ gives stronger privacy,
- but usually leads to lower accuracy due to stronger noise injection.

In general, the private optimizers converge more slowly than their non-private counterparts because DP training perturbs gradients after clipping and noise addition.

If the target privacy budget increases from

$$
\varepsilon = 1 \to 10,
$$

the model is allowed to use less restrictive noise relative to utility, and test accuracy typically improves.

The non-private baselines are expected to achieve higher final accuracy because they are not constrained by gradient clipping and privacy noise.

---

## Reproducibility

To reproduce the experiments:

1. Install the required libraries:
   - `torch`
   - `torchvision`
   - `opacus`
   - `matplotlib`
   - `pandas`
   - `numpy`

2. Run the script:
   ```bash
   python q1.py
