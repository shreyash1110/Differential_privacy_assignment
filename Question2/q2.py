"""
CS729 Question 2: Comparing Privacy Accountants for DP-SGD

This script compares four privacy accounting approaches for DP-SGD on MNIST
using a feed-forward neural network:
1. Advanced Composition Theorem (manual baseline)
2. RDP Accountant ('rdp')
3. GDP Accountant ('gdp')
4. PRV Accountant ('prv')

Target privacy budget: epsilon = 5, delta = 1e-5.
"""

import math
import random
import warnings
from dataclasses import dataclass

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision import datasets, transforms

from opacus import PrivacyEngine
from opacus.accountants.utils import get_noise_multiplier

# Suppress experimental/Opacus warnings for clean standard output
warnings.filterwarnings("ignore")

# =============================================================================
# 1. Experimental Setup & Configuration
# =============================================================================
SEED = 42

BATCH_SIZE = 64
TEST_BATCH_SIZE = 1024
EPOCHS = 10

LR = 0.1
MOMENTUM = 0.0

TARGET_EPSILON = 5.0
DELTA = 1e-5
MAX_GRAD_NORM = 1.0

ACCOUNTANTS = ["advanced", "rdp", "gdp", "prv"]

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {DEVICE}")

# =============================================================================
# 2. Reproducibility & Data Loading
# =============================================================================
def seed_everything(seed: int = SEED):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False

seed_everything(SEED)

transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.1307,), (0.3081,))
])

# Download once
train_dataset = datasets.MNIST(root="./data", train=True, download=True, transform=transform)
test_dataset  = datasets.MNIST(root="./data", train=False, download=True, transform=transform)

def make_train_loader(seed: int = SEED, shuffle: bool = True):
    generator = torch.Generator()
    generator.manual_seed(seed)
    return DataLoader(
        train_dataset,
        batch_size=BATCH_SIZE,
        shuffle=shuffle,
        generator=generator,
        num_workers=0,
        pin_memory=torch.cuda.is_available(),
    )

def make_test_loader():
    return DataLoader(
        test_dataset,
        batch_size=TEST_BATCH_SIZE,
        shuffle=False,
        num_workers=0,
        pin_memory=torch.cuda.is_available(),
    )

base_train_loader = make_train_loader()
test_loader = make_test_loader()

N_TRAIN = len(train_dataset)
STEPS_PER_EPOCH = len(base_train_loader)
TOTAL_STEPS = STEPS_PER_EPOCH * EPOCHS
SAMPLE_RATE = BATCH_SIZE / N_TRAIN

print(f"Training samples: {N_TRAIN}")
print(f"Batches per epoch: {STEPS_PER_EPOCH}")
print(f"Total steps: {TOTAL_STEPS}")
print(f"Sample rate: {SAMPLE_RATE:.6f}")
print("-" * 60)

# =============================================================================
# 3. Model Definition
# =============================================================================
class FeedForwardNet(nn.Module):
    def __init__(self):
        super().__init__()
        self.net = nn.Sequential(
            nn.Flatten(),
            nn.Linear(28 * 28, 256),
            nn.ReLU(),
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Linear(128, 10)
        )

    def forward(self, x):
        return self.net(x)

def build_model():
    return FeedForwardNet().to(DEVICE)

def build_optimizer(model):
    return optim.SGD(model.parameters(), lr=LR, momentum=MOMENTUM)

# =============================================================================
# 4. Manual Advanced Composition Baseline
# =============================================================================
def gaussian_eps_per_step(sigma: float, delta_step: float) -> float:
    """
    Classical Gaussian mechanism upper bound: eps <= sqrt(2 * log(1.25/delta_step)) / sigma
    """
    if sigma <= 0:
        raise ValueError("sigma must be positive")
    return math.sqrt(2.0 * math.log(1.25 / delta_step)) / sigma

def advanced_composition_epsilon(sigma: float, steps: int, delta_total: float, q: float) -> float:
    if steps <= 0:
        raise ValueError("steps must be positive")
        
    delta_prime = delta_total / 2.0
    delta_step = delta_total / (2.0 * steps)
    base_delta = delta_step / q
    
    raw_eps0 = gaussian_eps_per_step(sigma, base_delta)
    
    try:
        # Privacy amplification by subsampling
        eps_sub = math.log(1.0 + q * math.expm1(raw_eps0))
        # Advanced composition
        eps_total = (
            math.sqrt(2.0 * steps * math.log(1.0 / delta_prime)) * eps_sub 
            + steps * eps_sub * (math.exp(eps_sub) - 1.0)
        )
        return eps_total
        
    except OverflowError:
        return float('inf')

def solve_sigma_for_advanced_composition(
    target_epsilon: float, steps: int, delta_total: float, q: float, 
    sigma_low: float = 0.01, sigma_high: float = 1000.0, tol: float = 1e-4, max_iter: int = 200
):
    def f(sig):
        return advanced_composition_epsilon(sig, steps, delta_total, q)

    while f(sigma_high) > target_epsilon:
        sigma_high *= 2.0
        if sigma_high > 1e6:
            raise RuntimeError("Could not find a large enough sigma for advanced composition.")

    for _ in range(max_iter):
        mid = 0.5 * (sigma_low + sigma_high)
        eps_mid = f(mid)
        if abs(eps_mid - target_epsilon) < tol:
            return mid
        if eps_mid > target_epsilon:
            sigma_low = mid
        else:
            sigma_high = mid

    return sigma_high

# =============================================================================
# 5. Training and Evaluation Routines
# =============================================================================
criterion = nn.CrossEntropyLoss()

def train_one_epoch(model, loader, optimizer):
    model.train()
    running_loss = 0.0
    iteration_losses = []

    for x, y in loader:
        x, y = x.to(DEVICE), y.to(DEVICE)
        optimizer.zero_grad()
        logits = model(x)
        loss = criterion(logits, y)
        loss.backward()
        optimizer.step()

        loss_val = loss.item()
        running_loss += loss_val
        iteration_losses.append(loss_val)

    epoch_loss = running_loss / len(loader)
    return iteration_losses, epoch_loss

@torch.no_grad()
def evaluate(model, loader):
    model.eval()
    total_loss = 0.0
    correct = 0
    total = 0

    for x, y in loader:
        x, y = x.to(DEVICE), y.to(DEVICE)
        logits = model(x)
        loss = criterion(logits, y)

        total_loss += loss.item() * x.size(0)
        preds = logits.argmax(dim=1)
        correct += (preds == y).sum().item()
        total += x.size(0)

    avg_loss = total_loss / total
    acc = 100.0 * correct / total
    return avg_loss, acc

@dataclass
class RunMetrics:
    accountant: str
    noise_multiplier: float
    iteration_losses: list
    epoch_losses: list
    epsilons: list
    test_losses: list
    test_accuracies: list
    final_test_loss: float
    final_accuracy: float
    final_epsilon: float

def advanced_epsilon_after_epochs(sigma: float, current_epoch: int) -> float:
    steps_so_far = current_epoch * STEPS_PER_EPOCH
    return advanced_composition_epsilon(
        sigma=sigma,
        steps=steps_so_far,
        delta_total=DELTA,
        q=SAMPLE_RATE
    )

def run_dp_experiment(accountant_name: str, sigma: float, seed: int = SEED):
    seed_everything(seed)

    model = build_model()
    optimizer = build_optimizer(model)
    train_loader = make_train_loader(seed=seed, shuffle=True)
    test_loader = make_test_loader()

    if accountant_name == "advanced":
        # Use Opacus for DP training mechanics, but report epsilon manually
        privacy_engine = PrivacyEngine(accountant="rdp")
        model, optimizer, private_loader = privacy_engine.make_private(
            module=model,
            optimizer=optimizer,
            data_loader=train_loader,
            noise_multiplier=sigma,
            max_grad_norm=MAX_GRAD_NORM,
        )
        epsilon_fn = lambda epoch: advanced_epsilon_after_epochs(sigma, epoch)
    else:
        privacy_engine = PrivacyEngine(accountant=accountant_name)
        model, optimizer, private_loader = privacy_engine.make_private(
            module=model,
            optimizer=optimizer,
            data_loader=train_loader,
            noise_multiplier=sigma,
            max_grad_norm=MAX_GRAD_NORM,
        )
        epsilon_fn = lambda epoch: privacy_engine.get_epsilon(delta=DELTA)

    metrics = {
        "accountant": accountant_name,
        "noise_multiplier": sigma,
        "iteration_losses": [],
        "epoch_losses": [],
        "epsilons": [],
        "test_losses": [],
        "test_accuracies": [],
    }

    for epoch in range(1, EPOCHS + 1):
        iter_losses, epoch_loss = train_one_epoch(model, private_loader, optimizer)
        test_loss, test_acc = evaluate(model, test_loader)

        eps_spent = epsilon_fn(epoch)

        metrics["iteration_losses"].extend(iter_losses)
        metrics["epoch_losses"].append(epoch_loss)
        metrics["epsilons"].append(eps_spent)
        metrics["test_losses"].append(test_loss)
        metrics["test_accuracies"].append(test_acc)

        print(
            f"[{accountant_name.upper()}] Epoch {epoch:02d}/{EPOCHS} | "
            f"Train Loss: {epoch_loss:.4f} | "
            f"Epsilon Spent: {eps_spent:.4f} | "
            f"Test Loss: {test_loss:.4f} | "
            f"Test Acc: {test_acc:.2f}%"
        )

    return RunMetrics(
        accountant=metrics["accountant"],
        noise_multiplier=metrics["noise_multiplier"],
        iteration_losses=metrics["iteration_losses"],
        epoch_losses=metrics["epoch_losses"],
        epsilons=metrics["epsilons"],
        test_losses=metrics["test_losses"],
        test_accuracies=metrics["test_accuracies"],
        final_test_loss=metrics["test_losses"][-1],
        final_accuracy=metrics["test_accuracies"][-1],
        final_epsilon=metrics["epsilons"][-1],
    )

def moving_average(a, window=50):
    a = np.array(a)
    ret = np.cumsum(a, dtype=float)
    ret[window:] = ret[window:] - ret[:-window]
    return ret[window - 1:] / window

# =============================================================================
# 6. Main Execution Block
# =============================================================================
if __name__ == "__main__":
    
    print("Calibrating Noise Multipliers...")
    noise_multipliers = {}
    for acc in ACCOUNTANTS:
        if acc == "advanced":
            sigma = solve_sigma_for_advanced_composition(
                target_epsilon=TARGET_EPSILON,
                steps=TOTAL_STEPS,
                delta_total=DELTA,
                q=SAMPLE_RATE
            )
        else:
            sigma = get_noise_multiplier(
                target_epsilon=TARGET_EPSILON,
                target_delta=DELTA,
                sample_rate=SAMPLE_RATE,
                epochs=EPOCHS,
                accountant=acc,
                epsilon_tolerance=0.01,
            )
        noise_multipliers[acc] = sigma

    # Print noise calibration results
    noise_df = pd.DataFrame({
        "accountant": list(noise_multipliers.keys()),
        "noise_multiplier": list(noise_multipliers.values())
    }).sort_values("noise_multiplier")
    print("\nCalibrated Noise Multipliers:")
    print(noise_df.to_string(index=False))
    print("-" * 60)

    # Run experiments
    results = {}
    for acc in ACCOUNTANTS:
        print(f"\nRunning experiment for accountant: {acc}")
        print(f"Noise multiplier: {noise_multipliers[acc]:.6f}")
        results[acc] = run_dp_experiment(acc, noise_multipliers[acc])

    print("\nAll Q2 experiments completed.")
    print("-" * 60)

    # Summary Table
    summary_rows = []
    for acc, res in results.items():
        summary_rows.append({
            "accountant": acc,
            "noise_multiplier": res.noise_multiplier,
            "final_epsilon": res.final_epsilon,
            "final_test_loss": res.final_test_loss,
            "final_test_accuracy": res.final_accuracy,
        })

    summary_df = pd.DataFrame(summary_rows).sort_values("noise_multiplier")
    print("\nSummary of Results:")
    print(summary_df.to_string(index=False))

    # Plot 1: Epsilon Spent vs Epochs
    epochs_axis = np.arange(1, EPOCHS + 1)
    plt.figure(figsize=(10, 5))
    for acc, res in results.items():
        plt.plot(epochs_axis, res.epsilons, marker="o", label=acc)
    plt.xlabel("Epoch")
    plt.ylabel("Epsilon spent")
    plt.title("Privacy spent vs Epochs for different accountants")
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig("epsilon_vs_epochs.png", dpi=300)
    print("Saved plot: epsilon_vs_epochs.png")

    # Plot 2: Iterations vs Training Loss (smoothed)
    plt.figure(figsize=(10, 5))
    for acc, res in results.items():
        smoothed = moving_average(res.iteration_losses, window=50)
        plt.plot(smoothed, label=acc)
    plt.xlabel("Iteration")
    plt.ylabel("Training loss (smoothed)")
    plt.title("Iterations vs Training Loss for different accountants")
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig("iterations_vs_loss.png", dpi=300)
    print("Saved plot: iterations_vs_loss.png")
    
    plt.show()
