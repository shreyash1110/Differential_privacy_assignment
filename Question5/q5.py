"""CS729 Assignment 1 - Question 5

Compare SGD, DP-SGD, FTRL, and DP-FTRL on MNIST using a feed-forward network.

This script produces:
1. Iterations vs training loss
2. Epsilon spent vs epochs
3. Target epsilon vs test accuracy

DP-SGD uses the RDP accountant from Opacus.
DP-FTRL uses helper modules from the public Google Research DP-FTRL implementation.
"""

from __future__ import annotations

import math
import random
import warnings
from dataclasses import dataclass, asdict
from pathlib import Path
from typing import Callable

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.func import functional_call, grad, vmap
from torch.utils.data import DataLoader
from torchvision import datasets, transforms

from opacus import PrivacyEngine
from opacus.accountants.utils import get_noise_multiplier

# These helper files should be placed alongside q5.py in the submission zip.
from dpftrl_optimizers import FTRLOptimizer
from dpftrl_noise import CummuNoiseEffTorch
from dpftrl_privacy import compute_epsilon_tree

warnings.filterwarnings("ignore")


# =========================
# Configuration
# =========================
SEED = 42
BATCH_SIZE = 250
TEST_BATCH_SIZE = 1000
EPOCHS = 10

LR_SGD = 0.1
ALPHA_FTRL = 100.0
FTRL_MOMENTUM = 0.9

TARGET_EPSILONS = [1.0, 10.0]
DELTA = 1e-5
MAX_GRAD_NORM = 1.0
ACCOUNTANT = "rdp"

DP_FTRL_RESTART_EVERY = 1
DP_FTRL_TREE_COMPLETION = True

DATA_ROOT = "./data"
OUTPUT_DIR = Path("./q5_outputs")
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


# =========================
# Utilities
# =========================
def seed_everything(seed: int = SEED) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


seed_everything(SEED)


@dataclass
class RunMetrics:
    method: str
    target_epsilon: float | None
    noise_multiplier: float | None
    iteration_losses: list[float]
    epoch_losses: list[float]
    epsilons: list[float | None]
    test_losses: list[float]
    test_accuracies: list[float]
    final_test_loss: float
    final_accuracy: float
    final_epsilon: float | None


criterion = nn.CrossEntropyLoss()


def moving_average(x: list[float], window: int = 20) -> np.ndarray:
    arr = np.asarray(x, dtype=float)
    if len(arr) < window:
        return arr
    return np.convolve(arr, np.ones(window) / window, mode="valid")


# =========================
# Data
# =========================
transform = transforms.Compose(
    [
        transforms.ToTensor(),
        transforms.Normalize((0.1307,), (0.3081,)),
    ]
)

train_dataset = datasets.MNIST(root=DATA_ROOT, train=True, download=True, transform=transform)
test_dataset = datasets.MNIST(root=DATA_ROOT, train=False, download=True, transform=transform)


def make_train_loader(seed: int = SEED, shuffle: bool = True) -> DataLoader:
    generator = torch.Generator()
    generator.manual_seed(seed)
    return DataLoader(
        train_dataset,
        batch_size=BATCH_SIZE,
        shuffle=shuffle,
        generator=generator,
        num_workers=0,
        pin_memory=torch.cuda.is_available(),
        drop_last=True,
    )



def make_test_loader() -> DataLoader:
    return DataLoader(
        test_dataset,
        batch_size=TEST_BATCH_SIZE,
        shuffle=False,
        num_workers=0,
        pin_memory=torch.cuda.is_available(),
    )


base_train_loader_sgd = make_train_loader(seed=SEED, shuffle=True)
base_train_loader_ftrl = make_train_loader(seed=SEED, shuffle=False)
test_loader = make_test_loader()

N_TRAIN = len(train_dataset)
NUM_BATCHES = len(base_train_loader_sgd)
SAMPLE_RATE = BATCH_SIZE / N_TRAIN


# =========================
# Model and optimizers
# =========================
class FeedForwardNet(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.net = nn.Sequential(
            nn.Flatten(),
            nn.Linear(28 * 28, 256),
            nn.ReLU(),
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Linear(128, 10),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)



def build_model() -> nn.Module:
    return FeedForwardNet().to(DEVICE)



def build_sgd_optimizer(model: nn.Module) -> optim.Optimizer:
    return optim.SGD(model.parameters(), lr=LR_SGD, momentum=0.0)



def build_ftrl_optimizer(model: nn.Module, record_last_noise: bool = True) -> FTRLOptimizer:
    return FTRLOptimizer(
        model.parameters(),
        momentum=FTRL_MOMENTUM,
        record_last_noise=record_last_noise,
    )


# =========================
# Evaluation
# =========================
@torch.no_grad()
def evaluate(model: nn.Module, loader: DataLoader) -> tuple[float, float]:
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

    return total_loss / total, 100.0 * correct / total


# =========================
# SGD / DP-SGD
# =========================
def train_one_epoch_sgd(
    model: nn.Module,
    loader: DataLoader,
    optimizer: optim.Optimizer,
) -> tuple[list[float], float]:
    model.train()
    iteration_losses: list[float] = []
    running_loss = 0.0

    for x, y in loader:
        x, y = x.to(DEVICE), y.to(DEVICE)

        optimizer.zero_grad()
        logits = model(x)
        loss = criterion(logits, y)
        loss.backward()
        optimizer.step()

        loss_value = float(loss.item())
        iteration_losses.append(loss_value)
        running_loss += loss_value

    return iteration_losses, running_loss / len(loader)



def run_nonprivate_sgd(seed: int = SEED) -> RunMetrics:
    seed_everything(seed)

    model = build_model()
    optimizer = build_sgd_optimizer(model)
    train_loader = make_train_loader(seed=seed, shuffle=True)

    iteration_losses, epoch_losses = [], []
    epsilons, test_losses, test_accuracies = [], [], []

    for epoch in range(1, EPOCHS + 1):
        iter_losses, epoch_loss = train_one_epoch_sgd(model, train_loader, optimizer)
        test_loss, test_acc = evaluate(model, test_loader)

        iteration_losses.extend(iter_losses)
        epoch_losses.append(epoch_loss)
        epsilons.append(None)
        test_losses.append(test_loss)
        test_accuracies.append(test_acc)

        print(
            f"[SGD] Epoch {epoch:02d}/{EPOCHS} | "
            f"Train Loss: {epoch_loss:.4f} | "
            f"Test Loss: {test_loss:.4f} | "
            f"Test Acc: {test_acc:.2f}%"
        )

    return RunMetrics(
        method="SGD",
        target_epsilon=None,
        noise_multiplier=None,
        iteration_losses=iteration_losses,
        epoch_losses=epoch_losses,
        epsilons=epsilons,
        test_losses=test_losses,
        test_accuracies=test_accuracies,
        final_test_loss=test_losses[-1],
        final_accuracy=test_accuracies[-1],
        final_epsilon=None,
    )



def run_dp_sgd(target_epsilon: float, seed: int = SEED) -> RunMetrics:
    seed_everything(seed)

    model = build_model()
    optimizer = build_sgd_optimizer(model)
    train_loader = make_train_loader(seed=seed, shuffle=True)

    noise_multiplier = get_noise_multiplier(
        target_epsilon=target_epsilon,
        target_delta=DELTA,
        sample_rate=SAMPLE_RATE,
        epochs=EPOCHS,
        accountant=ACCOUNTANT,
        epsilon_tolerance=0.01,
    )

    privacy_engine = PrivacyEngine(accountant=ACCOUNTANT)
    model, optimizer, private_train_loader = privacy_engine.make_private(
        module=model,
        optimizer=optimizer,
        data_loader=train_loader,
        noise_multiplier=noise_multiplier,
        max_grad_norm=MAX_GRAD_NORM,
    )

    iteration_losses, epoch_losses = [], []
    epsilons, test_losses, test_accuracies = [], [], []

    for epoch in range(1, EPOCHS + 1):
        iter_losses, epoch_loss = train_one_epoch_sgd(model, private_train_loader, optimizer)
        test_loss, test_acc = evaluate(model, test_loader)
        eps_spent = float(privacy_engine.get_epsilon(delta=DELTA))

        iteration_losses.extend(iter_losses)
        epoch_losses.append(epoch_loss)
        epsilons.append(eps_spent)
        test_losses.append(test_loss)
        test_accuracies.append(test_acc)

        print(
            f"[DP-SGD, target ε={target_epsilon}] Epoch {epoch:02d}/{EPOCHS} | "
            f"Train Loss: {epoch_loss:.4f} | "
            f"Epsilon Spent: {eps_spent:.4f} | "
            f"Test Loss: {test_loss:.4f} | "
            f"Test Acc: {test_acc:.2f}%"
        )

    return RunMetrics(
        method="DP-SGD",
        target_epsilon=target_epsilon,
        noise_multiplier=float(noise_multiplier),
        iteration_losses=iteration_losses,
        epoch_losses=epoch_losses,
        epsilons=epsilons,
        test_losses=test_losses,
        test_accuracies=test_accuracies,
        final_test_loss=test_losses[-1],
        final_accuracy=test_accuracies[-1],
        final_epsilon=epsilons[-1],
    )


# =========================
# FTRL / DP-FTRL helpers
# =========================
def make_per_sample_grad_fn(model: nn.Module):
    def single_loss(params, buffers, sample, target):
        logits = functional_call(model, (params, buffers), (sample.unsqueeze(0),))
        return F.cross_entropy(logits, target.unsqueeze(0), reduction="mean")

    return vmap(grad(single_loss), in_dims=(None, None, 0, 0))



def compute_clipped_mean_gradients(
    model: nn.Module,
    per_sample_grad_fn,
    x: torch.Tensor,
    y: torch.Tensor,
    max_grad_norm: float,
) -> dict[str, torch.Tensor]:
    params = {name: p for name, p in model.named_parameters()}
    buffers = {name: b for name, b in model.named_buffers()}

    per_sample_grads = per_sample_grad_fn(params, buffers, x, y)
    batch_size = x.size(0)
    per_sample_sq_norms = torch.zeros(batch_size, device=x.device)

    for g in per_sample_grads.values():
        per_sample_sq_norms += g.reshape(batch_size, -1).pow(2).sum(dim=1)

    per_sample_norms = torch.sqrt(per_sample_sq_norms + 1e-12)
    clip_factors = (max_grad_norm / (per_sample_norms + 1e-12)).clamp(max=1.0)

    clipped_mean_grads: dict[str, torch.Tensor] = {}
    for name, g in per_sample_grads.items():
        view_shape = [batch_size] + [1] * (g.dim() - 1)
        clipped_mean_grads[name] = (g * clip_factors.view(*view_shape)).mean(dim=0)

    return clipped_mean_grads



def train_one_epoch_ftrl(
    model: nn.Module,
    loader: DataLoader,
    optimizer: FTRLOptimizer,
    alpha: float,
    private: bool = False,
    noise_source: Callable | None = None,
    max_grad_norm: float | None = None,
    per_sample_grad_fn=None,
) -> tuple[list[float], float]:
    model.train()
    iteration_losses: list[float] = []
    running_loss = 0.0

    for x, y in loader:
        x, y = x.to(DEVICE), y.to(DEVICE)

        optimizer.zero_grad()
        logits = model(x)
        loss = criterion(logits, y)

        if torch.isnan(loss) or torch.isinf(loss):
            raise RuntimeError("Loss became NaN/Inf before optimizer step.")

        if not private:
            loss.backward()
        else:
            clipped_mean_grads = compute_clipped_mean_gradients(
                model=model,
                per_sample_grad_fn=per_sample_grad_fn,
                x=x,
                y=y,
                max_grad_norm=max_grad_norm,
            )
            with torch.no_grad():
                for name, p in model.named_parameters():
                    p.grad = clipped_mean_grads[name].detach().clone()

        noise = noise_source() if noise_source is not None else [torch.zeros_like(p) for p in model.parameters()]
        optimizer.step((alpha, noise))

        with torch.no_grad():
            for p in model.parameters():
                if torch.isnan(p).any() or torch.isinf(p).any():
                    raise RuntimeError("Model parameters became NaN/Inf after optimizer step.")

        loss_value = float(loss.item())
        iteration_losses.append(loss_value)
        running_loss += loss_value

    return iteration_losses, running_loss / len(loader)



def get_epochs_between_restarts(completed_epochs: int, restart_every: int) -> list[int]:
    schedule = []
    full = completed_epochs // restart_every
    rem = completed_epochs % restart_every

    schedule.extend([restart_every] * full)
    if rem > 0:
        schedule.append(rem)
    return schedule



def dpftrl_epsilon_from_noise(noise_multiplier: float, completed_epochs: int) -> float:
    schedule = get_epochs_between_restarts(completed_epochs, DP_FTRL_RESTART_EVERY)
    return float(
        compute_epsilon_tree(
            num_batches=NUM_BATCHES,
            epochs_between_restarts=schedule,
            noise=noise_multiplier,
            delta=DELTA,
            tree_completion=DP_FTRL_TREE_COMPLETION,
            verbose=False,
        )
    )



def solve_dpftrl_noise_multiplier(target_epsilon: float, tol: float = 1e-4, max_iter: int = 80) -> float:
    def eps_from_noise(noise: float) -> float:
        return dpftrl_epsilon_from_noise(noise, EPOCHS)

    low, high = 1e-3, 1.0
    while eps_from_noise(high) > target_epsilon:
        high *= 2.0
        if high > 1e5:
            raise RuntimeError("Could not find a large enough DP-FTRL noise multiplier.")

    for _ in range(max_iter):
        mid = 0.5 * (low + high)
        eps_mid = eps_from_noise(mid)

        if abs(eps_mid - target_epsilon) < tol:
            return float(mid)
        if eps_mid > target_epsilon:
            low = mid
        else:
            high = mid

    return float(high)


# =========================
# FTRL / DP-FTRL runs
# =========================
def run_nonprivate_ftrl(seed: int = SEED) -> RunMetrics:
    seed_everything(seed)

    model = build_model()
    optimizer = build_ftrl_optimizer(model, record_last_noise=False)
    train_loader = make_train_loader(seed=seed, shuffle=False)

    zero_noise = [torch.zeros_like(p, device=DEVICE) for p in model.parameters()]

    iteration_losses, epoch_losses = [], []
    epsilons, test_losses, test_accuracies = [], [], []

    for epoch in range(1, EPOCHS + 1):
        iter_losses, epoch_loss = train_one_epoch_ftrl(
            model=model,
            loader=train_loader,
            optimizer=optimizer,
            alpha=ALPHA_FTRL,
            private=False,
            noise_source=lambda: zero_noise,
        )
        test_loss, test_acc = evaluate(model, test_loader)

        iteration_losses.extend(iter_losses)
        epoch_losses.append(epoch_loss)
        epsilons.append(None)
        test_losses.append(test_loss)
        test_accuracies.append(test_acc)

        with torch.no_grad():
            max_abs_param = max(p.abs().max().item() for p in model.parameters())

        print(
            f"[FTRL] Epoch {epoch:02d}/{EPOCHS} | "
            f"Train Loss: {epoch_loss:.4f} | "
            f"Test Loss: {test_loss:.4f} | "
            f"Test Acc: {test_acc:.2f}% | "
            f"Max |param|: {max_abs_param:.4e}"
        )

    return RunMetrics(
        method="FTRL",
        target_epsilon=None,
        noise_multiplier=None,
        iteration_losses=iteration_losses,
        epoch_losses=epoch_losses,
        epsilons=epsilons,
        test_losses=test_losses,
        test_accuracies=test_accuracies,
        final_test_loss=test_losses[-1],
        final_accuracy=test_accuracies[-1],
        final_epsilon=None,
    )



def run_dp_ftrl(target_epsilon: float, seed: int = SEED) -> RunMetrics:
    seed_everything(seed)

    model = build_model()
    optimizer = build_ftrl_optimizer(model, record_last_noise=True)
    per_sample_grad_fn = make_per_sample_grad_fn(model)
    noise_multiplier = solve_dpftrl_noise_multiplier(target_epsilon)
    train_loader = make_train_loader(seed=seed, shuffle=False)

    shapes = [p.shape for p in model.parameters()]
    cumm_noise = CummuNoiseEffTorch(
        std=noise_multiplier * MAX_GRAD_NORM / BATCH_SIZE,
        shapes=shapes,
        device=DEVICE,
    )

    iteration_losses, epoch_losses = [], []
    epsilons, test_losses, test_accuracies = [], [], []

    for epoch in range(1, EPOCHS + 1):
        iter_losses, epoch_loss = train_one_epoch_ftrl(
            model=model,
            loader=train_loader,
            optimizer=optimizer,
            alpha=ALPHA_FTRL,
            private=True,
            noise_source=cumm_noise,
            max_grad_norm=MAX_GRAD_NORM,
            per_sample_grad_fn=per_sample_grad_fn,
        )
        test_loss, test_acc = evaluate(model, test_loader)
        eps_spent = dpftrl_epsilon_from_noise(noise_multiplier, epoch)

        iteration_losses.extend(iter_losses)
        epoch_losses.append(epoch_loss)
        epsilons.append(eps_spent)
        test_losses.append(test_loss)
        test_accuracies.append(test_acc)

        print(
            f"[DP-FTRL, target ε={target_epsilon}] Epoch {epoch:02d}/{EPOCHS} | "
            f"Train Loss: {epoch_loss:.4f} | "
            f"Epsilon Spent: {eps_spent:.4f} | "
            f"Test Loss: {test_loss:.4f} | "
            f"Test Acc: {test_acc:.2f}%"
        )

        if epoch < EPOCHS:
            cumm_noise = CummuNoiseEffTorch(
                std=noise_multiplier * MAX_GRAD_NORM / BATCH_SIZE,
                shapes=shapes,
                device=DEVICE,
            )

    return RunMetrics(
        method="DP-FTRL",
        target_epsilon=target_epsilon,
        noise_multiplier=noise_multiplier,
        iteration_losses=iteration_losses,
        epoch_losses=epoch_losses,
        epsilons=epsilons,
        test_losses=test_losses,
        test_accuracies=test_accuracies,
        final_test_loss=test_losses[-1],
        final_accuracy=test_accuracies[-1],
        final_epsilon=epsilons[-1],
    )


# =========================
# Reporting / plotting
# =========================
def summarize_results(results: dict[str, RunMetrics]) -> pd.DataFrame:
    rows = []
    for key, res in results.items():
        rows.append(
            {
                "run": key,
                "method": res.method,
                "target_epsilon": res.target_epsilon,
                "noise_multiplier": res.noise_multiplier,
                "final_epsilon": res.final_epsilon,
                "final_test_loss": res.final_test_loss,
                "final_test_accuracy": res.final_accuracy,
            }
        )
    df = pd.DataFrame(rows)
    df.to_csv(OUTPUT_DIR / "q5_summary.csv", index=False)
    return df



def plot_iterations_vs_training_loss(results: dict[str, RunMetrics]) -> None:
    plt.figure(figsize=(10, 6))
    for key, res in results.items():
        if res.method in ["DP-SGD", "DP-FTRL"]:
            plt.plot(moving_average(res.iteration_losses, window=20), label=key)
    plt.xlabel("Iteration")
    plt.ylabel("Training Loss (smoothed)")
    plt.title("Iterations vs Training Loss for DP-SGD and DP-FTRL")
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / "q5_private_iterations_vs_loss.png", dpi=300)
    plt.close()

    plt.figure(figsize=(10, 6))
    for key in ["SGD", "FTRL"]:
        res = results[key]
        plt.plot(moving_average(res.iteration_losses, window=20), label=key)
    plt.xlabel("Iteration")
    plt.ylabel("Training Loss (smoothed)")
    plt.title("Iterations vs Training Loss for Non-Private Baselines")
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / "q5_nonprivate_iterations_vs_loss.png", dpi=300)
    plt.close()



def plot_privacy_spent_vs_epochs(results: dict[str, RunMetrics]) -> None:
    epochs_axis = np.arange(1, EPOCHS + 1)
    plt.figure(figsize=(9, 6))
    for key, res in results.items():
        if res.method in ["DP-SGD", "DP-FTRL"]:
            plt.plot(epochs_axis, res.epsilons, marker="o", label=key)
    plt.xlabel("Epoch")
    plt.ylabel("Epsilon spent")
    plt.title("Privacy Spent vs Epochs for DP-SGD and DP-FTRL")
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / "q5_privacy_spent_vs_epochs.png", dpi=300)
    plt.close()



def plot_target_epsilon_vs_test_accuracy(summary_df: pd.DataFrame) -> None:
    dp_sgd_plot = summary_df[summary_df["method"] == "DP-SGD"].sort_values("target_epsilon")
    dp_ftrl_plot = summary_df[summary_df["method"] == "DP-FTRL"].sort_values("target_epsilon")

    sgd_acc = summary_df.loc[summary_df["method"] == "SGD", "final_test_accuracy"].iloc[0]
    ftrl_acc = summary_df.loc[summary_df["method"] == "FTRL", "final_test_accuracy"].iloc[0]

    plt.figure(figsize=(9, 6))
    plt.plot(dp_sgd_plot["target_epsilon"], dp_sgd_plot["final_test_accuracy"], marker="o", label="DP-SGD")
    plt.plot(dp_ftrl_plot["target_epsilon"], dp_ftrl_plot["final_test_accuracy"], marker="o", label="DP-FTRL")
    plt.axhline(y=sgd_acc, linestyle="--", label=f"SGD (non-private): {sgd_acc:.2f}%")
    plt.axhline(y=ftrl_acc, linestyle="--", label=f"FTRL (non-private): {ftrl_acc:.2f}%")
    plt.xlabel("Target epsilon")
    plt.ylabel("Final test accuracy (%)")
    plt.title("Target Epsilon vs Test Accuracy")
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / "q5_target_epsilon_vs_test_accuracy.png", dpi=300)
    plt.close()


# =========================
# Main
# =========================
def main() -> None:
    print(f"Using device: {DEVICE}")
    print(f"Training samples: {N_TRAIN}")
    print(f"Test samples: {len(test_dataset)}")
    print(f"Batches per epoch: {NUM_BATCHES}")
    print(f"Sample rate: {SAMPLE_RATE:.6f}")
    print()

    print("Solved DP-SGD noise multipliers:")
    for eps in TARGET_EPSILONS:
        nm = get_noise_multiplier(
            target_epsilon=eps,
            target_delta=DELTA,
            sample_rate=SAMPLE_RATE,
            epochs=EPOCHS,
            accountant=ACCOUNTANT,
        )
        print(f"  target ε={eps}: noise_multiplier={nm:.6f}")

    print("\nSolved DP-FTRL noise multipliers:")
    for eps in TARGET_EPSILONS:
        nm = solve_dpftrl_noise_multiplier(eps)
        print(f"  target ε={eps}: noise_multiplier={nm:.6f}")

    results: dict[str, RunMetrics] = {}

    print("\n" + "=" * 100)
    results["SGD"] = run_nonprivate_sgd()

    print("=" * 100)
    results["FTRL"] = run_nonprivate_ftrl()

    for eps in TARGET_EPSILONS:
        print("=" * 100)
        results[f"DP-SGD-ε{int(eps)}"] = run_dp_sgd(eps)

    for eps in TARGET_EPSILONS:
        print("=" * 100)
        results[f"DP-FTRL-ε{int(eps)}"] = run_dp_ftrl(eps)

    print("=" * 100)
    print("All Question 5 experiments completed.")

    summary_df = summarize_results(results)
    print("\nSummary:")
    print(summary_df.to_string(index=False))

    plot_iterations_vs_training_loss(results)
    plot_privacy_spent_vs_epochs(results)
    plot_target_epsilon_vs_test_accuracy(summary_df)

    print(f"\nSaved outputs to: {OUTPUT_DIR.resolve()}")


if __name__ == "__main__":
    main()
