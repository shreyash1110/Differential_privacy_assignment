#!/usr/bin/env python3
"""
Question 3: Effect of Gradient Clipping Norm in DP-SGD

This script trains a feed-forward neural network on MNIST using DP-SGD
with two clipping norms C in {0.1, 10.0}, while keeping the noise multiplier
and batch size fixed. It saves:
    1. Iteration vs Training Loss plot
    2. Summary CSV
    3. Full JSON results

Author: Shreyash
"""

from __future__ import annotations

import argparse
import json
import random
from pathlib import Path
from typing import Dict, List, Tuple

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision import datasets, transforms

try:
    from opacus import PrivacyEngine
    from opacus.accountants.utils import get_noise_multiplier
except ImportError as exc:
    raise ImportError(
        "Opacus is not installed. Install it with: pip install opacus"
    ) from exc


# =========================
# Default configuration
# =========================
SEED = 42

BATCH_SIZE = 64
TEST_BATCH_SIZE = 1024
EPOCHS = 10

LR = 0.1
MOMENTUM = 0.0

DELTA = 1e-5
ACCOUNTANT = "rdp"

CLIP_NORMS = [0.1, 10.0]
TARGET_EPSILON_FOR_FIXED_SIGMA = 5.0

CLIPPING_MODE = "flat"
GRAD_SAMPLE_MODE = "ghost"

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def seed_everything(seed: int = SEED) -> None:
    """Seed Python, NumPy, and PyTorch for reproducibility."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)

    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False


def get_data_loaders(
    batch_size: int,
    test_batch_size: int,
    data_root: str,
    seed: int,
) -> Tuple[datasets.MNIST, datasets.MNIST, DataLoader, DataLoader]:
    """Create MNIST datasets and data loaders."""
    transform = transforms.Compose(
        [
            transforms.ToTensor(),
            transforms.Normalize((0.1307,), (0.3081,)),
        ]
    )

    train_dataset = datasets.MNIST(
        root=data_root,
        train=True,
        download=True,
        transform=transform,
    )
    test_dataset = datasets.MNIST(
        root=data_root,
        train=False,
        download=True,
        transform=transform,
    )

    generator = torch.Generator()
    generator.manual_seed(seed)

    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        generator=generator,
        num_workers=0,
        pin_memory=torch.cuda.is_available(),
    )
    test_loader = DataLoader(
        test_dataset,
        batch_size=test_batch_size,
        shuffle=False,
        num_workers=0,
        pin_memory=torch.cuda.is_available(),
    )

    return train_dataset, test_dataset, train_loader, test_loader


class FeedForwardNet(nn.Module):
    """Feed-forward neural network for MNIST."""

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
    """Build and move the model to the configured device."""
    return FeedForwardNet().to(DEVICE)


def build_optimizer(model: nn.Module, lr: float, momentum: float) -> optim.Optimizer:
    """Build the SGD optimizer."""
    return optim.SGD(model.parameters(), lr=lr, momentum=momentum)


def train_one_epoch(
    model: nn.Module,
    loader: DataLoader,
    optimizer: optim.Optimizer,
    criterion_private: nn.Module,
) -> Tuple[List[float], float]:
    """Train the model for one epoch and return per-iteration and average epoch loss."""
    model.train()
    iteration_losses: List[float] = []
    running_loss = 0.0

    for x, y in loader:
        x, y = x.to(DEVICE), y.to(DEVICE)

        optimizer.zero_grad()
        logits = model(x)
        loss = criterion_private(logits, y)
        loss.backward()
        optimizer.step()

        loss_value = float(loss.item())
        iteration_losses.append(loss_value)
        running_loss += loss_value

    epoch_loss = running_loss / len(loader)
    return iteration_losses, epoch_loss


@torch.no_grad()
def evaluate(model: nn.Module, loader: DataLoader) -> Tuple[float, float]:
    """Evaluate the model and return average loss and accuracy."""
    criterion_eval = nn.CrossEntropyLoss()

    model.eval()
    total_loss = 0.0
    correct = 0
    total = 0

    for x, y in loader:
        x, y = x.to(DEVICE), y.to(DEVICE)
        logits = model(x)
        loss = criterion_eval(logits, y)

        total_loss += float(loss.item()) * x.size(0)
        preds = logits.argmax(dim=1)
        correct += int((preds == y).sum().item())
        total += int(x.size(0))

    avg_loss = total_loss / total
    accuracy = 100.0 * correct / total
    return avg_loss, accuracy


def compute_fixed_noise_multiplier(
    sample_rate: float,
    epochs: int,
    target_epsilon: float,
    delta: float,
    accountant: str,
) -> float:
    """Calibrate one fixed noise multiplier and reuse it across all clip norms."""
    return float(
        get_noise_multiplier(
            target_epsilon=target_epsilon,
            target_delta=delta,
            sample_rate=sample_rate,
            epochs=epochs,
            accountant=accountant,
        )
    )


def run_dp_experiment_for_clip_norm(
    clip_norm: float,
    train_loader: DataLoader,
    test_loader: DataLoader,
    fixed_noise_multiplier: float,
    epochs: int,
    delta: float,
    accountant: str,
    clipping_mode: str,
    grad_sample_mode: str,
    lr: float,
    momentum: float,
    seed: int,
) -> Dict[str, object]:
    """Run DP-SGD training for one clipping norm and collect metrics."""
    seed_everything(seed)

    model = build_model()
    optimizer = build_optimizer(model, lr=lr, momentum=momentum)
    criterion = nn.CrossEntropyLoss()

    privacy_engine = PrivacyEngine(accountant=accountant)

    model, optimizer, criterion_private, private_train_loader = privacy_engine.make_private(
        module=model,
        optimizer=optimizer,
        criterion=criterion,
        data_loader=train_loader,
        noise_multiplier=fixed_noise_multiplier,
        max_grad_norm=clip_norm,
        clipping=clipping_mode,
        grad_sample_mode=grad_sample_mode,
    )

    metrics: Dict[str, object] = {
        "clip_norm": clip_norm,
        "iteration_losses": [],
        "epoch_losses": [],
        "epsilons": [],
        "test_losses": [],
        "test_accuracies": [],
    }

    for epoch in range(1, epochs + 1):
        iter_losses, epoch_loss = train_one_epoch(
            model=model,
            loader=private_train_loader,
            optimizer=optimizer,
            criterion_private=criterion_private,
        )

        test_loss, test_acc = evaluate(model, test_loader)
        eps_spent = float(privacy_engine.get_epsilon(delta=delta))

        metrics["iteration_losses"].extend(iter_losses)
        metrics["epoch_losses"].append(epoch_loss)
        metrics["epsilons"].append(eps_spent)
        metrics["test_losses"].append(test_loss)
        metrics["test_accuracies"].append(test_acc)

        print(
            f"[C = {clip_norm}] Epoch {epoch:02d}/{epochs} | "
            f"Train Loss: {epoch_loss:.4f} | "
            f"Epsilon Spent: {eps_spent:.4f} | "
            f"Test Loss: {test_loss:.4f} | "
            f"Test Acc: {test_acc:.2f}%"
        )

    metrics["final_epsilon"] = metrics["epsilons"][-1]
    metrics["final_test_loss"] = metrics["test_losses"][-1]
    metrics["final_test_accuracy"] = metrics["test_accuracies"][-1]

    return metrics


def save_plot(results: Dict[float, Dict[str, object]], output_path: Path) -> None:
    """Save the iterations vs training loss plot."""
    plt.figure(figsize=(9, 5))

    for clip_norm, res in results.items():
        iteration_losses = res["iteration_losses"]
        iterations = np.arange(1, len(iteration_losses) + 1)
        plt.plot(iterations, iteration_losses, label=f"C = {clip_norm}")

    plt.xlabel("Iteration")
    plt.ylabel("Training Loss")
    plt.title("Iterations vs Training Loss for Different Clipping Norms")
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(output_path, dpi=300)
    plt.close()


def save_summary(
    results: Dict[float, Dict[str, object]],
    fixed_noise_multiplier: float,
    output_csv: Path,
) -> pd.DataFrame:
    """Save a summary CSV of final metrics."""
    summary_rows = []
    for clip_norm, res in results.items():
        summary_rows.append(
            {
                "clip_norm": clip_norm,
                "noise_multiplier": fixed_noise_multiplier,
                "final_epsilon": res["final_epsilon"],
                "final_test_loss": res["final_test_loss"],
                "final_test_accuracy": res["final_test_accuracy"],
            }
        )

    summary_df = pd.DataFrame(summary_rows).sort_values("clip_norm")
    summary_df.to_csv(output_csv, index=False)
    return summary_df


def save_full_results(results: Dict[float, Dict[str, object]], output_json: Path) -> None:
    """Save full experiment results as JSON."""
    serializable_results = {str(k): v for k, v in results.items()}
    with output_json.open("w", encoding="utf-8") as f:
        json.dump(serializable_results, f, indent=2)


def parse_args() -> argparse.Namespace:
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(
        description="Run Question 3 experiments for DP-SGD gradient clipping on MNIST."
    )
    parser.add_argument("--data-root", type=str, default="./data", help="Directory for MNIST data.")
    parser.add_argument("--output-dir", type=str, default="./q3_outputs", help="Directory for outputs.")
    parser.add_argument("--epochs", type=int, default=EPOCHS, help="Number of training epochs.")
    parser.add_argument("--batch-size", type=int, default=BATCH_SIZE, help="Training batch size.")
    parser.add_argument(
        "--test-batch-size",
        type=int,
        default=TEST_BATCH_SIZE,
        help="Test batch size.",
    )
    parser.add_argument("--lr", type=float, default=LR, help="Learning rate.")
    parser.add_argument("--momentum", type=float, default=MOMENTUM, help="SGD momentum.")
    parser.add_argument("--delta", type=float, default=DELTA, help="Target delta for DP.")
    parser.add_argument(
        "--target-epsilon",
        type=float,
        default=TARGET_EPSILON_FOR_FIXED_SIGMA,
        help="Target epsilon used to calibrate one fixed noise multiplier.",
    )
    parser.add_argument(
        "--accountant",
        type=str,
        default=ACCOUNTANT,
        choices=["rdp", "gdp", "prv"],
        help="Privacy accountant.",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=SEED,
        help="Random seed.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    seed_everything(args.seed)

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    print(f"Using device: {DEVICE}")
    print(f"Accountant: {args.accountant}")
    print(f"Clipping mode: {CLIPPING_MODE}")
    print(f"Grad sample mode: {GRAD_SAMPLE_MODE}")

    train_dataset, test_dataset, train_loader, test_loader = get_data_loaders(
        batch_size=args.batch_size,
        test_batch_size=args.test_batch_size,
        data_root=args.data_root,
        seed=args.seed,
    )

    print(f"Training samples: {len(train_dataset)}")
    print(f"Test samples: {len(test_dataset)}")
    print(f"Batches per epoch: {len(train_loader)}")

    sample_rate = 1.0 / len(train_loader)
    fixed_noise_multiplier = compute_fixed_noise_multiplier(
        sample_rate=sample_rate,
        epochs=args.epochs,
        target_epsilon=args.target_epsilon,
        delta=args.delta,
        accountant=args.accountant,
    )

    print(f"Fixed noise multiplier used for both runs: {fixed_noise_multiplier:.6f}")

    results: Dict[float, Dict[str, object]] = {}
    for clip_norm in CLIP_NORMS:
        print("=" * 80)
        print(f"Running DP-SGD with clipping norm C = {clip_norm}")
        results[clip_norm] = run_dp_experiment_for_clip_norm(
            clip_norm=clip_norm,
            train_loader=train_loader,
            test_loader=test_loader,
            fixed_noise_multiplier=fixed_noise_multiplier,
            epochs=args.epochs,
            delta=args.delta,
            accountant=args.accountant,
            clipping_mode=CLIPPING_MODE,
            grad_sample_mode=GRAD_SAMPLE_MODE,
            lr=args.lr,
            momentum=args.momentum,
            seed=args.seed,
        )

    plot_path = output_dir / "q3_iterations_vs_training_loss.png"
    csv_path = output_dir / "q3_summary.csv"
    json_path = output_dir / "q3_full_results.json"

    save_plot(results, plot_path)
    summary_df = save_summary(results, fixed_noise_multiplier, csv_path)
    save_full_results(results, json_path)

    print("=" * 80)
    print("All Question 3 experiments completed.")
    print("\nSummary:")
    print(summary_df.to_string(index=False))
    print(f"\nSaved plot to: {plot_path}")
    print(f"Saved summary CSV to: {csv_path}")
    print(f"Saved full JSON results to: {json_path}")


if __name__ == "__main__":
    main()
