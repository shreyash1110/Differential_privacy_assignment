"""
CS729 Differential Privacy in Machine Learning
Question 1: Optimizer comparison for non-private and private training on MNIST.

This script modularizes the notebook solution into reusable functions and a clean main() entrypoint.
It runs:
  - non-private SGD
  - non-private Adam
  - DP-SGD with target epsilon in {1, 10}
  - DP-Adam with target epsilon in {1, 10}

It produces:
  (a) Number of Iterations vs Training Loss
  (b) Epsilon Spent vs Epochs
  (c) Target Epsilon vs Test Accuracy (with non-private baselines)
  - a summary CSV
  - a text summary with key observations
"""

from __future__ import annotations

import argparse
import random
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

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
except ImportError as exc:
    raise ImportError(
        "Opacus is required for this script. Install it with: pip install opacus"
    ) from exc


@dataclass
class Config:
    seed: int = 42
    batch_size: int = 64
    test_batch_size: int = 1024
    epochs: int = 10
    delta: float = 1e-5
    target_epsilons: Tuple[float, ...] = (1.0, 10.0)
    max_grad_norm: float = 1.0
    lr_sgd: float = 0.1
    lr_adam: float = 1e-3
    accountant: str = "rdp"
    smoothing_window: int = 50
    data_root: str = "./data"
    output_dir: str = "./q1_outputs"

    @property
    def device(self) -> torch.device:
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")


class MNISTFFNN(nn.Module):
    """Simple feed-forward neural network for MNIST."""

    def __init__(self) -> None:
        super().__init__()
        self.flatten = nn.Flatten()
        self.network = nn.Sequential(
            nn.Linear(28 * 28, 512),
            nn.ReLU(),
            nn.Linear(512, 256),
            nn.ReLU(),
            nn.Linear(256, 10),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.flatten(x)
        return self.network(x)


def seed_everything(seed: int) -> None:
    """Seed all relevant RNGs for reproducibility."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def build_transforms() -> transforms.Compose:
    """Build the MNIST preprocessing pipeline."""
    return transforms.Compose(
        [
            transforms.ToTensor(),
            transforms.Normalize((0.1307,), (0.3081,)),
        ]
    )


def load_datasets(config: Config) -> Tuple[datasets.MNIST, datasets.MNIST]:
    """Load MNIST train and test datasets."""
    transform = build_transforms()
    train_dataset = datasets.MNIST(
        root=config.data_root,
        train=True,
        download=True,
        transform=transform,
    )
    test_dataset = datasets.MNIST(
        root=config.data_root,
        train=False,
        download=True,
        transform=transform,
    )
    return train_dataset, test_dataset


def make_train_loader(
    train_dataset: datasets.MNIST,
    config: Config,
    seed: Optional[int] = None,
    shuffle: bool = True,
) -> DataLoader:
    """Create a reproducible training DataLoader."""
    generator = torch.Generator()
    generator.manual_seed(config.seed if seed is None else seed)
    return DataLoader(
        train_dataset,
        batch_size=config.batch_size,
        shuffle=shuffle,
        generator=generator,
        num_workers=0,
        pin_memory=torch.cuda.is_available(),
    )


def make_test_loader(test_dataset: datasets.MNIST, config: Config) -> DataLoader:
    """Create the evaluation DataLoader."""
    return DataLoader(
        test_dataset,
        batch_size=config.test_batch_size,
        shuffle=False,
        num_workers=0,
        pin_memory=torch.cuda.is_available(),
    )


def build_model(config: Config) -> nn.Module:
    """Instantiate the model on the configured device."""
    return MNISTFFNN().to(config.device)


def build_optimizer(opt_name: str, model: nn.Module, config: Config) -> optim.Optimizer:
    """Construct either SGD or Adam."""
    if opt_name == "SGD":
        return optim.SGD(model.parameters(), lr=config.lr_sgd)
    if opt_name == "Adam":
        return optim.Adam(model.parameters(), lr=config.lr_adam)
    raise ValueError(f"Unsupported optimizer: {opt_name}")


def train_one_epoch(
    model: nn.Module,
    train_loader: DataLoader,
    optimizer: optim.Optimizer,
    config: Config,
    privacy_engine: Optional[PrivacyEngine] = None,
    start_iteration: int = 0,
) -> Tuple[List[int], List[float], float, Optional[float]]:
    """
    Train for one epoch and collect:
      - per-iteration losses,
      - corresponding global iteration indices,
      - epoch-average training loss,
      - epsilon spent so far (if private).
    """
    model.train()
    criterion = nn.CrossEntropyLoss()

    iteration_losses: List[float] = []
    iteration_steps: List[int] = []
    running_loss = 0.0
    num_examples = 0
    iteration = start_iteration

    for data, target in train_loader:
        data, target = data.to(config.device), target.to(config.device)

        optimizer.zero_grad()
        output = model(data)
        loss = criterion(output, target)
        loss.backward()
        optimizer.step()

        loss_value = float(loss.item())
        batch_size = int(data.size(0))
        iteration += 1

        iteration_steps.append(iteration)
        iteration_losses.append(loss_value)
        running_loss += loss_value * batch_size
        num_examples += batch_size

    epoch_loss = running_loss / max(num_examples, 1)
    epsilon = None if privacy_engine is None else float(privacy_engine.get_epsilon(config.delta))
    return iteration_steps, iteration_losses, epoch_loss, epsilon


@torch.no_grad()
def evaluate(model: nn.Module, data_loader: DataLoader, config: Config) -> Tuple[float, float]:
    """Evaluate average test loss and test accuracy."""
    model.eval()
    criterion = nn.CrossEntropyLoss(reduction="sum")

    total_loss = 0.0
    correct = 0
    total_examples = 0

    for data, target in data_loader:
        data, target = data.to(config.device), target.to(config.device)
        output = model(data)
        total_loss += float(criterion(output, target).item())
        pred = output.argmax(dim=1)
        correct += int((pred == target).sum().item())
        total_examples += int(target.size(0))

    avg_loss = total_loss / max(total_examples, 1)
    accuracy = 100.0 * correct / max(total_examples, 1)
    return avg_loss, accuracy


def smooth_curve(values: List[float], window: int) -> np.ndarray:
    """Moving-average smoothing used for the training-loss curves."""
    array = np.asarray(values, dtype=float)
    if len(array) == 0:
        return array
    if len(array) < window:
        return array
    kernel = np.ones(window, dtype=float) / window
    return np.convolve(array, kernel, mode="valid")


def initialize_metrics(opt_name: str, private: bool, target_epsilon: float) -> Dict[str, Any]:
    """Create a metrics dictionary for one run."""
    return {
        "optimizer": opt_name,
        "private": private,
        "target_epsilon": target_epsilon,
        "noise_multiplier": None,
        "iteration_steps": [],
        "iteration_losses": [],
        "epoch_losses": [],
        "epsilons": [],
        "test_losses": [],
        "test_accuracies": [],
    }


def finalize_metrics(metrics: Dict[str, Any]) -> Dict[str, Any]:
    """Add final summary fields to a completed metrics dictionary."""
    metrics["final_test_loss"] = metrics["test_losses"][-1]
    metrics["final_accuracy"] = metrics["test_accuracies"][-1]
    metrics["final_epsilon"] = (
        metrics["epsilons"][-1] if metrics["private"] else float("inf")
    )
    return metrics


def run_non_private_experiment(
    opt_name: str,
    train_dataset: datasets.MNIST,
    test_dataset: datasets.MNIST,
    config: Config,
    seed: Optional[int] = None,
) -> Dict[str, Any]:
    """Run non-private training for one optimizer."""
    run_seed = config.seed if seed is None else seed
    seed_everything(run_seed)

    model = build_model(config)
    optimizer = build_optimizer(opt_name, model, config)
    train_loader = make_train_loader(train_dataset, config, seed=run_seed, shuffle=True)
    test_loader = make_test_loader(test_dataset, config)

    metrics = initialize_metrics(opt_name=opt_name, private=False, target_epsilon=float("inf"))
    global_iteration = 0

    for epoch in range(1, config.epochs + 1):
        iter_steps, iter_losses, epoch_loss, _ = train_one_epoch(
            model=model,
            train_loader=train_loader,
            optimizer=optimizer,
            config=config,
            privacy_engine=None,
            start_iteration=global_iteration,
        )

        global_iteration = iter_steps[-1]
        test_loss, test_acc = evaluate(model, test_loader, config)

        metrics["iteration_steps"].extend(iter_steps)
        metrics["iteration_losses"].extend(iter_losses)
        metrics["epoch_losses"].append(epoch_loss)
        metrics["epsilons"].append(float("inf"))
        metrics["test_losses"].append(test_loss)
        metrics["test_accuracies"].append(test_acc)

        print(
            f"[Non-Private {opt_name}] Epoch {epoch:02d}/{config.epochs} | "
            f"Train Loss: {epoch_loss:.4f} | Test Loss: {test_loss:.4f} | Test Acc: {test_acc:.2f}%"
        )

    return finalize_metrics(metrics)


def make_private_training_components(
    model: nn.Module,
    optimizer: optim.Optimizer,
    train_loader: DataLoader,
    target_epsilon: float,
    config: Config,
) -> Tuple[PrivacyEngine, nn.Module, optim.Optimizer, DataLoader]:
    """Wrap model/optimizer/loader using Opacus' make_private_with_epsilon."""
    privacy_engine = PrivacyEngine(accountant=config.accountant)
    model, optimizer, private_train_loader = privacy_engine.make_private_with_epsilon(
        module=model,
        optimizer=optimizer,
        data_loader=train_loader,
        target_epsilon=target_epsilon,
        target_delta=config.delta,
        epochs=config.epochs,
        max_grad_norm=config.max_grad_norm,
    )
    return privacy_engine, model, optimizer, private_train_loader


def run_private_experiment(
    opt_name: str,
    target_epsilon: float,
    train_dataset: datasets.MNIST,
    test_dataset: datasets.MNIST,
    config: Config,
    seed: Optional[int] = None,
) -> Dict[str, Any]:
    """Run private training for one optimizer and one target epsilon."""
    run_seed = config.seed if seed is None else seed
    seed_everything(run_seed)

    model = build_model(config)
    optimizer = build_optimizer(opt_name, model, config)
    train_loader = make_train_loader(train_dataset, config, seed=run_seed, shuffle=True)
    test_loader = make_test_loader(test_dataset, config)

    privacy_engine, model, optimizer, private_train_loader = make_private_training_components(
        model=model,
        optimizer=optimizer,
        train_loader=train_loader,
        target_epsilon=target_epsilon,
        config=config,
    )

    metrics = initialize_metrics(opt_name=opt_name, private=True, target_epsilon=target_epsilon)
    metrics["noise_multiplier"] = getattr(privacy_engine, "noise_multiplier", None)
    global_iteration = 0

    for epoch in range(1, config.epochs + 1):
        iter_steps, iter_losses, epoch_loss, epsilon = train_one_epoch(
            model=model,
            train_loader=private_train_loader,
            optimizer=optimizer,
            config=config,
            privacy_engine=privacy_engine,
            start_iteration=global_iteration,
        )

        global_iteration = iter_steps[-1]
        test_loss, test_acc = evaluate(model, test_loader, config)

        metrics["iteration_steps"].extend(iter_steps)
        metrics["iteration_losses"].extend(iter_losses)
        metrics["epoch_losses"].append(epoch_loss)
        metrics["epsilons"].append(epsilon)
        metrics["test_losses"].append(test_loss)
        metrics["test_accuracies"].append(test_acc)

        print(
            f"[DP-{opt_name} | target eps={target_epsilon}] Epoch {epoch:02d}/{config.epochs} | "
            f"Train Loss: {epoch_loss:.4f} | Epsilon Spent: {epsilon:.4f} | "
            f"Test Loss: {test_loss:.4f} | Test Acc: {test_acc:.2f}%"
        )

    return finalize_metrics(metrics)


def run_full_experiment_grid(
    train_dataset: datasets.MNIST,
    test_dataset: datasets.MNIST,
    config: Config,
) -> Dict[str, Dict[str, Any]]:
    """Run all six Q1 experiments."""
    results: Dict[str, Dict[str, Any]] = {}

    results["SGD_non_private"] = run_non_private_experiment(
        opt_name="SGD",
        train_dataset=train_dataset,
        test_dataset=test_dataset,
        config=config,
    )
    results["Adam_non_private"] = run_non_private_experiment(
        opt_name="Adam",
        train_dataset=train_dataset,
        test_dataset=test_dataset,
        config=config,
    )

    for eps in config.target_epsilons:
        eps_label = int(eps) if float(eps).is_integer() else eps
        results[f"SGD_dp_eps{eps_label}"] = run_private_experiment(
            opt_name="SGD",
            target_epsilon=eps,
            train_dataset=train_dataset,
            test_dataset=test_dataset,
            config=config,
        )
        results[f"Adam_dp_eps{eps_label}"] = run_private_experiment(
            opt_name="Adam",
            target_epsilon=eps,
            train_dataset=train_dataset,
            test_dataset=test_dataset,
            config=config,
        )

    print("All Question 1 experiments completed.")
    return results


def build_summary_dataframe(results: Dict[str, Dict[str, Any]]) -> pd.DataFrame:
    """Convert the experiment dictionary into a compact summary table."""
    summary_rows: List[Dict[str, Any]] = []
    for run_name, metrics in results.items():
        summary_rows.append(
            {
                "Run": run_name,
                "Optimizer": metrics["optimizer"],
                "Private": metrics["private"],
                "Target Epsilon": metrics["target_epsilon"],
                "Final Epsilon Spent": metrics["final_epsilon"],
                "Noise Multiplier": metrics["noise_multiplier"],
                "Final Test Loss": metrics["final_test_loss"],
                "Final Test Accuracy (%)": metrics["final_accuracy"],
            }
        )

    summary_df = pd.DataFrame(summary_rows)
    summary_df = summary_df.sort_values(
        by=["Private", "Optimizer", "Target Epsilon"]
    ).reset_index(drop=True)
    return summary_df


def format_summary_dataframe(summary_df: pd.DataFrame) -> pd.DataFrame:
    """Create a display-friendly version of the summary table."""
    display_df = summary_df.copy()
    display_df["Target Epsilon"] = display_df["Target Epsilon"].replace(float("inf"), "non-private")
    display_df["Final Epsilon Spent"] = display_df["Final Epsilon Spent"].apply(
        lambda x: "inf" if np.isinf(x) else round(float(x), 4)
    )
    display_df["Noise Multiplier"] = display_df["Noise Multiplier"].apply(
        lambda x: "-" if x is None else round(float(x), 4)
    )
    display_df["Final Test Loss"] = display_df["Final Test Loss"].round(4)
    display_df["Final Test Accuracy (%)"] = display_df["Final Test Accuracy (%)"].round(2)
    return display_df


def plot_smoothed_curve(
    ax: plt.Axes,
    steps: List[int],
    values: List[float],
    label: str,
    smoothing_window: int,
    linestyle: str = "-",
) -> None:
    """Plot a smoothed line on an existing axis."""
    smoothed = smooth_curve(values, window=smoothing_window)
    if len(smoothed) == 0:
        return
    aligned_steps = np.asarray(steps[-len(smoothed):])
    ax.plot(aligned_steps, smoothed, label=label, linestyle=linestyle, linewidth=2)


def plot_training_loss_vs_iterations(
    results: Dict[str, Dict[str, Any]],
    config: Config,
    output_dir: Path,
) -> Path:
    """Plot Number of Iterations vs Training Loss for each target epsilon."""
    fig, axes = plt.subplots(1, 2, figsize=(16, 5), sharey=True)

    for ax, eps in zip(axes, config.target_epsilons):
        eps_label = int(eps) if float(eps).is_integer() else eps
        plot_smoothed_curve(
            ax,
            results["SGD_non_private"]["iteration_steps"],
            results["SGD_non_private"]["iteration_losses"],
            "SGD (non-private)",
            smoothing_window=config.smoothing_window,
            linestyle="--",
        )
        plot_smoothed_curve(
            ax,
            results["Adam_non_private"]["iteration_steps"],
            results["Adam_non_private"]["iteration_losses"],
            "Adam (non-private)",
            smoothing_window=config.smoothing_window,
            linestyle=":",
        )
        plot_smoothed_curve(
            ax,
            results[f"SGD_dp_eps{eps_label}"]["iteration_steps"],
            results[f"SGD_dp_eps{eps_label}"]["iteration_losses"],
            f"DP-SGD (target eps={eps_label})",
            smoothing_window=config.smoothing_window,
        )
        plot_smoothed_curve(
            ax,
            results[f"Adam_dp_eps{eps_label}"]["iteration_steps"],
            results[f"Adam_dp_eps{eps_label}"]["iteration_losses"],
            f"DP-Adam (target eps={eps_label})",
            smoothing_window=config.smoothing_window,
        )
        ax.set_title(f"Training Loss vs Iterations (target eps={eps_label})")
        ax.set_xlabel("Number of Iterations")
        ax.set_ylabel("Training Loss")
        ax.legend()

    plt.tight_layout()
    output_path = output_dir / "plot_a_iterations_vs_training_loss.png"
    plt.savefig(output_path, dpi=300, bbox_inches="tight")
    plt.close(fig)
    return output_path


def plot_epsilon_vs_epochs(
    results: Dict[str, Dict[str, Any]],
    config: Config,
    output_dir: Path,
) -> Path:
    """Plot epsilon spent vs epochs for each target epsilon."""
    epoch_axis = np.arange(1, config.epochs + 1)
    fig, axes = plt.subplots(1, 2, figsize=(16, 5), sharey=False)

    for ax, eps in zip(axes, config.target_epsilons):
        eps_label = int(eps) if float(eps).is_integer() else eps
        ax.plot(
            epoch_axis,
            results[f"SGD_dp_eps{eps_label}"]["epsilons"],
            marker="o",
            linewidth=2,
            label="DP-SGD",
        )
        ax.plot(
            epoch_axis,
            results[f"Adam_dp_eps{eps_label}"]["epsilons"],
            marker="s",
            linewidth=2,
            label="DP-Adam",
        )
        ax.axhline(
            eps,
            color="black",
            linestyle="--",
            linewidth=1.5,
            alpha=0.7,
            label="Target epsilon",
        )
        ax.set_title(f"Epsilon Spent vs Epochs (target eps={eps_label})")
        ax.set_xlabel("Epochs")
        ax.set_ylabel("Epsilon Spent")
        ax.set_xticks(epoch_axis)
        ax.legend()

    plt.tight_layout()
    output_path = output_dir / "plot_b_epsilon_vs_epochs.png"
    plt.savefig(output_path, dpi=300, bbox_inches="tight")
    plt.close(fig)
    return output_path


def plot_target_epsilon_vs_test_accuracy(
    results: Dict[str, Dict[str, Any]],
    config: Config,
    output_dir: Path,
) -> Path:
    """Plot target epsilon vs final test accuracy, including non-private baselines."""
    eps_values = np.array(config.target_epsilons, dtype=float)
    sgd_dp_accs = []
    adam_dp_accs = []

    for eps in config.target_epsilons:
        eps_label = int(eps) if float(eps).is_integer() else eps
        sgd_dp_accs.append(results[f"SGD_dp_eps{eps_label}"]["final_accuracy"])
        adam_dp_accs.append(results[f"Adam_dp_eps{eps_label}"]["final_accuracy"])

    fig = plt.figure(figsize=(10, 5))
    plt.plot(eps_values, sgd_dp_accs, marker="o", linewidth=2, label="DP-SGD")
    plt.plot(eps_values, adam_dp_accs, marker="s", linewidth=2, label="DP-Adam")
    plt.axhline(
        results["SGD_non_private"]["final_accuracy"],
        linestyle="--",
        linewidth=1.8,
        label="SGD (non-private)",
    )
    plt.axhline(
        results["Adam_non_private"]["final_accuracy"],
        linestyle=":",
        linewidth=1.8,
        label="Adam (non-private)",
    )
    plt.xticks(eps_values, [str(int(eps)) if float(eps).is_integer() else str(eps) for eps in eps_values])
    plt.xlabel("Target Epsilon")
    plt.ylabel("Test Accuracy (%)")
    plt.title("Target Epsilon vs Test Accuracy")
    plt.legend()

    output_path = output_dir / "plot_c_target_epsilon_vs_test_accuracy.png"
    plt.savefig(output_path, dpi=300, bbox_inches="tight")
    plt.close(fig)
    return output_path


def generate_observations(results: Dict[str, Dict[str, Any]], config: Config) -> List[str]:
    """Generate concise written observations matching the notebook discussion."""
    observations = [
        "Adam converges faster than SGD in the non-private setting.",
        "Differential privacy slows convergence and introduces additional noise in the optimization process.",
        "For both optimizers, the run with target epsilon = 10 achieves better test accuracy than the run with target epsilon = 1, showing the privacy-utility tradeoff.",
        "The epsilon spent increases monotonically with training epochs.",
        "The non-private models achieve the highest final accuracy, while the private models sacrifice some utility in exchange for privacy guarantees.",
        f"Accountant used throughout Q1: {config.accountant}",
        f"Delta used throughout Q1: {config.delta}",
    ]

    private_run_names = [name for name, metrics in results.items() if metrics["private"]]
    best_private_run = max(private_run_names, key=lambda name: results[name]["final_accuracy"])
    best_private_acc = results[best_private_run]["final_accuracy"]
    observations.append(f"Best private run by final test accuracy: {best_private_run}")
    observations.append(f"Best private accuracy: {best_private_acc:.2f}%")
    return observations


def save_summary_and_notes(
    results: Dict[str, Dict[str, Any]],
    config: Config,
    output_dir: Path,
) -> Tuple[Path, Path, Path]:
    """Save the raw summary, display summary, and a notes text file."""
    summary_df = build_summary_dataframe(results)
    display_df = format_summary_dataframe(summary_df)

    raw_csv_path = output_dir / "q1_summary_raw.csv"
    display_csv_path = output_dir / "q1_summary_display.csv"
    notes_path = output_dir / "q1_notes.txt"

    summary_df.to_csv(raw_csv_path, index=False)
    display_df.to_csv(display_csv_path, index=False)

    observations = generate_observations(results, config)
    with notes_path.open("w", encoding="utf-8") as handle:
        handle.write("Question 1 Notes\n")
        handle.write("=================\n\n")
        handle.write("Accountant Used\n")
        handle.write(f"- {config.accountant}\n")
        handle.write(f"- Delta: {config.delta}\n\n")
        handle.write("Observations\n")
        for item in observations[:5]:
            handle.write(f"- {item}\n")
        handle.write("\nBest Private Run\n")
        for item in observations[7:]:
            handle.write(f"- {item}\n")

    print("\nSummary Table:\n")
    print(display_df.to_string(index=False))
    return raw_csv_path, display_csv_path, notes_path


def parse_args() -> argparse.Namespace:
    """Parse optional command-line overrides."""
    parser = argparse.ArgumentParser(
        description="Run CS729 Q1 experiments for non-private and private SGD/Adam on MNIST."
    )
    parser.add_argument("--epochs", type=int, default=10, help="Number of training epochs.")
    parser.add_argument("--batch-size", type=int, default=64, help="Training batch size.")
    parser.add_argument("--test-batch-size", type=int, default=1024, help="Test batch size.")
    parser.add_argument("--seed", type=int, default=42, help="Random seed.")
    parser.add_argument("--delta", type=float, default=1e-5, help="Target delta for privacy accounting.")
    parser.add_argument("--max-grad-norm", type=float, default=1.0, help="Gradient clipping norm.")
    parser.add_argument("--lr-sgd", type=float, default=0.1, help="Learning rate for SGD.")
    parser.add_argument("--lr-adam", type=float, default=1e-3, help="Learning rate for Adam.")
    parser.add_argument(
        "--target-epsilons",
        type=float,
        nargs="+",
        default=[1.0, 10.0],
        help="Target epsilon values for private runs.",
    )
    parser.add_argument(
        "--accountant",
        type=str,
        default="rdp",
        choices=["rdp", "gdp", "prv"],
        help="Privacy accountant used by Opacus.",
    )
    parser.add_argument(
        "--smoothing-window",
        type=int,
        default=50,
        help="Window size for smoothing the training-loss curves.",
    )
    parser.add_argument("--data-root", type=str, default="./data", help="Directory for MNIST downloads.")
    parser.add_argument(
        "--output-dir",
        type=str,
        default="./q1_outputs",
        help="Directory where plots and summaries will be saved.",
    )
    return parser.parse_args()


def make_config_from_args(args: argparse.Namespace) -> Config:
    """Convert parsed CLI arguments into a Config object."""
    return Config(
        seed=args.seed,
        batch_size=args.batch_size,
        test_batch_size=args.test_batch_size,
        epochs=args.epochs,
        delta=args.delta,
        target_epsilons=tuple(args.target_epsilons),
        max_grad_norm=args.max_grad_norm,
        lr_sgd=args.lr_sgd,
        lr_adam=args.lr_adam,
        accountant=args.accountant,
        smoothing_window=args.smoothing_window,
        data_root=args.data_root,
        output_dir=args.output_dir,
    )


def print_run_header(config: Config, train_dataset: datasets.MNIST, test_dataset: datasets.MNIST) -> None:
    """Print experiment configuration and dataset information."""
    base_train_loader = make_train_loader(train_dataset, config, seed=config.seed, shuffle=True)
    print(f"Using device: {config.device}")
    print(f"Accountant used for Q1: {config.accountant}")
    print(f"Training samples: {len(train_dataset)}")
    print(f"Test samples: {len(test_dataset)}")
    print(f"Batches per epoch (non-private loader): {len(base_train_loader)}")
    print("\nConfiguration:")
    for key, value in asdict(config).items():
        print(f"  {key}: {value}")


def main() -> None:
    """Main entrypoint: run all experiments, generate tables, and save plots."""
    args = parse_args()
    config = make_config_from_args(args)
    output_dir = Path(config.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    seed_everything(config.seed)
    train_dataset, test_dataset = load_datasets(config)
    print_run_header(config, train_dataset, test_dataset)

    results = run_full_experiment_grid(train_dataset, test_dataset, config)

    raw_csv_path, display_csv_path, notes_path = save_summary_and_notes(
        results=results,
        config=config,
        output_dir=output_dir,
    )

    plot_a_path = plot_training_loss_vs_iterations(results, config, output_dir)
    plot_b_path = plot_epsilon_vs_epochs(results, config, output_dir)
    plot_c_path = plot_target_epsilon_vs_test_accuracy(results, config, output_dir)

    print("\nSaved files:")
    for path in [
        raw_csv_path,
        display_csv_path,
        notes_path,
        plot_a_path,
        plot_b_path,
        plot_c_path,
    ]:
        print(f"  - {path.resolve()}")


if __name__ == "__main__":
    main()
