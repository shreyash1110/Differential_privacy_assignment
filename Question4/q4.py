#!/usr/bin/env python3
"""
CS729 Question 4: Membership Inference Attack (MIA)

This script implements a loss-based membership inference attack on MNIST using:
1. A non-private model trained with SGD.
2. A private model trained with DP-SGD via Opacus.

It performs the following steps:
- splits the MNIST training set into Din (members) and Dout (non-members)
- trains both models on Din
- computes per-sample cross-entropy losses on Din and Dout
- selects an attack threshold tau using a calibration split
- reports confusion matrices for both models
- plots ROC curves and computes AUC
- reports 10 randomly selected samples (5 members + 5 non-members)

Outputs are saved to the chosen output directory.
"""

from __future__ import annotations

import argparse
import json
import os
import random
from dataclasses import dataclass, asdict
from pathlib import Path
from typing import Dict, List, Tuple

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset

try:
    from torchvision import datasets, transforms
except Exception as exc:
    raise SystemExit(
        "torchvision could not be imported. This usually means your torch/torchvision versions are mismatched. "        "Try reinstalling matching versions, then rerun the script."
    ) from exc

try:
    from opacus import PrivacyEngine
    from opacus.accountants.utils import get_noise_multiplier
except ImportError as exc:
    raise SystemExit(
        "Opacus is required for this script. Install it with: pip install opacus"
    ) from exc


# =========================
# Configuration
# =========================
SEED = 42
DELTA = 1e-5
ACCOUNTANT = "rdp"
MAX_GRAD_NORM = 1.0
DIN_FRACTION = 0.5


@dataclass
class Config:
    batch_size: int = 64
    eval_batch_size: int = 1024
    epochs_nonprivate: int = 10
    epochs_private: int = 10
    lr: float = 0.1
    momentum: float = 0.0
    target_epsilon: float = 5.0
    data_dir: str = "./data"
    output_dir: str = "./q4_outputs"
    device: str = "cuda" if torch.cuda.is_available() else "cpu"


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


class IndexedSubset(Dataset):
    def __init__(self, base_dataset: Dataset, indices: List[int]):
        self.base_dataset = base_dataset
        self.indices = [int(i) for i in indices]

    def __len__(self) -> int:
        return len(self.indices)

    def __getitem__(self, idx: int):
        original_idx = self.indices[idx]
        x, y = self.base_dataset[original_idx]
        return x, y, original_idx


class FeedForwardNet(nn.Module):
    def __init__(self):
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


@dataclass
class ExperimentObjects:
    config: Config
    device: torch.device
    din_dataset: Dataset
    dout_dataset: Dataset
    din_train_loader: DataLoader
    din_eval_loader: DataLoader
    dout_eval_loader: DataLoader
    sample_rate: float
    fixed_noise_multiplier: float


def make_loader(dataset: Dataset, batch_size: int, shuffle: bool, seed: int = SEED) -> DataLoader:
    generator = torch.Generator()
    generator.manual_seed(seed)
    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        generator=generator,
        num_workers=0,
        pin_memory=torch.cuda.is_available(),
    )


def setup_experiment(config: Config) -> ExperimentObjects:
    seed_everything(SEED)
    device = torch.device(config.device)

    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.1307,), (0.3081,)),
    ])

    full_train_dataset = datasets.MNIST(
        root=config.data_dir,
        train=True,
        download=True,
        transform=transform,
    )

    split_generator = torch.Generator().manual_seed(SEED)
    all_indices = torch.randperm(len(full_train_dataset), generator=split_generator).tolist()

    din_size = int(DIN_FRACTION * len(all_indices))
    din_indices = all_indices[:din_size]
    dout_indices = all_indices[din_size:]

    din_dataset = IndexedSubset(full_train_dataset, din_indices)
    dout_dataset = IndexedSubset(full_train_dataset, dout_indices)

    din_train_loader = make_loader(din_dataset, config.batch_size, shuffle=True, seed=SEED)
    din_eval_loader = make_loader(din_dataset, config.eval_batch_size, shuffle=False, seed=SEED)
    dout_eval_loader = make_loader(dout_dataset, config.eval_batch_size, shuffle=False, seed=SEED)

    sample_rate = config.batch_size / len(din_dataset)
    fixed_noise_multiplier = get_noise_multiplier(
        target_epsilon=config.target_epsilon,
        target_delta=DELTA,
        sample_rate=sample_rate,
        epochs=config.epochs_private,
        accountant=ACCOUNTANT,
        epsilon_tolerance=0.01,
    )

    return ExperimentObjects(
        config=config,
        device=device,
        din_dataset=din_dataset,
        dout_dataset=dout_dataset,
        din_train_loader=din_train_loader,
        din_eval_loader=din_eval_loader,
        dout_eval_loader=dout_eval_loader,
        sample_rate=sample_rate,
        fixed_noise_multiplier=fixed_noise_multiplier,
    )


def build_model(device: torch.device) -> nn.Module:
    return FeedForwardNet().to(device)


def build_optimizer(model: nn.Module, config: Config) -> optim.Optimizer:
    return optim.SGD(model.parameters(), lr=config.lr, momentum=config.momentum)


# =========================
# Training helpers
# =========================
criterion = nn.CrossEntropyLoss()
loss_criterion_none = nn.CrossEntropyLoss(reduction="none")


def train_one_epoch(model: nn.Module, loader: DataLoader, optimizer: optim.Optimizer, device: torch.device) -> float:
    model.train()
    running_loss = 0.0

    for x, y, _ in loader:
        x, y = x.to(device), y.to(device)
        optimizer.zero_grad()
        logits = model(x)
        loss = criterion(logits, y)
        loss.backward()
        optimizer.step()
        running_loss += loss.item()

    return running_loss / len(loader)


@torch.no_grad()
def compute_accuracy(model: nn.Module, loader: DataLoader, device: torch.device) -> float:
    model.eval()
    correct = 0
    total = 0
    for x, y, _ in loader:
        x, y = x.to(device), y.to(device)
        logits = model(x)
        preds = logits.argmax(dim=1)
        correct += (preds == y).sum().item()
        total += x.size(0)
    return 100.0 * correct / total


def train_nonprivate_model(exp: ExperimentObjects) -> Tuple[nn.Module, pd.DataFrame]:
    seed_everything(SEED)
    model = build_model(exp.device)
    optimizer = build_optimizer(model, exp.config)
    train_loader = make_loader(exp.din_dataset, exp.config.batch_size, shuffle=True, seed=SEED)

    history: List[Dict[str, float]] = []
    for epoch in range(1, exp.config.epochs_nonprivate + 1):
        train_loss = train_one_epoch(model, train_loader, optimizer, exp.device)
        din_acc = compute_accuracy(model, exp.din_eval_loader, exp.device)
        dout_acc = compute_accuracy(model, exp.dout_eval_loader, exp.device)
        history.append({
            "epoch": epoch,
            "train_loss": train_loss,
            "din_accuracy": din_acc,
            "dout_accuracy": dout_acc,
        })
        print(
            f"[Non-private SGD] Epoch {epoch:02d}/{exp.config.epochs_nonprivate} | "
            f"Train Loss: {train_loss:.4f} | Din Acc: {din_acc:.2f}% | Dout Acc: {dout_acc:.2f}%"
        )

    return model, pd.DataFrame(history)


def train_private_model(exp: ExperimentObjects) -> Tuple[nn.Module, pd.DataFrame, PrivacyEngine]:
    seed_everything(SEED)
    model = build_model(exp.device)
    optimizer = build_optimizer(model, exp.config)
    train_loader = make_loader(exp.din_dataset, exp.config.batch_size, shuffle=True, seed=SEED)

    privacy_engine = PrivacyEngine(accountant=ACCOUNTANT)
    model, optimizer, private_train_loader = privacy_engine.make_private(
        module=model,
        optimizer=optimizer,
        data_loader=train_loader,
        noise_multiplier=exp.fixed_noise_multiplier,
        max_grad_norm=MAX_GRAD_NORM,
    )

    history: List[Dict[str, float]] = []
    for epoch in range(1, exp.config.epochs_private + 1):
        train_loss = train_one_epoch(model, private_train_loader, optimizer, exp.device)
        din_acc = compute_accuracy(model, exp.din_eval_loader, exp.device)
        dout_acc = compute_accuracy(model, exp.dout_eval_loader, exp.device)
        eps_spent = privacy_engine.get_epsilon(delta=DELTA)
        history.append({
            "epoch": epoch,
            "train_loss": train_loss,
            "din_accuracy": din_acc,
            "dout_accuracy": dout_acc,
            "epsilon_spent": eps_spent,
        })
        print(
            f"[DP-SGD] Epoch {epoch:02d}/{exp.config.epochs_private} | Train Loss: {train_loss:.4f} | "
            f"Epsilon Spent: {eps_spent:.4f} | Din Acc: {din_acc:.2f}% | Dout Acc: {dout_acc:.2f}%"
        )

    return model, pd.DataFrame(history), privacy_engine


# =========================
# Loss computation
# =========================
@torch.no_grad()
def compute_per_sample_losses(model: nn.Module, dataset: Dataset, eval_batch_size: int, device: torch.device) -> pd.DataFrame:
    model.eval()
    loader = make_loader(dataset, eval_batch_size, shuffle=False, seed=SEED)
    rows: List[Dict[str, float]] = []

    for x, y, idx in loader:
        x, y = x.to(device), y.to(device)
        logits = model(x)
        losses = loss_criterion_none(logits, y).detach().cpu().numpy()
        preds = logits.argmax(dim=1).detach().cpu().numpy()
        labels = y.detach().cpu().numpy()
        indices = idx.detach().cpu().numpy()

        for original_idx, label, pred, loss_value in zip(indices, labels, preds, losses):
            rows.append({
                "original_index": int(original_idx),
                "true_label": int(label),
                "predicted_label": int(pred),
                "loss": float(loss_value),
            })

    return pd.DataFrame(rows).sort_values("original_index").reset_index(drop=True)


def build_attack_dataframe(nonprivate_model: nn.Module, private_model: nn.Module, exp: ExperimentObjects) -> pd.DataFrame:
    nonprivate_din_df = compute_per_sample_losses(nonprivate_model, exp.din_dataset, exp.config.eval_batch_size, exp.device)
    nonprivate_dout_df = compute_per_sample_losses(nonprivate_model, exp.dout_dataset, exp.config.eval_batch_size, exp.device)
    private_din_df = compute_per_sample_losses(private_model, exp.din_dataset, exp.config.eval_batch_size, exp.device)
    private_dout_df = compute_per_sample_losses(private_model, exp.dout_dataset, exp.config.eval_batch_size, exp.device)

    for df, membership, status in [
        (nonprivate_din_df, 1, "Member"),
        (nonprivate_dout_df, 0, "Not-Member"),
        (private_din_df, 1, "Member"),
        (private_dout_df, 0, "Not-Member"),
    ]:
        df["true_membership"] = membership
        df["true_status"] = status

    nonprivate_attack_df = pd.concat([nonprivate_din_df, nonprivate_dout_df], ignore_index=True).rename(
        columns={"predicted_label": "nonprivate_predicted_label", "loss": "nonprivate_loss"}
    )
    private_attack_df = pd.concat([private_din_df, private_dout_df], ignore_index=True).rename(
        columns={"predicted_label": "private_predicted_label", "loss": "private_loss"}
    )

    attack_df = nonprivate_attack_df[
        [
            "original_index",
            "true_label",
            "true_membership",
            "true_status",
            "nonprivate_predicted_label",
            "nonprivate_loss",
        ]
    ].merge(
        private_attack_df[["original_index", "private_predicted_label", "private_loss"]],
        on="original_index",
        how="inner",
    )

    return attack_df.sort_values("original_index").reset_index(drop=True)


# =========================
# MIA helpers
# =========================
def split_member_nonmember_losses(member_losses: np.ndarray, nonmember_losses: np.ndarray, seed: int = SEED, calib_fraction: float = 0.5) -> Dict[str, np.ndarray]:
    rng = np.random.default_rng(seed)
    member_losses = np.asarray(member_losses)
    nonmember_losses = np.asarray(nonmember_losses)

    member_losses = member_losses[rng.permutation(len(member_losses))]
    nonmember_losses = nonmember_losses[rng.permutation(len(nonmember_losses))]

    m_calib = int(calib_fraction * len(member_losses))
    n_calib = int(calib_fraction * len(nonmember_losses))

    return {
        "member_calib": member_losses[:m_calib],
        "member_eval": member_losses[m_calib:],
        "nonmember_calib": nonmember_losses[:n_calib],
        "nonmember_eval": nonmember_losses[n_calib:],
    }


def get_candidate_thresholds(member_losses: np.ndarray, nonmember_losses: np.ndarray) -> np.ndarray:
    all_losses = np.unique(np.concatenate([member_losses, nonmember_losses]))
    if len(all_losses) == 1:
        eps = 1e-8
        return np.array([all_losses[0] - eps, all_losses[0] + eps])
    mids = (all_losses[:-1] + all_losses[1:]) / 2.0
    eps = 1e-8
    return np.concatenate([[all_losses[0] - eps], mids, [all_losses[-1] + eps]])


def sweep_thresholds(member_losses: np.ndarray, nonmember_losses: np.ndarray) -> pd.DataFrame:
    thresholds = get_candidate_thresholds(np.asarray(member_losses), np.asarray(nonmember_losses))
    num_members = len(member_losses)
    num_nonmembers = len(nonmember_losses)
    rows: List[Dict[str, float]] = []

    for tau in thresholds:
        tp = int(np.sum(member_losses < tau))
        fn = int(num_members - tp)
        fp = int(np.sum(nonmember_losses < tau))
        tn = int(num_nonmembers - fp)

        tpr = tp / num_members if num_members > 0 else 0.0
        fpr = fp / num_nonmembers if num_nonmembers > 0 else 0.0
        tnr = tn / num_nonmembers if num_nonmembers > 0 else 0.0
        balanced_accuracy = 0.5 * (tpr + tnr)
        youdens_j = tpr - fpr

        rows.append({
            "threshold": float(tau),
            "TP": tp,
            "FN": fn,
            "FP": fp,
            "TN": tn,
            "TPR": tpr,
            "FPR": fpr,
            "TNR": tnr,
            "balanced_accuracy": balanced_accuracy,
            "youdens_j": youdens_j,
        })

    return pd.DataFrame(rows)


def choose_tau_from_calibration(member_losses_calib: np.ndarray, nonmember_losses_calib: np.ndarray) -> Tuple[float, pd.DataFrame]:
    calib_curve_df = sweep_thresholds(member_losses_calib, nonmember_losses_calib)
    best_row = calib_curve_df.sort_values(
        by=["youdens_j", "balanced_accuracy", "threshold"],
        ascending=[False, False, True],
    ).iloc[0]
    return float(best_row["threshold"]), calib_curve_df


def evaluate_attack_at_tau(member_losses_eval: np.ndarray, nonmember_losses_eval: np.ndarray, tau: float) -> Dict[str, float]:
    tp = int(np.sum(member_losses_eval < tau))
    fn = int(len(member_losses_eval) - tp)
    fp = int(np.sum(nonmember_losses_eval < tau))
    tn = int(len(nonmember_losses_eval) - fp)

    tpr = tp / len(member_losses_eval) if len(member_losses_eval) > 0 else 0.0
    fpr = fp / len(nonmember_losses_eval) if len(nonmember_losses_eval) > 0 else 0.0
    tnr = tn / len(nonmember_losses_eval) if len(nonmember_losses_eval) > 0 else 0.0
    balanced_accuracy = 0.5 * (tpr + tnr)
    youdens_j = tpr - fpr

    return {
        "TP": tp,
        "FN": fn,
        "FP": fp,
        "TN": tn,
        "TPR": tpr,
        "FPR": fpr,
        "TNR": tnr,
        "balanced_accuracy": balanced_accuracy,
        "youdens_j": youdens_j,
        "threshold": float(tau),
    }


def build_roc_and_auc(member_losses_eval: np.ndarray, nonmember_losses_eval: np.ndarray) -> Tuple[pd.DataFrame, float]:
    roc_df = sweep_thresholds(member_losses_eval, nonmember_losses_eval)
    roc_df = roc_df.sort_values(["FPR", "TPR"]).reset_index(drop=True)
    roc_df = roc_df.drop_duplicates(subset=["FPR", "TPR"])
    auc_value = np.trapezoid(roc_df["TPR"].to_numpy(), roc_df["FPR"].to_numpy())
    return roc_df, float(auc_value)


def confusion_matrix_dataframe(cm_dict: Dict[str, float]) -> pd.DataFrame:
    return pd.DataFrame(
        [[cm_dict["TP"], cm_dict["FN"]], [cm_dict["FP"], cm_dict["TN"]]],
        index=["Member (Truth)", "Not-Member (Truth)"],
        columns=["Member (Predicted)", "Not-Member (Predicted)"],
    )


# =========================
# Reporting helpers
# =========================
def unnormalize_mnist(x: np.ndarray) -> np.ndarray:
    return x * 0.3081 + 0.1307


def save_loss_histograms(attack_df: pd.DataFrame, output_dir: Path) -> None:
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))

    axes[0].hist(
        attack_df.loc[attack_df["true_membership"] == 1, "nonprivate_loss"],
        bins=60,
        alpha=0.6,
        density=True,
        label="Din (members)",
    )
    axes[0].hist(
        attack_df.loc[attack_df["true_membership"] == 0, "nonprivate_loss"],
        bins=60,
        alpha=0.6,
        density=True,
        label="Dout (non-members)",
    )
    axes[0].set_xlabel("Cross-entropy loss")
    axes[0].set_ylabel("Density")
    axes[0].set_title("Non-private SGD: Loss Distribution")
    axes[0].legend()
    axes[0].grid(True, alpha=0.3)

    axes[1].hist(
        attack_df.loc[attack_df["true_membership"] == 1, "private_loss"],
        bins=60,
        alpha=0.6,
        density=True,
        label="Din (members)",
    )
    axes[1].hist(
        attack_df.loc[attack_df["true_membership"] == 0, "private_loss"],
        bins=60,
        alpha=0.6,
        density=True,
        label="Dout (non-members)",
    )
    axes[1].set_xlabel("Cross-entropy loss")
    axes[1].set_ylabel("Density")
    axes[1].set_title("DP-SGD: Loss Distribution")
    axes[1].legend()
    axes[1].grid(True, alpha=0.3)

    fig.tight_layout()
    fig.savefig(output_dir / "loss_distributions.png", dpi=200, bbox_inches="tight")
    plt.close(fig)


def save_roc_curve(nonprivate_curve_df: pd.DataFrame, nonprivate_auc: float, private_curve_df: pd.DataFrame, private_auc: float, output_dir: Path) -> None:
    plt.figure(figsize=(8, 6))
    plt.plot(nonprivate_curve_df["FPR"], nonprivate_curve_df["TPR"], label=f"Non-private SGD (AUC = {nonprivate_auc:.4f})")
    plt.plot(private_curve_df["FPR"], private_curve_df["TPR"], label=f"DP-SGD (AUC = {private_auc:.4f})")
    plt.plot([0, 1], [0, 1], linestyle="--", linewidth=1, label="Random Guess")
    plt.xlabel("False Positive Rate (FPR)")
    plt.ylabel("True Positive Rate (TPR)")
    plt.title("ROC Curve for Loss-based Membership Inference Attack")
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.savefig(output_dir / "roc_curve.png", dpi=200, bbox_inches="tight")
    plt.close()


def build_sample_report(attack_df: pd.DataFrame, exp: ExperimentObjects, nonprivate_tau: float, private_tau: float, output_dir: Path) -> pd.DataFrame:
    rng = np.random.default_rng(SEED)
    member_local_indices = rng.choice(len(exp.din_dataset), size=5, replace=False)
    nonmember_local_indices = rng.choice(len(exp.dout_dataset), size=5, replace=False)

    selected_samples = []
    for local_idx in member_local_indices:
        x, y, original_idx = exp.din_dataset[int(local_idx)]
        selected_samples.append({"image": x, "label": int(y), "original_index": int(original_idx), "true_status": "Member"})
    for local_idx in nonmember_local_indices:
        x, y, original_idx = exp.dout_dataset[int(local_idx)]
        selected_samples.append({"image": x, "label": int(y), "original_index": int(original_idx), "true_status": "Not-Member"})

    attack_df = attack_df.copy()
    attack_df["nonprivate_attack_prediction"] = np.where(attack_df["nonprivate_loss"] < nonprivate_tau, "Member", "Not-Member")
    attack_df["private_attack_prediction"] = np.where(attack_df["private_loss"] < private_tau, "Member", "Not-Member")

    sample_rows: List[Dict[str, object]] = []
    for item in selected_samples:
        row = attack_df.loc[attack_df["original_index"] == item["original_index"]].iloc[0]
        sample_rows.append({
            "original_index": item["original_index"],
            "true_label": item["label"],
            "true_status": item["true_status"],
            "nonprivate_loss": row["nonprivate_loss"],
            "nonprivate_prediction": row["nonprivate_attack_prediction"],
            "private_loss": row["private_loss"],
            "private_prediction": row["private_attack_prediction"],
        })

    sample_report_df = pd.DataFrame(sample_rows)

    fig, axes = plt.subplots(2, 5, figsize=(14, 8))
    axes = axes.flatten()
    for ax, item in zip(axes, selected_samples):
        img = item["image"].squeeze().cpu().numpy()
        img = unnormalize_mnist(img)
        ax.imshow(img, cmap="gray")
        ax.set_title(f"idx={item['original_index']}\n{item['true_status']}, y={item['label']}")
        ax.axis("off")
    plt.suptitle("10 Randomly Selected Samples (5 Members + 5 Non-Members)", fontsize=14)
    plt.tight_layout()
    plt.savefig(output_dir / "sample_images.png", dpi=200, bbox_inches="tight")
    plt.close()

    return sample_report_df


# =========================
# Main pipeline
# =========================
def main() -> None:
    parser = argparse.ArgumentParser(description="Run CS729 Q4 MIA experiment.")
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--eval-batch-size", type=int, default=1024)
    parser.add_argument("--epochs-nonprivate", type=int, default=10)
    parser.add_argument("--epochs-private", type=int, default=10)
    parser.add_argument("--lr", type=float, default=0.1)
    parser.add_argument("--momentum", type=float, default=0.0)
    parser.add_argument("--target-epsilon", type=float, default=5.0)
    parser.add_argument("--data-dir", type=str, default="./data")
    parser.add_argument("--output-dir", type=str, default="./q4_outputs")
    parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")
    args = parser.parse_args()

    config = Config(
        batch_size=args.batch_size,
        eval_batch_size=args.eval_batch_size,
        epochs_nonprivate=args.epochs_nonprivate,
        epochs_private=args.epochs_private,
        lr=args.lr,
        momentum=args.momentum,
        target_epsilon=args.target_epsilon,
        data_dir=args.data_dir,
        output_dir=args.output_dir,
        device=args.device,
    )

    output_dir = Path(config.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    print("=" * 100)
    print("CS729 Q4: Membership Inference Attack")
    print("=" * 100)
    print(f"Using device: {config.device}")
    print(f"Saving outputs to: {output_dir.resolve()}")
    print(f"Configuration: {asdict(config)}")

    exp = setup_experiment(config)
    print(f"Din size: {len(exp.din_dataset)}")
    print(f"Dout size: {len(exp.dout_dataset)}")
    print(f"Din batches per epoch: {len(exp.din_train_loader)}")
    print(f"Sample rate: {exp.sample_rate:.8f}")
    print(f"Fixed noise multiplier for DP-SGD: {exp.fixed_noise_multiplier:.6f}")

    nonprivate_model, nonprivate_history = train_nonprivate_model(exp)
    print("=" * 100)
    private_model, private_history, _ = train_private_model(exp)

    attack_df = build_attack_dataframe(nonprivate_model, private_model, exp)

    nonprivate_member_losses = attack_df.loc[attack_df["true_membership"] == 1, "nonprivate_loss"].to_numpy()
    nonprivate_nonmember_losses = attack_df.loc[attack_df["true_membership"] == 0, "nonprivate_loss"].to_numpy()
    private_member_losses = attack_df.loc[attack_df["true_membership"] == 1, "private_loss"].to_numpy()
    private_nonmember_losses = attack_df.loc[attack_df["true_membership"] == 0, "private_loss"].to_numpy()

    nonprivate_split = split_member_nonmember_losses(nonprivate_member_losses, nonprivate_nonmember_losses, seed=SEED, calib_fraction=0.5)
    private_split = split_member_nonmember_losses(private_member_losses, private_nonmember_losses, seed=SEED, calib_fraction=0.5)

    nonprivate_tau, nonprivate_calib_curve_df = choose_tau_from_calibration(
        nonprivate_split["member_calib"], nonprivate_split["nonmember_calib"]
    )
    private_tau, private_calib_curve_df = choose_tau_from_calibration(
        private_split["member_calib"], private_split["nonmember_calib"]
    )

    nonprivate_cm_eval = evaluate_attack_at_tau(
        nonprivate_split["member_eval"], nonprivate_split["nonmember_eval"], nonprivate_tau
    )
    private_cm_eval = evaluate_attack_at_tau(
        private_split["member_eval"], private_split["nonmember_eval"], private_tau
    )

    nonprivate_cm_full = evaluate_attack_at_tau(nonprivate_member_losses, nonprivate_nonmember_losses, nonprivate_tau)
    private_cm_full = evaluate_attack_at_tau(private_member_losses, private_nonmember_losses, private_tau)

    nonprivate_curve_df, nonprivate_auc = build_roc_and_auc(nonprivate_member_losses, nonprivate_nonmember_losses)
    private_curve_df, private_auc = build_roc_and_auc(private_member_losses, private_nonmember_losses)

    save_loss_histograms(attack_df, output_dir)
    save_roc_curve(nonprivate_curve_df, nonprivate_auc, private_curve_df, private_auc, output_dir)
    sample_report_df = build_sample_report(attack_df, exp, nonprivate_tau, private_tau, output_dir)

    loss_summary_df = pd.DataFrame([
        {
            "group": "Din (members)",
            "nonprivate_mean_loss": attack_df.loc[attack_df["true_membership"] == 1, "nonprivate_loss"].mean(),
            "private_mean_loss": attack_df.loc[attack_df["true_membership"] == 1, "private_loss"].mean(),
        },
        {
            "group": "Dout (non-members)",
            "nonprivate_mean_loss": attack_df.loc[attack_df["true_membership"] == 0, "nonprivate_loss"].mean(),
            "private_mean_loss": attack_df.loc[attack_df["true_membership"] == 0, "private_loss"].mean(),
        },
    ])

    tau_summary_df = pd.DataFrame([
        {
            "model": "Non-private SGD",
            "tau_from_calibration": nonprivate_tau,
            "best_calibration_youdens_j": nonprivate_calib_curve_df["youdens_j"].max(),
            "best_calibration_balanced_accuracy": nonprivate_calib_curve_df["balanced_accuracy"].max(),
            "AUC_full_dataset": nonprivate_auc,
        },
        {
            "model": "DP-SGD",
            "tau_from_calibration": private_tau,
            "best_calibration_youdens_j": private_calib_curve_df["youdens_j"].max(),
            "best_calibration_balanced_accuracy": private_calib_curve_df["balanced_accuracy"].max(),
            "AUC_full_dataset": private_auc,
        },
    ])

    summary_attack_df = pd.DataFrame([
        {
            "model": "Non-private SGD",
            "chosen_tau": nonprivate_tau,
            "AUC": nonprivate_auc,
            "TP": nonprivate_cm_full["TP"],
            "FN": nonprivate_cm_full["FN"],
            "FP": nonprivate_cm_full["FP"],
            "TN": nonprivate_cm_full["TN"],
            "TPR": nonprivate_cm_full["TPR"],
            "FPR": nonprivate_cm_full["FPR"],
            "balanced_accuracy": nonprivate_cm_full["balanced_accuracy"],
        },
        {
            "model": "DP-SGD",
            "chosen_tau": private_tau,
            "AUC": private_auc,
            "TP": private_cm_full["TP"],
            "FN": private_cm_full["FN"],
            "FP": private_cm_full["FP"],
            "TN": private_cm_full["TN"],
            "TPR": private_cm_full["TPR"],
            "FPR": private_cm_full["FPR"],
            "balanced_accuracy": private_cm_full["balanced_accuracy"],
        },
    ])

    # Save tables
    nonprivate_history.to_csv(output_dir / "nonprivate_history.csv", index=False)
    private_history.to_csv(output_dir / "private_history.csv", index=False)
    attack_df.to_csv(output_dir / "attack_dataframe.csv", index=False)
    loss_summary_df.to_csv(output_dir / "loss_summary.csv", index=False)
    nonprivate_calib_curve_df.to_csv(output_dir / "nonprivate_calibration_curve.csv", index=False)
    private_calib_curve_df.to_csv(output_dir / "private_calibration_curve.csv", index=False)
    nonprivate_curve_df.to_csv(output_dir / "nonprivate_roc_curve.csv", index=False)
    private_curve_df.to_csv(output_dir / "private_roc_curve.csv", index=False)
    confusion_matrix_dataframe(nonprivate_cm_full).to_csv(output_dir / "nonprivate_confusion_matrix_full.csv")
    confusion_matrix_dataframe(private_cm_full).to_csv(output_dir / "private_confusion_matrix_full.csv")
    confusion_matrix_dataframe(nonprivate_cm_eval).to_csv(output_dir / "nonprivate_confusion_matrix_eval.csv")
    confusion_matrix_dataframe(private_cm_eval).to_csv(output_dir / "private_confusion_matrix_eval.csv")
    tau_summary_df.to_csv(output_dir / "tau_summary.csv", index=False)
    summary_attack_df.to_csv(output_dir / "attack_summary.csv", index=False)
    sample_report_df.to_csv(output_dir / "sample_report.csv", index=False)

    result_summary = {
        "config": asdict(config),
        "accountant": ACCOUNTANT,
        "delta": DELTA,
        "max_grad_norm": MAX_GRAD_NORM,
        "din_size": len(exp.din_dataset),
        "dout_size": len(exp.dout_dataset),
        "sample_rate": exp.sample_rate,
        "fixed_noise_multiplier": exp.fixed_noise_multiplier,
        "nonprivate_tau": nonprivate_tau,
        "private_tau": private_tau,
        "nonprivate_auc": nonprivate_auc,
        "private_auc": private_auc,
        "more_vulnerable_model": "Non-private SGD" if nonprivate_auc > private_auc else "DP-SGD",
        "nonprivate_cm_full": nonprivate_cm_full,
        "private_cm_full": private_cm_full,
        "nonprivate_cm_eval": nonprivate_cm_eval,
        "private_cm_eval": private_cm_eval,
    }
    with open(output_dir / "results_summary.json", "w", encoding="utf-8") as f:
        json.dump(result_summary, f, indent=2)

    print("\n" + "=" * 100)
    print("Final summary")
    print("=" * 100)
    print(f"Chosen tau for non-private model: {nonprivate_tau:.8f}")
    print(f"Chosen tau for private model: {private_tau:.8f}")
    print(f"AUC of Non-private SGD: {nonprivate_auc:.6f}")
    print(f"AUC of DP-SGD: {private_auc:.6f}")
    print(f"More vulnerable model according to AUC: {result_summary['more_vulnerable_model']}")
    print("\nConfusion matrix for Non-private SGD (full Din/Dout attack dataset):")
    print(confusion_matrix_dataframe(nonprivate_cm_full))
    print("\nConfusion matrix for DP-SGD (full Din/Dout attack dataset):")
    print(confusion_matrix_dataframe(private_cm_full))
    print(f"\nAll outputs saved to: {output_dir.resolve()}")


if __name__ == "__main__":
    main()
