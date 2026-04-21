from __future__ import annotations

import argparse
import random
import sys
import time
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix


SCRIPT_DIR = Path(__file__).resolve().parent
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))

OUTPUT_DIR = SCRIPT_DIR.parent / "output"
CAPTION_FONT_SIZE = 11

from nce_model import NoiseGeneration, TrainNCE
from stochastic_airlift import (
    MultinomialLogitAircraftModel,
)
from unified_data_loader import UnifiedDataSplit, build_shared_split


def configure_determinism(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

    # Keep algorithmic behavior deterministic where possible while avoiding hard failures.
    torch.use_deterministic_algorithms(True, warn_only=True)
    torch.set_num_threads(1)
    # torch.set_num_interop_threads(1)
    if hasattr(torch.backends, "cudnn"):
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False


class TorchMultinomialLogit(nn.Module):
    def __init__(self, input_dim: int, num_classes: int):
        super().__init__()
        self.linear = nn.Linear(input_dim, num_classes)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.linear(x)

# NOTE class weight alpha is a scalar exponent applied to the inverse-frequency class weights. 0.0 means no weighting; 1.0 means full inverse-frequency weighting.
def train_torch_mnlr(
    split: UnifiedDataSplit,
    epochs: int = 250,
    learning_rate: float = 0.05,
    class_weight_alpha: float = 0.75,
    verbose: bool = False,
    log_every: int = 25,
) -> tuple[TorchMultinomialLogit, np.ndarray, np.ndarray]:
    torch.manual_seed(split.seed)
    np.random.seed(split.seed)

    input_dim = int(split.train_features.shape[1])
    num_classes = len(split.label_names)

    model = TorchMultinomialLogit(input_dim=input_dim, num_classes=num_classes)
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)

    class_counts = torch.bincount(split.train_targets, minlength=num_classes).float()
    base_class_weights = class_counts.sum() / (num_classes * class_counts.clamp_min(1.0))
    class_weights = torch.pow(base_class_weights, class_weight_alpha)
    criterion = nn.CrossEntropyLoss(weight=class_weights)

    if verbose:
        print("=== Config Prediction Training (MNLR) ===")
        print(
            f"epochs={epochs}, lr={learning_rate}, class_weight_alpha={class_weight_alpha}, "
            f"train_rows={len(split.train_targets)}, test_rows={len(split.test_targets)}"
        )

    train_start = time.perf_counter()
    for epoch in range(epochs):
        optimizer.zero_grad()
        logits = model(split.train_features)
        loss = criterion(logits, split.train_targets)
        loss.backward()
        optimizer.step()

        if verbose and ((epoch + 1) == 1 or (epoch + 1) % max(1, log_every) == 0 or (epoch + 1) == epochs):
            elapsed = time.perf_counter() - train_start
            print(f"[config-mnlr] epoch={epoch + 1}/{epochs} loss={loss.item():.6f} elapsed={elapsed:.1f}s")

    model.eval()
    with torch.no_grad():
        test_logits = model(split.test_features)
        y_pred = test_logits.argmax(dim=1).cpu().numpy()

    y_true = split.test_targets.cpu().numpy()
    if verbose:
        elapsed = time.perf_counter() - train_start
        print(f"[config-mnlr] complete in {elapsed:.1f}s")
    return model, y_true, y_pred


def plot_nce_summary_figures(
    train_probs: np.ndarray,
    test_probs: np.ndarray,
    noise_probs: np.ndarray,
) -> list[Path]:
    if train_probs.size == 0 or test_probs.size == 0 or noise_probs.size == 0:
        return []

    output_dir = OUTPUT_DIR / "figures"
    output_dir.mkdir(parents=True, exist_ok=True)

    saved_paths: list[Path] = []

    sns.set_theme(style="darkgrid")
    colors = sns.color_palette("deep")

    plt.figure()
    sns.kdeplot(train_probs, color=colors[2], fill=True, alpha=0.5, label="Training Set")
    sns.kdeplot(test_probs, color=colors[1], fill=True, alpha=0.5, label="Testing Set")
    sns.kdeplot(noise_probs, color=colors[3], fill=True, alpha=0.5, label="Random Noise")
    plt.xlabel("NCE Probability", fontsize=CAPTION_FONT_SIZE)
    plt.ylabel("Density", fontsize=CAPTION_FONT_SIZE)
    plt.legend(loc="upper left")
    plt.tight_layout()
    distribution_path = output_dir / "nce_probability_distribution.pdf"
    plt.savefig(distribution_path, bbox_inches="tight", dpi=300)
    plt.close()
    saved_paths.append(distribution_path)

    summary_frame = pd.DataFrame(
        [
            {"Split": "Train", "Statistic": "Mean", "Value": float(train_probs.mean())},
            {"Split": "Train", "Statistic": "Max", "Value": float(train_probs.max())},
            {"Split": "Test", "Statistic": "Mean", "Value": float(test_probs.mean())},
            {"Split": "Test", "Statistic": "Max", "Value": float(test_probs.max())},
            {"Split": "Noise", "Statistic": "Mean", "Value": float(noise_probs.mean())},
            {"Split": "Noise", "Statistic": "Max", "Value": float(noise_probs.max())},
        ]
    )

    plt.figure()
    ax = sns.barplot(data=summary_frame, x="Split", y="Value", hue="Statistic", palette="deep")
    ax.set_xlabel("", fontsize=CAPTION_FONT_SIZE)
    ax.set_ylabel("Probability", fontsize=CAPTION_FONT_SIZE)
    ax.set_ylim(0, 1)
    plt.tight_layout()
    summary_path = output_dir / "nce_probability_summary.pdf"
    plt.savefig(summary_path, bbox_inches="tight", dpi=300)
    plt.close()
    saved_paths.append(summary_path)

    return saved_paths


def summarize_mnlr_results(split: UnifiedDataSplit, class_weight_alpha: float = 0.75) -> dict[str, Any]:
    print("Training MNLR model with class weight alpha =", class_weight_alpha)
    model, y_true, y_pred = train_torch_mnlr(split, class_weight_alpha=class_weight_alpha)

    accuracy = accuracy_score(y_true, y_pred)
    class_counts = np.bincount(split.targets.cpu().numpy().astype(int))
    baseline_accuracy = class_counts.max() / class_counts.sum()
    labels = list(range(len(split.label_names)))
    report = classification_report(
        y_true,
        y_pred,
        labels=labels,
        target_names=split.label_names,
        output_dict=True,
        zero_division=0,
    )

    print("=== Balanced MNLR Aircraft Type ===")
    print(f"Class-weight alpha: {class_weight_alpha:.2f}")
    print(f"Baseline Accuracy (Majority Class - Widebody): {baseline_accuracy:.2%}")
    if accuracy < baseline_accuracy:
        print("Model is performing worse than baseline model with majority class (naive guess).")
    print(f"Test Accuracy: {accuracy:.2%}")
    print(
        f"Shared split info: source={split.data_path}, train={len(split.train_indices)}, "
        f"test={len(split.test_indices)}, ratio={split.train_ratio:.2f}/{1 - split.train_ratio:.2f}, seed={split.seed}"
    )
    print(f"Shared input dimensionality: {split.features.shape[1]}")
    print("\nDetailed Performance Metrics:")
    print(
        classification_report(
            y_true,
            y_pred,
            labels=labels,
            target_names=split.label_names,
            zero_division=0,
        )
    )
    print("Confusion Matrix")
    matrix = confusion_matrix(y_true, y_pred, labels=labels)
    print(matrix)

    weight = model.linear.weight.detach().cpu().numpy()
    bias = model.linear.bias.detach().cpu().numpy()
    aircraft_model = MultinomialLogitAircraftModel(
        coef_=weight,
        intercept_=bias,
        classes_=split.label_names,
    )
    # Use an observed row so the example vector always matches the model's fitted feature dimension.
    example_vector = split.features[0].detach().cpu().numpy().astype(np.float32)
    example_probability = aircraft_model.predict_proba(example_vector)
    sampled_class, sampled_probability, _ = aircraft_model.sample(example_vector, rng=np.random.default_rng(42))

    print("\nExample stochastic aircraft draw")
    print(f"Predicted class probabilities: {dict(zip(split.label_names, example_probability.round(4)))}")
    print(f"Sampled aircraft type: {sampled_class} (p={sampled_probability:.4f})")

    # Plot confusion matrix using seaborn
    plt.figure()
    sns.heatmap(
        matrix,
        annot=True,
        fmt="d",
        cmap="Blues",
        xticklabels=split.label_names,
        yticklabels=split.label_names,
        cbar_kws={"label": "Count"},
    )
    plt.xlabel("Predicted", fontsize=CAPTION_FONT_SIZE)
    plt.ylabel("Actual", fontsize=CAPTION_FONT_SIZE)
    # plt.title(f"Confusion Matrix - MNLR Aircraft Type (Accuracy: {accuracy:.2%})")
    plt.tight_layout()
    confusion_path = OUTPUT_DIR / 'figures' / "mnlr_confusion_matrix.pdf"
    plt.savefig(confusion_path, bbox_inches="tight", dpi=300)
    plt.close()
    # plt.show()
    # print(f"\nSaved confusion matrix plot: {confusion_path}")
    plt.close()

    # Plot normalized confusion matrix (row-wise normalization)
    normalized_matrix = matrix.astype('float') / matrix.sum(axis=1, keepdims=True)
    plt.figure()
    sns.heatmap(
        normalized_matrix,
        annot=True,
        fmt=".2f",
        cmap="Blues",
        xticklabels=split.label_names,
        yticklabels=split.label_names,
        cbar_kws={"label": "Proportion"},
    )
    plt.xlabel("Predicted", fontsize=CAPTION_FONT_SIZE)
    plt.ylabel("Actual", fontsize=CAPTION_FONT_SIZE)
    # plt.title(f"Normalized Confusion Matrix - MNLR Aircraft Type (Accuracy: {accuracy:.2%})")
    plt.tight_layout()
    normalized_confusion_path = OUTPUT_DIR / 'figures' / "mnlr_confusion_matrix_normalized.pdf"
    plt.savefig(normalized_confusion_path, bbox_inches="tight", dpi=300)
    plt.close()

    return {
        "model": model,
        "accuracy": accuracy,
        "baseline_accuracy": baseline_accuracy,
        "report": report,
        "confusion_matrix": matrix,
        "aircraft_model": aircraft_model,
    }


def summarize_nce_results(split: UnifiedDataSplit, n_trials: int = 50, seed: int = 42) -> dict[str, Any]:
    torch.manual_seed(seed)
    np.random.seed(seed)

    print("=== NCE Acceptance Model ===")
    print(
        f"Shared split info: source={split.data_path}, train={len(split.train_indices)}, "
        f"test={len(split.test_indices)}, ratio={split.train_ratio:.2f}/{1 - split.train_ratio:.2f}, seed={split.seed}"
    )
    print(f"Shared input dimensionality: {split.features.shape[1]}")

    train_tensor = split.train_features
    test_tensor = split.test_features

    trainer = TrainNCE(train_tensor, test_tensor)
    study = trainer.run_optimization(n_trials=n_trials)
    best_params = study.best_trial.params

    print("Best trial:")
    print(f"  Value: {study.best_trial.value}")
    print("  Params:")
    for key, value in best_params.items():
        print(f"    {key}: {value}")

    best_model = trainer.train_final_model(best_params)
    noise_gen = NoiseGeneration(train_tensor, inflation=1.5)
    nce_network = getattr(best_model, "model", None)

    if nce_network is not None:
        with torch.no_grad():
            score_train = nce_network(train_tensor).squeeze()
            score_test = nce_network(test_tensor).squeeze()
            noise_tensor = noise_gen.sample(len(test_tensor))
            score_noise = nce_network(noise_tensor).squeeze()
            real_probs = torch.sigmoid(score_train - noise_gen.log_prob(train_tensor)).cpu().numpy()
            test_probs = torch.sigmoid(score_test - noise_gen.log_prob(test_tensor)).cpu().numpy()
            noise_probs = torch.sigmoid(score_noise - noise_gen.log_prob(noise_tensor)).cpu().numpy()

        print("\nTrain/Test/Noise probability summary from the trained NCE model")
        print(f"Training set mean probability: {real_probs.mean():.2%}")
        print(f"Testing set mean probability: {test_probs.mean():.2%}")
        print(f"Noise set mean probability: {noise_probs.mean():.2%}")
        print(f"Training set max probability: {real_probs.max():.2%}")
        print(f"Testing set max probability: {test_probs.max():.2%}")
        print(f"Noise set max probability: {noise_probs.max():.2%}")
        saved_paths = plot_nce_summary_figures(real_probs, test_probs, noise_probs)
        if saved_paths:
            print("Saved seaborn summary figures:")
            for path in saved_paths:
                print(f"  {path}")
    else:
        real_probs = np.array([])
        test_probs = np.array([])
        noise_probs = np.array([])
        print("\nTrain/Test/Noise probability summary from the trained NCE model")
        print("Skipped: trained model does not expose an underlying network.")

    return {
        "study": study,
        "best_model": best_model,
        "train_probs": real_probs,
        "test_probs": test_probs,
        "noise_probs": noise_probs,
    }


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Reproduce the airlift results using the stochastic airlift module.")
    parser.add_argument(
        "--aircraft-data",
        type=Path,
        default=None,
        help="Optional input workbook/CSV path. If set, this source is used for both MNLR and NCE to keep inputs aligned.",
    )
    parser.add_argument(
        "--nce-data",
        type=Path,
        default=None,
        help="Optional input workbook/CSV path. If set, this source is used for both MNLR and NCE to keep inputs aligned.",
    )
    parser.add_argument(
        "--run",
        choices=["mnlr", "nce", "all"],
        default="nce",
        help="Which reproduction experiment to execute.",
    )
    parser.add_argument("--n-trials", type=int, default=50, help="Optuna trials for the NCE experiment.")
    parser.add_argument("--seed", type=int, default=42, help="Random seed shared by the split and model training.")
    parser.add_argument(
        "--alpha",
        type=float,
        default=0.75,
        help=(
            "Exponent applied to inverse-frequency class weights for MNLR. "
            "0.0 means no weighting; 1.0 means full inverse-frequency weighting."
        ),
    )
    parser.add_argument(
        "--save-figs",
        type=bool,
        default=False,
        help="Whether to save the generated figures.",
    )

    return parser


def main() -> None:
    args = build_parser().parse_args()

    configure_determinism(args.seed)

    if args.aircraft_data is not None and args.nce_data is not None and args.aircraft_data != args.nce_data:
        raise ValueError(
            "--aircraft-data and --nce-data must match because MNLR and NCE now share one data source and split."
        )

    shared_data_path = args.aircraft_data if args.aircraft_data is not None else args.nce_data
    split = build_shared_split(data_path=shared_data_path, train_ratio=0.7, seed=args.seed)

    if args.run in {"mnlr", "all"}:
        if not (0.0 <= args.alpha <= 1.0):
            raise ValueError("--alpha must be between 0.0 and 1.0.")
        summarize_mnlr_results(split, class_weight_alpha=args.alpha)

    if args.run in {"nce", "all"}:
        summarize_nce_results(split, n_trials=args.n_trials, seed=args.seed)


if __name__ == "__main__":
    main()