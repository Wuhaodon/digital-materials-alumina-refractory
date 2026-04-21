"""Generate the reordered train-versus-test ablation comparison figure."""

from __future__ import annotations

import argparse
import warnings
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_RESULTS_DIR = PROJECT_ROOT / "results" / "ablation"
DEFAULT_TARGET_ORDER = ["WE", "hm-hf", "Gf", "Pmax", "WPP", "KIC", "Stiffness", "Modulus"]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Create the reordered train-versus-test ablation comparison figure."
    )
    parser.add_argument("--results-dir", type=Path, default=DEFAULT_RESULTS_DIR)
    return parser.parse_args()


def configure_plotting() -> None:
    warnings.filterwarnings("ignore")
    plt.rcParams["font.family"] = "Times New Roman"
    plt.rcParams["axes.unicode_minus"] = False


def load_inputs(results_dir: Path) -> tuple[pd.DataFrame, pd.DataFrame]:
    train_path = results_dir / "baseline_vs_ablation_train.csv"
    test_path = results_dir / "baseline_vs_ablation_test.csv"
    if not train_path.exists() or not test_path.exists():
        raise FileNotFoundError(
            "Missing ablation comparison tables. Run scripts/03_ablation_stability.py first."
        )
    return pd.read_csv(train_path), pd.read_csv(test_path)


def reorder_targets(train_df: pd.DataFrame, test_df: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    train_df = train_df.copy()
    test_df = test_df.copy()
    train_df["target"] = pd.Categorical(train_df["target"], categories=DEFAULT_TARGET_ORDER, ordered=True)
    test_df["target"] = pd.Categorical(test_df["target"], categories=DEFAULT_TARGET_ORDER, ordered=True)
    return (
        train_df.sort_values("target").reset_index(drop=True),
        test_df.sort_values("target").reset_index(drop=True),
    )


def main() -> None:
    args = parse_args()
    configure_plotting()

    train_df, test_df = reorder_targets(*load_inputs(args.results_dir))
    colors = {
        "train_baseline": "#7BA3D0",
        "train_ablation": "#F4A460",
        "test_baseline": "#9B7EBD",
        "test_ablation": "#E8A87C",
    }

    fig, ax = plt.subplots(figsize=(14, 6))
    targets = train_df["target"].to_numpy()
    x = np.arange(len(targets))
    width = 0.35
    gap = 0.08

    baseline_train = train_df["baseline_R2"].to_numpy()
    ablation_train = train_df["ablation_R2"].to_numpy()
    baseline_test = test_df["baseline_R2"].to_numpy()
    ablation_test = test_df["ablation_R2"].to_numpy()

    for index in range(len(targets)):
        pos_train = x[index] - width / 2 - gap / 2
        pos_test = x[index] + width / 2 + gap / 2

        ax.bar(
            pos_train,
            baseline_train[index],
            width,
            color=colors["train_baseline"],
            alpha=0.9,
            edgecolor="white",
            linewidth=1.2,
            zorder=2,
        )
        if not np.isnan(ablation_train[index]):
            ax.bar(
                pos_train,
                ablation_train[index],
                width,
                color=colors["train_ablation"],
                alpha=0.75,
                edgecolor="white",
                linewidth=1.2,
                zorder=3,
            )

        ax.bar(
            pos_test,
            baseline_test[index],
            width,
            color=colors["test_baseline"],
            alpha=0.9,
            edgecolor="white",
            linewidth=1.2,
            zorder=2,
        )
        if not np.isnan(ablation_test[index]):
            ax.bar(
                pos_test,
                ablation_test[index],
                width,
                color=colors["test_ablation"],
                alpha=0.75,
                edgecolor="white",
                linewidth=1.2,
                zorder=3,
            )

    legend_elements = [
        plt.Rectangle((0, 0), 1, 1, fc=colors["train_baseline"], alpha=0.9, edgecolor="white", linewidth=1.2, label="Train - Baseline"),
        plt.Rectangle((0, 0), 1, 1, fc=colors["train_ablation"], alpha=0.75, edgecolor="white", linewidth=1.2, label="Train - Ablation"),
        plt.Rectangle((0, 0), 1, 1, fc=colors["test_baseline"], alpha=0.9, edgecolor="white", linewidth=1.2, label="Test - Baseline"),
        plt.Rectangle((0, 0), 1, 1, fc=colors["test_ablation"], alpha=0.75, edgecolor="white", linewidth=1.2, label="Test - Ablation"),
    ]
    ax.legend(
        handles=legend_elements,
        fontsize=10,
        frameon=True,
        fancybox=True,
        shadow=True,
        loc="lower left",
        edgecolor="gray",
        framealpha=0.95,
        ncol=2,
    )

    ax.set_xlabel("Target Variable", fontsize=13, fontweight="bold")
    ax.set_ylabel("R² Score", fontsize=13, fontweight="bold")
    ax.set_title("Ablation Study: Train vs Test Performance Comparison", fontsize=14, fontweight="bold", pad=15)
    ax.set_xticks(x)
    ax.set_xticklabels(targets, fontsize=11, fontweight="bold")
    ax.set_ylim([0.60, 1.05])
    ax.grid(axis="y", alpha=0.25, linestyle="--", linewidth=0.8, color="gray", zorder=0)
    ax.set_axisbelow(True)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.spines["left"].set_linewidth(1.5)
    ax.spines["bottom"].set_linewidth(1.5)
    ax.tick_params(labelsize=11, width=1.5, length=5)

    for index in range(len(targets)):
        y_pos = 0.58
        ax.plot([x[index] - width - gap / 2, x[index] - gap / 2], [y_pos, y_pos], color="gray", linewidth=2, alpha=0.5, clip_on=False)
        ax.plot([x[index] + gap / 2, x[index] + width + gap / 2], [y_pos, y_pos], color="gray", linewidth=2, alpha=0.5, clip_on=False)
        ax.text(
            x[index] - width / 2 - gap / 2,
            y_pos - 0.02,
            "Train",
            ha="center",
            va="top",
            fontsize=8,
            color="gray",
            style="italic",
            clip_on=False,
        )
        ax.text(
            x[index] + width / 2 + gap / 2,
            y_pos - 0.02,
            "Test",
            ha="center",
            va="top",
            fontsize=8,
            color="gray",
            style="italic",
            clip_on=False,
        )

    fig.tight_layout()
    fig.savefig(args.results_dir / "ablation_train_test_reordered.pdf", dpi=300, bbox_inches="tight")
    fig.savefig(args.results_dir / "ablation_train_test_reordered.png", dpi=300, bbox_inches="tight")
    plt.close(fig)

    print("Generated reordered train/test comparison figure:")
    print(f"  - {args.results_dir / 'ablation_train_test_reordered.pdf'}")
    print(f"  - {args.results_dir / 'ablation_train_test_reordered.png'}")
    print("\nColor palette:")
    print(f"  Train Baseline: {colors['train_baseline']}")
    print(f"  Train Ablation: {colors['train_ablation']}")
    print(f"  Test Baseline:  {colors['test_baseline']}")
    print(f"  Test Ablation:  {colors['test_ablation']}")


if __name__ == "__main__":
    main()
