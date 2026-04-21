"""Generate the overlay-style ablation comparison figure from precomputed CSV outputs."""

from __future__ import annotations

import argparse
import warnings
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_RESULTS_DIR = PROJECT_ROOT / "results" / "ablation"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Create the two-panel overlay figure for ablation train/test comparisons."
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


def build_summary_table(
    train_df: pd.DataFrame,
    test_df: pd.DataFrame,
    targets: np.ndarray,
) -> pd.DataFrame:
    summary_rows: list[dict[str, str]] = []
    for index, target in enumerate(targets):
        if np.isnan(test_df.loc[index, "ablation_R2"]):
            continue
        summary_rows.append(
            {
                "Target": target,
                "Baseline_Train": f"{train_df.loc[index, 'baseline_R2']:.4f}+/-{train_df.loc[index, 'baseline_R2_std']:.4f}",
                "Ablation_Train": f"{train_df.loc[index, 'ablation_R2']:.4f}+/-{train_df.loc[index, 'ablation_R2_std']:.4f}",
                "Train_Drop": f"{train_df.loc[index, 'baseline_R2'] - train_df.loc[index, 'ablation_R2']:.4f}",
                "Baseline_Test": f"{test_df.loc[index, 'baseline_R2']:.4f}+/-{test_df.loc[index, 'baseline_R2_std']:.4f}",
                "Ablation_Test": f"{test_df.loc[index, 'ablation_R2']:.4f}+/-{test_df.loc[index, 'ablation_R2_std']:.4f}",
                "Test_Drop": f"{test_df.loc[index, 'baseline_R2'] - test_df.loc[index, 'ablation_R2']:.4f}",
                "Removed_Features": test_df.loc[index, "removed_features"],
            }
        )
    return pd.DataFrame(summary_rows)


def main() -> None:
    args = parse_args()
    configure_plotting()

    train_df, test_df = load_inputs(args.results_dir)
    colors = {
        "baseline": "#4A90E2",
        "ablation": "#E8743B",
    }

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))
    targets = train_df["target"].to_numpy()
    x = np.arange(len(targets))
    width = 0.65

    baseline_train = train_df["baseline_R2"].to_numpy()
    baseline_train_std = train_df["baseline_R2_std"].to_numpy()
    ablation_train = train_df["ablation_R2"].to_numpy()
    ablation_train_std = train_df["ablation_R2_std"].to_numpy()

    ax1.bar(
        x,
        baseline_train,
        width,
        label="Baseline (All Features)",
        color=colors["baseline"],
        alpha=0.85,
        edgecolor="white",
        linewidth=1.5,
        zorder=2,
    )
    for index in range(len(targets)):
        if not np.isnan(ablation_train[index]):
            ax1.bar(
                x[index],
                ablation_train[index],
                width,
                color=colors["ablation"],
                alpha=0.75,
                edgecolor="white",
                linewidth=1.5,
                zorder=3,
            )
    ax1.bar([], [], width, label="Ablation (Remove |r|>0.8)", color=colors["ablation"], alpha=0.75)
    for index in range(len(targets)):
        ax1.errorbar(
            x[index],
            baseline_train[index],
            yerr=baseline_train_std[index],
            fmt="none",
            ecolor="#2C3E50",
            capsize=4,
            capthick=1.5,
            alpha=0.6,
            zorder=4,
        )
        if not np.isnan(ablation_train[index]):
            ax1.errorbar(
                x[index],
                ablation_train[index],
                yerr=ablation_train_std[index],
                fmt="none",
                ecolor="#8B4513",
                capsize=4,
                capthick=1.5,
                alpha=0.6,
                zorder=5,
            )
    ax1.set_xlabel("Target Variable", fontsize=12, fontweight="bold")
    ax1.set_ylabel("Train R² Score", fontsize=12, fontweight="bold")
    ax1.set_title("(a) Training Performance", fontsize=13, fontweight="bold", loc="left", pad=10)
    ax1.set_xticks(x)
    ax1.set_xticklabels(targets, rotation=45, ha="right", fontsize=10)
    ax1.set_ylim([0.65, 1.03])
    ax1.legend(fontsize=10, frameon=True, fancybox=True, shadow=True, loc="lower left", edgecolor="gray", framealpha=0.95)
    ax1.grid(axis="y", alpha=0.25, linestyle="--", linewidth=0.8, color="gray")
    ax1.set_axisbelow(True)
    ax1.spines["top"].set_visible(False)
    ax1.spines["right"].set_visible(False)
    ax1.spines["left"].set_linewidth(1.2)
    ax1.spines["bottom"].set_linewidth(1.2)
    ax1.tick_params(labelsize=10, width=1.2)

    baseline_test = test_df["baseline_R2"].to_numpy()
    baseline_test_std = test_df["baseline_R2_std"].to_numpy()
    ablation_test = test_df["ablation_R2"].to_numpy()
    ablation_test_std = test_df["ablation_R2_std"].to_numpy()

    ax2.bar(
        x,
        baseline_test,
        width,
        label="Baseline (All Features)",
        color=colors["baseline"],
        alpha=0.85,
        edgecolor="white",
        linewidth=1.5,
        zorder=2,
    )
    for index in range(len(targets)):
        if not np.isnan(ablation_test[index]):
            ax2.bar(
                x[index],
                ablation_test[index],
                width,
                color=colors["ablation"],
                alpha=0.75,
                edgecolor="white",
                linewidth=1.5,
                zorder=3,
            )
    ax2.bar([], [], width, label="Ablation (Remove |r|>0.8)", color=colors["ablation"], alpha=0.75)
    for index in range(len(targets)):
        ax2.errorbar(
            x[index],
            baseline_test[index],
            yerr=baseline_test_std[index],
            fmt="none",
            ecolor="#2C3E50",
            capsize=4,
            capthick=1.5,
            alpha=0.6,
            zorder=4,
        )
        if not np.isnan(ablation_test[index]):
            ax2.errorbar(
                x[index],
                ablation_test[index],
                yerr=ablation_test_std[index],
                fmt="none",
                ecolor="#8B4513",
                capsize=4,
                capthick=1.5,
                alpha=0.6,
                zorder=5,
            )
    ax2.set_xlabel("Target Variable", fontsize=12, fontweight="bold")
    ax2.set_ylabel("Test R² Score", fontsize=12, fontweight="bold")
    ax2.set_title("(b) Test Performance", fontsize=13, fontweight="bold", loc="left", pad=10)
    ax2.set_xticks(x)
    ax2.set_xticklabels(targets, rotation=45, ha="right", fontsize=10)
    ax2.set_ylim([0.60, 1.03])
    ax2.legend(fontsize=10, frameon=True, fancybox=True, shadow=True, loc="lower left", edgecolor="gray", framealpha=0.95)
    ax2.grid(axis="y", alpha=0.25, linestyle="--", linewidth=0.8, color="gray")
    ax2.set_axisbelow(True)
    ax2.spines["top"].set_visible(False)
    ax2.spines["right"].set_visible(False)
    ax2.spines["left"].set_linewidth(1.2)
    ax2.spines["bottom"].set_linewidth(1.2)
    ax2.tick_params(labelsize=10, width=1.2)

    fig.tight_layout()
    fig.savefig(args.results_dir / "ablation_combined_overlay.pdf", dpi=300, bbox_inches="tight")
    fig.savefig(args.results_dir / "ablation_combined_overlay.png", dpi=300, bbox_inches="tight")
    plt.close(fig)

    summary_df = build_summary_table(train_df=train_df, test_df=test_df, targets=targets)
    summary_df.to_csv(args.results_dir / "ablation_performance_summary.csv", index=False, encoding="utf-8-sig")

    print("Generated overlay comparison figure:")
    print(f"  - {args.results_dir / 'ablation_combined_overlay.pdf'}")
    print(f"  - {args.results_dir / 'ablation_combined_overlay.png'}")
    print("\nGenerated performance summary table:")
    print(f"  - {args.results_dir / 'ablation_performance_summary.csv'}")
    print("\n" + summary_df.to_string(index=False))


if __name__ == "__main__":
    main()
