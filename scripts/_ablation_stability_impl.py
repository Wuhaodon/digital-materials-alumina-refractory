"""Implementation for the ablation stability command-line entrypoint."""

from __future__ import annotations

import argparse
import random
import warnings
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
import torch
from sklearn.impute import SimpleImputer
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.model_selection import KFold
from sklearn.multioutput import MultiOutputRegressor
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from tabpfn import TabPFNRegressor

from _selected_feature_io import read_selected_features, resolve_default_modeling_path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_INPUT_PATH = resolve_default_modeling_path("merged_features.xlsx")
DEFAULT_OUTPUT_DIR = PROJECT_ROOT / "results" / "ablation"
TARGETS = read_selected_features()


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run the baseline-versus-ablation stability analysis for the public eight-target workflow."
    )
    parser.add_argument("--input-path", type=Path, default=DEFAULT_INPUT_PATH)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--corr-threshold", type=float, default=0.8)
    parser.add_argument("--seed", type=int, default=42)
    return parser.parse_args()


def seed_everything(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False


def configure_plotting() -> None:
    warnings.filterwarnings("ignore")
    plt.rcParams["font.family"] = "Times New Roman"
    plt.rcParams["axes.unicode_minus"] = False


def load_dataset(input_path: Path) -> tuple[pd.DataFrame, pd.DataFrame, list[str]]:
    if not input_path.exists():
        raise FileNotFoundError(f"Missing modeling table: {input_path}")
    data = pd.read_excel(input_path)
    data = data.dropna(subset=TARGETS).drop_duplicates(ignore_index=True)
    features = [column for column in data.columns if column not in TARGETS]
    return data[features].copy(), data[TARGETS].copy(), features


def compute_correlations(
    x_frame: pd.DataFrame,
    y_frame: pd.DataFrame,
    threshold: float,
    output_dir: Path,
) -> dict[str, list[str]]:
    corr_matrix = pd.DataFrame(index=x_frame.columns, columns=TARGETS)
    for target in TARGETS:
        for feature in x_frame.columns:
            corr_matrix.loc[feature, target] = x_frame[feature].corr(y_frame[target])
    corr_matrix = corr_matrix.astype(float)
    corr_matrix.to_csv(output_dir / "feature_target_correlation.csv", encoding="utf-8-sig")

    fig, ax = plt.subplots(figsize=(12, 10))
    sns.heatmap(
        corr_matrix,
        annot=True,
        fmt=".2f",
        cmap="coolwarm",
        center=0,
        vmin=-1,
        vmax=1,
        cbar_kws={"label": "Pearson Correlation"},
        ax=ax,
    )
    ax.set_title("Feature-Target Correlation Matrix")
    ax.set_xlabel("Targets")
    ax.set_ylabel("Features")
    fig.tight_layout()
    fig.savefig(output_dir / "correlation_heatmap.png", dpi=300)
    fig.savefig(output_dir / "correlation_heatmap.pdf")
    plt.close(fig)

    high_corr_features: dict[str, list[str]] = {}
    lines: list[str] = []
    for target in TARGETS:
        high_corr = corr_matrix[target][corr_matrix[target].abs() > threshold].sort_values(key=abs, ascending=False)
        high_corr_features[target] = high_corr.index.tolist()
        lines.append(f"\n{target}:")
        if high_corr.empty:
            lines.append(f"  none (|r|>{threshold})")
            continue
        for feature in high_corr.index:
            lines.append(f"  - {feature}: {float(high_corr[feature]):.3f}")
    (output_dir / "high_correlation_features.txt").write_text("\n".join(lines).lstrip(), encoding="utf-8")
    return high_corr_features


def make_pipeline(device: torch.device, seed: int) -> Pipeline:
    base = TabPFNRegressor(device=device, random_state=seed)
    model = MultiOutputRegressor(estimator=base, n_jobs=(1 if device.type == "cuda" else -1))
    return Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="mean")),
            ("scaler", StandardScaler()),
            ("model", model),
        ]
    )


def run_cv(
    x_frame: pd.DataFrame,
    y_values: np.ndarray,
    target_names: list[str],
    experiment: str,
    device: torch.device,
    seed: int,
) -> pd.DataFrame:
    kfold = KFold(n_splits=5, shuffle=True, random_state=seed)
    rows: list[dict[str, object]] = []
    for fold_id, (train_idx, valid_idx) in enumerate(kfold.split(x_frame), start=1):
        x_train = x_frame.iloc[train_idx]
        x_valid = x_frame.iloc[valid_idx]
        y_train = y_values[train_idx]
        y_valid = y_values[valid_idx]
        pipeline = make_pipeline(device, seed)
        pipeline.fit(x_train, y_train)
        y_pred_train = pipeline.predict(x_train)
        y_pred_valid = pipeline.predict(x_valid)
        for index, target in enumerate(target_names):
            rows.append({"experiment": experiment, "fold": fold_id, "target": target, "split": "train", "R2": r2_score(y_train[:, index], y_pred_train[:, index]), "MSE": mean_squared_error(y_train[:, index], y_pred_train[:, index])})
            rows.append({"experiment": experiment, "fold": fold_id, "target": target, "split": "test", "R2": r2_score(y_valid[:, index], y_pred_valid[:, index]), "MSE": mean_squared_error(y_valid[:, index], y_pred_valid[:, index])})
    return pd.DataFrame(rows)


def summarize(metrics_df: pd.DataFrame, experiment: str, split: str, n_features: int, removed: str) -> pd.DataFrame:
    summary = (
        metrics_df[metrics_df["split"] == split]
        .groupby("target")
        .agg(R2_mean=("R2", "mean"), R2_std=("R2", "std"), MSE_mean=("MSE", "mean"), MSE_std=("MSE", "std"))
        .reset_index()
    )
    summary.insert(0, "experiment", experiment)
    summary.insert(1, "split", split)
    summary.insert(3, "n_features", n_features)
    summary.insert(4, "removed_features", removed)
    return summary


def build_comparison(
    baseline_train: pd.DataFrame,
    baseline_test: pd.DataFrame,
    ablation_train_frames: list[pd.DataFrame],
    ablation_test_frames: list[pd.DataFrame],
) -> tuple[pd.DataFrame, pd.DataFrame]:
    train_rows: list[dict[str, object]] = []
    test_rows: list[dict[str, object]] = []
    for target in TARGETS:
        base_train = baseline_train[baseline_train["target"] == target].iloc[0]
        base_test = baseline_test[baseline_test["target"] == target].iloc[0]
        train_match = next((frame[frame["target"] == target].iloc[0] for frame in ablation_train_frames if target in frame["target"].values), None)
        test_match = next((frame[frame["target"] == target].iloc[0] for frame in ablation_test_frames if target in frame["target"].values), None)
        train_rows.append({"target": target, "baseline_R2": base_train["R2_mean"], "baseline_R2_std": base_train["R2_std"], "ablation_R2": (train_match["R2_mean"] if train_match is not None else np.nan), "ablation_R2_std": (train_match["R2_std"] if train_match is not None else np.nan), "R2_drop": (base_train["R2_mean"] - train_match["R2_mean"] if train_match is not None else np.nan), "removed_features": (train_match["removed_features"] if train_match is not None else "None (no high corr)")})
        test_rows.append({"target": target, "baseline_R2": base_test["R2_mean"], "baseline_R2_std": base_test["R2_std"], "ablation_R2": (test_match["R2_mean"] if test_match is not None else np.nan), "ablation_R2_std": (test_match["R2_std"] if test_match is not None else np.nan), "R2_drop": (base_test["R2_mean"] - test_match["R2_mean"] if test_match is not None else np.nan), "removed_features": (test_match["removed_features"] if test_match is not None else "None (no high corr)")})
    return pd.DataFrame(train_rows), pd.DataFrame(test_rows)


def plot_r2_comparison(comparison_df: pd.DataFrame, output_dir: Path, split: str) -> None:
    fig, ax = plt.subplots(figsize=(12, 6))
    x = np.arange(len(TARGETS))
    width = 0.35
    ax.bar(x - width / 2, comparison_df["baseline_R2"].values, width, label="Baseline (All Features)", yerr=comparison_df["baseline_R2_std"].values, capsize=5, alpha=0.8, color="#1f77b4")
    ax.bar(x + width / 2, comparison_df["ablation_R2"].values, width, label="Ablation (Remove |r|>0.8)", yerr=comparison_df["ablation_R2_std"].values, capsize=5, alpha=0.8, color="#ff7f0e")
    ax.set_xlabel("Target", fontsize=12, fontweight="bold")
    ax.set_ylabel(f"{split} R² Score", fontsize=12, fontweight="bold")
    ax.set_title(f"Baseline vs Ablation: 5-Fold CV {split} Performance", fontsize=14, fontweight="bold")
    ax.set_xticks(x)
    ax.set_xticklabels(TARGETS, rotation=45, ha="right")
    ax.legend(fontsize=11)
    ax.grid(axis="y", alpha=0.3)
    fig.tight_layout()
    stem = f"baseline_vs_ablation_{split.lower()}_barplot"
    fig.savefig(output_dir / f"{stem}.png", dpi=300)
    fig.savefig(output_dir / f"{stem}.pdf")
    plt.close(fig)


def plot_r2_drop(comparison_df: pd.DataFrame, output_dir: Path, split: str) -> None:
    fig, ax = plt.subplots(figsize=(10, 6))
    r2_drop = comparison_df["R2_drop"].values
    colors = ["#d62728" if value > 0 else "#2ca02c" for value in np.nan_to_num(r2_drop, nan=0.0)]
    ax.barh(TARGETS, r2_drop, color=colors, alpha=0.7)
    ax.axvline(0, color="black", linewidth=1.5, linestyle="--")
    ax.set_xlabel(f"{split} R² Drop (Baseline - Ablation)", fontsize=12, fontweight="bold")
    ax.set_ylabel("Target", fontsize=12, fontweight="bold")
    ax.set_title(f"Performance Impact of Removing High-Correlation Features ({split})", fontsize=14, fontweight="bold")
    ax.grid(axis="x", alpha=0.3)
    fig.tight_layout()
    stem = f"r2_drop_{split.lower()}_barplot"
    fig.savefig(output_dir / f"{stem}.png", dpi=300)
    fig.savefig(output_dir / f"{stem}.pdf")
    plt.close(fig)


def main() -> None:
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    seed_everything(args.seed)
    configure_plotting()

    x_frame, y_frame, features = load_dataset(args.input_path)
    high_corr_features = compute_correlations(x_frame, y_frame, args.corr_threshold, args.output_dir)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    baseline_metrics = run_cv(x_frame, y_frame.to_numpy(), TARGETS, "Baseline", device, args.seed)
    baseline_train = summarize(baseline_metrics, "Baseline", "train", len(features), "None")
    baseline_test = summarize(baseline_metrics, "Baseline", "test", len(features), "None")

    ablation_metrics_frames: list[pd.DataFrame] = []
    ablation_train_frames: list[pd.DataFrame] = []
    ablation_test_frames: list[pd.DataFrame] = []
    for target in TARGETS:
        removed = high_corr_features[target]
        if not removed:
            continue
        remaining = [feature for feature in features if feature not in removed]
        metrics = run_cv(x_frame[remaining].copy(), y_frame[[target]].to_numpy(), [target], f"Ablation_{target}", device, args.seed)
        ablation_metrics_frames.append(metrics)
        removed_text = ", ".join(removed)
        ablation_train_frames.append(summarize(metrics, f"Ablation_{target}", "train", len(remaining), removed_text))
        ablation_test_frames.append(summarize(metrics, f"Ablation_{target}", "test", len(remaining), removed_text))

    pd.concat([baseline_metrics] + ablation_metrics_frames, ignore_index=True).to_csv(args.output_dir / "all_fold_metrics.csv", index=False, encoding="utf-8-sig")
    all_train = pd.concat([baseline_train] + ablation_train_frames, ignore_index=True)
    all_test = pd.concat([baseline_test] + ablation_test_frames, ignore_index=True)
    pd.concat([all_train, all_test], ignore_index=True).to_csv(args.output_dir / "ablation_summary.csv", index=False, encoding="utf-8-sig")
    all_train.to_csv(args.output_dir / "ablation_summary_train.csv", index=False, encoding="utf-8-sig")
    all_test.to_csv(args.output_dir / "ablation_summary_test.csv", index=False, encoding="utf-8-sig")

    comparison_train, comparison_test = build_comparison(baseline_train, baseline_test, ablation_train_frames, ablation_test_frames)
    comparison_train.to_csv(args.output_dir / "baseline_vs_ablation_train.csv", index=False, encoding="utf-8-sig")
    comparison_test.to_csv(args.output_dir / "baseline_vs_ablation_test.csv", index=False, encoding="utf-8-sig")
    plot_r2_comparison(comparison_test, args.output_dir, "Test")
    plot_r2_comparison(comparison_train, args.output_dir, "Train")
    plot_r2_drop(comparison_test, args.output_dir, "Test")
    plot_r2_drop(comparison_train, args.output_dir, "Train")


if __name__ == "__main__":
    main()
