"""Run SHAP exports for the ablation feature subsets used in the manuscript follow-up."""

from __future__ import annotations

import argparse
import os
import random
import warnings
from importlib.metadata import version
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import shap
import torch
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler
from tabpfn import TabPFNRegressor

from _selected_feature_io import read_selected_features, resolve_default_modeling_path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_INPUT_PATH = resolve_default_modeling_path("merged_features.xlsx")
DEFAULT_OUTPUT_DIR = PROJECT_ROOT / "results" / "shap_ablation_outputs"
DEFAULT_MODEL_CKPT_PATH = Path("/root/.cache/tabpfn/models/tabpfn-v2.5-regressor-v2.5_default.ckpt")
TARGETS = read_selected_features()
DEFAULT_CORR_THRESHOLD = 0.8
DEFAULT_SEED = 42
DEFAULT_BACKGROUND_SIZE = 200


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run SHAP exports for the predefined ablation feature subsets."
    )
    parser.add_argument("--input-path", type=Path, default=DEFAULT_INPUT_PATH)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--model-ckpt-path", type=Path, default=DEFAULT_MODEL_CKPT_PATH)
    parser.add_argument("--background-size", type=int, default=DEFAULT_BACKGROUND_SIZE)
    parser.add_argument("--corr-threshold", type=float, default=DEFAULT_CORR_THRESHOLD)
    parser.add_argument("--seed", type=int, default=DEFAULT_SEED)
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
        raise FileNotFoundError(
            f"Missing modeling table: {input_path}. Restore the private dataset before running this script."
        )
    data = pd.read_excel(input_path)
    data = data.dropna(subset=TARGETS).drop_duplicates(ignore_index=True)
    all_features = [column for column in data.columns if column not in TARGETS]
    return data[all_features].copy(), data[TARGETS].copy(), all_features


def build_ablation_feature_map(
    x_full_df: pd.DataFrame,
    y_full_df: pd.DataFrame,
    threshold: float,
) -> dict[str, list[str]]:
    feature_map: dict[str, list[str]] = {}
    for target in TARGETS:
        correlations = []
        for feature in x_full_df.columns:
            corr_value = x_full_df[feature].corr(y_full_df[target])
            if pd.isna(corr_value) or abs(float(corr_value)) <= threshold:
                continue
            correlations.append((feature, abs(float(corr_value))))
        correlations.sort(key=lambda item: (-item[1], item[0]))
        if correlations:
            feature_map[target] = [feature for feature, _ in correlations]
    return feature_map


def sample_array_with_idx(array: np.ndarray, n_rows: int, seed: int) -> tuple[np.ndarray, np.ndarray]:
    n_rows = min(n_rows, array.shape[0])
    if n_rows == array.shape[0]:
        indices = np.arange(array.shape[0])
    else:
        indices = np.random.RandomState(seed).choice(array.shape[0], n_rows, replace=False)
    return array[indices], indices


def build_model(device: torch.device, seed: int, model_ckpt_path: Path | None) -> TabPFNRegressor:
    if model_ckpt_path:
        return TabPFNRegressor(device=device, random_state=seed, model_path=model_ckpt_path)
    return TabPFNRegressor(device=device, random_state=seed)


def run_shap_analysis(
    target_name: str,
    features_to_use: list[str],
    y_target: np.ndarray,
    suffix: str,
    x_full_df: pd.DataFrame,
    all_features: list[str],
    output_dir: Path,
    background_size: int,
    seed: int,
    device: torch.device,
    model_ckpt_path: Path | None,
) -> shap.Explanation:
    print(f"\n{'=' * 60}")
    print(f"Target: {target_name}{suffix}")
    print(f"Using {len(features_to_use)} features")
    if suffix:
        removed = [feature for feature in all_features if feature not in features_to_use]
        print(f"Removed features: {removed}")
    print(f"{'=' * 60}")

    x_frame = x_full_df[features_to_use].copy()
    imputer = SimpleImputer(strategy="mean")
    scaler = StandardScaler()
    x_standardized = scaler.fit_transform(imputer.fit_transform(x_frame))

    print("Training TabPFN model...")
    model = build_model(device=device, seed=seed, model_ckpt_path=model_ckpt_path)
    model.fit(x_standardized, y_target)

    n_samples = x_standardized.shape[0]
    x_background, _ = sample_array_with_idx(x_standardized, min(background_size, n_samples), seed=seed)
    x_explain, row_indices = sample_array_with_idx(x_standardized, n_samples, seed=seed)
    x_explain_original = scaler.inverse_transform(x_explain)

    print(f"Computing SHAP values (background={x_background.shape[0]}, explain={x_explain.shape[0]})...")
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        torch.cuda.synchronize()

    explainer = shap.Explainer(model.predict, x_background)
    shap_values = explainer(x_explain, batch_size=8)

    del explainer, model
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    target_dir = output_dir / f"{target_name}{suffix}"
    export_dir = target_dir / "shap_exports"
    export_dir.mkdir(parents=True, exist_ok=True)

    x_explain_df_std = pd.DataFrame(x_explain, columns=features_to_use)
    x_explain_df_std.insert(0, "row_index", row_indices)
    x_explain_df_orig = pd.DataFrame(x_explain_original, columns=features_to_use)
    x_explain_df_orig.insert(0, "row_index", row_indices)
    x_explain_df_std.to_csv(export_dir / "X_explain_standardized.csv", index=False, encoding="utf-8-sig")
    x_explain_df_orig.to_csv(export_dir / "X_explain_original_units.csv", index=False, encoding="utf-8-sig")

    shap_values_df = pd.DataFrame(shap_values.values, columns=features_to_use)
    shap_values_df.insert(0, "row_index", row_indices)
    shap_values_df.to_csv(export_dir / "shap_values.csv", index=False, encoding="utf-8-sig")

    base_values = np.atleast_1d(shap_values.base_values)
    pd.DataFrame({"base_value": base_values}).to_csv(
        export_dir / "shap_base_values.csv",
        index=False,
        encoding="utf-8-sig",
    )

    print("Generating SHAP figures...")
    fig = plt.figure(figsize=(10, 6))
    shap.summary_plot(shap_values.values, x_explain_original, feature_names=features_to_use, show=False)
    plt.title(f"SHAP Summary for {target_name}{suffix}")
    plt.tight_layout()
    fig.savefig(target_dir / "SHAP_Summary.pdf", format="pdf", bbox_inches="tight")
    fig.savefig(target_dir / "SHAP_Summary.png", dpi=300)
    plt.close(fig)

    fig = plt.figure(figsize=(10, 6))
    shap.summary_plot(
        shap_values.values,
        x_explain_original,
        feature_names=features_to_use,
        plot_type="bar",
        show=False,
    )
    plt.title(f"SHAP Summary (Bar) for {target_name}{suffix}")
    plt.tight_layout()
    fig.savefig(target_dir / "SHAP_Summary_Bar.pdf", format="pdf", bbox_inches="tight")
    fig.savefig(target_dir / "SHAP_Summary_Bar.png", dpi=300)
    plt.close(fig)

    feature_importance = np.abs(shap_values.values).mean(axis=0)
    top_features = list(pd.Series(features_to_use)[np.argsort(feature_importance)[::-1][:3]])
    for feature in top_features:
        feature_index = features_to_use.index(feature)
        fig = plt.figure(figsize=(8, 6))
        shap.dependence_plot(
            feature_index,
            shap_values.values,
            x_explain_original,
            feature_names=features_to_use,
            interaction_index="auto",
            show=False,
        )
        plt.title(f"SHAP Dependence: {target_name}{suffix} - {feature}")
        plt.tight_layout()
        fig.savefig(target_dir / f"SHAP_Dependency_{feature}.pdf", format="pdf", bbox_inches="tight")
        fig.savefig(target_dir / f"SHAP_Dependency_{feature}.png", dpi=300)
        plt.close(fig)

    try:
        sample_index = 0
        base_value = (
            shap_values.base_values[sample_index]
            if hasattr(shap_values.base_values, "__len__")
            else shap_values.base_values
        )
        explanation = shap.Explanation(
            values=shap_values.values[sample_index, :],
            base_values=base_value,
            data=x_explain_original[sample_index],
            feature_names=features_to_use,
        )
        fig = plt.figure(figsize=(8, 6))
        shap.plots.waterfall(explanation, show=False)
        plt.title(f"SHAP Waterfall: {target_name}{suffix}")
        plt.tight_layout()
        fig.savefig(target_dir / "SHAP_Waterfall.pdf", format="pdf", bbox_inches="tight")
        fig.savefig(target_dir / "SHAP_Waterfall.png", dpi=300)
        plt.close(fig)
    except Exception as exc:
        print(f"Waterfall plot failed: {exc}")

    print(f"Completed {target_name}{suffix}; outputs saved to {target_dir}")
    return shap_values


def main() -> None:
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)

    os.environ["CUDA_LAUNCH_BLOCKING"] = "1"
    os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"

    seed_everything(args.seed)
    configure_plotting()

    x_full_df, y_full_df, all_features = load_dataset(args.input_path)
    print(f"Dataset: {x_full_df.shape[0]} samples, {len(all_features)} features, {len(TARGETS)} targets")
    print(f"shap version: {shap.__version__}")
    print(f"tabpfn version: {version('tabpfn')}")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    ablation_feature_map = build_ablation_feature_map(x_full_df, y_full_df, args.corr_threshold)
    print(f"Ablation targets above |r|>{args.corr_threshold}: {sorted(ablation_feature_map)}")

    for target, removed_features in ablation_feature_map.items():
        y_target = y_full_df[target].to_numpy()
        remaining_features = [feature for feature in all_features if feature not in removed_features]
        run_shap_analysis(
            target_name=target,
            features_to_use=remaining_features,
            y_target=y_target,
            suffix="_ablation",
            x_full_df=x_full_df,
            all_features=all_features,
            output_dir=args.output_dir,
            background_size=args.background_size,
            seed=args.seed,
            device=device,
            model_ckpt_path=args.model_ckpt_path,
        )
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    print(f"\n{'=' * 60}")
    print("All ablation SHAP analyses completed.")
    print(f"Results saved to: {args.output_dir.resolve()}")
    print(f"{'=' * 60}")


if __name__ == "__main__":
    main()

