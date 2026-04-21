"""Run the manuscript-aligned TabPFN regression workflow for the final 8-variable set."""

import os
import random
import sys
import warnings
from importlib.metadata import version
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
import shap
import torch
from sklearn.impute import SimpleImputer
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.model_selection import KFold
from sklearn.multioutput import MultiOutputRegressor
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from tabpfn import TabPFNRegressor, model_loading
from tabpfn.settings import settings

from _selected_feature_io import read_selected_features, resolve_default_modeling_path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
LOCAL_TABPFN_EXT = PROJECT_ROOT / "external" / "temp_tabpfn_ext" / "src"
DATA_PATH = resolve_default_modeling_path("merged_features.xlsx")
SAVE_DIR = PROJECT_ROOT / "results" / "cv"
TARGETS = read_selected_features()
SEED = 42
BACKGROUND_SAMPLE_SIZE = 50
EXPLANATION_SAMPLE_SIZE = 80
SHAPIQ_BUDGET = 200
COLOR_BY_FEATURE = None


def _coerce_path(value):
    return Path(value) if isinstance(value, str) else value


def patch_tabpfn_model_loading():
    original_load_config = model_loading.load_model_criterion_config
    original_resolve_model_path = model_loading.resolve_model_path

    def patched_load_config(*args, **kwargs):
        if "model_path" in kwargs:
            kwargs["model_path"] = _coerce_path(kwargs["model_path"])
        return original_load_config(*args, **kwargs)

    def patched_resolve_model_path(*args, **kwargs):
        result = original_resolve_model_path(*args, **kwargs)
        for attr in ("dirs", "paths", "model_dirs", "resolved_model_dirs"):
            if not hasattr(result, attr):
                continue
            value = getattr(result, attr)
            if isinstance(value, (list, tuple)):
                setattr(result, attr, [_coerce_path(item) for item in value])
            else:
                setattr(result, attr, _coerce_path(value))
        return result

    model_loading.load_model_criterion_config = patched_load_config
    model_loading.resolve_model_path = patched_resolve_model_path


patch_tabpfn_model_loading()
settings.tabpfn.model_cache_dir = Path(os.path.expanduser("~/.cache/tabpfn/models"))

if LOCAL_TABPFN_EXT.exists():
    sys.path.append(str(LOCAL_TABPFN_EXT))

try:
    from tabpfn_extensions.interpretability import shapiq as tabpfn_shapiq
except ImportError as exc:
    raise ImportError(
        "tabpfn_extensions is required. Install it or clone it under external/temp_tabpfn_ext."
    ) from exc


def seed_everything(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False



def build_pipeline(device):
    base_model = TabPFNRegressor(device=device, random_state=SEED)
    wrapped_model = MultiOutputRegressor(
        estimator=base_model,
        n_jobs=(1 if device.type == "cuda" else -1),
    )
    return Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="mean")),
            ("scaler", StandardScaler()),
            ("model", wrapped_model),
        ]
    )



def sample_rows(array, size, seed, y=None):
    size = min(size, array.shape[0])
    if size == array.shape[0]:
        indices = np.arange(array.shape[0])
    else:
        indices = np.random.RandomState(seed).choice(array.shape[0], size, replace=False)

    if y is not None:
        return array[indices], y[indices], indices
    return array[indices], indices



def load_dataset():
    if not DATA_PATH.exists():
        raise FileNotFoundError(
            f"Missing modeling table: {DATA_PATH}. Restore the private dataset before running this script."
        )

    data = pd.read_excel(DATA_PATH)
    data = data.dropna(subset=TARGETS).drop_duplicates(ignore_index=True)
    feature_names = [column for column in data.columns if column not in TARGETS]
    x_frame = data[feature_names].copy()
    y_values = data[TARGETS].to_numpy()
    return x_frame, y_values, feature_names



def run_cross_validation(x_frame, y_values, device):
    kfold = KFold(n_splits=5, shuffle=True, random_state=SEED)
    fold_metrics = []

    for fold_id, (train_idx, valid_idx) in enumerate(kfold.split(x_frame), start=1):
        x_train = x_frame.iloc[train_idx]
        x_valid = x_frame.iloc[valid_idx]
        y_train = y_values[train_idx]
        y_valid = y_values[valid_idx]

        pipeline = build_pipeline(device)
        pipeline.fit(x_train, y_train)
        y_pred = pipeline.predict(x_valid)

        for target_index, target_name in enumerate(TARGETS):
            fold_metrics.append(
                {
                    "fold": fold_id,
                    "target": target_name,
                    "R2": r2_score(y_valid[:, target_index], y_pred[:, target_index]),
                    "MSE": mean_squared_error(y_valid[:, target_index], y_pred[:, target_index]),
                }
            )

    metrics_df = pd.DataFrame(fold_metrics)
    summary_df = (
        metrics_df.groupby("target")
        .agg(
            R2_mean=("R2", "mean"),
            R2_std=("R2", "std"),
            MSE_mean=("MSE", "mean"),
            MSE_std=("MSE", "std"),
        )
        .reset_index()
    )
    return metrics_df, summary_df



def save_cv_outputs(metrics_df, summary_df):
    SAVE_DIR.mkdir(parents=True, exist_ok=True)
    metrics_df.to_csv(SAVE_DIR / "cv_fold_metrics.csv", index=False, encoding="utf-8-sig")
    summary_df.to_csv(SAVE_DIR / "cv_summary_by_target.csv", index=False, encoding="utf-8-sig")

    r2_pivot = metrics_df.pivot(index="target", columns="fold", values="R2").loc[TARGETS]
    means = r2_pivot.mean(axis=1)
    stds = r2_pivot.std(axis=1)
    r2_augmented = r2_pivot.copy()
    r2_augmented["Mean"] = means

    fold_means = r2_pivot.mean(axis=0)
    fold_means["Mean"] = means.mean()
    r2_augmented.loc["(Fold mean)"] = fold_means

    annotation = r2_augmented.copy().astype(object)
    for target_name in TARGETS:
        annotation.loc[target_name, "Mean"] = f"{means.loc[target_name]:.3f}+/-{stds.loc[target_name]:.3f}"
    annotation.loc["(Fold mean)"] = [
        f"{value:.3f}" if isinstance(value, (int, float, np.floating)) else value
        for value in r2_augmented.loc["(Fold mean)"]
    ]

    plt.figure(figsize=(1.2 * len(r2_augmented.columns) + 6, 0.6 * len(r2_augmented.index) + 3), dpi=160)
    sns.heatmap(
        r2_augmented.astype(float),
        annot=annotation,
        fmt="",
        cmap="coolwarm",
        vmin=0.0,
        vmax=1.0,
        cbar_kws={"label": "R2"},
        linewidths=0.5,
        linecolor="white",
        square=False,
    )
    plt.title("Five-fold cross-validation: target R2 summary", fontsize=12)
    plt.ylabel("")
    plt.xlabel("")
    plt.tight_layout()
    plt.savefig(SAVE_DIR / "cv_r2_heatmap_with_means.png", dpi=300)
    plt.savefig(SAVE_DIR / "cv_r2_heatmap_with_means.pdf")
    plt.close()



def fit_final_model(x_frame, y_values, device):
    imputer = SimpleImputer(strategy="mean")
    scaler = StandardScaler()
    x_standardized = scaler.fit_transform(imputer.fit_transform(x_frame))

    base_model = TabPFNRegressor(device=device, random_state=SEED)
    model = MultiOutputRegressor(
        estimator=base_model,
        n_jobs=(1 if device.type == "cuda" else -1),
    )
    model.fit(x_standardized, y_values)
    return model, scaler, x_standardized



def compute_shap_values(model, x_standardized, y_values, feature_names):
    os.environ["CUDA_LAUNCH_BLOCKING"] = "1"
    os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"

    x_background, y_background, _ = sample_rows(
        x_standardized,
        BACKGROUND_SAMPLE_SIZE,
        seed=SEED,
        y=y_values,
    )
    x_explain, row_indices = sample_rows(
        x_standardized,
        EXPLANATION_SAMPLE_SIZE,
        seed=SEED,
    )

    shap_values_list = []

    for target_index, target_name in enumerate(TARGETS):
        separator = "=" * 60
        print()
        print(separator)
        print(f"[{target_index + 1}/{len(TARGETS)}] Processing {target_name}")
        print(separator)

        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            torch.cuda.synchronize()

        estimator = model.estimators_[target_index]
        explainer = tabpfn_shapiq.get_tabpfn_explainer(
            model=estimator,
            data=x_background,
            labels=y_background[:, target_index],
            index="SV",
            max_order=1,
            verbose=False,
        )

        sample_values = []
        base_values = []
        total_samples = len(x_explain)
        print(f"  Calculating {total_samples} explanation samples with budget={SHAPIQ_BUDGET}...")

        for sample_number, sample in enumerate(x_explain, start=1):
            interaction_values = explainer.explain(sample, budget=SHAPIQ_BUDGET)

            if hasattr(interaction_values, "baseline_value"):
                base_value = interaction_values.baseline_value
            else:
                values_map = getattr(interaction_values, "values", {})
                base_value = values_map.get(tuple(), 0.0) if hasattr(values_map, "get") else 0.0

            row_values = [
                interaction_values[(feature_index,)]
                for feature_index in range(len(feature_names))
            ]
            sample_values.append(row_values)
            base_values.append(base_value)

            if sample_number % 10 == 0 or sample_number == total_samples:
                print(f"    Sample {sample_number}/{total_samples} complete")

        shap_values = shap.Explanation(
            values=np.array(sample_values),
            base_values=np.array(base_values),
            data=x_explain,
            feature_names=feature_names,
        )
        shap_values_list.append(shap_values)

        del explainer
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            torch.cuda.synchronize()

        print(f"  Finished {target_name}.")

    separator = "=" * 60
    print()
    print(separator)
    print(f"All {len(TARGETS)} targets completed.")
    print(separator)
    print()
    return shap_values_list, x_explain, row_indices



def export_explanation_tables(shap_values_list, x_explain, x_explain_original, row_indices, feature_names):
    export_dir = SAVE_DIR / "shap_exports"
    export_dir.mkdir(parents=True, exist_ok=True)

    x_explain_df_std = pd.DataFrame(x_explain, columns=feature_names)
    x_explain_df_orig = pd.DataFrame(x_explain_original, columns=feature_names)
    x_explain_df_std.insert(0, "row_index", row_indices)
    x_explain_df_orig.insert(0, "row_index", row_indices)
    x_explain_df_std.to_csv(export_dir / "X_explain_standardized.csv", index=False, encoding="utf-8-sig")
    x_explain_df_orig.to_csv(export_dir / "X_explain_original_units.csv", index=False, encoding="utf-8-sig")

    for target_name, shap_values in zip(TARGETS, shap_values_list):
        shap_frame = pd.DataFrame(shap_values.values, columns=feature_names)
        shap_frame.insert(0, "row_index", row_indices)
        shap_frame.to_csv(export_dir / f"shap_values_{target_name}.csv", index=False, encoding="utf-8-sig")

        base_values = np.atleast_1d(shap_values.base_values)
        pd.DataFrame({"base_value": base_values}).to_csv(
            export_dir / f"shap_base_values_{target_name}.csv",
            index=False,
            encoding="utf-8-sig",
        )

    return export_dir



def build_shap_range_table(shap_values, x_original, feature_names, target_name, n_bins=10):
    tables = []
    for feature_index, feature_name in enumerate(feature_names):
        values = x_original[:, feature_index]
        shap_column = shap_values[:, feature_index]
        try:
            bins = pd.qcut(values, q=n_bins, duplicates="drop")
        except ValueError:
            bins = pd.cut(values, bins=n_bins, include_lowest=True)

        frame = pd.DataFrame({"bin": bins, "value": values, "shap": shap_column}).dropna()
        summary = (
            frame.groupby("bin", observed=False)
            .agg(
                n=("shap", "size"),
                value_min=("value", "min"),
                value_max=("value", "max"),
                value_mean=("value", "mean"),
                shap_mean=("shap", "mean"),
                shap_median=("shap", "median"),
            )
            .reset_index()
        )
        summary["effect"] = np.where(summary["shap_mean"] > 0, "positive", "negative")
        summary.insert(0, "target", target_name)
        summary.insert(1, "feature", feature_name)
        tables.append(summary)

    return pd.concat(tables, ignore_index=True)



def select_dependence_features(shap_values, feature_names):
    top_feature_indices = np.argsort(np.abs(shap_values.values).mean(axis=0))[::-1][:3]
    candidates = ["Thermal shock", "Low carbon"] + [feature_names[index] for index in top_feature_indices]

    ordered = []
    seen = set()
    for feature_name in candidates:
        if feature_name in seen:
            continue
        seen.add(feature_name)
        ordered.append(feature_name)
    return ordered



def save_shap_figures(shap_values_list, x_explain, x_explain_original, feature_names):
    if COLOR_BY_FEATURE is not None and COLOR_BY_FEATURE in feature_names:
        interaction_index = feature_names.index(COLOR_BY_FEATURE)
    else:
        interaction_index = "auto"

    for target_name, shap_values in zip(TARGETS, shap_values_list):
        print(f"Generating SHAP plots for {target_name}...")

        plt.figure(figsize=(10, 6))
        shap.summary_plot(shap_values.values, x_explain_original, feature_names=feature_names, show=False)
        plt.title(f"SHAP Summary Plot for {target_name} (original units)")
        plt.tight_layout()
        plt.savefig(SAVE_DIR / f"SHAP_Summary_{target_name}.pdf", format="pdf", bbox_inches="tight")
        plt.close()

        plt.figure(figsize=(10, 6))
        shap.summary_plot(
            shap_values.values,
            x_explain_original,
            feature_names=feature_names,
            plot_type="bar",
            show=False,
        )
        plt.title(f"SHAP Summary Plot (Bar) for {target_name}")
        plt.tight_layout()
        plt.savefig(SAVE_DIR / f"SHAP_Summary_Bar_{target_name}.pdf", format="pdf", bbox_inches="tight")
        plt.close()

        for feature_name in select_dependence_features(shap_values, feature_names):
            if feature_name not in feature_names:
                print(f"  Skip dependence plot for {feature_name}: feature not found.")
                continue

            plt.figure(figsize=(8, 6))
            shap.dependence_plot(
                feature_names.index(feature_name),
                shap_values.values,
                x_explain_original,
                feature_names=feature_names,
                interaction_index=interaction_index,
                show=False,
            )
            if interaction_index != "auto":
                try:
                    plt.gcf().axes[-1].set_ylabel(COLOR_BY_FEATURE)
                except Exception:
                    pass
            plt.title(f"SHAP Dependence Plot for {target_name} - {feature_name} (original units)")
            plt.tight_layout()
            plt.savefig(SAVE_DIR / f"SHAP_Dependency_{target_name}_{feature_name}.pdf", format="pdf", bbox_inches="tight")
            plt.close()

        try:
            sample_index = 0
            base_value = (
                shap_values.base_values[sample_index]
                if hasattr(shap_values.base_values, "__len__")
                else shap_values.base_values
            )
            x_row_original = pd.Series(x_explain_original[sample_index], index=feature_names)

            plt.figure(figsize=(10, 6))
            shap.force_plot(
                base_value,
                shap_values.values[sample_index, :],
                x_row_original,
                feature_names=feature_names,
                matplotlib=True,
                show=False,
            )
            plt.title(f"SHAP Force Plot for {target_name} (original units)")
            plt.tight_layout()
            plt.savefig(SAVE_DIR / f"SHAP_Force_{target_name}.pdf", format="pdf", bbox_inches="tight")
            plt.close()

            explanation = shap.Explanation(
                values=shap_values.values[sample_index, :],
                base_values=base_value,
                data=x_row_original.values,
                feature_names=feature_names,
            )
            plt.figure(figsize=(8, 6))
            shap.plots.waterfall(explanation, show=False)
            plt.title(f"SHAP Waterfall Plot for {target_name} (original units)")
            plt.tight_layout()
            plt.savefig(SAVE_DIR / f"SHAP_Waterfall_{target_name}.pdf", format="pdf", bbox_inches="tight")
            plt.close()
        except Exception as exc:
            print(f"  Force/Waterfall export failed for {target_name}: {exc}")



def main():
    warnings.filterwarnings("ignore")
    plt.rcParams["font.family"] = "Times New Roman"
    plt.rcParams["axes.unicode_minus"] = False

    seed_everything(SEED)
    SAVE_DIR.mkdir(parents=True, exist_ok=True)

    x_frame, y_values, feature_names = load_dataset()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    metrics_df, summary_df = run_cross_validation(x_frame, y_values, device)
    save_cv_outputs(metrics_df, summary_df)

    print("shap version:", shap.__version__)
    print("tabpfn version:", version("tabpfn"))
    print("Device:", device)

    model, scaler, x_standardized = fit_final_model(x_frame, y_values, device)
    shap_values_list, x_explain, row_indices = compute_shap_values(
        model,
        x_standardized,
        y_values,
        feature_names,
    )

    x_explain_original = scaler.inverse_transform(x_explain)
    export_dir = export_explanation_tables(
        shap_values_list,
        x_explain,
        x_explain_original,
        row_indices,
        feature_names,
    )

    if "WE" in TARGETS:
        we_index = TARGETS.index("WE")
        we_table = build_shap_range_table(
            shap_values_list[we_index].values,
            x_explain_original,
            feature_names,
            target_name="WE",
            n_bins=10,
        )
        we_table.to_csv(export_dir / "WE_shap_value_ranges.csv", index=False, encoding="utf-8-sig")

    save_shap_figures(shap_values_list, x_explain, x_explain_original, feature_names)

    print("All done. Exports saved to:", SAVE_DIR.resolve())
    print("SHAP raw exports saved to:", export_dir.resolve())


if __name__ == "__main__":
    main()
