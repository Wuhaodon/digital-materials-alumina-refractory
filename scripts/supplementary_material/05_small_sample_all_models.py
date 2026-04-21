from __future__ import annotations

import importlib.util
import json
import math
import sys
import warnings
from pathlib import Path
from typing import Dict, Iterable, List

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from sklearn.impute import SimpleImputer
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score, roc_auc_score
from sklearn.model_selection import StratifiedShuffleSplit
from sklearn.preprocessing import StandardScaler

from _selected_feature_io import DEFAULT_FINAL_FEATURES_CSV, resolve_default_modeling_path


PROJECT_ROOT = Path(__file__).resolve().parents[2]
REVISION_DIR = PROJECT_ROOT / "results" / "supplementary_material"
BENCHMARK_OUTPUT_DIR = PROJECT_ROOT / "results" / "table6_benchmark"
OUT_DIR = REVISION_DIR / "small_sample_all_models"
OUT_DIR.mkdir(parents=True, exist_ok=True)

BENCHMARK_SCRIPT = PROJECT_ROOT / "scripts" / "01_feature_selection_table6_benchmark.py"
TABPFN_MODEL = PROJECT_ROOT / "external" / "tabpfn_cache" / "tabpfn-v2-classifier.ckpt"

RANDOM_STATE = 42
REPEATED_SPLITS = 30
LEARNING_SPLITS = 20
TEST_SIZE = 0.30
LEARNING_FRACTIONS = [0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 1.0]
METRICS = ["Accuracy", "Precision", "Recall", "F1-score", "ROC-AUC"]
PLOT_METRICS = ["Accuracy", "F1-score", "ROC-AUC"]
MODEL_ORDER = ["RandomForest", "AdaBoost", "LightGBM", "CatBoost", "XGBoost", "TabPFN"]
PALETTE = dict(zip(MODEL_ORDER, sns.color_palette("Set2", n_colors=len(MODEL_ORDER))))

sns.set_theme(style="whitegrid")
plt.rcParams["font.family"] = "Times New Roman"
plt.rcParams["axes.unicode_minus"] = False
warnings.filterwarnings("ignore")

TITLE_SIZE = 50
SUPTITLE_SIZE = 60
AXIS_LABEL_SIZE = 46
TICK_LABEL_SIZE = 36
LEGEND_SIZE = 30


def load_benchmark_module():
    spec = importlib.util.spec_from_file_location("benchmark_table6_module_all_models", BENCHMARK_SCRIPT)
    if spec is None or spec.loader is None:
        raise ImportError(f"Unable to load benchmark script: {BENCHMARK_SCRIPT}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


benchmark = load_benchmark_module()


def preprocess_train_test(
    x_train_df: pd.DataFrame,
    x_test_df: pd.DataFrame,
) -> tuple[np.ndarray, np.ndarray]:
    imputer = SimpleImputer(strategy="mean")
    scaler = StandardScaler()
    x_train = imputer.fit_transform(x_train_df)
    x_test = imputer.transform(x_test_df)
    x_train = scaler.fit_transform(x_train)
    x_test = scaler.transform(x_test)
    return x_train, x_test


def evaluate_predictions(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    y_prob: np.ndarray | None,
) -> Dict[str, float]:
    row: Dict[str, float] = {
        "Accuracy": accuracy_score(y_true, y_pred),
        "Precision": precision_score(y_true, y_pred, average="weighted", zero_division=0),
        "Recall": recall_score(y_true, y_pred, average="weighted", zero_division=0),
        "F1-score": f1_score(y_true, y_pred, average="weighted", zero_division=0),
        "ROC-AUC": np.nan,
    }
    if y_prob is not None:
        row["ROC-AUC"] = roc_auc_score(y_true, y_prob, multi_class="ovr")
    return row


def summarize_metrics(df: pd.DataFrame, group_cols: List[str]) -> pd.DataFrame:
    records: List[Dict[str, float | int | str]] = []
    grouped = df.groupby(group_cols, sort=False)
    for key, grp in grouped:
        key_tuple = key if isinstance(key, tuple) else (key,)
        row: Dict[str, float | int | str] = {}
        for col_name, value in zip(group_cols, key_tuple):
            row[col_name] = value
        for metric in METRICS:
            row[f"{metric}_mean"] = grp[metric].mean()
            row[f"{metric}_std"] = grp[metric].std(ddof=1)
        row["n_runs"] = len(grp)
        records.append(row)
    out = pd.DataFrame(records)
    if "Model" in out.columns:
        order_map = {name: idx for idx, name in enumerate(MODEL_ORDER)}
        out = out.assign(_order=out["Model"].map(order_map)).sort_values("_order").drop(columns="_order").reset_index(drop=True)
    return out


def get_feature_list(df: pd.DataFrame) -> List[str]:
    return benchmark.infer_all_feature_order(df)


def get_model_factories(config):
    factories = benchmark.make_models(config)
    return {name: factories[name] for name in MODEL_ORDER}


def evaluate_model_split(
    model_name: str,
    factory,
    x_train_df: pd.DataFrame,
    x_test_df: pd.DataFrame,
    y_train: np.ndarray,
    y_test: np.ndarray,
) -> Dict[str, float]:
    x_train, x_test = preprocess_train_test(x_train_df, x_test_df)
    model = factory()
    model.fit(x_train, y_train)
    y_pred = model.predict(x_test)
    y_prob = model.predict_proba(x_test) if hasattr(model, "predict_proba") else None
    return evaluate_predictions(y_test, y_pred, y_prob)


def run_repeated_subsampling(
    df: pd.DataFrame,
    feature_list: List[str],
    config,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    x_df = df[feature_list].copy()
    y, _ = benchmark.encode_target(df)
    model_factories = get_model_factories(config)
    splitter = StratifiedShuffleSplit(
        n_splits=REPEATED_SPLITS,
        test_size=TEST_SIZE,
        random_state=RANDOM_STATE,
    )
    rows: List[Dict[str, float | int | str]] = []

    for split_id, (train_idx, test_idx) in enumerate(splitter.split(x_df, y), start=1):
        x_train_df = x_df.iloc[train_idx]
        x_test_df = x_df.iloc[test_idx]
        y_train = y[train_idx]
        y_test = y[test_idx]

        for model_name, factory in model_factories.items():
            metrics = evaluate_model_split(
                model_name,
                factory,
                x_train_df,
                x_test_df,
                y_train,
                y_test,
            )
            metrics.update(
                {
                    "Model": model_name,
                    "Split": split_id,
                    "TrainSamples": len(train_idx),
                    "TestSamples": len(test_idx),
                }
            )
            rows.append(metrics)

    raw_df = pd.DataFrame(rows)
    summary_df = summarize_metrics(raw_df, ["Model"])
    return raw_df, summary_df


def stratified_subsample_indices(y_train: np.ndarray, train_fraction: float, seed: int) -> np.ndarray:
    if math.isclose(train_fraction, 1.0):
        return np.arange(len(y_train))
    splitter = StratifiedShuffleSplit(
        n_splits=1,
        train_size=train_fraction,
        random_state=seed,
    )
    sub_idx, _ = next(splitter.split(np.zeros(len(y_train)), y_train))
    return sub_idx


def run_learning_curves(
    df: pd.DataFrame,
    feature_list: List[str],
    config,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    x_df = df[feature_list].copy()
    y, _ = benchmark.encode_target(df)
    model_factories = get_model_factories(config)
    outer_splitter = StratifiedShuffleSplit(
        n_splits=LEARNING_SPLITS,
        test_size=TEST_SIZE,
        random_state=RANDOM_STATE,
    )
    rows: List[Dict[str, float | int | str]] = []

    for split_id, (train_idx, test_idx) in enumerate(outer_splitter.split(x_df, y), start=1):
        x_train_pool = x_df.iloc[train_idx].reset_index(drop=True)
        x_test_df = x_df.iloc[test_idx].reset_index(drop=True)
        y_train_pool = y[train_idx]
        y_test = y[test_idx]

        for frac in LEARNING_FRACTIONS:
            sub_idx = stratified_subsample_indices(
                y_train_pool,
                train_fraction=frac,
                seed=RANDOM_STATE + split_id,
            )
            x_train_df = x_train_pool.iloc[sub_idx]
            y_train = y_train_pool[sub_idx]

            for model_name, factory in model_factories.items():
                metrics = evaluate_model_split(
                    model_name,
                    factory,
                    x_train_df,
                    x_test_df,
                    y_train,
                    y_test,
                )
                metrics.update(
                    {
                        "Model": model_name,
                        "Split": split_id,
                        "TrainFraction": frac,
                        "TrainSamples": len(sub_idx),
                        "TestSamples": len(test_idx),
                    }
                )
                rows.append(metrics)

    raw_df = pd.DataFrame(rows)
    summary_df = summarize_metrics(raw_df, ["Model", "TrainFraction", "TrainSamples"])
    return raw_df, summary_df


def load_official_baseline_all_models() -> pd.DataFrame:
    path = BENCHMARK_OUTPUT_DIR / "table6_single_split_all_features.csv"
    return pd.read_csv(path)


def plot_repeated_subsampling_boxplots(raw_df: pd.DataFrame) -> Path:
    fig, axes = plt.subplots(1, len(PLOT_METRICS), figsize=(22.5, 7.4), dpi=280)
    for idx, metric in enumerate(PLOT_METRICS):
        ax = axes[idx]
        sns.boxplot(
            data=raw_df,
            x="Model",
            y=metric,
            order=MODEL_ORDER,
            hue="Model",
            dodge=False,
            palette=PALETTE,
            ax=ax,
        )
        sns.stripplot(
            data=raw_df,
            x="Model",
            y=metric,
            order=MODEL_ORDER,
            color="black",
            size=1.8,
            alpha=0.25,
            ax=ax,
        )
        ax.set_title(metric, fontsize=TITLE_SIZE)
        ax.set_xlabel("")
        ax.set_ylabel(metric, fontsize=AXIS_LABEL_SIZE)
        ax.tick_params(axis="x", rotation=30, labelsize=TICK_LABEL_SIZE)
        ax.tick_params(axis="y", labelsize=TICK_LABEL_SIZE)
        legend = ax.get_legend()
        if legend is not None:
            legend.remove()
    fig.suptitle("Repeated Stratified Resampling Benchmark", fontsize=SUPTITLE_SIZE)
    fig.tight_layout()
    out_path = OUT_DIR / "FigS_all_models_repeated_resampling_boxplots.png"
    fig.savefig(out_path, bbox_inches="tight")
    plt.close(fig)
    return out_path


def plot_learning_curves(summary_df: pd.DataFrame) -> Path:
    fig, axes = plt.subplots(1, len(PLOT_METRICS), figsize=(22.5, 7.4), dpi=280)
    for idx, metric in enumerate(PLOT_METRICS):
        ax = axes[idx]
        for model_name in MODEL_ORDER:
            model_df = summary_df[summary_df["Model"] == model_name].sort_values("TrainSamples")
            x = model_df["TrainSamples"].to_numpy()
            y = model_df[f"{metric}_mean"].to_numpy()
            yerr = model_df[f"{metric}_std"].fillna(0).to_numpy()
            ax.plot(x, y, marker="o", linewidth=2.8, markersize=7, label=model_name, color=PALETTE[model_name])
            ax.fill_between(x, y - yerr, y + yerr, alpha=0.10, color=PALETTE[model_name])
        ax.set_title(metric, fontsize=TITLE_SIZE)
        ax.set_xlabel("Training samples", fontsize=AXIS_LABEL_SIZE)
        ax.set_ylabel(metric, fontsize=AXIS_LABEL_SIZE)
        ax.tick_params(axis="x", labelsize=TICK_LABEL_SIZE)
        ax.tick_params(axis="y", labelsize=TICK_LABEL_SIZE)
    axes[0].legend(frameon=True, fontsize=LEGEND_SIZE)
    fig.suptitle("Learning Curves of the Benchmark Models", fontsize=SUPTITLE_SIZE)
    fig.tight_layout()
    out_path = OUT_DIR / "FigS_all_models_learning_curves.png"
    fig.savefig(out_path, bbox_inches="tight")
    plt.close(fig)
    return out_path


def build_manuscript_ready_summary(
    official_baseline: pd.DataFrame,
    repeated_summary: pd.DataFrame,
    learning_summary: pd.DataFrame,
) -> pd.DataFrame:
    rows: List[Dict[str, str]] = []

    for _, row in official_baseline.iterrows():
        rows.append(
            {
                "Section": "official_single_split_baseline",
                "Model": str(row["Model"]),
                "KeyMetrics": ", ".join(
                    [
                        f"Accuracy={row['Accuracy']:.4f}",
                        f"F1={row['F1-score']:.4f}",
                        f"ROC-AUC={row['ROC-AUC']:.4f}",
                    ]
                ),
            }
        )

    for _, row in repeated_summary.iterrows():
        rows.append(
            {
                "Section": "repeated_resampling",
                "Model": str(row["Model"]),
                "KeyMetrics": ", ".join(
                    [
                        f"Accuracy={row['Accuracy_mean']:.4f}±{row['Accuracy_std']:.4f}",
                        f"F1={row['F1-score_mean']:.4f}±{row['F1-score_std']:.4f}",
                        f"ROC-AUC={row['ROC-AUC_mean']:.4f}±{row['ROC-AUC_std']:.4f}",
                    ]
                ),
            }
        )

    for model_name in MODEL_ORDER:
        sub = learning_summary[learning_summary["Model"] == model_name].sort_values("TrainSamples")
        first_row = sub.iloc[0]
        last_row = sub.iloc[-1]
        rows.append(
            {
                "Section": "learning_curve",
                "Model": model_name,
                "KeyMetrics": ", ".join(
                    [
                        f"Accuracy {first_row['Accuracy_mean']:.4f}->{last_row['Accuracy_mean']:.4f}",
                        f"F1 {first_row['F1-score_mean']:.4f}->{last_row['F1-score_mean']:.4f}",
                        f"ROC-AUC {first_row['ROC-AUC_mean']:.4f}->{last_row['ROC-AUC_mean']:.4f}",
                    ]
                ),
            }
        )
    return pd.DataFrame(rows)


def write_manifest(files: Iterable[Path], feature_list: List[str], data_path: Path) -> None:
    payload = {
        "benchmark_script": str(BENCHMARK_SCRIPT),
        "data_path": str(data_path),
        "tabpfn_model": str(TABPFN_MODEL),
        "repeated_splits": REPEATED_SPLITS,
        "learning_splits": LEARNING_SPLITS,
        "learning_fractions": LEARNING_FRACTIONS,
        "feature_list": feature_list,
        "generated_files": [str(path) for path in files],
    }
    (OUT_DIR / "manifest.json").write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")


def main() -> None:
    config = benchmark.BenchmarkConfig(
        data_path=resolve_default_modeling_path("table6_legacy.xlsx"),
        output_dir=BENCHMARK_OUTPUT_DIR,
        tabpfn_model_path=TABPFN_MODEL,
        tabpfn_device="cpu",
        random_state=RANDOM_STATE,
        test_size=TEST_SIZE,
    )

    df = benchmark.load_dataset(config.data_path)
    feature_list = get_feature_list(df)

    official_baseline = load_official_baseline_all_models()
    repeated_raw, repeated_summary = run_repeated_subsampling(df, feature_list, config)
    learning_raw, learning_summary = run_learning_curves(df, feature_list, config)
    manuscript_summary = build_manuscript_ready_summary(
        official_baseline,
        repeated_summary,
        learning_summary,
    )

    repeated_plot = plot_repeated_subsampling_boxplots(repeated_raw)
    learning_plot = plot_learning_curves(learning_summary)

    files_to_write = {
        "official_baseline_all_models.csv": official_baseline,
        "repeated_resampling_metrics.csv": repeated_raw,
        "repeated_resampling_summary.csv": repeated_summary,
        "learning_curve_metrics.csv": learning_raw,
        "learning_curve_summary.csv": learning_summary,
        "manuscript_ready_summary.csv": manuscript_summary,
    }

    written_files: List[Path] = []
    for filename, table in files_to_write.items():
        out_path = OUT_DIR / filename
        table.to_csv(out_path, index=False, encoding="utf-8-sig")
        written_files.append(out_path)

    written_files.extend([repeated_plot, learning_plot])
    write_manifest(written_files, feature_list, config.data_path)

    print("Reviewer #5 all-model extension finished.")
    print(f"Data: {config.data_path}")
    print(f"Output directory: {OUT_DIR}")
    print(f"Feature list: {feature_list}")


if __name__ == "__main__":
    main()
