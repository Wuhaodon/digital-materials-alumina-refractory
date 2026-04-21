from __future__ import annotations

import json
import math
import os
import warnings
from pathlib import Path
from typing import Dict, List, Tuple

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from sklearn.impute import SimpleImputer
from sklearn.metrics import (
    accuracy_score,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.model_selection import StratifiedKFold, StratifiedShuffleSplit
from sklearn.preprocessing import LabelEncoder, StandardScaler

from _selected_feature_io import resolve_default_modeling_path
from tabpfn import TabPFNClassifier
from xgboost import XGBClassifier


PROJECT_ROOT = Path(__file__).resolve().parents[2]
OUT_DIR = PROJECT_ROOT / "results" / "supplementary_material" / "small_sample_tabpfn_xgboost"
OUT_DIR.mkdir(parents=True, exist_ok=True)
LOCAL_TABPFN_CACHE = PROJECT_ROOT / "external" / "tabpfn_cache"
LOCAL_TABPFN_CACHE.mkdir(parents=True, exist_ok=True)
LOCAL_TABPFN_MODEL = LOCAL_TABPFN_CACHE / "tabpfn-v2-classifier.ckpt"

DATA_FILE = resolve_default_modeling_path("table6_legacy.xlsx")
ORIGINAL_RESULT_CSV = PROJECT_ROOT / "backup" / "release_prep_20260413" / "results" / "table6_benchmark_public_trim" / "table6_cv5_all_features.csv"

RANDOM_STATE = 42
REPEATED_SPLITS = 30
LEARNING_SPLITS = 20
TEST_SIZE = 0.30
LEARNING_FRACTIONS = [0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 1.0]

METRICS = ["Accuracy", "Precision", "Recall", "F1-score", "ROC-AUC"]
PLOT_METRICS = ["Accuracy", "F1-score", "ROC-AUC"]

sns.set_theme(style="whitegrid")
plt.rcParams["font.family"] = "Times New Roman"
plt.rcParams["axes.unicode_minus"] = False

os.environ["TABPFN_MODEL_CACHE_DIR"] = str(LOCAL_TABPFN_CACHE)
warnings.filterwarnings("ignore")


def create_group_from_flags(df: pd.DataFrame) -> pd.Series:
    mapping = {
        "00": "F2-TP",
        "01": "E1-TP",
        "10": "F2-RE",
        "11": "E1-RE",
    }
    codes = (
        df["热震"].astype(int).astype(str)
        + df["低碳"].astype(int).astype(str)
    )
    return codes.map(mapping)


def load_classification_dataset() -> Tuple[pd.DataFrame, np.ndarray, LabelEncoder, List[str]]:
    df = pd.read_excel(DATA_FILE).copy()

    if "group" not in df.columns:
        df["group"] = create_group_from_flags(df)

    drop_cols = {"group", "group_str", "热震", "低碳"}
    feature_cols = [c for c in df.columns if c not in drop_cols]
    feature_df = df[feature_cols].copy()

    # Keep only numeric columns to match the ML workflow.
    numeric_cols = []
    for col in feature_df.columns:
        try:
            feature_df[col] = pd.to_numeric(feature_df[col], errors="coerce")
            numeric_cols.append(col)
        except Exception:
            continue
    feature_df = feature_df[numeric_cols]

    label_encoder = LabelEncoder()
    y = label_encoder.fit_transform(df["group"])
    return feature_df, y, label_encoder, numeric_cols


def make_xgb(random_state: int) -> XGBClassifier:
    return XGBClassifier(
        random_state=random_state,
        eval_metric="mlogloss",
        use_label_encoder=False,
    )


def make_tabpfn(random_state: int) -> TabPFNClassifier:
    try:
        return TabPFNClassifier(
            device="cpu",
            random_state=random_state,
            model_path=LOCAL_TABPFN_MODEL,
        )
    except TypeError:
        return TabPFNClassifier(device="cpu", model_path=LOCAL_TABPFN_MODEL)


def preprocess_train_test(
    X_train: pd.DataFrame, X_test: pd.DataFrame
) -> Tuple[np.ndarray, np.ndarray]:
    imputer = SimpleImputer(strategy="mean")
    scaler = StandardScaler()
    X_train_imp = imputer.fit_transform(X_train)
    X_test_imp = imputer.transform(X_test)
    X_train_scaled = scaler.fit_transform(X_train_imp)
    X_test_scaled = scaler.transform(X_test_imp)
    return X_train_scaled, X_test_scaled


def evaluate_predictions(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    y_prob: np.ndarray,
) -> Dict[str, float]:
    result = {
        "Accuracy": accuracy_score(y_true, y_pred),
        "Precision": precision_score(y_true, y_pred, average="weighted", zero_division=0),
        "Recall": recall_score(y_true, y_pred, average="weighted", zero_division=0),
        "F1-score": f1_score(y_true, y_pred, average="weighted", zero_division=0),
    }
    try:
        result["ROC-AUC"] = roc_auc_score(
            y_true,
            y_prob,
            multi_class="ovr",
            average="weighted",
        )
    except Exception:
        result["ROC-AUC"] = np.nan
    return result


def evaluate_model_on_split(
    model_name: str,
    X_train: pd.DataFrame,
    X_test: pd.DataFrame,
    y_train: np.ndarray,
    y_test: np.ndarray,
    random_state: int,
) -> Dict[str, float]:
    X_train_scaled, X_test_scaled = preprocess_train_test(X_train, X_test)

    if model_name == "XGBoost":
        model = make_xgb(random_state)
    elif model_name == "TabPFN":
        model = make_tabpfn(random_state)
    else:
        raise ValueError(f"Unsupported model: {model_name}")

    model.fit(X_train_scaled, y_train)
    y_pred = model.predict(X_test_scaled)
    y_prob = model.predict_proba(X_test_scaled)
    return evaluate_predictions(y_test, y_pred, y_prob)


def summarize_metrics(df: pd.DataFrame, group_cols: List[str]) -> pd.DataFrame:
    records: List[Dict[str, float]] = []
    grouped = df.groupby(group_cols, sort=False)
    for key, grp in grouped:
        if not isinstance(key, tuple):
            key = (key,)
        row: Dict[str, float] = {}
        for col_name, value in zip(group_cols, key):
            row[col_name] = value
        for metric in METRICS:
            row[f"{metric}_mean"] = grp[metric].mean()
            row[f"{metric}_std"] = grp[metric].std(ddof=1)
        row["n_runs"] = len(grp)
        records.append(row)
    return pd.DataFrame(records)


def run_baseline_5fold(
    X: pd.DataFrame,
    y: np.ndarray,
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=RANDOM_STATE)
    rows: List[Dict[str, float]] = []
    for fold_id, (train_idx, test_idx) in enumerate(skf.split(X, y), start=1):
        X_train = X.iloc[train_idx]
        X_test = X.iloc[test_idx]
        y_train = y[train_idx]
        y_test = y[test_idx]
        for model_name in ["XGBoost", "TabPFN"]:
            metrics = evaluate_model_on_split(
                model_name,
                X_train,
                X_test,
                y_train,
                y_test,
                random_state=RANDOM_STATE + fold_id,
            )
            metrics.update({"Model": model_name, "Fold": fold_id})
            rows.append(metrics)
    df = pd.DataFrame(rows)
    summary = summarize_metrics(df, ["Model"])
    return df, summary


def run_repeated_subsampling(
    X: pd.DataFrame,
    y: np.ndarray,
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    splitter = StratifiedShuffleSplit(
        n_splits=REPEATED_SPLITS,
        test_size=TEST_SIZE,
        random_state=RANDOM_STATE,
    )
    rows: List[Dict[str, float]] = []
    for split_id, (train_idx, test_idx) in enumerate(splitter.split(X, y), start=1):
        X_train = X.iloc[train_idx]
        X_test = X.iloc[test_idx]
        y_train = y[train_idx]
        y_test = y[test_idx]
        for model_name in ["XGBoost", "TabPFN"]:
            metrics = evaluate_model_on_split(
                model_name,
                X_train,
                X_test,
                y_train,
                y_test,
                random_state=RANDOM_STATE + split_id,
            )
            metrics.update({"Model": model_name, "Split": split_id})
            rows.append(metrics)
    df = pd.DataFrame(rows)
    summary = summarize_metrics(df, ["Model"])
    return df, summary


def stratified_subsample_indices(
    y_train: np.ndarray, train_fraction: float, seed: int
) -> np.ndarray:
    if math.isclose(train_fraction, 1.0):
        return np.arange(len(y_train))
    inner_splitter = StratifiedShuffleSplit(
        n_splits=1,
        train_size=train_fraction,
        random_state=seed,
    )
    sub_idx, _ = next(inner_splitter.split(np.zeros(len(y_train)), y_train))
    return sub_idx


def run_learning_curve(
    X: pd.DataFrame,
    y: np.ndarray,
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    outer_splitter = StratifiedShuffleSplit(
        n_splits=LEARNING_SPLITS,
        test_size=TEST_SIZE,
        random_state=RANDOM_STATE,
    )
    rows: List[Dict[str, float]] = []
    for split_id, (train_idx, test_idx) in enumerate(outer_splitter.split(X, y), start=1):
        X_train_pool = X.iloc[train_idx].reset_index(drop=True)
        X_test = X.iloc[test_idx].reset_index(drop=True)
        y_train_pool = y[train_idx]
        y_test = y[test_idx]
        for frac in LEARNING_FRACTIONS:
            sub_idx = stratified_subsample_indices(
                y_train_pool,
                train_fraction=frac,
                seed=RANDOM_STATE + split_id,
            )
            X_train = X_train_pool.iloc[sub_idx]
            y_train = y_train_pool[sub_idx]
            for model_name in ["XGBoost", "TabPFN"]:
                metrics = evaluate_model_on_split(
                    model_name,
                    X_train,
                    X_test,
                    y_train,
                    y_test,
                    random_state=RANDOM_STATE + split_id,
                )
                metrics.update(
                    {
                        "Model": model_name,
                        "Split": split_id,
                        "TrainFraction": frac,
                        "TrainSamples": len(sub_idx),
                    }
                )
                rows.append(metrics)
    df = pd.DataFrame(rows)
    summary = summarize_metrics(df, ["Model", "TrainFraction", "TrainSamples"])
    return df, summary


def plot_repeated_subsampling_boxplots(df: pd.DataFrame) -> None:
    fig, axes = plt.subplots(1, len(PLOT_METRICS), figsize=(15, 4.5), dpi=220)
    for ax, metric in zip(axes, PLOT_METRICS):
        sns.boxplot(
            data=df,
            x="Model",
            y=metric,
            hue="Model",
            dodge=False,
            ax=ax,
            palette="Set2",
        )
        sns.stripplot(
            data=df,
            x="Model",
            y=metric,
            ax=ax,
            color="black",
            size=2.2,
            alpha=0.35,
        )
        ax.set_title(metric)
        ax.set_xlabel("")
        legend = ax.get_legend()
        if legend is not None:
            legend.remove()
    fig.suptitle("Repeated Stratified Random Subsampling", fontsize=13)
    fig.tight_layout()
    fig.savefig(OUT_DIR / "FigS_repeated_subsampling_boxplots.png", bbox_inches="tight")
    plt.close(fig)


def plot_learning_curves(summary_df: pd.DataFrame) -> None:
    fig, axes = plt.subplots(1, len(PLOT_METRICS), figsize=(16, 4.8), dpi=220)
    for ax, metric in zip(axes, PLOT_METRICS):
        for model_name, marker in [("XGBoost", "o"), ("TabPFN", "s")]:
            sub = summary_df[summary_df["Model"] == model_name].sort_values("TrainSamples")
            x = sub["TrainSamples"].to_numpy()
            y = sub[f"{metric}_mean"].to_numpy()
            yerr = sub[f"{metric}_std"].fillna(0).to_numpy()
            ax.plot(x, y, marker=marker, linewidth=2, label=model_name)
            ax.fill_between(x, y - yerr, y + yerr, alpha=0.15)
        ax.set_title(metric)
        ax.set_xlabel("Training samples")
        ax.set_ylabel(metric)
    axes[0].legend(frameon=True)
    fig.suptitle("Learning Curves Under Small-Sample Conditions", fontsize=13)
    fig.tight_layout()
    fig.savefig(OUT_DIR / "FigS_learning_curves_tabpfn_vs_xgboost.png", bbox_inches="tight")
    plt.close(fig)


def build_original_workflow_report(
    X: pd.DataFrame,
    y: np.ndarray,
    label_encoder: LabelEncoder,
    baseline_summary: pd.DataFrame,
) -> str:
    lines = []
    lines.append("Original workflow located for reviewer #5")
    lines.append("")
    lines.append(f"Dataset file: {DATA_FILE}")
    lines.append(f"Total samples: {len(X)}")
    lines.append(f"Total numeric descriptors used as candidate features: {X.shape[1]}")
    lines.append(f"Classes: {list(label_encoder.classes_)}")
    lines.append("Task: predict processing-condition group from micromechanical descriptors.")
    lines.append("Target construction: group is created from the thermal-shock and low-carbon flags.")
    lines.append("Baseline preprocessing reproduced here: mean imputation + StandardScaler.")
    lines.append("Baseline comparison reproduced here: XGBoost vs TabPFN classification.")
    lines.append("")
    lines.append("Reproduced 5-fold summary:")
    lines.append(baseline_summary.to_string(index=False))
    if ORIGINAL_RESULT_CSV.exists():
        original = pd.read_csv(ORIGINAL_RESULT_CSV)
        lines.append("")
        lines.append("Original stored benchmark csv found at:")
        lines.append(str(ORIGINAL_RESULT_CSV))
        lines.append(original.to_string(index=False))
    return "\n".join(lines)


def save_json_manifest(files: List[Path]) -> None:
    payload = {
        "generated_files": [str(p) for p in files],
        "data_file": str(DATA_FILE),
        "original_result_csv": str(ORIGINAL_RESULT_CSV),
        "repeated_splits": REPEATED_SPLITS,
        "learning_splits": LEARNING_SPLITS,
        "learning_fractions": LEARNING_FRACTIONS,
    }
    (OUT_DIR / "manifest.json").write_text(
        json.dumps(payload, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )


def main() -> None:
    X, y, label_encoder, feature_cols = load_classification_dataset()

    baseline_df, baseline_summary = run_baseline_5fold(X, y)
    repeated_df, repeated_summary = run_repeated_subsampling(X, y)
    learning_df, learning_summary = run_learning_curve(X, y)

    baseline_df.to_csv(OUT_DIR / "baseline_5fold_metrics.csv", index=False, encoding="utf-8-sig")
    baseline_summary.to_csv(OUT_DIR / "baseline_5fold_summary.csv", index=False, encoding="utf-8-sig")
    repeated_df.to_csv(OUT_DIR / "repeated_subsampling_metrics.csv", index=False, encoding="utf-8-sig")
    repeated_summary.to_csv(OUT_DIR / "repeated_subsampling_summary.csv", index=False, encoding="utf-8-sig")
    learning_df.to_csv(OUT_DIR / "learning_curve_metrics.csv", index=False, encoding="utf-8-sig")
    learning_summary.to_csv(OUT_DIR / "learning_curve_summary.csv", index=False, encoding="utf-8-sig")

    original_report = build_original_workflow_report(X, y, label_encoder, baseline_summary)
    (OUT_DIR / "original_workflow_report.txt").write_text(original_report, encoding="utf-8")

    feature_report = pd.DataFrame({"Feature": feature_cols})
    feature_report.to_csv(OUT_DIR / "candidate_feature_columns.csv", index=False, encoding="utf-8-sig")

    if ORIGINAL_RESULT_CSV.exists():
        original = pd.read_csv(ORIGINAL_RESULT_CSV)
        compare = baseline_summary.copy()
        compare["Model"] = compare["Model"].astype(str)
        merged = compare.merge(original, on="Model", how="outer", suffixes=("_reproduced", "_original"))
        merged.to_csv(OUT_DIR / "baseline_vs_original_reported_metrics.csv", index=False, encoding="utf-8-sig")

    plot_repeated_subsampling_boxplots(repeated_df)
    plot_learning_curves(learning_summary)

    save_json_manifest(
        [
            OUT_DIR / "baseline_5fold_metrics.csv",
            OUT_DIR / "baseline_5fold_summary.csv",
            OUT_DIR / "repeated_subsampling_metrics.csv",
            OUT_DIR / "repeated_subsampling_summary.csv",
            OUT_DIR / "learning_curve_metrics.csv",
            OUT_DIR / "learning_curve_summary.csv",
            OUT_DIR / "FigS_repeated_subsampling_boxplots.png",
            OUT_DIR / "FigS_learning_curves_tabpfn_vs_xgboost.png",
            OUT_DIR / "original_workflow_report.txt",
            OUT_DIR / "candidate_feature_columns.csv",
        ]
    )

    print("Finished reviewer #5 small-sample comparison.")
    print(f"Output directory: {OUT_DIR}")


if __name__ == "__main__":
    main()
