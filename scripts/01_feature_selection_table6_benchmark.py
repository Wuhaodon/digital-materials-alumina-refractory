from __future__ import annotations

import argparse
from dataclasses import dataclass
from pathlib import Path
from typing import Callable

import numpy as np
import pandas as pd
from catboost import CatBoostClassifier
from lightgbm import LGBMClassifier
from sklearn.ensemble import AdaBoostClassifier, RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score, roc_auc_score
from sklearn.model_selection import StratifiedKFold, train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler
from tabpfn import TabPFNClassifier
from xgboost import XGBClassifier

from _selected_feature_io import (
    ALL_FEATURES,
    DEFAULT_FINAL_FEATURES_CSV,
    DEFAULT_TABPFN_MODEL_PATH,
    GROUP_COL,
    RANDOM_STATE,
    TEST_SIZE,
    infer_all_feature_order,
    load_dataset,
    read_selected_features,
    resolve_default_modeling_path,
)

PROJECT_ROOT = Path(__file__).resolve().parents[1]
LEGACY_TABLE6_DATA_PATH = resolve_default_modeling_path('table6_legacy.xlsx')
DEFAULT_DATA_PATH = LEGACY_TABLE6_DATA_PATH
DEFAULT_OUTPUT_DIR = PROJECT_ROOT / 'results' / 'table6_benchmark'

MODEL_ORDER = [
    'RandomForest',
    'AdaBoost',
    'LightGBM',
    'CatBoost',
    'XGBoost',
    'TabPFN',
]


@dataclass(frozen=True)
class BenchmarkConfig:
    data_path: Path
    output_dir: Path
    tabpfn_model_path: Path
    selected_features_csv: Path
    tabpfn_device: str = 'cpu'
    random_state: int = RANDOM_STATE
    test_size: float = TEST_SIZE


def parse_args() -> BenchmarkConfig:
    parser = argparse.ArgumentParser(
        description=(
            'Run the 6-model classification benchmark on all 13 features and on the '
            'feature subset exported by scripts/01_feature_selection_dual_model_rfe.py.'
        )
    )
    parser.add_argument(
        '--data',
        type=Path,
        default=DEFAULT_DATA_PATH,
        help='Path to the classification dataset.',
    )
    parser.add_argument(
        '--outdir',
        type=Path,
        default=DEFAULT_OUTPUT_DIR,
        help='Directory for CSV outputs.',
    )
    parser.add_argument(
        '--tabpfn-model',
        type=Path,
        default=DEFAULT_TABPFN_MODEL_PATH,
        help='Path to the local TabPFN checkpoint file.',
    )
    parser.add_argument(
        '--selected-features-csv',
        type=Path,
        default=DEFAULT_FINAL_FEATURES_CSV,
        help='Feature-selection artifact exported by the dual-model RFE script.',
    )
    parser.add_argument(
        '--tabpfn-device',
        default='cpu',
        choices=['cpu', 'cuda'],
        help='TabPFN device. Default is cpu because the local environment is CPU-only.',
    )
    args = parser.parse_args()
    return BenchmarkConfig(
        data_path=args.data,
        output_dir=args.outdir,
        tabpfn_model_path=args.tabpfn_model,
        selected_features_csv=args.selected_features_csv,
        tabpfn_device=args.tabpfn_device,
    )


def make_models(config: BenchmarkConfig) -> dict[str, Callable[[], object]]:
    def make_tabpfn() -> TabPFNClassifier:
        if not config.tabpfn_model_path.exists():
            raise FileNotFoundError(
                'TabPFN checkpoint not found. Expected at '
                f'{config.tabpfn_model_path}'
            )
        return TabPFNClassifier(
            device=config.tabpfn_device,
            model_path=config.tabpfn_model_path,
            random_state=config.random_state,
            n_jobs=1,
        )

    return {
        'RandomForest': lambda: RandomForestClassifier(random_state=config.random_state),
        'AdaBoost': lambda: AdaBoostClassifier(random_state=config.random_state, algorithm='SAMME'),
        'LightGBM': lambda: LGBMClassifier(random_state=config.random_state, verbose=-1),
        'CatBoost': lambda: CatBoostClassifier(verbose=0, random_state=config.random_state),
        'XGBoost': lambda: XGBClassifier(
            use_label_encoder=False,
            eval_metric='logloss',
            random_state=config.random_state,
        ),
        'TabPFN': make_tabpfn,
    }


def encode_target(df: pd.DataFrame) -> tuple[np.ndarray, LabelEncoder]:
    encoder = LabelEncoder()
    return encoder.fit_transform(df[GROUP_COL]), encoder


def evaluate_predictions(
    model_name: str,
    y_true: np.ndarray,
    y_pred: np.ndarray,
    y_prob: np.ndarray | None,
) -> dict[str, float | str]:
    row: dict[str, float | str] = {
        'Model': model_name,
        'Accuracy': accuracy_score(y_true, y_pred),
        'Precision': precision_score(y_true, y_pred, average='weighted', zero_division=0),
        'Recall': recall_score(y_true, y_pred, average='weighted', zero_division=0),
        'F1-score': f1_score(y_true, y_pred, average='weighted', zero_division=0),
        'ROC-AUC': np.nan,
    }
    if y_prob is not None:
        row['ROC-AUC'] = roc_auc_score(y_true, y_prob, multi_class='ovr')
    return row


def reorder_models(df: pd.DataFrame) -> pd.DataFrame:
    order_index = {name: idx for idx, name in enumerate(MODEL_ORDER)}
    return (
        df.assign(_order=df['Model'].map(order_index))
        .sort_values('_order')
        .drop(columns='_order')
        .reset_index(drop=True)
    )


def run_single_split(
    df: pd.DataFrame,
    feature_list: list[str],
    config: BenchmarkConfig,
) -> pd.DataFrame:
    y, _ = encode_target(df)
    x = df[feature_list].copy()

    imputer = SimpleImputer(strategy='mean')
    x = imputer.fit_transform(x)
    scaler = StandardScaler()
    x = scaler.fit_transform(x)

    x_train, x_test, y_train, y_test = train_test_split(
        x,
        y,
        stratify=y,
        random_state=config.random_state,
        test_size=config.test_size,
    )

    rows: list[dict[str, float | str]] = []
    for model_name, factory in make_models(config).items():
        model = factory()
        model.fit(x_train, y_train)
        y_pred = model.predict(x_test)
        y_prob = model.predict_proba(x_test) if hasattr(model, 'predict_proba') else None
        rows.append(evaluate_predictions(model_name, y_test, y_pred, y_prob))

    return reorder_models(pd.DataFrame(rows))


def run_stratified_cv5(
    df: pd.DataFrame,
    feature_list: list[str],
    config: BenchmarkConfig,
) -> pd.DataFrame:
    x_df = df[feature_list].copy()
    y, _ = encode_target(df)
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=config.random_state)

    rows: list[dict[str, float | str]] = []
    for model_name, factory in make_models(config).items():
        accs: list[float] = []
        pres: list[float] = []
        recs: list[float] = []
        f1s: list[float] = []
        aucs: list[float] = []

        for train_idx, test_idx in cv.split(x_df, y):
            x_train_df = x_df.iloc[train_idx]
            x_test_df = x_df.iloc[test_idx]
            y_train = y[train_idx]
            y_test = y[test_idx]

            imputer = SimpleImputer(strategy='mean')
            x_train = imputer.fit_transform(x_train_df)
            x_test = imputer.transform(x_test_df)
            scaler = StandardScaler()
            x_train = scaler.fit_transform(x_train)
            x_test = scaler.transform(x_test)

            model = factory()
            model.fit(x_train, y_train)
            y_pred = model.predict(x_test)
            y_prob = model.predict_proba(x_test) if hasattr(model, 'predict_proba') else None

            accs.append(accuracy_score(y_test, y_pred))
            pres.append(precision_score(y_test, y_pred, average='weighted', zero_division=0))
            recs.append(recall_score(y_test, y_pred, average='weighted', zero_division=0))
            f1s.append(f1_score(y_test, y_pred, average='weighted', zero_division=0))
            if y_prob is not None:
                aucs.append(roc_auc_score(y_test, y_prob, multi_class='ovr'))

        rows.append(
            {
                'Model': model_name,
                'Accuracy_mean': np.mean(accs),
                'Accuracy_std': np.std(accs),
                'Precision_mean': np.mean(pres),
                'Precision_std': np.std(pres),
                'Recall_mean': np.mean(recs),
                'Recall_std': np.std(recs),
                'F1-score_mean': np.mean(f1s),
                'F1-score_std': np.std(f1s),
                'ROC-AUC_mean': np.mean(aucs) if aucs else np.nan,
                'ROC-AUC_std': np.std(aucs) if aucs else np.nan,
            }
        )

    return reorder_models(pd.DataFrame(rows))


def build_tabpfn_selected_vs_baseline_summary(
    single_all: pd.DataFrame,
    cv5_all: pd.DataFrame,
    single_selected: pd.DataFrame,
    cv5_selected: pd.DataFrame,
) -> pd.DataFrame:
    base_single = single_all.loc[single_all['Model'] == 'TabPFN'].iloc[0]
    selected_single = single_selected.loc[single_selected['Model'] == 'TabPFN'].iloc[0]
    base_cv5 = cv5_all.loc[cv5_all['Model'] == 'TabPFN'].iloc[0]
    selected_cv5 = cv5_selected.loc[cv5_selected['Model'] == 'TabPFN'].iloc[0]

    rows = [
        {
            'setting': 'single_split',
            'feature_set': 'all_13',
            'Accuracy': float(base_single['Accuracy']),
            'Precision': float(base_single['Precision']),
            'Recall': float(base_single['Recall']),
            'F1-score': float(base_single['F1-score']),
            'ROC-AUC': float(base_single['ROC-AUC']),
        },
        {
            'setting': 'single_split',
            'feature_set': 'selected_features',
            'Accuracy': float(selected_single['Accuracy']),
            'Precision': float(selected_single['Precision']),
            'Recall': float(selected_single['Recall']),
            'F1-score': float(selected_single['F1-score']),
            'ROC-AUC': float(selected_single['ROC-AUC']),
        },
        {
            'setting': 'single_split',
            'feature_set': 'delta_selected_minus_all',
            'Accuracy': float(selected_single['Accuracy']) - float(base_single['Accuracy']),
            'Precision': float(selected_single['Precision']) - float(base_single['Precision']),
            'Recall': float(selected_single['Recall']) - float(base_single['Recall']),
            'F1-score': float(selected_single['F1-score']) - float(base_single['F1-score']),
            'ROC-AUC': float(selected_single['ROC-AUC']) - float(base_single['ROC-AUC']),
        },
        {
            'setting': 'cv5_mean',
            'feature_set': 'all_13',
            'Accuracy': float(base_cv5['Accuracy_mean']),
            'Precision': float(base_cv5['Precision_mean']),
            'Recall': float(base_cv5['Recall_mean']),
            'F1-score': float(base_cv5['F1-score_mean']),
            'ROC-AUC': float(base_cv5['ROC-AUC_mean']),
        },
        {
            'setting': 'cv5_mean',
            'feature_set': 'selected_features',
            'Accuracy': float(selected_cv5['Accuracy_mean']),
            'Precision': float(selected_cv5['Precision_mean']),
            'Recall': float(selected_cv5['Recall_mean']),
            'F1-score': float(selected_cv5['F1-score_mean']),
            'ROC-AUC': float(selected_cv5['ROC-AUC_mean']),
        },
        {
            'setting': 'cv5_mean',
            'feature_set': 'delta_selected_minus_all',
            'Accuracy': float(selected_cv5['Accuracy_mean']) - float(base_cv5['Accuracy_mean']),
            'Precision': float(selected_cv5['Precision_mean']) - float(base_cv5['Precision_mean']),
            'Recall': float(selected_cv5['Recall_mean']) - float(base_cv5['Recall_mean']),
            'F1-score': float(selected_cv5['F1-score_mean']) - float(base_cv5['F1-score_mean']),
            'ROC-AUC': float(selected_cv5['ROC-AUC_mean']) - float(base_cv5['ROC-AUC_mean']),
        },
    ]
    return pd.DataFrame(rows)


def write_outputs(
    output_dir: Path,
    selected_features: list[str],
    selected_features_csv: Path,
    single_all: pd.DataFrame,
    cv5_all: pd.DataFrame,
    single_selected: pd.DataFrame,
    cv5_selected: pd.DataFrame,
) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    single_all.to_csv(output_dir / 'table6_single_split_all_features.csv', index=False, encoding='utf-8-sig')
    cv5_all.to_csv(output_dir / 'table6_cv5_all_features.csv', index=False, encoding='utf-8-sig')
    single_selected.to_csv(output_dir / 'selected_features_single_split.csv', index=False, encoding='utf-8-sig')
    cv5_selected.to_csv(output_dir / 'selected_features_cv5.csv', index=False, encoding='utf-8-sig')

    legacy_prefix = f'selected{len(selected_features)}'
    single_selected.to_csv(output_dir / f'{legacy_prefix}_single_split.csv', index=False, encoding='utf-8-sig')
    cv5_selected.to_csv(output_dir / f'{legacy_prefix}_cv5.csv', index=False, encoding='utf-8-sig')

    build_tabpfn_selected_vs_baseline_summary(
        single_all,
        cv5_all,
        single_selected,
        cv5_selected,
    ).to_csv(
        output_dir / 'tabpfn_selected_vs_baseline_summary.csv',
        index=False,
        encoding='utf-8-sig',
    )

    pd.DataFrame({'feature': selected_features}).to_csv(
        output_dir / 'selected_features_used.csv',
        index=False,
        encoding='utf-8-sig',
    )

    (output_dir / 'selected_features_source.txt').write_text(
        str(selected_features_csv),
        encoding='utf-8',
    )


def print_section(title: str, df: pd.DataFrame) -> None:
    print(f'\n=== {title} ===')
    print(df.to_string(index=False))


def print_selected_tabpfn_summary(cv5_all: pd.DataFrame, cv5_selected: pd.DataFrame) -> None:
    base_row = cv5_all.loc[cv5_all['Model'] == 'TabPFN'].iloc[0]
    selected_row = cv5_selected.loc[cv5_selected['Model'] == 'TabPFN'].iloc[0]
    print('\n=== Selected-feature TabPFN CV5 summary ===')
    print(
        'All 13: '
        f"Accuracy {base_row['Accuracy_mean']:.4f} +- {base_row['Accuracy_std']:.4f}, "
        f"Precision {base_row['Precision_mean']:.4f} +- {base_row['Precision_std']:.4f}, "
        f"Recall {base_row['Recall_mean']:.4f} +- {base_row['Recall_std']:.4f}, "
        f"F1 {base_row['F1-score_mean']:.4f} +- {base_row['F1-score_std']:.4f}, "
        f"ROC-AUC {base_row['ROC-AUC_mean']:.4f} +- {base_row['ROC-AUC_std']:.4f}"
    )
    print(
        'Selected: '
        f"Accuracy {selected_row['Accuracy_mean']:.4f} +- {selected_row['Accuracy_std']:.4f}, "
        f"Precision {selected_row['Precision_mean']:.4f} +- {selected_row['Precision_std']:.4f}, "
        f"Recall {selected_row['Recall_mean']:.4f} +- {selected_row['Recall_std']:.4f}, "
        f"F1 {selected_row['F1-score_mean']:.4f} +- {selected_row['F1-score_std']:.4f}, "
        f"ROC-AUC {selected_row['ROC-AUC_mean']:.4f} +- {selected_row['ROC-AUC_std']:.4f}"
    )


def main() -> None:
    config = parse_args()
    df = load_dataset(config.data_path)
    all_feature_order = infer_all_feature_order(df)
    selected_features = read_selected_features(config.selected_features_csv)

    missing = [feature for feature in selected_features if feature not in df.columns]
    if missing:
        raise KeyError(f'Selected-feature file contains columns missing from the dataset: {missing}')

    single_all = run_single_split(df, all_feature_order, config)
    cv5_all = run_stratified_cv5(df, all_feature_order, config)
    single_selected = run_single_split(df, selected_features, config)
    cv5_selected = run_stratified_cv5(df, selected_features, config)

    write_outputs(
        config.output_dir,
        selected_features,
        config.selected_features_csv,
        single_all,
        cv5_all,
        single_selected,
        cv5_selected,
    )

    print(f'Data: {config.data_path}')
    print(f'TabPFN checkpoint: {config.tabpfn_model_path}')
    print(f'Selected-feature artifact: {config.selected_features_csv}')
    print(f'Output directory: {config.output_dir}')
    print(f'All-feature order used for benchmark: {all_feature_order}')
    print(f'Selected features ({len(selected_features)}): {selected_features}')
    print_section('Single stratified split, all features', single_all)
    print_section('Leakage-free stratified 5-fold, all features', cv5_all)
    print_section('Single stratified split, selected features', single_selected)
    print_section('Leakage-free stratified 5-fold, selected features', cv5_selected)
    print_selected_tabpfn_summary(cv5_all, cv5_selected)


if __name__ == '__main__':
    main()