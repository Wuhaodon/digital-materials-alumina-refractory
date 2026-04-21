from __future__ import annotations

import argparse
import json
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.impute import SimpleImputer
from sklearn.inspection import permutation_importance
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score, roc_auc_score
from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import LabelEncoder, StandardScaler
from tabpfn import TabPFNClassifier
from xgboost import XGBClassifier

from _selected_feature_io import (
    ALL_FEATURES,
    DEFAULT_TABPFN_MODEL_PATH,
    GROUP_COL,
    RANDOM_STATE,
    infer_all_feature_order,
    load_dataset,
    resolve_default_modeling_path,
)

warnings.filterwarnings('ignore')

PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_DATA_PATH = resolve_default_modeling_path('table6_legacy.xlsx')
DEFAULT_OUTPUT_DIR = PROJECT_ROOT / 'results' / 'feature_selection_dual_model_rfe'
DEFAULT_POLICY_PATH = PROJECT_ROOT / 'configs' / 'feature_selection' / 'final_selection_policy.csv'
DEFAULT_ALLOWLIST_PATH = PROJECT_ROOT / 'configs' / 'feature_selection' / 'correlation_allowlist.csv'
DEFAULT_CORR_THRESHOLD = 0.8
DEFAULT_PERMUTATION_REPEATS = 3
MODEL_ORDER = ['XGBoost', 'TabPFN']


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            'Run dual-model recursive feature elimination with XGBoost and TabPFN, '
            'export the computed best-k preliminary set, and then apply the '
            'correlation-constraint filtering to obtain the final selected subset.'
        )
    )
    parser.add_argument('--data', type=Path, default=DEFAULT_DATA_PATH)
    parser.add_argument('--outdir', type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument('--tabpfn-model', type=Path, default=DEFAULT_TABPFN_MODEL_PATH)
    parser.add_argument('--policy-csv', type=Path, default=DEFAULT_POLICY_PATH)
    parser.add_argument('--allowlist-csv', type=Path, default=DEFAULT_ALLOWLIST_PATH)
    parser.add_argument('--correlation-threshold', type=float, default=DEFAULT_CORR_THRESHOLD)
    parser.add_argument('--permutation-repeats', type=int, default=DEFAULT_PERMUTATION_REPEATS)
    parser.add_argument('--tabpfn-device', choices=['cpu', 'cuda'], default='cpu')
    return parser.parse_args()


def make_model(model_name: str, tabpfn_model_path: Path, tabpfn_device: str):
    if model_name == 'XGBoost':
        return XGBClassifier(
            use_label_encoder=False,
            eval_metric='logloss',
            random_state=RANDOM_STATE,
        )
    if not tabpfn_model_path.exists():
        raise FileNotFoundError(f'Missing TabPFN checkpoint: {tabpfn_model_path}')
    return TabPFNClassifier(
        device=tabpfn_device,
        model_path=tabpfn_model_path,
        random_state=RANDOM_STATE,
        n_jobs=1,
    )


def encode_target(df: pd.DataFrame) -> np.ndarray:
    encoder = LabelEncoder()
    return encoder.fit_transform(df[GROUP_COL])


def evaluate_predictions(y_true: np.ndarray, y_pred: np.ndarray, y_prob: np.ndarray | None) -> dict[str, float]:
    metrics: dict[str, float] = {
        'accuracy': float(accuracy_score(y_true, y_pred)),
        'precision': float(precision_score(y_true, y_pred, average='weighted', zero_division=0)),
        'recall': float(recall_score(y_true, y_pred, average='weighted', zero_division=0)),
        'f1': float(f1_score(y_true, y_pred, average='weighted', zero_division=0)),
        'roc_auc': float('nan'),
    }
    if y_prob is not None:
        metrics['roc_auc'] = float(roc_auc_score(y_true, y_prob, multi_class='ovr'))
    return metrics


def run_model_cv(
    df: pd.DataFrame,
    feature_list: list[str],
    y: np.ndarray,
    model_name: str,
    tabpfn_model_path: Path,
    tabpfn_device: str,
) -> dict[str, float]:
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=RANDOM_STATE)
    x_df = df[feature_list].copy()
    fold_metrics: list[dict[str, float]] = []

    for train_idx, test_idx in cv.split(x_df, y):
        x_train_df = x_df.iloc[train_idx]
        x_test_df = x_df.iloc[test_idx]
        y_train = y[train_idx]
        y_test = y[test_idx]

        imputer = SimpleImputer(strategy='mean')
        scaler = StandardScaler()
        x_train = scaler.fit_transform(imputer.fit_transform(x_train_df))
        x_test = scaler.transform(imputer.transform(x_test_df))

        model = make_model(model_name, tabpfn_model_path, tabpfn_device)
        model.fit(x_train, y_train)
        y_pred = model.predict(x_test)
        y_prob = model.predict_proba(x_test) if hasattr(model, 'predict_proba') else None
        fold_metrics.append(evaluate_predictions(y_test, y_pred, y_prob))

    return {
        'accuracy': float(np.mean([row['accuracy'] for row in fold_metrics])),
        'precision': float(np.mean([row['precision'] for row in fold_metrics])),
        'recall': float(np.mean([row['recall'] for row in fold_metrics])),
        'f1': float(np.mean([row['f1'] for row in fold_metrics])),
        'roc_auc': float(np.mean([row['roc_auc'] for row in fold_metrics])),
    }


def fit_full_model(
    df: pd.DataFrame,
    feature_list: list[str],
    y: np.ndarray,
    model_name: str,
    tabpfn_model_path: Path,
    tabpfn_device: str,
):
    imputer = SimpleImputer(strategy='mean')
    scaler = StandardScaler()
    x_full = scaler.fit_transform(imputer.fit_transform(df[feature_list]))
    model = make_model(model_name, tabpfn_model_path, tabpfn_device)
    model.fit(x_full, y)
    return model, x_full


def permutation_scorer(estimator, x_values, y_true) -> float:
    y_pred = estimator.predict(x_values)
    return float(
        0.5 * accuracy_score(y_true, y_pred)
        + 0.5 * f1_score(y_true, y_pred, average='weighted', zero_division=0)
    )


def compute_permutation_importance(
    df: pd.DataFrame,
    feature_list: list[str],
    y: np.ndarray,
    model_name: str,
    tabpfn_model_path: Path,
    tabpfn_device: str,
    repeats: int,
) -> dict[str, float]:
    model, x_full = fit_full_model(df, feature_list, y, model_name, tabpfn_model_path, tabpfn_device)
    result = permutation_importance(
        model,
        x_full,
        y,
        scoring=permutation_scorer,
        n_repeats=repeats,
        random_state=RANDOM_STATE,
        n_jobs=1,
    )
    return {
        feature: float(score)
        for feature, score in zip(feature_list, result.importances_mean.tolist())
    }


def run_rfe(
    df: pd.DataFrame,
    tabpfn_model_path: Path,
    tabpfn_device: str,
    permutation_repeats: int,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    y = encode_target(df)
    current_features = infer_all_feature_order(df)
    path_rows: list[dict[str, object]] = []
    importance_rows: list[dict[str, object]] = []

    while True:
        xgb_metrics = run_model_cv(df, current_features, y, 'XGBoost', tabpfn_model_path, tabpfn_device)
        tab_metrics = run_model_cv(df, current_features, y, 'TabPFN', tabpfn_model_path, tabpfn_device)

        xgb_importance = compute_permutation_importance(
            df,
            current_features,
            y,
            'XGBoost',
            tabpfn_model_path,
            tabpfn_device,
            permutation_repeats,
        )
        tab_importance = compute_permutation_importance(
            df,
            current_features,
            y,
            'TabPFN',
            tabpfn_model_path,
            tabpfn_device,
            permutation_repeats,
        )

        for feature in current_features:
            importance_rows.append(
                {
                    'k': len(current_features),
                    'feature': feature,
                    'xgboost_importance': xgb_importance[feature],
                    'tabpfn_importance': tab_importance[feature],
                    'combined_importance': xgb_importance[feature] + tab_importance[feature],
                }
            )

        removed_feature = ''
        removed_by = ''
        if len(current_features) > 1:
            combined = sorted(
                (
                    {
                        'feature': feature,
                        'combined_importance': xgb_importance[feature] + tab_importance[feature],
                        'xgboost_importance': xgb_importance[feature],
                        'tabpfn_importance': tab_importance[feature],
                    }
                    for feature in current_features
                ),
                key=lambda row: (
                    row['combined_importance'],
                    row['xgboost_importance'],
                    row['tabpfn_importance'],
                    row['feature'],
                ),
            )
            removed_feature = str(combined[0]['feature'])
            removed_by = 'combined_permutation_importance'

        path_rows.append(
            {
                'k': len(current_features),
                'features': ','.join(current_features),
                'xgboost_accuracy': xgb_metrics['accuracy'],
                'xgboost_precision': xgb_metrics['precision'],
                'xgboost_recall': xgb_metrics['recall'],
                'xgboost_f1': xgb_metrics['f1'],
                'xgboost_roc_auc': xgb_metrics['roc_auc'],
                'tabpfn_accuracy': tab_metrics['accuracy'],
                'tabpfn_precision': tab_metrics['precision'],
                'tabpfn_recall': tab_metrics['recall'],
                'tabpfn_f1': tab_metrics['f1'],
                'tabpfn_roc_auc': tab_metrics['roc_auc'],
                'removed_feature_next': removed_feature,
                'removed_by': removed_by,
            }
        )

        if len(current_features) == 1:
            break
        current_features = [feature for feature in current_features if feature != removed_feature]

    path_df = pd.DataFrame(path_rows).sort_values('k', ascending=False, ignore_index=True)
    importance_df = pd.DataFrame(importance_rows).sort_values(
        ['k', 'combined_importance'],
        ascending=[False, False],
        ignore_index=True,
    )
    return path_df, importance_df


def select_best_k(path_df: pd.DataFrame) -> pd.Series:
    ranked = path_df.copy()
    ranked['tabpfn_primary_score'] = (ranked['tabpfn_accuracy'] + ranked['tabpfn_f1']).round(4)
    max_score = ranked['tabpfn_primary_score'].max()
    return ranked[ranked['tabpfn_primary_score'].eq(max_score)].sort_values('k', ascending=False).iloc[0]


def load_policy(policy_csv: Path) -> pd.DataFrame:
    if not policy_csv.exists():
        return pd.DataFrame(columns=['feature', 'priority', 'mandatory', 'rationale'])
    policy = pd.read_csv(policy_csv, encoding='utf-8-sig')
    expected = {'feature', 'priority', 'mandatory', 'rationale'}
    if not expected.issubset(policy.columns):
        raise ValueError(f'Policy file must contain columns {sorted(expected)}: {policy_csv}')
    return policy.sort_values('priority', ascending=False, ignore_index=True)


def load_allow_pairs(allowlist_csv: Path) -> tuple[set[frozenset[str]], dict[frozenset[str], str]]:
    if not allowlist_csv.exists():
        return set(), {}
    allow_df = pd.read_csv(allowlist_csv, encoding='utf-8-sig')
    pairs: set[frozenset[str]] = set()
    notes: dict[frozenset[str], str] = {}
    for _, row in allow_df.iterrows():
        pair = frozenset((str(row['feature_1']).strip(), str(row['feature_2']).strip()))
        pairs.add(pair)
        notes[pair] = str(row.get('rationale', '')).strip()
    return pairs, notes


def build_high_correlation_table(df: pd.DataFrame, threshold: float) -> pd.DataFrame:
    corr_matrix = df[ALL_FEATURES].corr(numeric_only=True)
    rows: list[dict[str, object]] = []
    for idx, left in enumerate(ALL_FEATURES):
        for right in ALL_FEATURES[idx + 1 :]:
            value = float(corr_matrix.loc[left, right])
            if np.isnan(value) or abs(value) < threshold:
                continue
            rows.append(
                {
                    'feature_1': left,
                    'feature_2': right,
                    'pearson_r': value,
                    'abs_pearson_r': abs(value),
                }
            )
    table = pd.DataFrame(rows, columns=['feature_1', 'feature_2', 'pearson_r', 'abs_pearson_r'])
    if table.empty:
        return table
    return table.sort_values('abs_pearson_r', ascending=False, ignore_index=True)


def build_global_ranking(path_df: pd.DataFrame) -> list[str]:
    ascending = path_df.sort_values('k', ascending=True, ignore_index=True)
    ranking: list[str] = []
    ranking.extend(feature for feature in str(ascending.iloc[0]['features']).split(',') if feature)
    ranking.extend(
        str(value).strip()
        for value in ascending['removed_feature_next'].tolist()
        if str(value).strip()
    )
    deduped: list[str] = []
    for feature in ranking:
        if feature not in deduped:
            deduped.append(feature)
    return deduped


def build_candidate_order(global_ranking: list[str], preliminary_features: list[str]) -> list[str]:
    preliminary = [feature for feature in global_ranking if feature in preliminary_features]
    backfill = [feature for feature in global_ranking if feature not in preliminary]
    return preliminary + backfill


def apply_correlation_constraint(
    df: pd.DataFrame,
    candidate_order: list[str],
    best_k: int,
    allowlist_csv: Path,
    threshold: float,
    outdir: Path,
) -> tuple[list[str], pd.DataFrame, pd.DataFrame]:
    allow_pairs, allow_notes = load_allow_pairs(allowlist_csv)
    high_corr_table = build_high_correlation_table(df, threshold=threshold)
    corr_lookup = {
        frozenset((str(row['feature_1']), str(row['feature_2']))): float(row['abs_pearson_r'])
        for _, row in high_corr_table.iterrows()
    }

    selected: list[str] = []
    audit_rows: list[dict[str, object]] = []
    for candidate_rank, feature in enumerate(candidate_order, start=1):
        if len(selected) >= best_k:
            break
        if feature in selected:
            audit_rows.append(
                {
                    'candidate_rank': candidate_rank,
                    'feature': feature,
                    'decision': 'skip_duplicate',
                    'conflict_with': '',
                    'correlation': np.nan,
                    'rationale': 'duplicate_in_candidate_order',
                }
            )
            continue

        blocked = False
        for existing in selected:
            pair = frozenset((feature, existing))
            abs_r = corr_lookup.get(pair, 0.0)
            if abs_r >= threshold and pair not in allow_pairs:
                audit_rows.append(
                    {
                        'candidate_rank': candidate_rank,
                        'feature': feature,
                        'decision': 'skip_correlation_conflict',
                        'conflict_with': existing,
                        'correlation': abs_r,
                        'rationale': 'blocked_by_correlation_threshold',
                    }
                )
                blocked = True
                break
        if blocked:
            continue

        allowlisted_conflicts = []
        allowlisted_corrs = []
        for existing in selected:
            pair = frozenset((feature, existing))
            abs_r = corr_lookup.get(pair, 0.0)
            if abs_r >= threshold and pair in allow_pairs:
                allowlisted_conflicts.append(f'{existing} ({abs_r:.3f})')
                allowlisted_corrs.append(abs_r)

        selected.append(feature)
        audit_rows.append(
            {
                'candidate_rank': candidate_rank,
                'feature': feature,
                'decision': 'selected',
                'conflict_with': '; '.join(allowlisted_conflicts),
                'correlation': np.nan if not allowlisted_corrs else max(allowlisted_corrs),
                'rationale': 'selected_from_rfe_candidate_order',
            }
        )

    if len(selected) < best_k:
        raise ValueError(
            f'Unable to fill {best_k} features after correlation filtering. Selected only {len(selected)}.'
        )

    allow_rows = []
    for pair, note in allow_notes.items():
        pair_list = sorted(pair)
        allow_rows.append(
            {
                'feature_1': pair_list[0],
                'feature_2': pair_list[1],
                'rationale': note,
            }
        )
    allow_df = pd.DataFrame(allow_rows)
    if not allow_df.empty:
        allow_df.to_csv(outdir / 'correlation_allowlist_used.csv', index=False, encoding='utf-8-sig')

    audit_df = pd.DataFrame(audit_rows)
    return selected, audit_df, high_corr_table


def write_feature_table(path: Path, features: list[str], stage: str, policy: pd.DataFrame | None = None) -> None:
    rows = []
    for rank, feature in enumerate(features, start=1):
        row = {'feature': feature, 'rank': rank, 'stage': stage}
        if policy is not None and feature in set(policy['feature']):
            policy_row = policy.loc[policy['feature'].eq(feature)].iloc[0]
            row['priority'] = int(policy_row['priority'])
            row['rationale'] = str(policy_row['rationale'])
        rows.append(row)
    pd.DataFrame(rows).to_csv(path, index=False, encoding='utf-8-sig')


def main() -> None:
    args = parse_args()
    args.outdir.mkdir(parents=True, exist_ok=True)

    df = load_dataset(args.data)
    path_df, importance_df = run_rfe(
        df,
        tabpfn_model_path=args.tabpfn_model,
        tabpfn_device=args.tabpfn_device,
        permutation_repeats=args.permutation_repeats,
    )

    best_row = select_best_k(path_df)
    best_k = int(best_row['k'])
    preliminary_features = str(best_row['features']).split(',')
    policy = load_policy(args.policy_csv)
    global_ranking = build_global_ranking(path_df)
    candidate_order = build_candidate_order(global_ranking, preliminary_features)
    final_features, audit_df, high_corr_table = apply_correlation_constraint(
        df,
        candidate_order=candidate_order,
        best_k=best_k,
        allowlist_csv=args.allowlist_csv,
        threshold=args.correlation_threshold,
        outdir=args.outdir,
    )

    path_df.to_csv(args.outdir / 'rfe_path_metrics.csv', index=False, encoding='utf-8-sig')
    importance_df.to_csv(args.outdir / 'permutation_importance_by_step.csv', index=False, encoding='utf-8-sig')
    high_corr_table.to_csv(args.outdir / 'high_correlation_pairs.csv', index=False, encoding='utf-8-sig')
    audit_df.to_csv(args.outdir / 'correlation_constraint_audit.csv', index=False, encoding='utf-8-sig')

    write_feature_table(args.outdir / 'selected_features_preliminary.csv', preliminary_features, 'preliminary_rfe_best_k', policy=policy)
    write_feature_table(args.outdir / 'selected_features_final.csv', final_features, 'final_correlation_constrained', policy=policy)
    pd.DataFrame({'feature': global_ranking, 'rank': range(1, len(global_ranking) + 1)}).to_csv(
        args.outdir / 'feature_global_ranking.csv',
        index=False,
        encoding='utf-8-sig',
    )

    best_summary = pd.DataFrame(
        [
            {
                'best_k': best_k,
                'criterion': 'maximize rounded(TabPFN accuracy + TabPFN F1); choose the largest k among ties',
                'tabpfn_accuracy': float(best_row['tabpfn_accuracy']),
                'tabpfn_f1': float(best_row['tabpfn_f1']),
                'xgboost_accuracy': float(best_row['xgboost_accuracy']),
                'xgboost_f1': float(best_row['xgboost_f1']),
                'preliminary_features': ','.join(preliminary_features),
                'final_features': ','.join(final_features),
                'candidate_order': ','.join(candidate_order),
            }
        ]
    )
    best_summary.to_csv(args.outdir / 'best_k_summary.csv', index=False, encoding='utf-8-sig')

    manifest = {
        'data_path': str(args.data),
        'tabpfn_model': str(args.tabpfn_model),
        'policy_csv': str(args.policy_csv),
        'allowlist_csv': str(args.allowlist_csv),
        'correlation_threshold': args.correlation_threshold,
        'permutation_repeats': args.permutation_repeats,
        'best_k': best_k,
        'preliminary_features': preliminary_features,
        'final_features': final_features,
    }
    (args.outdir / 'manifest.json').write_text(json.dumps(manifest, indent=2, ensure_ascii=False), encoding='utf-8')

    print(f'Data: {args.data}')
    print(f'TabPFN checkpoint: {args.tabpfn_model}')
    print(f'Output directory: {args.outdir}')
    print(f'Best k: {best_k}')
    print(f'Preliminary RFE features: {preliminary_features}')
    print(f'Final correlation-constrained features: {final_features}')


if __name__ == '__main__':
    main()
