"""Run per-sample clustering for the four manuscript specimens and export paper-facing summaries."""

from __future__ import annotations

import argparse
import os
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from sklearn.metrics import silhouette_score
from sklearn.preprocessing import QuantileTransformer, StandardScaler

os.environ.setdefault('OMP_NUM_THREADS', '1')

PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_INPUT_DIR = PROJECT_ROOT / 'data' / 'interim' / 'clustering_inputs'
DEFAULT_OUTPUT_DIR = PROJECT_ROOT / 'data' / 'processed' / 'clusters'
FEATURE_COLUMNS = ['Hardness', 'Modulus', 'GC', 'KIC']
SAMPLE_ORDER = ['00', '01', '10', '11']
SAMPLE_NAME_MAP = {
    '00': 'C0',
    '01': 'C1',
    '10': 'C0-R',
    '11': 'C1-R',
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description='Cluster each manuscript specimen separately and export cluster tables.'
    )
    parser.add_argument('--input-dir', type=Path, default=DEFAULT_INPUT_DIR)
    parser.add_argument('--output-dir', type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument('--min-k', type=int, default=3)
    parser.add_argument('--max-k', type=int, default=8)
    return parser.parse_args()


def safe_read_csv(path: Path) -> pd.DataFrame:
    for encoding in ('utf-8-sig', 'utf-8', 'gbk', 'ansi'):
        try:
            return pd.read_csv(path, encoding=encoding)
        except UnicodeDecodeError:
            continue
    return pd.read_csv(path)


def available_sample_files(input_dir: Path) -> dict[str, Path]:
    sample_files: dict[str, Path] = {}
    for sample_code in SAMPLE_ORDER:
        candidate = input_dir / f'{sample_code}_features.csv'
        if candidate.exists():
            sample_files[sample_code] = candidate
    if not sample_files:
        raise FileNotFoundError(f'No *_features.csv files found in {input_dir}')
    return sample_files


def transform_features(frame: pd.DataFrame) -> tuple[pd.DataFrame, np.ndarray, PCA]:
    transformed = frame.copy()
    for column in FEATURE_COLUMNS:
        transformed[column], _ = stats.boxcox(transformed[column] + 1)
    quantile = QuantileTransformer(output_distribution='normal', random_state=42)
    transformed.loc[:, FEATURE_COLUMNS] = quantile.fit_transform(transformed[FEATURE_COLUMNS])
    scaled = StandardScaler().fit_transform(transformed[FEATURE_COLUMNS])
    pca = PCA(n_components=2, random_state=42)
    pca.fit(scaled)
    return transformed, scaled, pca


def select_k(x_scaled: np.ndarray, min_k: int, max_k: int) -> tuple[int, list[dict[str, float]]]:
    rows: list[dict[str, float]] = []
    best_k = min_k
    best_score = float('-inf')
    for k in range(min_k, max_k + 1):
        model = KMeans(n_clusters=k, random_state=42)
        labels = model.fit_predict(x_scaled)
        score = silhouette_score(x_scaled, labels)
        rows.append({'k': k, 'silhouette_score': score, 'inertia': float(model.inertia_)})
        if score > best_score:
            best_score = score
            best_k = k
    return best_k, rows


def format_mean_std(mean_value: float, std_value: float) -> str:
    if np.isnan(std_value):
        return f'{mean_value:.2f}+/-nan'
    return f'{mean_value:.2f}+/-{std_value:.2f}'


def export_cluster_tables(
    sample_code: str,
    frame: pd.DataFrame,
    labels: np.ndarray,
    output_dir: Path,
) -> tuple[int, pd.DataFrame]:
    clustered = frame.copy()
    clustered['Cluster'] = labels.astype(int)
    cluster_path = output_dir / f'{sample_code}z.csv'
    clustered.to_csv(cluster_path, index=False, encoding='utf-8-sig')

    visual_dir = output_dir / 'visual_summaries'
    visual_dir.mkdir(parents=True, exist_ok=True)

    numeric_summary_rows = []
    formatted_summary_rows = []
    total_rows = len(clustered)
    for cluster_id in sorted(clustered['Cluster'].unique()):
        group = clustered[clustered['Cluster'] == cluster_id]
        proportion = len(group) / total_rows
        numeric_row = {'Cluster': int(cluster_id), 'Count': len(group), 'Proportion': proportion}
        formatted_row = {'Cluster': int(cluster_id)}
        for column in FEATURE_COLUMNS:
            mean_value = group[column].mean()
            std_value = group[column].std(ddof=1)
            numeric_row[f'{column}_mean'] = mean_value
            numeric_row[f'{column}_std'] = std_value
            formatted_row[column] = format_mean_std(mean_value, std_value)
        numeric_summary_rows.append(numeric_row)
        formatted_summary_rows.append(formatted_row)

    numeric_summary = pd.DataFrame(numeric_summary_rows)
    formatted_summary = pd.DataFrame(formatted_summary_rows)
    numeric_summary.to_csv(visual_dir / f'{sample_code}z_stats_numeric.csv', index=False, encoding='utf-8-sig')
    formatted_summary.to_csv(visual_dir / f'{sample_code}z_mean_std.csv', index=False, encoding='utf-8-sig')
    return int(clustered['Cluster'].nunique()), numeric_summary


def main() -> None:
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)

    manifest_rows = []
    silhouette_rows = []
    sample_files = available_sample_files(args.input_dir)
    for sample_code in SAMPLE_ORDER:
        input_path = sample_files.get(sample_code)
        if input_path is None:
            continue

        frame = safe_read_csv(input_path)
        frame = frame.loc[:, FEATURE_COLUMNS].copy()
        _, x_scaled, pca = transform_features(frame)
        best_k, k_rows = select_k(x_scaled, args.min_k, args.max_k)
        labels = KMeans(n_clusters=best_k, random_state=42).fit_predict(x_scaled)
        cluster_count, numeric_summary = export_cluster_tables(sample_code, frame, labels, args.output_dir)

        for row in k_rows:
            row['sample_code'] = sample_code
            row['sample_name'] = SAMPLE_NAME_MAP.get(sample_code, sample_code)
            silhouette_rows.append(row)

        manifest_rows.append(
            {
                'sample_code': sample_code,
                'sample_name': SAMPLE_NAME_MAP.get(sample_code, sample_code),
                'input_file': input_path.name,
                'best_k': best_k,
                'cluster_count': cluster_count,
                'n_rows': len(frame),
                'pc1_loading_Hardness': pca.components_[0, 0],
                'pc1_loading_Modulus': pca.components_[0, 1],
                'pc1_loading_GC': pca.components_[0, 2],
                'pc1_loading_KIC': pca.components_[0, 3],
            }
        )
        print(
            f"{SAMPLE_NAME_MAP.get(sample_code, sample_code)} ({sample_code}) -> best K = {best_k}, "
            f"cluster file {sample_code}z.csv, summary rows = {len(numeric_summary)}"
        )

    pd.DataFrame(manifest_rows).to_csv(
        args.output_dir / 'cluster_selection_summary.csv',
        index=False,
        encoding='utf-8-sig',
    )
    pd.DataFrame(silhouette_rows).to_csv(
        args.output_dir / 'cluster_k_search_scores.csv',
        index=False,
        encoding='utf-8-sig',
    )
    print(f'Finished per-sample clustering for {len(manifest_rows)} specimens.')


if __name__ == '__main__':
    main()
