"""Compute manuscript-style probabilistic cluster correspondence across specimen pairs."""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_CLUSTER_DIR = PROJECT_ROOT / 'data' / 'processed' / 'clusters'
DEFAULT_OUTPUT_DIR = PROJECT_ROOT / 'data' / 'processed' / 'unified_states'
FEATURES = ['Hardness', 'Modulus', 'GC', 'KIC']
WEIGHTS = {'Hardness': 0.21, 'Modulus': 0.18, 'GC': 0.47, 'KIC': 0.15}
PAIRS = [
    ('C0', 'C0-R', '00z.csv', '10z.csv', 'C0_to_C0R_report.md', 'C0_to_C0R_dominant.csv'),
    ('C0', 'C1', '00z.csv', '01z.csv', 'C0_to_C1_report.md', 'C0_to_C1_dominant.csv'),
    ('C1', 'C1-R', '01z.csv', '11z.csv', 'C1_to_C1R_report.md', 'C1_to_C1R_dominant.csv'),
]
EPS = 1e-6


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description='Run weighted soft correspondence across the manuscript specimen pairs.'
    )
    parser.add_argument('--cluster-dir', type=Path, default=DEFAULT_CLUSTER_DIR)
    parser.add_argument('--output-dir', type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument('--tau-percentile', type=float, default=70.0)
    parser.add_argument('--topk', type=int, default=3)
    return parser.parse_args()


def safe_read_csv(path: Path) -> pd.DataFrame:
    for encoding in ('utf-8-sig', 'utf-8', 'gbk', 'ansi'):
        try:
            return pd.read_csv(path, encoding=encoding)
        except UnicodeDecodeError:
            continue
    return pd.read_csv(path)


def load_cluster_frame(path: Path) -> pd.DataFrame:
    frame = safe_read_csv(path)
    required = FEATURES + ['Cluster']
    missing = [column for column in required if column not in frame.columns]
    if missing:
        raise KeyError(f'{path} is missing columns: {missing}')
    frame = frame.loc[:, required].copy()
    frame['Cluster'] = frame['Cluster'].astype(str)
    return frame


def sorted_cluster_labels(frame: pd.DataFrame) -> list[str]:
    return sorted(frame['Cluster'].unique(), key=lambda value: int(value))


def cluster_stats(frame: pd.DataFrame) -> dict[str, object]:
    labels = sorted_cluster_labels(frame)
    means = []
    stds = []
    fractions = []
    counts = []
    total_rows = len(frame)
    for label in labels:
        group = frame[frame['Cluster'] == label]
        values = group[FEATURES].to_numpy(float)
        means.append(values.mean(axis=0))
        stds.append(values.std(axis=0, ddof=1) if len(group) > 1 else np.zeros(values.shape[1]))
        fractions.append(len(group) / total_rows)
        counts.append(len(group))
    return {
        'labels': labels,
        'means': np.vstack(means),
        'stds': np.vstack(stds),
        'fractions': np.array(fractions, dtype=float),
        'counts': np.array(counts, dtype=int),
        'n_rows': total_rows,
    }


def weight_vector() -> np.ndarray:
    weights = np.array([WEIGHTS[column] for column in FEATURES], dtype=float)
    return weights / weights.sum()


def cost_matrix(src_stats: dict[str, object], dst_stats: dict[str, object]) -> np.ndarray:
    weights = weight_vector()
    src_means = np.asarray(src_stats['means'], dtype=float)
    src_stds = np.asarray(src_stats['stds'], dtype=float)
    dst_means = np.asarray(dst_stats['means'], dtype=float)
    dst_stds = np.asarray(dst_stats['stds'], dtype=float)
    cost = np.zeros((src_means.shape[0], dst_means.shape[0]), dtype=float)
    for i in range(src_means.shape[0]):
        for j in range(dst_means.shape[0]):
            numerator = (src_means[i] - dst_means[j]) ** 2
            denominator = src_stds[i] ** 2 + dst_stds[j] ** 2 + EPS
            cost[i, j] = float(np.sum(weights * (numerator / denominator)))
    return cost


def soft_alignment(cost: np.ndarray, tau_percentile: float) -> tuple[np.ndarray, float]:
    tau = float(np.percentile(cost, tau_percentile))
    scores = np.exp(-cost)
    mask = (cost <= tau).astype(float)
    probabilities = np.zeros_like(scores)
    for row_index in range(scores.shape[0]):
        row = scores[row_index] * mask[row_index]
        if row.sum() <= 0:
            probabilities[row_index, int(np.argmin(cost[row_index]))] = 1.0
        else:
            probabilities[row_index] = row / row.sum()
    return probabilities, tau


def dominant_summary(
    src_name: str,
    dst_name: str,
    src_stats: dict[str, object],
    dst_stats: dict[str, object],
    probabilities: np.ndarray,
) -> pd.DataFrame:
    dst_labels = list(dst_stats['labels'])
    rows = []
    for i, src_label in enumerate(src_stats['labels']):
        dominant_index = int(np.argmax(probabilities[i]))
        rows.append(
            {
                'source_sample': src_name,
                'target_sample': dst_name,
                'source_cluster': f'{src_name}-{src_label}',
                'dominant_target_cluster': f'{dst_name}-{dst_labels[dominant_index]}',
                'probability': float(probabilities[i, dominant_index]),
            }
        )
    return pd.DataFrame(rows)


def topk_table(
    src_stats: dict[str, object],
    dst_stats: dict[str, object],
    probabilities: np.ndarray,
    topk: int,
) -> pd.DataFrame:
    dst_labels = list(dst_stats['labels'])
    src_fractions = np.asarray(src_stats['fractions'], dtype=float)
    rows = []
    for i, src_label in enumerate(src_stats['labels']):
        ranked = np.argsort(probabilities[i])[::-1][:topk]
        for j in ranked:
            rows.append(
                {
                    'source_cluster': src_label,
                    'target_cluster': dst_labels[j],
                    'p_src': float(src_fractions[i]),
                    'P_ij': float(probabilities[i, j]),
                    'w_ij': float(src_fractions[i] * probabilities[i, j]),
                }
            )
    return pd.DataFrame(rows)


def expected_shift_table(
    src_stats: dict[str, object],
    dst_stats: dict[str, object],
    probabilities: np.ndarray,
) -> pd.DataFrame:
    src_means = np.asarray(src_stats['means'], dtype=float)
    src_stds = np.asarray(src_stats['stds'], dtype=float)
    dst_means = np.asarray(dst_stats['means'], dtype=float)
    dst_stds = np.asarray(dst_stats['stds'], dtype=float)
    src_fractions = np.asarray(src_stats['fractions'], dtype=float)
    rows = []
    for i, src_label in enumerate(src_stats['labels']):
        expected_mean = probabilities[i] @ dst_means
        expected_var = probabilities[i] @ (dst_stds ** 2)
        delta = expected_mean - src_means[i]
        delta_z = delta / np.sqrt(src_stds[i] ** 2 + expected_var + EPS)
        row = {
            'source_cluster': src_label,
            'p_src': float(src_fractions[i]),
        }
        for feature_index, feature_name in enumerate(FEATURES):
            row[f'mu_src_{feature_name}'] = float(src_means[i, feature_index])
            row[f'mu_exp_{feature_name}'] = float(expected_mean[feature_index])
            row[f'delta_{feature_name}'] = float(delta[feature_index])
            row[f'delta_z_{feature_name}'] = float(delta_z[feature_index])
        rows.append(row)
    return pd.DataFrame(rows)


def target_fraction_table(
    src_stats: dict[str, object],
    dst_stats: dict[str, object],
    probabilities: np.ndarray,
) -> pd.DataFrame:
    src_fractions = np.asarray(src_stats['fractions'], dtype=float)
    dst_fractions = np.asarray(dst_stats['fractions'], dtype=float)
    predicted = src_fractions @ probabilities
    rows = []
    for index, dst_label in enumerate(dst_stats['labels']):
        rows.append(
            {
                'target_cluster': dst_label,
                'p_dst_true': float(dst_fractions[index]),
                'p_dst_pred_from_src': float(predicted[index]),
                'delta_pred_minus_true': float(predicted[index] - dst_fractions[index]),
            }
        )
    return pd.DataFrame(rows)


def fractions_with_ci(sample_name: str, stats: dict[str, object]) -> pd.DataFrame:
    rows = []
    n_rows = int(stats['n_rows'])
    for label, fraction in zip(stats['labels'], stats['fractions']):
        z = 1.96
        denominator = 1 + z ** 2 / n_rows
        center = (fraction + z ** 2 / (2 * n_rows)) / denominator
        margin = z * np.sqrt(fraction * (1 - fraction) / n_rows + z ** 2 / (4 * n_rows ** 2)) / denominator
        rows.append(
            {
                'index': f'{sample_name}-{label}',
                'p': float(fraction),
                'CI_low': float(max(0.0, center - margin)),
                'CI_high': float(min(1.0, center + margin)),
                'N': n_rows,
            }
        )
    return pd.DataFrame(rows)


def frame_to_markdown(frame: pd.DataFrame, float_digits: int = 4) -> str:
    display = frame.copy()
    for column in display.columns:
        if pd.api.types.is_numeric_dtype(display[column]):
            display[column] = display[column].map(
                lambda value: f'{value:.{float_digits}f}' if pd.notnull(value) else ''
            )
    return display.to_markdown(index=False)


def write_report(
    output_path: Path,
    src_name: str,
    dst_name: str,
    tau_percentile: float,
    tau: float,
    topk_frame: pd.DataFrame,
    shift_frame: pd.DataFrame,
    fraction_frame: pd.DataFrame,
    ci_frame: pd.DataFrame,
) -> None:
    topk_size = int(topk_frame.groupby('source_cluster').size().max())
    text = f'# Probabilistic alignment report - {src_name} -> {dst_name}\n\n'
    text += 'Method summary:\n'
    text += '- Distance: variance-weighted (means+/-SD)\n'
    text += f'- Features: {FEATURES}\n'
    text += f'- Weights: {WEIGHTS}\n'
    text += f'- tau (percentile={tau_percentile:.1f}%): {tau:.6f}\n'
    text += f'- TOPK: {topk_size} (per source cluster)\n\n'
    text += '## 1) Top-K probabilistic correspondences and soft flows\n\n'
    text += frame_to_markdown(topk_frame) + '\n\n'
    text += '## 2) Expected feature shifts per source cluster (P-weighted)\n\n'
    text += frame_to_markdown(shift_frame) + '\n\n'
    text += '## 3) Target fraction: predicted (from soft flows) vs observed\n\n'
    text += frame_to_markdown(fraction_frame) + '\n\n'
    text += '## 4) Fractions with 95% Wilson CI\n\n'
    text += frame_to_markdown(ci_frame) + '\n'
    output_path.write_text(text, encoding='utf-8')


def main() -> None:
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    report_dir = args.output_dir / 'reports'
    report_dir.mkdir(parents=True, exist_ok=True)

    all_dominant = []
    for src_name, dst_name, src_file, dst_file, report_name, dominant_name in PAIRS:
        src_frame = load_cluster_frame(args.cluster_dir / src_file)
        dst_frame = load_cluster_frame(args.cluster_dir / dst_file)
        src_stats = cluster_stats(src_frame)
        dst_stats = cluster_stats(dst_frame)
        probabilities, tau = soft_alignment(cost_matrix(src_stats, dst_stats), args.tau_percentile)

        dominant = dominant_summary(src_name, dst_name, src_stats, dst_stats, probabilities)
        topk = topk_table(src_stats, dst_stats, probabilities, args.topk)
        shifts = expected_shift_table(src_stats, dst_stats, probabilities)
        fractions = target_fraction_table(src_stats, dst_stats, probabilities)
        ci = pd.concat(
            [fractions_with_ci(src_name, src_stats), fractions_with_ci(dst_name, dst_stats)],
            ignore_index=True,
        )

        dominant.to_csv(args.output_dir / dominant_name, index=False, encoding='utf-8-sig')
        write_report(
            report_dir / report_name,
            src_name=src_name,
            dst_name=dst_name,
            tau_percentile=args.tau_percentile,
            tau=tau,
            topk_frame=topk,
            shift_frame=shifts,
            fraction_frame=fractions,
            ci_frame=ci,
        )
        all_dominant.append(dominant)
        print(f'{src_name} -> {dst_name} complete. Report: {report_name}')

    pd.concat(all_dominant, ignore_index=True).to_csv(
        args.output_dir / 'dominant_correspondence_summary.csv',
        index=False,
        encoding='utf-8-sig',
    )
    print(f'Wrote correspondence outputs to {args.output_dir}')


if __name__ == '__main__':
    main()
