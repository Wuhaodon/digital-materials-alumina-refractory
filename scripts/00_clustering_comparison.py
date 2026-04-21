"""Summarize cluster-level properties across the four manuscript specimens."""

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_CLUSTER_DIR = PROJECT_ROOT / 'data' / 'processed' / 'clusters'
DEFAULT_OUTPUT_DIR = PROJECT_ROOT / 'results' / 'clustering'
FEATURE_COLUMNS = ['Hardness', 'Modulus', 'GC', 'KIC']
SAMPLE_ORDER = ['00', '01', '10', '11']
SAMPLE_NAME_MAP = {
    '00': 'C0',
    '01': 'C1',
    '10': 'C0-R',
    '11': 'C1-R',
}
FEATURE_COLORS = {
    'Hardness': '#9ad8df',
    'Modulus': '#f7b089',
    'GC': '#9bbce0',
    'KIC': '#d5c4f1',
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description='Create a concise four-panel cluster property summary figure.'
    )
    parser.add_argument('--cluster-dir', type=Path, default=DEFAULT_CLUSTER_DIR)
    parser.add_argument('--output-dir', type=Path, default=DEFAULT_OUTPUT_DIR)
    return parser.parse_args()


def build_summary_frame(cluster_dir: Path) -> pd.DataFrame:
    rows = []
    for sample_code in SAMPLE_ORDER:
        path = cluster_dir / f'{sample_code}z.csv'
        if not path.exists():
            continue
        frame = pd.read_csv(path)
        total_rows = len(frame)
        for cluster_id in sorted(frame['Cluster'].unique()):
            group = frame[frame['Cluster'] == cluster_id]
            row = {
                'sample_code': sample_code,
                'sample_name': SAMPLE_NAME_MAP[sample_code],
                'cluster_id': int(cluster_id),
                'count': len(group),
                'proportion_percent': 100.0 * len(group) / total_rows,
            }
            for feature in FEATURE_COLUMNS:
                row[feature] = float(group[feature].mean())
            rows.append(row)
    if not rows:
        raise FileNotFoundError(f'No cluster CSVs found in {cluster_dir}')
    return pd.DataFrame(rows)


def main() -> None:
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    summary = build_summary_frame(args.cluster_dir)
    summary.to_csv(args.output_dir / 'cluster_comparison_summary.csv', index=False, encoding='utf-8-sig')

    plt.rcParams['font.family'] = 'Times New Roman'
    fig, axes = plt.subplots(2, 2, figsize=(16, 12))
    axes = axes.flatten()
    for axis, sample_code in zip(axes, SAMPLE_ORDER):
        sample = summary[summary['sample_code'] == sample_code].sort_values('cluster_id')
        x = list(range(len(sample)))
        width = 0.18
        offsets = [-1.5 * width, -0.5 * width, 0.5 * width, 1.5 * width]
        for offset, feature in zip(offsets, FEATURE_COLUMNS):
            axis.bar(
                [value + offset for value in x],
                sample[feature],
                width=width,
                label=feature,
                color=FEATURE_COLORS[feature],
                edgecolor='black',
                linewidth=0.6,
            )
        axis.set_title(SAMPLE_NAME_MAP[sample_code])
        axis.set_xticks(x)
        axis.set_xticklabels(
            [
                f"{SAMPLE_NAME_MAP[sample_code]}-{cluster}\n{prop:.1f}%"
                for cluster, prop in zip(sample['cluster_id'], sample['proportion_percent'])
            ],
            fontsize=10,
        )
        axis.set_ylabel('Value')
        axis.grid(axis='y', linestyle='--', alpha=0.25)
    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc='upper center', ncol=4, frameon=False)
    plt.tight_layout(rect=(0, 0, 1, 0.95))
    plt.savefig(args.output_dir / 'cluster_comparison_overview.png', dpi=300)
    plt.savefig(args.output_dir / 'cluster_comparison_overview.pdf')
    plt.close(fig)
    print(f'Wrote cluster comparison summary to {args.output_dir}')


if __name__ == '__main__':
    main()
