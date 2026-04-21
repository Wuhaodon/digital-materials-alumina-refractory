from __future__ import annotations

import argparse
from math import atanh, tanh
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import pearsonr, spearmanr

PROJECT_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_PROPORTIONS_CSV = PROJECT_ROOT / 'data' / 'processed' / 'unified_states' / 'reports' / 'unified_state_proportions.csv'
DEFAULT_MACRO_CSV = PROJECT_ROOT / 'data' / 'processed' / 'unified_states' / 'reports' / 'specimen_macro_metrics.csv'
DEFAULT_OUTPUT_DIR = PROJECT_ROOT / 'results' / 'supplementary_material' / 'unified_state_correlations'
SPECIMEN_ORDER = ['C0', 'C0-R', 'C1', 'C1-R']


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            'Compute Table S2 from structured unified-state proportions and specimen-level '
            'macroscopic metrics. This script intentionally does not read the manuscript DOCX.'
        )
    )
    parser.add_argument('--proportions-csv', type=Path, default=DEFAULT_PROPORTIONS_CSV)
    parser.add_argument('--macro-csv', type=Path, default=DEFAULT_MACRO_CSV)
    parser.add_argument('--output-dir', type=Path, default=DEFAULT_OUTPUT_DIR)
    return parser.parse_args()


def parse_numeric(value) -> float:
    if isinstance(value, str):
        return float(value.replace('%', '').replace('+', '').strip())
    return float(value)


def fisher_ci(r_value: float, n_obs: int, confidence: float = 0.95) -> str:
    if n_obs <= 3:
        return '[nan, nan]'
    clipped = min(max(r_value, -0.999999), 0.999999)
    z_value = atanh(clipped)
    stderr = 1.0 / np.sqrt(n_obs - 3)
    z_crit = 1.959963984540054
    low = tanh(z_value - z_crit * stderr)
    high = tanh(z_value + z_crit * stderr)
    return f'[{low:+.2f}, {high:+.2f}]'


def load_proportions(path: Path) -> pd.DataFrame:
    if not path.exists():
        raise FileNotFoundError(
            'Missing unified-state proportions csv. Export the structured upstream table first: '
            f'{path}'
        )
    df = pd.read_csv(path, encoding='utf-8-sig')
    expected = {'State', 'C0', 'C0-R', 'C1', 'C1-R'}
    if not expected.issubset(df.columns):
        raise ValueError(f'Proportion table must contain columns {sorted(expected)}: {path}')
    out = df.copy()
    for column in SPECIMEN_ORDER:
        out[column] = out[column].map(parse_numeric)
    out['State'] = out['State'].astype(str).str.strip()
    return out


def load_macro_metrics(path: Path) -> pd.DataFrame:
    if not path.exists():
        raise FileNotFoundError(
            'Missing specimen macro-metrics csv. Provide structured macroscopic GC/KIC values first: '
            f'{path}'
        )
    df = pd.read_csv(path, encoding='utf-8-sig')
    expected = {'Specimen', 'GC', 'KIC'}
    if not expected.issubset(df.columns):
        raise ValueError(f'Macro-metric table must contain columns {sorted(expected)}: {path}')
    out = df.copy()
    out['Specimen'] = out['Specimen'].astype(str).str.strip()
    out['GC'] = out['GC'].map(parse_numeric)
    out['KIC'] = out['KIC'].map(parse_numeric)
    missing = [specimen for specimen in SPECIMEN_ORDER if specimen not in set(out['Specimen'])]
    if missing:
        raise ValueError(f'Macro-metric table is missing specimens {missing}: {path}')
    return out.set_index('Specimen').loc[SPECIMEN_ORDER].reset_index()


def build_summary(proportions: pd.DataFrame, macro_df: pd.DataFrame) -> pd.DataFrame:
    kic_values = macro_df['KIC'].to_numpy(dtype=float)
    gc_values = macro_df['GC'].to_numpy(dtype=float)
    rows: list[dict[str, object]] = []

    for _, row in proportions.iterrows():
        state_name = str(row['State'])
        state_values = row[SPECIMEN_ORDER].to_numpy(dtype=float)

        pearson_kic = pearsonr(state_values, kic_values)
        spearman_kic = spearmanr(state_values, kic_values)
        pearson_gc = pearsonr(state_values, gc_values)
        spearman_gc = spearmanr(state_values, gc_values)

        rows.append(
            {
                'State': state_name,
                'Pearson r (KIC)': round(float(pearson_kic.statistic), 2),
                'p (KIC)': round(float(pearson_kic.pvalue), 3),
                '95% CI (KIC)': fisher_ci(float(pearson_kic.statistic), len(state_values)),
                'Spearman rho (KIC)': round(float(spearman_kic.statistic), 2),
                'Pearson r (GC)': round(float(pearson_gc.statistic), 2),
                'p (GC)': round(float(pearson_gc.pvalue), 3),
                '95% CI (GC)': fisher_ci(float(pearson_gc.statistic), len(state_values)),
                'Spearman rho (GC)': round(float(spearman_gc.statistic), 2),
            }
        )

    return pd.DataFrame(rows)


def main() -> None:
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)

    proportions = load_proportions(args.proportions_csv)
    macro_df = load_macro_metrics(args.macro_csv)
    summary = build_summary(proportions, macro_df)

    out_path = args.output_dir / 'TableS2_unified_state_correlations.csv'
    summary.to_csv(out_path, index=False, encoding='utf-8-sig')

    print(f'Unified-state proportions: {args.proportions_csv}')
    print(f'Macro metrics: {args.macro_csv}')
    print(f'Wrote {out_path}')
    print(summary.to_string(index=False))


if __name__ == '__main__':
    main()