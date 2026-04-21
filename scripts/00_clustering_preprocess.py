"""Prepare specimen-specific clustering input tables from raw splits or frozen cleaned tables."""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Iterable

import numpy as np
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_INPUT_DIR = PROJECT_ROOT / 'data' / 'raw' / 'clustering_splits'
DEFAULT_OUTPUT_DIR = PROJECT_ROOT / 'data' / 'interim' / 'clustering_inputs'
FEATURE_COLUMNS = ['Hardness', 'Modulus', 'GC', 'KIC']
THERMAL_COL = 'Thermal shock'
LOW_CARBON_COL = 'Low carbon'
SAMPLE_NAME_MAP = {
    '00': 'C0',
    '01': 'C1',
    '10': 'C0-R',
    '11': 'C1-R',
}
FROZEN_PROCESSED_MAP = {
    'processed_data1.csv': '10',
    'processed_data2.csv': '11',
    'processed_data3.csv': '00',
    'processed_data4.csv': '01',
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description='Extract and normalize the four clustering features for each manuscript specimen.'
    )
    parser.add_argument('--input-dir', type=Path, default=DEFAULT_INPUT_DIR)
    parser.add_argument('--output-dir', type=Path, default=DEFAULT_OUTPUT_DIR)
    return parser.parse_args()


def safe_read_csv(path: Path) -> pd.DataFrame:
    for encoding in ('utf-8-sig', 'utf-8', 'gbk', 'ansi'):
        try:
            return pd.read_csv(path, encoding=encoding)
        except UnicodeDecodeError:
            continue
    return pd.read_csv(path)


def validate_feature_columns(frame: pd.DataFrame, feature_columns: Iterable[str]) -> None:
    missing = [column for column in feature_columns if column not in frame.columns]
    if missing:
        raise KeyError(f'Missing clustering features: {missing}')


def collect_processed_files(input_dir: Path) -> list[Path]:
    return [input_dir / name for name in FROZEN_PROCESSED_MAP if (input_dir / name).exists()]


def collect_split_files(input_dir: Path) -> list[Path]:
    return sorted(input_dir.glob('split_*.csv'))


def infer_sample_code(frame: pd.DataFrame) -> str:
    required = {THERMAL_COL, LOW_CARBON_COL}
    missing = required.difference(frame.columns)
    if missing:
        raise KeyError(f'Missing sample identity columns: {sorted(missing)}')

    thermal_values = frame[THERMAL_COL].dropna().astype(int).unique().tolist()
    low_carbon_values = frame[LOW_CARBON_COL].dropna().astype(int).unique().tolist()
    if len(thermal_values) != 1 or len(low_carbon_values) != 1:
        raise ValueError('Each split file must contain one unique specimen identity pair.')
    return f"{thermal_values[0]}{low_carbon_values[0]}"


def clean_raw_feature_table(frame: pd.DataFrame) -> tuple[pd.DataFrame, dict[str, int]]:
    cleaned = frame.loc[:, FEATURE_COLUMNS].copy()
    fill_counts: dict[str, int] = {}
    outlier_counts: dict[str, int] = {}
    nonpositive_counts: dict[str, int] = {}

    for column in FEATURE_COLUMNS:
        median_value = cleaned[column].median()
        missing_mask = cleaned[column].isna()
        fill_counts[column] = int(missing_mask.sum())
        if fill_counts[column]:
            cleaned.loc[missing_mask, column] = median_value

    for column in FEATURE_COLUMNS:
        mean_value = cleaned[column].mean()
        std_value = cleaned[column].std()
        upper_bound = mean_value + 3 * std_value
        lower_bound = mean_value - 3 * std_value
        median_value = cleaned[column].median()
        outlier_mask = (cleaned[column] > upper_bound) | (cleaned[column] < lower_bound)
        outlier_counts[column] = int(outlier_mask.sum())
        if outlier_counts[column]:
            cleaned.loc[outlier_mask, column] = median_value

    for column in FEATURE_COLUMNS:
        median_value = cleaned[column].median()
        nonpositive_mask = cleaned[column] <= 0
        nonpositive_counts[column] = int(nonpositive_mask.sum())
        if nonpositive_counts[column]:
            cleaned.loc[nonpositive_mask, column] = median_value

    metrics = {}
    for column in FEATURE_COLUMNS:
        metrics[f'filled_missing_{column}'] = fill_counts[column]
        metrics[f'replaced_outliers_{column}'] = outlier_counts[column]
        metrics[f'replaced_nonpositive_{column}'] = nonpositive_counts[column]
    return cleaned, metrics


def normalize_processed_frame(frame: pd.DataFrame) -> pd.DataFrame:
    trimmed = frame.loc[:, FEATURE_COLUMNS].copy()
    return trimmed


def export_frame(
    frame: pd.DataFrame,
    sample_code: str,
    sample_name: str,
    input_name: str,
    output_dir: Path,
    mode: str,
    metrics: dict[str, int] | None = None,
) -> dict[str, object]:
    output_path = output_dir / f'{sample_code}_features.csv'
    frame.to_csv(output_path, index=False, encoding='utf-8-sig')
    row: dict[str, object] = {
        'input_mode': mode,
        'input_file': input_name,
        'sample_code': sample_code,
        'sample_name': sample_name,
        'output_file': output_path.name,
        'n_rows': len(frame),
    }
    if metrics:
        row.update(metrics)
    print(f'Wrote {output_path.name} for {sample_name} ({sample_code}) with {len(frame)} rows [{mode}].')
    return row


def prepare_from_frozen_processed(input_dir: Path, output_dir: Path) -> list[dict[str, object]]:
    manifest_rows = []
    for input_path in [input_dir / name for name in FROZEN_PROCESSED_MAP if (input_dir / name).exists()]:
        sample_code = FROZEN_PROCESSED_MAP[input_path.name]
        sample_name = SAMPLE_NAME_MAP[sample_code]
        frame = normalize_processed_frame(safe_read_csv(input_path))
        validate_feature_columns(frame, FEATURE_COLUMNS)
        manifest_rows.append(
            export_frame(
                frame=frame,
                sample_code=sample_code,
                sample_name=sample_name,
                input_name=input_path.name,
                output_dir=output_dir,
                mode='frozen_cleaned',
            )
        )
    return manifest_rows


def prepare_from_raw_splits(input_dir: Path, output_dir: Path) -> list[dict[str, object]]:
    manifest_rows = []
    for split_path in collect_split_files(input_dir):
        frame = safe_read_csv(split_path)
        validate_feature_columns(frame, FEATURE_COLUMNS)
        sample_code = infer_sample_code(frame)
        sample_name = SAMPLE_NAME_MAP.get(sample_code, sample_code)
        cleaned, metrics = clean_raw_feature_table(frame)
        manifest_rows.append(
            export_frame(
                frame=cleaned,
                sample_code=sample_code,
                sample_name=sample_name,
                input_name=split_path.name,
                output_dir=output_dir,
                mode='raw_cleaned',
                metrics=metrics,
            )
        )
    return manifest_rows


def main() -> None:
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)

    processed_files = collect_processed_files(args.input_dir)
    split_files = collect_split_files(args.input_dir)

    if processed_files:
        manifest_rows = prepare_from_frozen_processed(args.input_dir, args.output_dir)
    elif split_files:
        manifest_rows = prepare_from_raw_splits(args.input_dir, args.output_dir)
    else:
        raise FileNotFoundError(
            f'No processed_data*.csv or split_*.csv files found in {args.input_dir}'
        )

    manifest = pd.DataFrame(manifest_rows)
    manifest.to_csv(args.output_dir / 'clustering_input_manifest.csv', index=False, encoding='utf-8-sig')
    print(f'Prepared {len(manifest_rows)} clustering input tables in {args.output_dir}')


if __name__ == '__main__':
    main()
