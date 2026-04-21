from __future__ import annotations

import csv
from pathlib import Path

import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[1]
DATA_ROOT = PROJECT_ROOT / 'data' / 'raw' / 'modeling'
BACKUP_DATA_ROOT = PROJECT_ROOT / 'backup' / 'release_prep_20260413' / 'data' / 'raw' / 'modeling'
SELECTION_ROOT = PROJECT_ROOT / 'results' / 'feature_selection_dual_model_rfe'

GROUP_COL = 'group'
RANDOM_STATE = 42
TEST_SIZE = 0.3

ALL_FEATURES = [
    'Irreversible',
    'hf',
    'hm',
    'hm-hf',
    'Pmax',
    'WPP',
    'GC',
    'KIC',
    'WE',
    'Gf',
    'Hardness',
    'Modulus',
    'Stiffness',
]

THERMAL_SHOCK_COL = 'Thermal shock'
LOW_CARBON_COL = 'Low carbon'
THERMAL_SHOCK_ALIASES = [THERMAL_SHOCK_COL, '热震']
LOW_CARBON_ALIASES = [LOW_CARBON_COL, '低碳']

LEGACY_TO_CANONICAL = {
    'Wt': 'Gf',
    'Wt ': 'Gf',
    'hmax': 'hm',
    'Wpp': 'WPP',
    'Wc': 'GC',
    'Kic': 'KIC',
    'We': 'WE',
    'We ': 'WE',
}

DEFAULT_TABPFN_MODEL_PATH = PROJECT_ROOT / 'external' / 'tabpfn_cache' / 'tabpfn-v2-classifier.ckpt'
DEFAULT_PRELIMINARY_FEATURES_CSV = SELECTION_ROOT / 'selected_features_preliminary.csv'
DEFAULT_FINAL_FEATURES_CSV = SELECTION_ROOT / 'selected_features_final.csv'


def resolve_default_modeling_path(filename: str) -> Path:
    candidates = [
        DATA_ROOT / filename,
        BACKUP_DATA_ROOT / filename,
    ]
    for path in candidates:
        if path.exists():
            return path
    return candidates[0]


def normalise_columns(df: pd.DataFrame) -> pd.DataFrame:
    renamed = {}
    for col in df.columns:
        clean = col.strip() if isinstance(col, str) else col
        renamed[col] = LEGACY_TO_CANONICAL.get(clean, clean)
    return df.rename(columns=renamed)


def resolve_column(df: pd.DataFrame, candidates: list[str], label: str) -> str:
    for candidate in candidates:
        if candidate in df.columns:
            return candidate
    raise KeyError(f'Missing {label} column. Tried: {candidates}')


def create_group(df: pd.DataFrame) -> pd.Series:
    thermal_col = resolve_column(df, THERMAL_SHOCK_ALIASES, 'thermal shock')
    low_carbon_col = resolve_column(df, LOW_CARBON_ALIASES, 'low carbon')
    mapping = {
        '00': 'F2-TP',
        '01': 'E1-TP',
        '10': 'F2-RE',
        '11': 'E1-RE',
    }
    keys = df[thermal_col].astype(int).astype(str) + df[low_carbon_col].astype(int).astype(str)
    return keys.map(mapping).fillna('Unknown')


def load_dataset(data_path: Path) -> pd.DataFrame:
    if not data_path.exists():
        raise FileNotFoundError(f'Dataset not found: {data_path}')

    if data_path.suffix.lower() == '.csv':
        df = pd.read_csv(data_path)
    else:
        df = pd.read_excel(data_path)

    df = normalise_columns(df)
    df[GROUP_COL] = create_group(df)

    required = set(ALL_FEATURES) | {GROUP_COL}
    missing = required.difference(df.columns)
    if missing:
        raise KeyError(
            'Dataset is missing required columns after normalisation: '
            f'{sorted(missing)}'
        )
    return df


def infer_all_feature_order(df: pd.DataFrame) -> list[str]:
    return [col for col in df.columns if col in ALL_FEATURES]


def read_selected_features(path: Path | None = None) -> list[str]:
    target_path = path or DEFAULT_FINAL_FEATURES_CSV
    if not target_path.exists():
        raise FileNotFoundError(
            'Missing selected-feature file. Run scripts/01_feature_selection_dual_model_rfe.py first: '
            f'{target_path}'
        )

    with target_path.open('r', encoding='utf-8-sig', newline='') as handle:
        reader = csv.DictReader(handle)
        if 'feature' not in (reader.fieldnames or []):
            raise ValueError(f'Selected-feature file must include a feature column: {target_path}')
        features = [str(row['feature']).strip() for row in reader if str(row.get('feature', '')).strip()]

    if not features:
        raise ValueError(f'No selected features found in {target_path}')
    return features