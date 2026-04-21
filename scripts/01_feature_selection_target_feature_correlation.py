from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns

from _selected_feature_io import PROJECT_ROOT, read_selected_features, resolve_default_modeling_path

DATA_PATH = resolve_default_modeling_path('merged_features.xlsx')
OUTPUT_CSV = PROJECT_ROOT / 'data' / 'processed' / 'correlations' / 'target_feature_full_correlation.csv'
OUTPUT_PNG = PROJECT_ROOT / 'data' / 'processed' / 'correlations' / 'target_feature_full_correlation.png'
TARGETS = read_selected_features()


def main() -> None:
    data = pd.read_excel(DATA_PATH)
    targets = [column for column in TARGETS if column in data.columns]
    if not targets:
        raise ValueError('No selected features from the RFE artifact were found in the modeling table.')

    features = [column for column in data.columns if column not in targets]
    corr_matrix = data[targets + features].corr(numeric_only=True).loc[targets, features]

    OUTPUT_CSV.parent.mkdir(parents=True, exist_ok=True)
    corr_matrix.to_csv(OUTPUT_CSV, encoding='utf-8-sig')

    plt.figure(figsize=(14, max(5, len(targets) * 0.7)))
    sns.heatmap(corr_matrix, annot=True, fmt='.2f', cmap='coolwarm', center=0, vmin=-1, vmax=1)
    plt.title('Selected-feature vs remaining-feature correlation matrix')
    plt.ylabel('Selected features')
    plt.xlabel('Remaining features')
    plt.tight_layout()
    plt.savefig(OUTPUT_PNG, dpi=300)
    plt.close()

    print(f'Data: {DATA_PATH}')
    print(f'Selected features: {targets}')
    print(f'Wrote correlation CSV to {OUTPUT_CSV}')
    print(f'Wrote heatmap to {OUTPUT_PNG}')


if __name__ == '__main__':
    main()