from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_DATA_PATH = PROJECT_ROOT / "data" / "raw" / "modeling" / "merged_features.xlsx"
DEFAULT_OUTPUT_PATH = (
    PROJECT_ROOT / "data" / "processed" / "correlations" / "high_correlation_pairs.csv"
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Scan the modeling table and export all feature pairs above a Pearson threshold."
    )
    parser.add_argument(
        "--data",
        type=Path,
        default=DEFAULT_DATA_PATH,
        help="Path to the modeling spreadsheet.",
    )
    parser.add_argument(
        "--threshold",
        type=float,
        default=0.8,
        help="Absolute Pearson-correlation threshold used to keep a pair.",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=DEFAULT_OUTPUT_PATH,
        help="CSV path for the exported pair list.",
    )
    return parser.parse_args()


def correlation_band(value: float) -> str:
    if value >= 0.95:
        return "critical"
    if value >= 0.90:
        return "high"
    return "moderate"


def build_pair_table(df: pd.DataFrame, threshold: float) -> pd.DataFrame:
    corr_matrix = df.corr(numeric_only=True)
    columns = corr_matrix.columns.tolist()
    rows: list[dict[str, float | str]] = []

    for i, left in enumerate(columns):
        for right in columns[i + 1 :]:
            r_value = corr_matrix.loc[left, right]
            if pd.isna(r_value) or abs(r_value) < threshold:
                continue
            rows.append(
                {
                    "feature_1": left,
                    "feature_2": right,
                    "pearson_r": float(r_value),
                    "abs_pearson_r": float(abs(r_value)),
                    "band": correlation_band(abs(r_value)),
                }
            )

    pair_table = pd.DataFrame(rows)
    if pair_table.empty:
        return pair_table
    return pair_table.sort_values("abs_pearson_r", ascending=False, ignore_index=True)


def main() -> None:
    args = parse_args()
    data = pd.read_excel(args.data)
    pair_table = build_pair_table(data, threshold=args.threshold)

    args.output.parent.mkdir(parents=True, exist_ok=True)
    pair_table.to_csv(args.output, index=False, encoding="utf-8-sig")

    print(f"Wrote {len(pair_table)} high-correlation pairs to {args.output}")
    if pair_table.empty:
        return

    print()
    print(pair_table.head(20).to_string(index=False))

    print()
    print("Counts by band:")
    print(pair_table["band"].value_counts().to_string())


if __name__ == "__main__":
    main()
