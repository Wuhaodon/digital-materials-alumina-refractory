from __future__ import annotations

from pathlib import Path

import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DATA_PATH = PROJECT_ROOT / "data" / "raw" / "modeling" / "merged_features.xlsx"
OUTPUT_PATH = PROJECT_ROOT / "data" / "processed" / "correlations" / "correlation_matrix.csv"

MANUSCRIPT_PAIRS = [
    ("WPP", "Irreversible"),
    ("Gf", "GC"),
    ("Gf", "Hardness"),
    ("Modulus", "Hardness"),
    ("Stiffness", "Hardness"),
    ("hf", "hm"),
    ("hm-hf", "hm"),
    ("hm-hf", "hf"),
]

DISPLAY_ORDER = [
    "Thermal shock",
    "Low carbon",
    "Irreversible",
    "hf",
    "hm",
    "GC",
    "Hardness",
    "WE",
    "hm-hf",
    "Gf",
    "Pmax",
    "WPP",
    "KIC",
    "Stiffness",
    "Modulus",
]


def classify_risk(value: float) -> str:
    score = abs(value)
    if score >= 0.95:
        return "critical"
    if score >= 0.85:
        return "high"
    if score >= 0.70:
        return "moderate"
    return "acceptable"


def main() -> None:
    data = pd.read_excel(DATA_PATH)

    rows: list[dict[str, str | float]] = []
    for left, right in MANUSCRIPT_PAIRS:
        if left not in data.columns or right not in data.columns:
            continue
        r_value = float(data[[left, right]].corr(numeric_only=True).iloc[0, 1])
        rows.append(
            {
                "feature_1": left,
                "feature_2": right,
                "pearson_r": r_value,
                "risk": classify_risk(r_value),
            }
        )

    summary = pd.DataFrame(rows).sort_values(
        by="pearson_r",
        key=lambda col: col.abs(),
        ascending=False,
        ignore_index=True,
    )

    print("Manuscript-relevant pairwise correlations")
    print("-" * 60)
    if summary.empty:
        print("No requested feature pairs were found in the dataset.")
    else:
        print(summary.to_string(index=False, formatters={"pearson_r": "{:.4f}".format}))

    available_columns = [column for column in DISPLAY_ORDER if column in data.columns]
    corr_matrix = data[available_columns].corr(numeric_only=True)
    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    corr_matrix.to_csv(OUTPUT_PATH, encoding="utf-8-sig")

    print()
    print(f"Wrote full correlation matrix to {OUTPUT_PATH}")


if __name__ == "__main__":
    main()
