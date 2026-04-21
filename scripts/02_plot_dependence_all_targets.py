from __future__ import annotations

import warnings
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd

from _selected_feature_io import read_selected_features

PROJECT_ROOT = Path(__file__).resolve().parents[1]
BASE_DIR = PROJECT_ROOT / "results" / "cv" / "shap_exports"
OUTPUT_BASE = PROJECT_ROOT / "results" / "legacy_shap_visualizations" / "shap_dependence_plots"

warnings.filterwarnings("ignore")
plt.rcParams["font.family"] = "Times New Roman"
plt.rcParams["axes.unicode_minus"] = False


def resolve_baseline_x_file(base_dir: Path) -> Path:
    candidates = [
        base_dir / "X_explain_original_units.csv",
        base_dir / "新建文件夹" / "X_explain_original_units.csv",
    ]
    for path in candidates:
        if path.exists():
            return path
    raise FileNotFoundError(
        "Missing baseline X_explain_original_units.csv under results/cv/shap_exports/."
    )


def discover_ablation_targets(base_dir: Path, baseline_targets: list[str]) -> list[str]:
    discovered = []
    for path in sorted(base_dir.glob("*_ablation")):
        export_dir = path / "shap_exports"
        if (export_dir / "shap_values.csv").exists():
            discovered.append(path.name.replace("_ablation", ""))
    ordered = [target for target in baseline_targets if target in discovered]
    extras = [target for target in discovered if target not in ordered]
    return ordered + extras


def plot_dependence(
    x_vals,
    shap_vals,
    feat_name: str,
    target_name: str,
    suffix: str,
    output_dir: Path,
) -> None:
    positive_mask = shap_vals > 0
    negative_mask = shap_vals <= 0

    fig, ax = plt.subplots(figsize=(8, 6))
    ax.scatter(
        x_vals[positive_mask],
        shap_vals[positive_mask],
        c="#d62728",
        s=80,
        alpha=0.7,
        label="SHAP > 0",
        edgecolors="none",
    )
    ax.scatter(
        x_vals[negative_mask],
        shap_vals[negative_mask],
        c="#1f77b4",
        s=80,
        alpha=0.7,
        label="SHAP <= 0",
        edgecolors="none",
    )
    ax.axhline(y=0, color="black", linewidth=2, linestyle="-")
    ax.set_xlabel(feat_name, fontsize=24, fontweight="bold")
    ax.set_ylabel(f"SHAP value for {feat_name}", fontsize=24, fontweight="bold")
    ax.set_title(
        f"SHAP Dependence Plot for {target_name}{suffix} - {feat_name}",
        fontsize=28,
        fontweight="bold",
        pad=15,
    )
    ax.legend(fontsize=22, frameon=True, fancybox=True, shadow=True, loc="best", markerscale=1.5, framealpha=0.95)
    ax.grid(False)
    for side in ["top", "right", "bottom", "left"]:
        ax.spines[side].set_linewidth(2)
    ax.tick_params(labelsize=26, width=2, length=8)
    for label in ax.get_xticklabels() + ax.get_yticklabels():
        label.set_fontweight("bold")
    plt.tight_layout()

    safe_feat = feat_name.replace("/", "_").replace(" ", "_")
    safe_target = target_name.replace("-", "_")
    filename = f"{safe_target}{suffix}_{safe_feat}"
    plt.savefig(output_dir / f"{filename}.pdf", format="pdf", bbox_inches="tight", dpi=300)
    plt.savefig(output_dir / f"{filename}.png", dpi=300, bbox_inches="tight")
    plt.close(fig)


def main() -> None:
    if not BASE_DIR.exists():
        raise FileNotFoundError(f"Missing SHAP export directory: {BASE_DIR}")

    OUTPUT_BASE.mkdir(parents=True, exist_ok=True)
    baseline_targets = read_selected_features()
    baseline_output = OUTPUT_BASE / "baseline"
    baseline_output.mkdir(parents=True, exist_ok=True)

    print("=" * 60)
    print("Generate baseline dependence plots")
    print("=" * 60)

    x_file = resolve_baseline_x_file(BASE_DIR)
    x_orig = pd.read_csv(x_file)
    features = [column for column in x_orig.columns if column != "row_index"]
    x_data = x_orig[features]

    for target in baseline_targets:
        print(f"\nProcessing {target} (baseline)...")
        shap_file = BASE_DIR / f"shap_values_{target}.csv"
        if not shap_file.exists():
            print(f"  Skip: missing {shap_file}")
            continue

        shap_data = pd.read_csv(shap_file)
        available = [feature for feature in features if feature in shap_data.columns]
        for feature in available:
            plot_dependence(
                x_data[feature].to_numpy(),
                shap_data[feature].to_numpy(),
                feature,
                target,
                "",
                baseline_output,
            )
        print(f"  Done {target}, generated {len(available)} figures")

    print("\n" + "=" * 60)
    print("Generate ablation dependence plots")
    print("=" * 60)

    ablation_output = OUTPUT_BASE / "ablation"
    ablation_output.mkdir(parents=True, exist_ok=True)
    ablation_targets = discover_ablation_targets(BASE_DIR, baseline_targets)

    for target in ablation_targets:
        print(f"\nProcessing {target} (ablation)...")
        target_dir = BASE_DIR / f"{target}_ablation" / "shap_exports"
        shap_file = target_dir / "shap_values.csv"
        x_file = target_dir / "X_explain_original_units.csv"
        if not shap_file.exists() or not x_file.exists():
            print(f"  Skip: missing SHAP exports for {target}")
            continue

        x_orig = pd.read_csv(x_file)
        shap_data = pd.read_csv(shap_file)
        features = [column for column in x_orig.columns if column != "row_index"]
        available = [feature for feature in features if feature in shap_data.columns]
        for feature in available:
            plot_dependence(
                x_orig[feature].to_numpy(),
                shap_data[feature].to_numpy(),
                feature,
                target,
                "_ablation",
                ablation_output,
            )
        print(f"  Done {target}, generated {len(available)} figures")

    baseline_count = len(list(baseline_output.glob("*.pdf")))
    ablation_count = len(list(ablation_output.glob("*.pdf")))
    print("\n" + "=" * 60)
    print("All figures generated")
    print("=" * 60)
    print(f"Baseline PDFs: {baseline_count}")
    print(f"Ablation PDFs: {ablation_count}")
    print(f"Total PDFs: {baseline_count + ablation_count}")


if __name__ == "__main__":
    main()
