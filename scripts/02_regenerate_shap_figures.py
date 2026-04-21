from __future__ import annotations

import warnings
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import shap
import seaborn as sns

PROJECT_ROOT = Path(__file__).resolve().parents[1]
BASE_DIR = PROJECT_ROOT / "results" / "shap_ablation_outputs"

warnings.filterwarnings("ignore")
plt.rcParams["font.family"] = "Times New Roman"
plt.rcParams["axes.unicode_minus"] = False
sns.set_theme(style="whitegrid")


def discover_targets(base_dir: Path) -> list[str]:
    targets = []
    for path in sorted(base_dir.glob("*_ablation")):
        export_dir = path / "shap_exports"
        if (export_dir / "shap_values.csv").exists() and (export_dir / "X_explain_original_units.csv").exists():
            targets.append(path.name.replace("_ablation", ""))
    return targets


def main() -> None:
    if not BASE_DIR.exists():
        raise FileNotFoundError(
            f"Missing saved SHAP directory: {BASE_DIR}. Restore shap_ablation_outputs first."
        )

    targets = discover_targets(BASE_DIR)
    if not targets:
        raise FileNotFoundError(f"No *_ablation SHAP exports found under {BASE_DIR}")

    print("Start regenerating figures from saved SHAP exports...")
    for target in targets:
        target_dir = BASE_DIR / f"{target}_ablation"
        export_dir = target_dir / "shap_exports"
        output_dir = target_dir / "regenerated_plots"
        output_dir.mkdir(parents=True, exist_ok=True)

        print(f"\nProcessing {target}...")
        try:
            x_explain_orig = pd.read_csv(export_dir / "X_explain_original_units.csv")
            shap_values_df = pd.read_csv(export_dir / "shap_values.csv")
            base_values_df = pd.read_csv(export_dir / "shap_base_values.csv")

            features = [column for column in x_explain_orig.columns if column != "row_index"]
            x_orig = x_explain_orig[features].to_numpy()
            shap_vals = shap_values_df[features].to_numpy()
            base_vals = base_values_df["base_value"].to_numpy()

            print(f"  Data: {x_orig.shape[0]} samples, {len(features)} features")

            plt.figure(figsize=(10, 6))
            shap.summary_plot(shap_vals, x_orig, feature_names=features, show=False)
            plt.title(f"SHAP Summary for {target} (Ablation)")
            plt.tight_layout()
            plt.savefig(output_dir / "SHAP_Summary.pdf", format="pdf", bbox_inches="tight")
            plt.savefig(output_dir / "SHAP_Summary.png", dpi=300)
            plt.close()

            plt.figure(figsize=(10, 6))
            shap.summary_plot(shap_vals, x_orig, feature_names=features, plot_type="bar", show=False)
            plt.title(f"SHAP Summary (Bar) for {target} (Ablation)")
            plt.tight_layout()
            plt.savefig(output_dir / "SHAP_Summary_Bar.pdf", format="pdf", bbox_inches="tight")
            plt.savefig(output_dir / "SHAP_Summary_Bar.png", dpi=300)
            plt.close()

            feature_importance = np.abs(shap_vals).mean(axis=0)
            top_features = list(pd.Series(features)[np.argsort(feature_importance)[::-1][:3]])
            for feature in top_features:
                feature_idx = features.index(feature)
                plt.figure(figsize=(8, 6))
                shap.dependence_plot(
                    feature_idx,
                    shap_vals,
                    x_orig,
                    feature_names=features,
                    interaction_index="auto",
                    show=False,
                )
                plt.title(f"SHAP Dependence: {target} (Ablation) - {feature}")
                plt.tight_layout()
                plt.savefig(output_dir / f"SHAP_Dependency_{feature}.pdf", format="pdf", bbox_inches="tight")
                plt.savefig(output_dir / f"SHAP_Dependency_{feature}.png", dpi=300)
                plt.close()

            try:
                sample_idx = 0
                base_val = base_vals[sample_idx] if len(base_vals) > sample_idx else base_vals[0]
                explanation = shap.Explanation(
                    values=shap_vals[sample_idx, :],
                    base_values=base_val,
                    data=x_orig[sample_idx, :],
                    feature_names=features,
                )
                plt.figure(figsize=(8, 6))
                shap.plots.waterfall(explanation, show=False)
                plt.title(f"SHAP Waterfall: {target} (Ablation)")
                plt.tight_layout()
                plt.savefig(output_dir / "SHAP_Waterfall.pdf", format="pdf", bbox_inches="tight")
                plt.savefig(output_dir / "SHAP_Waterfall.png", dpi=300)
                plt.close()
            except Exception as exc:
                print(f"  Waterfall error: {exc}")

            pd.DataFrame(
                {
                    "feature": features,
                    "mean_abs_shap": feature_importance,
                }
            ).sort_values("mean_abs_shap", ascending=False).to_csv(
                output_dir / "feature_importance.csv",
                index=False,
                encoding="utf-8-sig",
            )
            print(f"  Done, outputs saved to {output_dir}")
        except Exception as exc:
            print(f"  Error: {exc}")

    print("\n" + "=" * 60)
    print("All SHAP figures regenerated")
    print("=" * 60)


if __name__ == "__main__":
    main()
