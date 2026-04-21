from pathlib import Path
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns


PROJECT_ROOT = Path(__file__).resolve().parents[2]
BASE = PROJECT_ROOT / 'results' / 'supplementary_material' / 'weight_sensitivity'
IN_DIR = BASE / "weight_sensitivity_original_style"
OUT_DIR = IN_DIR / "visual_summary"
OUT_DIR.mkdir(parents=True, exist_ok=True)


def plot_weight_bars():
    df = pd.read_csv(IN_DIR / "weight_schemes_used.csv", encoding="utf-8-sig")
    long_df = df.melt(id_vars=["scheme"], var_name="feature", value_name="weight")
    plt.figure(figsize=(7.2, 4.5))
    sns.barplot(data=long_df, x="feature", y="weight", hue="scheme")
    plt.ylabel("Weight")
    plt.xlabel("Feature")
    plt.title("Weight definitions used in correspondence analysis")
    plt.tight_layout()
    plt.savefig(OUT_DIR / "Fig_weight_scheme_bars.png", dpi=500)
    plt.close()


def plot_agreement_heatmap():
    df = pd.read_csv(IN_DIR / "agreement_summary_vs_shap.csv", encoding="utf-8-sig")
    pivot = df.pivot(index="pair", columns="compare_scheme", values="dominant_mapping_agreement_vs_shap")
    plt.figure(figsize=(5.8, 2.8))
    sns.heatmap(pivot, annot=True, cmap="YlGnBu", vmin=0, vmax=1, fmt=".3f", cbar_kws={"label": "Agreement"})
    plt.xlabel("Compared with SHAP-based weighting")
    plt.ylabel("Specimen pair")
    plt.title("Dominant mapping agreement")
    plt.tight_layout()
    plt.savefig(OUT_DIR / "Fig_agreement_heatmap.png", dpi=500)
    plt.close()


def plot_pair_mapping(pair_name: str):
    df = pd.read_csv(IN_DIR / "dominant_mapping_all_schemes.csv", encoding="utf-8-sig")
    sub = df[df["pair"] == pair_name].copy()
    sub["label"] = sub["dominant_dst_cluster"].astype(str) + "\nP=" + sub["dominant_probability"].map(lambda x: f"{x:.2f}")
    table = sub.pivot(index="src_cluster", columns="scheme", values="dominant_dst_cluster")
    ann = sub.pivot(index="src_cluster", columns="scheme", values="label")

    scheme_order = [c for c in ["shap_based", "uniform", "raw_variance_based"] if c in table.columns]
    table = table[scheme_order]
    ann = ann[scheme_order]

    numeric = table.apply(pd.to_numeric, errors="coerce")
    plt.figure(figsize=(6.0, max(2.6, 0.45 * len(numeric.index) + 1.2)))
    sns.heatmap(numeric, annot=ann, fmt="", cmap="Pastel1", cbar=False, linewidths=0.5, linecolor="white")
    plt.xlabel("Weight scheme")
    plt.ylabel("Source cluster")
    plt.title(f"Dominant correspondence for {pair_name}")
    plt.tight_layout()
    safe_name = pair_name.replace("->", "_to_").replace("-", "")
    plt.savefig(OUT_DIR / f"Fig_mapping_{safe_name}.png", dpi=500)
    plt.close()


def main():
    plot_weight_bars()
    plot_agreement_heatmap()
    for pair in ["C0->C0-R", "C0->C1", "C1->C1-R"]:
        plot_pair_mapping(pair)


if __name__ == "__main__":
    main()
