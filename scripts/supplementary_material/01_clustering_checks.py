from pathlib import Path
import itertools
import os

os.environ.setdefault("OMP_NUM_THREADS", "1")

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy import stats
from sklearn.cluster import AgglomerativeClustering, KMeans, SpectralClustering
from sklearn.metrics import adjusted_rand_score, silhouette_score
from sklearn.mixture import GaussianMixture
from sklearn.preprocessing import QuantileTransformer, StandardScaler


PROJECT_ROOT = Path(__file__).resolve().parents[2]


def resolve_cluster_results_dir():
    candidates = [
        PROJECT_ROOT / "data" / "processed" / "clusters",
        PROJECT_ROOT / "backup" / "release_prep_20260413" / "data" / "processed" / "clusters",
    ]
    for path in candidates:
        if path.exists():
            return path
    return candidates[0]


DATA_DIR = resolve_cluster_results_dir()
OUT_DIR = PROJECT_ROOT / "results" / "supplementary_material" / "clustering_checks"
OUT_DIR.mkdir(parents=True, exist_ok=True)

FEATURES = ["Hardness", "Modulus", "GC", "KIC"]
SAMPLES = {
    "C0": {"code": "00", "file": "00z.csv"},
    "C1": {"code": "01", "file": "01z.csv"},
    "C0-R": {"code": "10", "file": "10z.csv"},
    "C1-R": {"code": "11", "file": "11z.csv"},
}
CLUSTER_COLORS = {
    0: "#BBDCEB",  # light blue
    1: "#BFE3A2",  # light green
    2: "#F3A0A0",  # soft red
    3: "#F4C27A",  # soft orange
    4: "#D7C6E6",  # light purple
    5: "#FFF7A8",  # pale yellow
    6: "#B87545",  # brown-orange
    7: "#4F9A46",  # dark green
}


def read_csv_any(path):
    for encoding in ("GBK", "utf-8-sig", "utf-8"):
        try:
            return pd.read_csv(path, encoding=encoding)
        except Exception:
            continue
    return pd.read_csv(path)



def load_cluster_result(path, specimen, code):
    df = read_csv_any(path)
    df = df.loc[:, ~df.columns.str.startswith("Unnamed")]
    df = df.loc[:, ~df.columns.duplicated()]
    df = df.rename(columns={"Gc": "GC", "Kic": "KIC"})

    if not {"XX", "YY"}.issubset(df.columns):
        fallback = (
            PROJECT_ROOT
            / "results"
            / "supplementary_material"
            / "final_release"
            / "clustering"
            / f"DataS_cluster_map_coords_{code}_{specimen}.csv"
        )
        if fallback.exists():
            df = read_csv_any(fallback)
            df = df.loc[:, ~df.columns.str.startswith("Unnamed")]
            df = df.loc[:, ~df.columns.duplicated()]
            df = df.rename(columns={"Gc": "GC", "Kic": "KIC"})

    numeric_cols = [col for col in FEATURES + ["Cluster", "XX", "YY"] if col in df.columns]
    for col in numeric_cols:
        df[col] = pd.to_numeric(df[col], errors="coerce")
    return df.dropna(subset=FEATURES + ["Cluster"]).reset_index(drop=True)

def preprocess_features(df):
    x = df[FEATURES].copy()
    for col in FEATURES:
        if not np.allclose(x[col].values, x[col].values[0]):
            x[col], _ = stats.boxcox(x[col] + 1)
    qt = QuantileTransformer(
        output_distribution="normal",
        n_quantiles=min(len(x), 1000),
        random_state=42,
    )
    x = pd.DataFrame(qt.fit_transform(x[FEATURES]), columns=FEATURES)
    return pd.DataFrame(StandardScaler().fit_transform(x), columns=FEATURES)


def safe_silhouette(x, labels):
    labels = np.asarray(labels)
    if len(np.unique(labels)) < 2 or len(np.unique(labels)) >= len(labels):
        return np.nan
    return silhouette_score(x, labels)


def add_grid_corrected_plot_coords(df):
    corrected = df.copy()
    corrected["XX_plot"] = np.nan
    corrected["YY_plot"] = np.nan

    if len(corrected) != 70:
        corrected["XX_plot"] = corrected["XX"]
        corrected["YY_plot"] = corrected["YY"]
        return corrected

    for start in (0, 35):
        block = corrected.iloc[start:start + 35].copy()

        # Each block is acquired as 7 columns × 5 rows with a serpentine path.
        x_centers = []
        row_groups = {i: [] for i in range(5)}

        for col in range(7):
            seg = block.iloc[col * 5:(col + 1) * 5]
            x_center = float(np.median(seg["XX"].values))
            x_centers.append(x_center)

            y_vals = seg["YY"].values
            if col % 2 == 0:
                y_ordered = y_vals
            else:
                y_ordered = y_vals[::-1]
            for row_idx, y in enumerate(y_ordered):
                row_groups[row_idx].append(float(y))

        y_centers = [float(np.median(row_groups[row_idx])) for row_idx in range(5)]

        for col in range(7):
            seg_idx = block.index[col * 5:(col + 1) * 5]
            if col % 2 == 0:
                row_order = [0, 1, 2, 3, 4]
            else:
                row_order = [4, 3, 2, 1, 0]
            for pos_in_seg, row_idx in enumerate(row_order):
                corrected.loc[seg_idx[pos_in_seg], "XX_plot"] = x_centers[col]
                corrected.loc[seg_idx[pos_in_seg], "YY_plot"] = y_centers[row_idx]

    return corrected


def plot_cluster_map(df, specimen, out_path):
    plt.rcParams.update({
        "font.family": "Times New Roman",
        "font.size": 10,
        "axes.titlesize": 12,
        "axes.labelsize": 11,
        "xtick.labelsize": 10,
        "ytick.labelsize": 10,
    })
    valid = df.dropna(subset=["XX_plot", "YY_plot", "Cluster"]).copy()
    labels = valid["Cluster"].astype(int).values
    uniq = np.sort(np.unique(labels))
    fallback = plt.cm.tab20(np.linspace(0, 1, max(8, len(uniq))))
    color_dict = {
        lab: CLUSTER_COLORS.get(int(lab), fallback[i % len(fallback)])
        for i, lab in enumerate(uniq)
    }

    fig, ax = plt.subplots(figsize=(6.6, 4.8))
    for _, row in valid.iterrows():
        lab = int(row["Cluster"])
        rect = plt.Rectangle(
            (row["XX_plot"] - 2.5, row["YY_plot"] - 2.5),
            5,
            5,
            facecolor=color_dict[lab],
            edgecolor="k",
            linewidth=0.7,
            alpha=0.88,
        )
        ax.add_patch(rect)
        ax.text(row["XX_plot"], row["YY_plot"], str(lab), fontsize=7, ha="center", va="center", color="black")

    ax.set_title(f"{specimen} cluster location map", fontsize=12)
    ax.set_xlabel("Measured X coordinate (um)", fontsize=11)
    ax.set_ylabel("Measured Y coordinate (um)", fontsize=11)
    ax.set_aspect("equal")
    ax.set_xlim(valid["XX_plot"].min() - 5, valid["XX_plot"].max() + 5)
    ax.set_ylim(valid["YY_plot"].min() - 5, valid["YY_plot"].max() + 5)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.tick_params(direction="out", length=3, width=0.8)
    handles = [
        plt.Line2D([0], [0], marker="s", color="w", markerfacecolor=color_dict[lab], markeredgecolor="k", markersize=8, label=f"Cluster {lab}")
        for lab in uniq
    ]
    ax.legend(
        handles=handles,
        title="Cluster",
        loc="upper left",
        bbox_to_anchor=(1.02, 1.0),
        fontsize=8,
        title_fontsize=9,
        frameon=False,
        borderaxespad=0.0,
    )
    fig.tight_layout()
    fig.savefig(out_path, dpi=600, bbox_inches="tight")
    plt.close(fig)


def plot_combined_cluster_maps(data_by_specimen, out_path):
    plt.rcParams.update({
        "font.family": "Times New Roman",
        "font.size": 9,
        "axes.titlesize": 11,
        "axes.labelsize": 9,
        "xtick.labelsize": 8,
        "ytick.labelsize": 8,
    })
    fig, axes = plt.subplots(2, 2, figsize=(9.2, 4.9))
    all_clusters = sorted(
        set(
            int(v)
            for df in data_by_specimen.values()
            for v in df["Cluster"].dropna().astype(int).unique()
        )
    )
    fallback = plt.cm.tab20(np.linspace(0, 1, max(8, len(all_clusters))))
    color_dict = {
        lab: CLUSTER_COLORS.get(int(lab), fallback[i % len(fallback)])
        for i, lab in enumerate(all_clusters)
    }

    for ax, (specimen, df) in zip(axes.ravel(), data_by_specimen.items()):
        valid = df.dropna(subset=["XX_plot", "YY_plot", "Cluster"]).copy()
        for _, row in valid.iterrows():
            lab = int(row["Cluster"])
            rect = plt.Rectangle(
                (row["XX_plot"] - 2.5, row["YY_plot"] - 2.5),
                5,
                5,
                facecolor=color_dict[lab],
                edgecolor="k",
                linewidth=0.55,
                alpha=0.88,
            )
            ax.add_patch(rect)
            ax.text(row["XX_plot"], row["YY_plot"], str(lab), fontsize=5.8, ha="center", va="center", color="black")
        ax.set_title(specimen, fontsize=12)
        ax.set_xlabel("Measured X coordinate (um)", fontsize=10)
        ax.set_ylabel("Measured Y coordinate (um)", fontsize=10)
        ax.set_aspect("equal")
        ax.set_xlim(valid["XX_plot"].min() - 5, valid["XX_plot"].max() + 5)
        ax.set_ylim(valid["YY_plot"].min() - 5, valid["YY_plot"].max() + 5)
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)
        ax.tick_params(direction="out", labelsize=8, length=3, width=0.8)

    handles = [
        plt.Line2D([0], [0], marker="s", color="w", markerfacecolor=color_dict[lab], markeredgecolor="k", markersize=8, label=f"Cluster {lab}")
        for lab in all_clusters
    ]
    fig.legend(handles=handles, title="Cluster", loc="center right", bbox_to_anchor=(1.02, 0.5), frameon=False, fontsize=8, title_fontsize=9)
    fig.tight_layout(rect=(0, 0, 0.9, 1))
    fig.savefig(out_path, dpi=600, bbox_inches="tight")
    plt.close(fig)


all_data = []
data_by_specimen = {}
cluster_count_rows = []
silhouette_rows = []
stability_rows = []
n_init_rows = []
alternative_rows = []
label_check_rows = []

for specimen, meta in SAMPLES.items():
    df = load_cluster_result(DATA_DIR / meta["file"], specimen, meta["code"])
    df = add_grid_corrected_plot_coords(df)
    df["Specimen"] = specimen
    df["SampleCode"] = meta["code"]
    all_data.append(df)
    data_by_specimen[specimen] = df.copy()
    x = preprocess_features(df)
    existing_labels = df["Cluster"].astype(int).values
    selected_k = len(np.unique(existing_labels))

    counts = df["Cluster"].astype(int).value_counts().sort_index()
    for cluster_id, n in counts.items():
        cluster_count_rows.append(
            {
                "Specimen": specimen,
                "SampleCode": meta["code"],
                "SelectedK": selected_k,
                "Cluster": int(cluster_id),
                "n": int(n),
                "Percent": 100 * int(n) / len(df),
            }
        )

    silhouette_rows.append(
        {
            "Specimen": specimen,
            "SampleCode": meta["code"],
            "K": len(np.unique(existing_labels)),
            "SilhouetteExistingLabels": safe_silhouette(x, existing_labels),
        }
    )

    for k in range(3, 9):
        km_labels = KMeans(n_clusters=k, random_state=42).fit_predict(x)
        silhouette_rows.append(
            {
                "Specimen": specimen,
                "SampleCode": meta["code"],
                "K": k,
                "KMeansSilhouette": safe_silhouette(x, km_labels),
                "SelectedK": k == selected_k,
            }
        )
        if k == selected_k:
            label_check_rows.append(
                {
                    "Specimen": specimen,
                    "SampleCode": meta["code"],
                    "SelectedK": k,
                    "ARI_existing_vs_recomputed_KMeans": adjusted_rand_score(existing_labels, km_labels),
                }
            )

    seed_labels = []
    for seed in range(100):
        labels = KMeans(n_clusters=selected_k, random_state=seed, n_init=1).fit_predict(x)
        seed_labels.append(labels)
        stability_rows.append(
            {
                "Specimen": specimen,
                "SampleCode": meta["code"],
                "K": selected_k,
                "Seed": seed,
                "Silhouette": safe_silhouette(x, labels),
                "PairwiseARI": np.nan,
            }
        )
    ari_values = [adjusted_rand_score(a, b) for a, b in itertools.combinations(seed_labels, 2)]
    stability_rows.append(
        {
            "Specimen": specimen,
            "SampleCode": meta["code"],
            "K": selected_k,
            "Seed": "mean_pairwise_ARI",
            "Silhouette": np.nan,
            "PairwiseARI": float(np.mean(ari_values)),
        }
    )

    for n_init in [1, 2, 5, 10, 20, 50, 100]:
        n_init_seed_labels = []
        for seed in range(50):
            model = KMeans(n_clusters=selected_k, random_state=seed, n_init=n_init)
            labels = model.fit_predict(x)
            n_init_seed_labels.append(labels)
            n_init_rows.append(
                {
                    "Specimen": specimen,
                    "SampleCode": meta["code"],
                    "K": selected_k,
                    "n_init": n_init,
                    "Seed": seed,
                    "Silhouette": safe_silhouette(x, labels),
                    "Inertia": float(model.inertia_),
                    "PairwiseARI": np.nan,
                    "MetricRow": "seed",
                }
            )
        n_init_ari_values = [
            adjusted_rand_score(a, b)
            for a, b in itertools.combinations(n_init_seed_labels, 2)
        ]
        n_init_rows.append(
            {
                "Specimen": specimen,
                "SampleCode": meta["code"],
                "K": selected_k,
                "n_init": n_init,
                "Seed": "mean_pairwise_ARI",
                "Silhouette": np.nan,
                "Inertia": np.nan,
                "PairwiseARI": float(np.mean(n_init_ari_values)),
                "MetricRow": "summary",
            }
        )

    for k in range(3, 9):
        method_labels = {
            "K-means": KMeans(n_clusters=k, random_state=42).fit_predict(x),
            "Agglomerative": AgglomerativeClustering(n_clusters=k, linkage="ward").fit_predict(x),
            "Spectral": SpectralClustering(
                n_clusters=k,
                random_state=42,
                assign_labels="kmeans",
                affinity="nearest_neighbors",
                n_neighbors=min(10, len(x) - 1),
            ).fit_predict(x),
            "GMM": GaussianMixture(n_components=k, random_state=42, n_init=10).fit_predict(x),
        }
        for method, labels in method_labels.items():
            alternative_rows.append(
                {
                    "Specimen": specimen,
                    "SampleCode": meta["code"],
                    "Method": method,
                    "K": k,
                    "Silhouette": safe_silhouette(x, labels),
                    "SelectedK": k == selected_k,
                }
            )

    plot_cluster_map(df, specimen, OUT_DIR / f"FigS_cluster_location_map_{meta['code']}_{specimen}.png")
    df.to_csv(OUT_DIR / f"DataS_cluster_map_coords_{meta['code']}_{specimen}.csv", index=False, encoding="GBK")

plot_combined_cluster_maps(data_by_specimen, OUT_DIR / "FigS_cluster_location_maps_combined.png")

cluster_counts = pd.DataFrame(cluster_count_rows)
silhouette = pd.DataFrame(silhouette_rows)
stability = pd.DataFrame(stability_rows)
n_init_df = pd.DataFrame(n_init_rows)
alternatives = pd.DataFrame(alternative_rows)
label_check = pd.DataFrame(label_check_rows)
combined = pd.concat(all_data, ignore_index=True)

stability_summary = (
    stability[pd.to_numeric(stability["Seed"], errors="coerce").notna()]
    .groupby(["Specimen", "SampleCode", "K"], as_index=False)
    .agg(SilhouetteMean=("Silhouette", "mean"), SilhouetteSD=("Silhouette", "std"))
)
ari_summary = stability[stability["Seed"].eq("mean_pairwise_ARI")][
    ["Specimen", "SampleCode", "K", "PairwiseARI"]
]
stability_summary = stability_summary.merge(ari_summary, on=["Specimen", "SampleCode", "K"], how="left")

n_init_summary = (
    n_init_df[n_init_df["MetricRow"].eq("seed")]
    .groupby(["Specimen", "SampleCode", "K", "n_init"], as_index=False)
    .agg(
        SilhouetteMean=("Silhouette", "mean"),
        SilhouetteSD=("Silhouette", "std"),
        InertiaMean=("Inertia", "mean"),
        InertiaSD=("Inertia", "std"),
    )
)
n_init_ari_summary = n_init_df[n_init_df["MetricRow"].eq("summary")][
    ["Specimen", "SampleCode", "K", "n_init", "PairwiseARI"]
]
n_init_summary = n_init_summary.merge(
    n_init_ari_summary,
    on=["Specimen", "SampleCode", "K", "n_init"],
    how="left",
)

cluster_stats = (
    combined.groupby(["Specimen", "SampleCode", "Cluster"])[FEATURES]
    .agg(["mean", "std"])
    .reset_index()
)

cluster_counts.to_csv(OUT_DIR / "TableS_cluster_population_counts.csv", index=False, encoding="GBK")
cluster_stats.to_csv(OUT_DIR / "TableS_cluster_feature_mean_std.csv", index=False, encoding="GBK")
silhouette.to_csv(OUT_DIR / "TableS_silhouette_existing_and_kmeans_curve.csv", index=False, encoding="GBK")
stability_summary.to_csv(OUT_DIR / "TableS_kmeans_seed_stability_summary.csv", index=False, encoding="GBK")
n_init_df.to_csv(OUT_DIR / "TableS_kmeans_n_init_sensitivity_raw.csv", index=False, encoding="GBK")
n_init_summary.to_csv(OUT_DIR / "TableS_kmeans_n_init_sensitivity_summary.csv", index=False, encoding="GBK")
alternatives.to_csv(OUT_DIR / "TableS_alternative_clustering_comparison.csv", index=False, encoding="GBK")
label_check.to_csv(OUT_DIR / "TableS_label_check_existing_vs_recomputed.csv", index=False, encoding="GBK")

plt.rcParams["font.family"] = "Times New Roman"

fig, axes = plt.subplots(2, 2, figsize=(7.2, 5.6), sharey=True)
for ax, specimen in zip(axes.ravel(), SAMPLES.keys()):
    group = cluster_counts[cluster_counts["Specimen"].eq(specimen)]
    ax.bar(group["Cluster"].astype(str), group["n"], color="#4C78A8")
    ax.set_title(specimen)
    ax.set_xlabel("Cluster")
    ax.set_ylabel("Number of data points")
    for x_pos, n in enumerate(group["n"]):
        ax.text(x_pos, n + 0.5, str(int(n)), ha="center", va="bottom", fontsize=8)
    ax.grid(False)
fig.tight_layout()
fig.savefig(OUT_DIR / "FigS_cluster_population_counts.png", dpi=600)
plt.close(fig)

km_curve = silhouette.dropna(subset=["KMeansSilhouette"]).copy()
fig, ax = plt.subplots(figsize=(6.8, 4.8))
for specimen, group in km_curve.groupby("Specimen"):
    ax.plot(group["K"], group["KMeansSilhouette"], marker="o", linewidth=1.8, label=specimen)
ax.set_xlabel("Number of clusters K")
ax.set_ylabel("Silhouette coefficient")
ax.set_xticks(list(range(3, 9)))
ax.legend(frameon=False)
ax.grid(False)
fig.tight_layout()
fig.savefig(OUT_DIR / "FigS_kmeans_silhouette_curve.png", dpi=600)
plt.close(fig)

selected_alt = alternatives[alternatives["SelectedK"]].copy()
fig, ax = plt.subplots(figsize=(7.2, 4.8))
methods = ["K-means", "Agglomerative", "Spectral", "GMM"]
specimens = list(SAMPLES.keys())
width = 0.18
xpos = np.arange(len(specimens))
for i, method in enumerate(methods):
    vals = [
        selected_alt[(selected_alt["Specimen"].eq(s)) & (selected_alt["Method"].eq(method))]["Silhouette"].iloc[0]
        for s in specimens
    ]
    ax.bar(xpos + (i - 1.5) * width, vals, width=width, label=method)
ax.set_xticks(xpos)
ax.set_xticklabels(specimens)
ax.set_ylabel("Silhouette coefficient at selected K")
ax.legend(frameon=False, ncols=2)
ax.grid(False)
fig.tight_layout()
fig.savefig(OUT_DIR / "FigS_alternative_clustering_selected_k.png", dpi=600)
plt.close(fig)

fig, ax = plt.subplots(figsize=(6.8, 4.8))
ax.bar(stability_summary["Specimen"], stability_summary["PairwiseARI"], color="#4C78A8")
ax.set_ylabel("Mean pairwise ARI across 100 seeds")
ax.set_ylim(0, 1.05)
ax.grid(False)
fig.tight_layout()
fig.savefig(OUT_DIR / "FigS_kmeans_seed_stability_ari.png", dpi=600)
plt.close(fig)

fig, ax = plt.subplots(figsize=(5.0, 4.0))
style_map = {
    "C0": {"color": "#4D4D4D", "marker": "o"},
    "C1": {"color": "#FF3B30", "marker": "o"},
    "C0-R": {"color": "#0066FF", "marker": "o"},
    "C1-R": {"color": "#17A65B", "marker": "o"},
}
for specimen in ["C0", "C1", "C0-R", "C1-R"]:
    group = n_init_summary[n_init_summary["Specimen"].eq(specimen)]
    ax.plot(
        group["n_init"],
        group["PairwiseARI"],
        marker=style_map[specimen]["marker"],
        markersize=7,
        markerfacecolor="white",
        markeredgewidth=1.2,
        color=style_map[specimen]["color"],
        linewidth=1.6,
        label=specimen,
    )
ax.set_xscale("log")
ax.set_xticks([1, 2, 5, 10, 20, 50, 100])
ax.get_xaxis().set_major_formatter(plt.ScalarFormatter())
ax.set_xlabel("n_init setting", fontsize=18)
ax.set_ylabel("Mean pairwise ARI", fontsize=14)
ax.set_ylim(0, 1.05)
ax.tick_params(axis="both", which="major", labelsize=12, direction="in", length=4, width=1.2)
ax.tick_params(axis="both", which="minor", direction="in", length=2.5, width=1.0)
for spine in ax.spines.values():
    spine.set_linewidth(1.4)
    spine.set_color("black")
ax.legend(frameon=False, fontsize=11, loc="lower right")
ax.grid(False)
fig.tight_layout()
fig.savefig(OUT_DIR / "FigS_kmeans_n_init_sensitivity_ari.png", dpi=600)
plt.close(fig)

fig, axes = plt.subplots(1, 2, figsize=(10.0, 4.2))
for specimen, group in n_init_summary.groupby("Specimen"):
    axes[0].errorbar(group["n_init"], group["SilhouetteMean"], yerr=group["SilhouetteSD"], marker="o", linewidth=1.5, capsize=3, label=specimen)
    axes[1].errorbar(group["n_init"], group["InertiaMean"], yerr=group["InertiaSD"], marker="o", linewidth=1.5, capsize=3, label=specimen)
for ax in axes:
    ax.set_xscale("log")
    ax.set_xticks([1, 2, 5, 10, 20, 50, 100])
    ax.get_xaxis().set_major_formatter(plt.ScalarFormatter())
    ax.set_xlabel("n_init setting", fontsize=16)
    ax.tick_params(axis="both", labelsize=14, direction="out", length=4, width=1.0)
    ax.grid(False)
axes[0].set_ylabel("Silhouette coefficient", fontsize=16)
axes[1].set_ylabel("Inertia", fontsize=16)
axes[0].legend(frameon=False, fontsize=12)
fig.tight_layout()
fig.savefig(OUT_DIR / "FigS_kmeans_n_init_sensitivity_silhouette_inertia.png", dpi=600)
plt.close(fig)

readme = """Supplementary clustering checks

Data source: current clustered result tables under data/processed/clusters/ (fallback: backup release_prep_20260413).
Sample mapping: 00 = C0, 01 = C1, 10 = C0-R, 11 = C1-R.

Use TableS_cluster_population_counts.csv for the cluster population counts derived from the current clustering outputs.
Use FigS_cluster_population_counts.png if a figure is preferred over a table.
Use FigS_cluster_location_map_*.png for the specimen-wise cluster maps derived from the current clustering outputs.
Use TableS_kmeans_seed_stability_summary.csv and FigS_kmeans_seed_stability_ari.png for K-means initialization stability over 100 random seeds at the currently selected K.
Use TableS_kmeans_n_init_sensitivity_summary.csv, FigS_kmeans_n_init_sensitivity_ari.png, and FigS_kmeans_n_init_sensitivity_silhouette_inertia.png to show whether increasing n_init improves K-means stability.
Use TableS_alternative_clustering_comparison.csv and FigS_alternative_clustering_selected_k.png to compare K-means with agglomerative clustering, spectral clustering, and GMM under the same preprocessing.
Use TableS_silhouette_existing_and_kmeans_curve.csv and FigS_kmeans_silhouette_curve.png to document the currently selected K values and their silhouette curves.
Use TableS_label_check_existing_vs_recomputed.csv only as an internal check, not necessarily for the manuscript.
"""
(OUT_DIR / "README.txt").write_text(readme, encoding="utf-8")

print("Saved outputs to:", OUT_DIR)
print("\nCluster population counts:")
print(cluster_counts.to_string(index=False))
print("\nK-means seed stability summary:")
print(stability_summary.to_string(index=False))
print("\nK-means n_init sensitivity summary:")
print(n_init_summary.to_string(index=False))
print("\nAlternative clustering at selected K:")
print(selected_alt.to_string(index=False))
print("\nExisting labels vs recomputed K-means ARI:")
print(label_check.to_string(index=False))


