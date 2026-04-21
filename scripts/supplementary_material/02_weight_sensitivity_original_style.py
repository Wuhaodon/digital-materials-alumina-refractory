from pathlib import Path
import numpy as np
import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parents[2]
BASE_DIR = PROJECT_ROOT / 'results' / 'supplementary_material' / 'weight_sensitivity'
OUT_DIR = BASE_DIR / "weight_sensitivity_original_style"
OUT_DIR.mkdir(parents=True, exist_ok=True)

DATA_DIR = PROJECT_ROOT / "backup" / "release_prep_20260413" / "data" / "processed" / "clusters"

PATHS = {
    "C0": DATA_DIR / "00z.csv",
    "C1": DATA_DIR / "01z.csv",
    "C0-R": DATA_DIR / "10z.csv",
    "C1-R": DATA_DIR / "11z.csv",
}

FEATURES = ["Hardness", "Modulus", "GC", "KIC"]
SHAP_WEIGHTS = {"Hardness": 0.21, "Modulus": 0.18, "GC": 0.47, "KIC": 0.15}
DIST_METHOD = "var"
TAU_PCT = 70.0
TOPK = 3
SHOW_FULL_MATRICES = False
EPS = 1e-6

PAIR_GROUPS = [
    ("C0", "C0-R"),
    ("C0", "C1"),
    ("C1", "C1-R"),
]


def safe_read_csv(path: Path) -> pd.DataFrame:
    for enc in ("utf-8-sig", "utf-8", "gbk", "ansi"):
        try:
            return pd.read_csv(path, encoding=enc)
        except Exception:
            continue
    return pd.read_csv(path)


def load_dataset(path: Path, feature_cols=FEATURES, cluster_col="Cluster") -> pd.DataFrame:
    df = safe_read_csv(path)
    missing = [c for c in (feature_cols + [cluster_col]) if c not in df.columns]
    if missing:
        raise ValueError(f"{path} is missing columns: {missing}. Existing columns: {list(df.columns)}")
    df = df[feature_cols + [cluster_col]].copy()
    df["Cluster"] = df[cluster_col].astype(str)
    return df


def shrink_cov(S, alpha=0.1):
    d = S.shape[0]
    mu = np.trace(S) / max(d, 1)
    return (1 - alpha) * S + alpha * mu * np.eye(d)


def cluster_stats(df: pd.DataFrame, feature_cols=FEATURES):
    groups = df.groupby("Cluster", sort=False)
    labels, n_list, frac_list = [], [], []
    mean_list, std_list, cov_list = [], [], []
    N = len(df)
    for lab, g in groups:
        X = g[feature_cols].to_numpy(float)
        n = len(g)
        mu = X.mean(axis=0)
        sd = X.std(axis=0, ddof=1) if n > 1 else np.zeros_like(mu)
        if n > 1:
            S = np.cov(X, rowvar=False, ddof=1)
        else:
            S = np.zeros((X.shape[1], X.shape[1]))
        S_reg = shrink_cov(S, alpha=0.1)
        labels.append(str(lab))
        n_list.append(n)
        frac_list.append(n / max(N, 1))
        mean_list.append(mu)
        std_list.append(sd)
        cov_list.append(S_reg)
    return {
        "labels": labels,
        "n": np.array(n_list, int),
        "frac": np.array(frac_list, float),
        "mean": np.vstack(mean_list) if mean_list else np.zeros((0, len(feature_cols))),
        "std": np.vstack(std_list) if std_list else np.zeros((0, len(feature_cols))),
        "cov": np.stack(cov_list) if cov_list else np.zeros((0, len(feature_cols), len(feature_cols))),
        "N": N,
    }


def weights_vector(weights_dict, feature_cols=FEATURES):
    w = np.array([weights_dict[c] for c in feature_cols], float)
    s = w.sum()
    return w / s if s > 0 else np.ones_like(w) / len(w)


def cost_matrix(stats_A, stats_B, weights, method=DIST_METHOD, eps=EPS):
    muA, sdA, covA = stats_A["mean"], stats_A["std"], stats_A["cov"]
    muB, sdB, covB = stats_B["mean"], stats_B["std"], stats_B["cov"]
    kA, d = muA.shape
    kB = muB.shape[0]
    w = weights_vector(weights)
    C = np.zeros((kA, kB), float)

    if method == "var":
        for i in range(kA):
            for j in range(kB):
                num = (muA[i] - muB[j]) ** 2
                den = sdA[i] ** 2 + sdB[j] ** 2 + eps
                C[i, j] = float(np.sum(w * (num / den)))
        return C

    if method == "maha":
        L = np.diag(np.sqrt(w + eps))
        I = np.eye(d)
        for i in range(kA):
            for j in range(kB):
                dm = (muA[i] - muB[j]).reshape(-1, 1)
                dm_w = L @ dm
                Sigma = covA[i] + covB[j] + eps * I
                try:
                    inv = np.linalg.inv(Sigma)
                except np.linalg.LinAlgError:
                    inv = np.linalg.pinv(Sigma)
                C[i, j] = float((dm_w.T @ inv @ dm_w).ravel())
        return C

    raise ValueError("method must be 'var' or 'maha'")


def soft_alignment_from_cost(C, tau_percentile=TAU_PCT):
    tau = np.percentile(C, tau_percentile)
    S = np.exp(-C)
    mask = C <= tau
    S_masked = S * mask
    P = np.zeros_like(S_masked)
    for i in range(S.shape[0]):
        row = S_masked[i]
        s = row.sum()
        if s <= 0:
            j = int(np.argmin(C[i]))
            P[i, j] = 1.0
        else:
            P[i] = row / s
    return P, S, tau


def wilson_ci(p, n, z=1.96):
    denom = 1 + z**2 / n
    center = (p + z**2 / (2 * n)) / denom
    pm = z * np.sqrt(p * (1 - p) / n + z**2 / (4 * n**2)) / denom
    return max(0.0, center - pm), min(1.0, center + pm)


def df_to_md_table(df: pd.DataFrame, float_fmt="%.6f") -> str:
    fmt_df = df.copy()
    for c in fmt_df.columns:
        if pd.api.types.is_float_dtype(fmt_df[c]) or pd.api.types.is_integer_dtype(fmt_df[c]):
            fmt_df[c] = fmt_df[c].map(lambda x: float_fmt % x if pd.notnull(x) else "")
    header = "| " + " | ".join(["index"] + list(fmt_df.columns)) + " |"
    sep = "| " + " | ".join(["---"] * (len(fmt_df.columns) + 1)) + " |"
    rows = []
    for idx, row in fmt_df.iterrows():
        rows.append("| " + " | ".join([str(idx)] + [str(v) for v in row.values]) + " |")
    return "\n".join([header, sep] + rows)


def raw_variance_weights(datasets: dict) -> dict:
    combined = pd.concat([df[FEATURES] for df in datasets.values()], ignore_index=True)
    variances = combined.var(ddof=1)
    weights = variances / variances.sum()
    return {feat: float(weights[feat]) for feat in FEATURES}


def expected_shift_table(P, stats_src, stats_dst):
    fr_src = stats_src["frac"]
    exp_rows = []
    for i, s_lab in enumerate(stats_src["labels"]):
        mu_src = stats_src["mean"][i]
        sd_src = stats_src["std"][i]
        mu_exp = np.sum(P[i].reshape(-1, 1) * stats_dst["mean"], axis=0)
        delta = mu_exp - mu_src
        pooled = np.sqrt(sd_src**2 + np.sum(P[i].reshape(-1, 1) * (stats_dst["std"] ** 2), axis=0) + EPS)
        delta_z = delta / pooled
        row = {"src": s_lab, "p_src": fr_src[i]}
        for k, f in enumerate(FEATURES):
            row[f"mu_src_{f}"] = mu_src[k]
            row[f"mu_exp_{f}"] = mu_exp[k]
            row[f"delta_{f}"] = delta[k]
            row[f"delta_z_{f}"] = delta_z[k]
        exp_rows.append(row)
    return pd.DataFrame(exp_rows)


def run_pair(name_src, name_dst, df_src, df_dst, out_file: Path, weights: dict, scheme_name: str,
             dist_method=DIST_METHOD, tau_pct=TAU_PCT, topk=TOPK, show_full=SHOW_FULL_MATRICES):
    stats_src = cluster_stats(df_src, FEATURES)
    stats_dst = cluster_stats(df_dst, FEATURES)

    C = cost_matrix(stats_src, stats_dst, weights, method=dist_method, eps=EPS)
    P, S, tau = soft_alignment_from_cost(C, tau_percentile=tau_pct)

    fr_src, fr_dst = stats_src["frac"], stats_dst["frac"]
    W = fr_src.reshape(-1, 1) * P

    rows_flow = []
    for i, s_lab in enumerate(stats_src["labels"]):
        order = np.argsort(-P[i])
        count = 0
        for j in order:
            if P[i, j] <= 0:
                continue
            rows_flow.append({
                "src": s_lab,
                "dst": stats_dst["labels"][j],
                "p_src": fr_src[i],
                "P_ij": P[i, j],
                "w_ij": W[i, j],
            })
            count += 1
            if count >= topk:
                break
    flow_df = pd.DataFrame(rows_flow)

    exp_df = expected_shift_table(P, stats_src, stats_dst)

    pred_dst_frac = W.sum(axis=0)
    comp_rows = []
    for j, t_lab in enumerate(stats_dst["labels"]):
        comp_rows.append({
            "dst": t_lab,
            "p_dst_true": fr_dst[j],
            "p_dst_pred_from_src": pred_dst_frac[j],
            "delta_p_pred_minus_true": pred_dst_frac[j] - fr_dst[j],
        })
    comp_df = pd.DataFrame(comp_rows)

    cost_df = pd.DataFrame(C, index=stats_src["labels"], columns=stats_dst["labels"])
    P_df = pd.DataFrame(P, index=stats_src["labels"], columns=stats_dst["labels"])

    ci_rows = []
    for i, s_lab in enumerate(stats_src["labels"]):
        lo, hi = wilson_ci(fr_src[i], max(stats_src["N"], 1))
        ci_rows.append({"set": name_src, "cluster": s_lab, "p": fr_src[i], "CI_low": lo, "CI_high": hi, "N": stats_src["N"]})
    for j, t_lab in enumerate(stats_dst["labels"]):
        lo, hi = wilson_ci(fr_dst[j], max(stats_dst["N"], 1))
        ci_rows.append({"set": name_dst, "cluster": t_lab, "p": fr_dst[j], "CI_low": lo, "CI_high": hi, "N": stats_dst["N"]})
    ci_df = pd.DataFrame(ci_rows)

    with open(out_file, "w", encoding="utf-8") as f:
        f.write(f"# Probabilistic alignment report: {name_src} -> {name_dst}\n\n")
        f.write("Method summary:\n")
        f.write(f"- Weight scheme: {scheme_name}\n")
        f.write(f"- Distance: {'variance-weighted (means±SD)' if dist_method == 'var' else 'Mahalanobis (covariance-regularized)'}\n")
        f.write(f"- Features: {FEATURES}\n")
        f.write(f"- Weights: {weights}\n")
        f.write(f"- tau (percentile={tau_pct:.1f}%): {tau:.6f}\n")
        f.write(f"- TOPK: {topk} (per source cluster)\n\n")

        f.write("## 1) Top-K probabilistic correspondences and soft flows\n\n")
        f.write(df_to_md_table(flow_df.set_index(["src", "dst"])))
        f.write("\n\n---\n\n")

        f.write("## 2) Expected feature shifts per source cluster (P-weighted)\n\n")
        f.write(df_to_md_table(exp_df.set_index("src")))
        f.write("\n\n---\n\n")

        f.write("## 3) Target fraction: predicted (from soft flows) vs observed\n\n")
        f.write(df_to_md_table(comp_df.set_index("dst")))
        f.write("\n\n---\n\n")

        if show_full:
            f.write("## A) Full cost matrix C (variance-weighted distance)\n\n")
            f.write(df_to_md_table(cost_df))
            f.write("\n\n---\n\n")
            f.write("## B) Full probabilistic alignment matrix P\n\n")
            f.write(df_to_md_table(P_df))
            f.write("\n\n---\n\n")

        f.write("## C) Fractions with 95% Wilson CI\n\n")
        f.write(df_to_md_table(ci_df.set_index(["set", "cluster"])))
        f.write("\n")

    return {
        "stats_src": stats_src,
        "stats_dst": stats_dst,
        "C": C,
        "P": P,
        "tau": tau,
        "flow_df": flow_df,
        "exp_df": exp_df,
        "comp_df": comp_df,
        "cost_df": cost_df,
        "P_df": P_df,
    }


def dominant_mapping(P_df: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for src in P_df.index:
        dst = P_df.loc[src].idxmax()
        rows.append({
            "src_cluster": str(src),
            "dominant_dst_cluster": str(dst),
            "dominant_probability": float(P_df.loc[src, dst]),
        })
    return pd.DataFrame(rows)


def agreement_against_shap(shap_map: pd.DataFrame, other_map: pd.DataFrame) -> float:
    merged = shap_map.merge(other_map, on="src_cluster", suffixes=("_shap", "_other"))
    return float(np.mean(merged["dominant_dst_cluster_shap"] == merged["dominant_dst_cluster_other"]))


def main():
    datasets = {name: load_dataset(path) for name, path in PATHS.items()}

    uniform_weights = {feat: 1.0 / len(FEATURES) for feat in FEATURES}
    variance_weights = raw_variance_weights(datasets)
    weight_schemes = {
        "shap_based": SHAP_WEIGHTS,
        "uniform": uniform_weights,
        "raw_variance_based": variance_weights,
    }

    pd.DataFrame(
        [{"scheme": name, **weights} for name, weights in weight_schemes.items()]
    ).to_csv(OUT_DIR / "weight_schemes_used.csv", index=False, encoding="utf-8-sig")

    summary_rows = []
    agreement_rows = []

    for name_src, name_dst in PAIR_GROUPS:
        pair_tag = f"{name_src}_to_{name_dst}".replace("-", "")
        pair_dir = OUT_DIR / pair_tag
        pair_dir.mkdir(parents=True, exist_ok=True)

        pair_maps = {}
        for scheme_name, weights in weight_schemes.items():
            out_file = pair_dir / f"{name_src}_to_{name_dst}_{scheme_name}_report.md".replace("-", "")
            result = run_pair(
                name_src=name_src,
                name_dst=name_dst,
                df_src=datasets[name_src],
                df_dst=datasets[name_dst],
                out_file=out_file,
                weights=weights,
                scheme_name=scheme_name,
            )
            result["cost_df"].to_csv(pair_dir / f"{name_src}_to_{name_dst}_{scheme_name}_cost_matrix.csv".replace("-", ""), encoding="utf-8-sig")
            result["P_df"].to_csv(pair_dir / f"{name_src}_to_{name_dst}_{scheme_name}_probability_matrix.csv".replace("-", ""), encoding="utf-8-sig")

            dom_df = dominant_mapping(result["P_df"])
            dom_df["pair"] = f"{name_src}->{name_dst}"
            dom_df["scheme"] = scheme_name
            dom_df["tau"] = result["tau"]
            dom_df.to_csv(pair_dir / f"{name_src}_to_{name_dst}_{scheme_name}_dominant_mapping.csv".replace("-", ""), index=False, encoding="utf-8-sig")
            pair_maps[scheme_name] = dom_df
            summary_rows.append(dom_df)

        shap_map = pair_maps["shap_based"][["src_cluster", "dominant_dst_cluster"]]
        for cmp_scheme in ("uniform", "raw_variance_based"):
            rate = agreement_against_shap(shap_map, pair_maps[cmp_scheme][["src_cluster", "dominant_dst_cluster"]])
            agreement_rows.append({
                "pair": f"{name_src}->{name_dst}",
                "compare_scheme": cmp_scheme,
                "dominant_mapping_agreement_vs_shap": rate,
            })

    pd.concat(summary_rows, ignore_index=True).to_csv(
        OUT_DIR / "dominant_mapping_all_schemes.csv", index=False, encoding="utf-8-sig"
    )
    pd.DataFrame(agreement_rows).to_csv(
        OUT_DIR / "agreement_summary_vs_shap.csv", index=False, encoding="utf-8-sig"
    )

    readme = (
        "Weight sensitivity analysis rebuilt with the same historical cross-sample alignment workflow.\n\n"
        "What is kept identical to the January notebook workflow:\n"
        "- data source: D:\\machinelearning\\nano2\\聚类结果\\8.28\\00z/01z/10z/11z.csv\n"
        "- features: Hardness, Modulus, GC, KIC\n"
        "- distance form: variance-weighted cluster-mean distance\n"
        "- tau percentile: 70.0\n"
        "- TOPK per source cluster: 3\n"
        "- output style: Markdown alignment reports plus cost/probability matrices\n\n"
        "What is changed on purpose:\n"
        "- only the feature-weight scheme wk is varied\n"
        "- schemes compared: shap_based, uniform, raw_variance_based\n\n"
        "Interpretation:\n"
        "- if the dominant source-to-target mapping remains similar under uniform weighting,\n"
        "  the main correspondence pattern is not solely imposed by SHAP weights.\n"
    )
    (OUT_DIR / "README.txt").write_text(readme, encoding="utf-8")


if __name__ == "__main__":
    main()
