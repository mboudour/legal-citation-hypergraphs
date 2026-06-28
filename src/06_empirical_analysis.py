"""
06_empirical_analysis.py
========================
Step 5 of the pipeline — Empirical Analysis.

Produces the core empirical discoveries of the paper:

  A. Structural Outliers
     Identify lifted hyperedges that occupy extreme positions in Legal
     Argument Space (high/low closure, high/low brokerage, etc.).

  B. Temporal Trajectories
     Track how the mean structural profile of the corpus shifts over
     decades.  Produces a time-series plot for each metric.

  C. Cross-Court Comparison (SCOTUS vs ECHR)
     Compare the distribution of structural profiles between the two
     corpora using summary statistics and distribution plots.

  D. Cluster Characterisation
     For each K-Means cluster, report mean profile values and identify
     representative hyperedges.

Inputs:
  data/processed/{dataset}/profiles.pkl
  data/processed/{dataset}/E0.pkl          (for timestamps)
  data/processed/{dataset}/las_features.pkl
  data/processed/{dataset}/las_pca2d.pkl
  data/processed/{dataset}/las_clusters.pkl

Outputs:
  results/{dataset}/outliers_high_closure.csv
  results/{dataset}/outliers_high_brokerage.csv
  results/{dataset}/temporal_trajectory.png
  results/{dataset}/cluster_summary.csv
  results/cross_court_comparison.png         (written once, uses both datasets)

Usage:
  env/bin/python src/06_empirical_analysis.py [--dataset scotus|echr|both]
                                               [--top-n N]
"""

import argparse
import os
import pickle

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

def processed_dir(dataset):
    return os.path.join(ROOT, "data", "processed", dataset)

def results_dir(dataset=""):
    p = os.path.join(ROOT, "results", dataset) if dataset else os.path.join(ROOT, "results")
    os.makedirs(p, exist_ok=True)
    return p


# ---------------------------------------------------------------------------
# Load helpers
# ---------------------------------------------------------------------------
def load_profiles_df(dataset: str) -> pd.DataFrame:
    path = os.path.join(processed_dir(dataset), "profiles.pkl")
    with open(path, "rb") as f:
        profiles = pickle.load(f)
    rows = []
    for H_key, prof in profiles.items():
        row = {k: v for k, v in prof.items() if k != "H_key"}
        row["H_key"] = str(sorted(H_key))
        rows.append(row)
    return pd.DataFrame(rows)


def load_E0(dataset: str) -> dict:
    path = os.path.join(processed_dir(dataset), "E0.pkl")
    with open(path, "rb") as f:
        return pickle.load(f)


def load_clusters(dataset: str) -> dict:
    path = os.path.join(processed_dir(dataset), "las_clusters.pkl")
    with open(path, "rb") as f:
        return pickle.load(f)


def load_las_features(dataset: str) -> dict:
    path = os.path.join(processed_dir(dataset), "las_features.pkl")
    with open(path, "rb") as f:
        return pickle.load(f)


# ---------------------------------------------------------------------------
# A. Structural Outliers
# ---------------------------------------------------------------------------
def find_outliers(df: pd.DataFrame, metric: str, top_n: int, high: bool) -> pd.DataFrame:
    col = df[metric].dropna()
    if high:
        idx = col.nlargest(top_n).index
    else:
        idx = col.nsmallest(top_n).index
    return df.loc[idx, ["H_key", "tail_size", "head_size", metric]]


def run_outliers(dataset: str, top_n: int):
    rdir = results_dir(dataset)
    df   = load_profiles_df(dataset)

    for metric in ("closure", "brokerage", "authority_concentration"):
        for high, label in ((True, "high"), (False, "low")):
            out = find_outliers(df, metric, top_n, high)
            path = os.path.join(rdir, f"outliers_{label}_{metric}.csv")
            out.to_csv(path, index=False)
            print(f"  [{dataset.upper()}] Saved {path}")


# ---------------------------------------------------------------------------
# B. Temporal Trajectories
# ---------------------------------------------------------------------------
def run_temporal_trajectory(dataset: str, E0: dict):
    rdir = results_dir(dataset)
    df   = load_profiles_df(dataset)

    # Attach year of the citing opinion (from E0 tau)
    # H_key in profiles corresponds to the seed set H; we use the earliest
    # tail opinion's timestamp as the representative year for the hyperedge.
    # Since profiles don't store tail members directly, we reload G_hat.
    ghat_path = os.path.join(processed_dir(dataset), "G_hat.pkl")
    if not os.path.exists(ghat_path):
        print(f"  [{dataset.upper()}] G_hat.pkl not found; skipping temporal trajectory.")
        return

    with open(ghat_path, "rb") as f:
        G_hat = pickle.load(f)

    years = {}
    for H_key, edge in G_hat.items():
        tail = edge["tail"]
        taus = []
        for u in tail:
            tau_str = E0.get(u, {}).get("tau", "")
            if tau_str:
                try:
                    taus.append(int(tau_str[:4]))
                except ValueError:
                    pass
        if taus:
            years[str(sorted(H_key))] = min(taus)

    df["year"] = df["H_key"].map(years)
    df = df.dropna(subset=["year"])
    df["year"] = df["year"].astype(int)

    metrics = ["closure", "brokerage", "authority_concentration", "community_dispersion"]
    decade_df = df.copy()
    decade_df["decade"] = (decade_df["year"] // 10) * 10
    grouped = decade_df.groupby("decade")[metrics].mean()

    fig, axes = plt.subplots(len(metrics), 1, figsize=(10, 3 * len(metrics)), sharex=True)
    for ax, metric in zip(axes, metrics):
        ax.plot(grouped.index, grouped[metric], marker="o")
        ax.set_ylabel(metric.replace("_", " ").title())
        ax.grid(True, alpha=0.3)
    axes[-1].set_xlabel("Decade")
    fig.suptitle(f"{dataset.upper()} — Temporal Trajectory of Structural Profile")
    plt.tight_layout()
    out_path = os.path.join(rdir, "temporal_trajectory.png")
    plt.savefig(out_path, dpi=150)
    plt.close()
    print(f"  [{dataset.upper()}] Saved {out_path}")


# ---------------------------------------------------------------------------
# C. Cross-Court Comparison
# ---------------------------------------------------------------------------
def run_cross_court_comparison(datasets: list):
    rdir = results_dir()
    dfs  = {}
    for ds in datasets:
        try:
            dfs[ds] = load_profiles_df(ds)
        except FileNotFoundError:
            print(f"  Skipping {ds}: profiles.pkl not found.")

    if len(dfs) < 2:
        print("  Cross-court comparison requires both datasets.")
        return

    metrics = ["closure", "brokerage", "authority_concentration", "community_dispersion"]
    fig, axes = plt.subplots(1, len(metrics), figsize=(4 * len(metrics), 5))

    for ax, metric in zip(axes, metrics):
        data   = [dfs[ds][metric].dropna().values for ds in datasets if ds in dfs]
        labels = [ds.upper() for ds in datasets if ds in dfs]
        ax.boxplot(data, labels=labels, notch=True, patch_artist=True)
        ax.set_title(metric.replace("_", " ").title())
        ax.set_ylabel("Value")
        ax.grid(True, alpha=0.3)

    fig.suptitle("Cross-Court Structural Profile Comparison")
    plt.tight_layout()
    out_path = os.path.join(rdir, "cross_court_comparison.png")
    plt.savefig(out_path, dpi=150)
    plt.close()
    print(f"  Saved {out_path}")


# ---------------------------------------------------------------------------
# D. Cluster Characterisation
# ---------------------------------------------------------------------------
def run_cluster_summary(dataset: str):
    rdir = results_dir(dataset)
    df   = load_profiles_df(dataset)

    las  = load_las_features(dataset)
    keys = [str(sorted(k)) for k in las["keys"]]
    clusters = load_clusters(dataset)

    km_labels = clusters.get("kmeans", None)
    if km_labels is None:
        return

    df_las = pd.DataFrame({"H_key": keys, "cluster": km_labels})
    df_merged = df.merge(df_las, on="H_key", how="inner")

    metrics = ["closure", "brokerage", "authority_concentration",
               "temporal_span", "community_dispersion"]
    summary = df_merged.groupby("cluster")[metrics].mean()
    summary["count"] = df_merged.groupby("cluster").size()

    out_path = os.path.join(rdir, "cluster_summary.csv")
    summary.to_csv(out_path)
    print(f"  [{dataset.upper()}] Saved {out_path}")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def process(dataset: str, top_n: int):
    pdir = processed_dir(dataset)
    for fname in ("profiles.pkl", "E0.pkl"):
        if not os.path.exists(os.path.join(pdir, fname)):
            raise FileNotFoundError(
                f"[{dataset.upper()}] {fname} not found. Run previous steps first."
            )

    print(f"[{dataset.upper()}] Running empirical analysis …")
    E0 = load_E0(dataset)
    run_outliers(dataset, top_n)
    run_temporal_trajectory(dataset, E0)
    run_cluster_summary(dataset)
    print(f"[{dataset.upper()}] Empirical analysis complete.")


def main():
    parser = argparse.ArgumentParser(
        description="Empirical analysis of the lifted hypergraph structural profiles."
    )
    parser.add_argument("--dataset", choices=["scotus", "echr", "both"], default="both")
    parser.add_argument("--top-n",   type=int, default=20,
                        help="Number of outliers to report per metric (default: 20)")
    args = parser.parse_args()

    datasets = []
    if args.dataset in ("scotus", "both"):
        datasets.append("scotus")
    if args.dataset in ("echr", "both"):
        datasets.append("echr")

    for ds in datasets:
        process(ds, args.top_n)

    if len(datasets) == 2:
        print("[CROSS-COURT] Generating comparison plot …")
        run_cross_court_comparison(datasets)


if __name__ == "__main__":
    main()
