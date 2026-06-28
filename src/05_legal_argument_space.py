"""
05_legal_argument_space.py
==========================
Step 4 of the pipeline — Legal Argument Space.

Each lifted hyperedge ê(H) in Ĝ has been assigned a structural profile
(closure, brokerage, authority_concentration, temporal_span,
community_dispersion).  We now:

  1. Assemble the feature matrix X from the profiles.
  2. Standardise X (zero mean, unit variance).
  3. Reduce to 2D for visualisation using PCA and UMAP.
     (PCA and UMAP are used strictly as visualisation tools;
      the full-dimensional profile is the Legal Argument Space.)
  4. Cluster hyperedges in the full-dimensional space using K-Means
     and Agglomerative Clustering.
  5. Save the embedded coordinates and cluster labels.

Inputs:
  data/processed/{dataset}/profiles.pkl
  data/processed/{dataset}/profiles.csv

Outputs:
  data/processed/{dataset}/las_features.pkl   — standardised feature matrix (numpy)
  data/processed/{dataset}/las_pca2d.pkl      — PCA 2D coordinates (numpy)
  data/processed/{dataset}/las_umap2d.pkl     — UMAP 2D coordinates (numpy)
  data/processed/{dataset}/las_clusters.pkl   — cluster label arrays (dict)
  results/{dataset}/las_pca.png               — PCA scatter plot
  results/{dataset}/las_umap.png              — UMAP scatter plot

Usage:
  env/bin/python src/05_legal_argument_space.py [--dataset scotus|echr|both]
                                                 [--n-clusters K]
"""

import argparse
import os
import pickle

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.cluster import AgglomerativeClustering, KMeans
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

def processed_dir(dataset):
    return os.path.join(ROOT, "data", "processed", dataset)

def results_dir(dataset):
    p = os.path.join(ROOT, "results", dataset)
    os.makedirs(p, exist_ok=True)
    return p


# ---------------------------------------------------------------------------
# Feature assembly
# ---------------------------------------------------------------------------
FEATURE_COLS = [
    "closure",
    "brokerage",
    "authority_concentration",
    "temporal_span",
    "community_dispersion",
]

def assemble_features(profiles: dict) -> tuple:
    """
    Build a feature matrix from the profile dict.

    Returns
    -------
    keys     : list of H_key values (row index)
    X_raw    : numpy array shape (n, len(FEATURE_COLS))
    X_scaled : standardised version of X_raw
    """
    keys  = []
    rows  = []
    for H_key, prof in profiles.items():
        row = []
        valid = True
        for col in FEATURE_COLS:
            val = prof.get(col, None)
            if val is None or val == -1:
                valid = False
                break
            row.append(float(val))
        if valid:
            keys.append(H_key)
            rows.append(row)

    X_raw    = np.array(rows, dtype=float)
    scaler   = StandardScaler()
    X_scaled = scaler.fit_transform(X_raw)
    return keys, X_raw, X_scaled


# ---------------------------------------------------------------------------
# Dimensionality reduction
# ---------------------------------------------------------------------------
def reduce_pca(X_scaled: np.ndarray) -> np.ndarray:
    pca = PCA(n_components=2, random_state=42)
    return pca.fit_transform(X_scaled)


def reduce_umap(X_scaled: np.ndarray) -> np.ndarray:
    try:
        import umap
        reducer = umap.UMAP(n_components=2, random_state=42)
        return reducer.fit_transform(X_scaled)
    except ImportError:
        print("  [WARNING] umap-learn not available; skipping UMAP.")
        return None


# ---------------------------------------------------------------------------
# Clustering
# ---------------------------------------------------------------------------
def cluster(X_scaled: np.ndarray, n_clusters: int) -> dict:
    km = KMeans(n_clusters=n_clusters, random_state=42, n_init=10)
    agg = AgglomerativeClustering(n_clusters=n_clusters)
    return {
        "kmeans":        km.fit_predict(X_scaled),
        "agglomerative": agg.fit_predict(X_scaled),
    }


# ---------------------------------------------------------------------------
# Plotting
# ---------------------------------------------------------------------------
def scatter_plot(coords_2d: np.ndarray, labels: np.ndarray,
                 title: str, out_path: str):
    fig, ax = plt.subplots(figsize=(8, 6))
    sc = ax.scatter(
        coords_2d[:, 0], coords_2d[:, 1],
        c=labels, cmap="tab10", s=8, alpha=0.6, linewidths=0
    )
    plt.colorbar(sc, ax=ax, label="Cluster")
    ax.set_title(title)
    ax.set_xlabel("Dimension 1")
    ax.set_ylabel("Dimension 2")
    plt.tight_layout()
    plt.savefig(out_path, dpi=150)
    plt.close()
    print(f"  Saved {out_path}")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def process(dataset: str, n_clusters: int):
    pdir = processed_dir(dataset)
    rdir = results_dir(dataset)

    prof_path = os.path.join(pdir, "profiles.pkl")
    if not os.path.exists(prof_path):
        raise FileNotFoundError(
            f"[{dataset.upper()}] profiles.pkl not found. "
            f"Run 04_structural_profiling.py first."
        )

    print(f"[{dataset.upper()}] Loading profiles …")
    with open(prof_path, "rb") as f:
        profiles = pickle.load(f)

    print(f"[{dataset.upper()}] Assembling feature matrix …")
    keys, X_raw, X_scaled = assemble_features(profiles)
    print(f"[{dataset.upper()}] Feature matrix shape: {X_scaled.shape}")

    print(f"[{dataset.upper()}] PCA 2D reduction …")
    pca2d = reduce_pca(X_scaled)

    print(f"[{dataset.upper()}] UMAP 2D reduction …")
    umap2d = reduce_umap(X_scaled)

    print(f"[{dataset.upper()}] Clustering (K={n_clusters}) …")
    cluster_labels = cluster(X_scaled, n_clusters)

    # Save
    with open(os.path.join(pdir, "las_features.pkl"), "wb") as f:
        pickle.dump({"keys": keys, "X_raw": X_raw, "X_scaled": X_scaled}, f)
    with open(os.path.join(pdir, "las_pca2d.pkl"), "wb") as f:
        pickle.dump(pca2d, f)
    with open(os.path.join(pdir, "las_clusters.pkl"), "wb") as f:
        pickle.dump(cluster_labels, f)
    if umap2d is not None:
        with open(os.path.join(pdir, "las_umap2d.pkl"), "wb") as f:
            pickle.dump(umap2d, f)

    # Plots
    scatter_plot(
        pca2d, cluster_labels["kmeans"],
        f"{dataset.upper()} — Legal Argument Space (PCA, K-Means K={n_clusters})",
        os.path.join(rdir, "las_pca.png")
    )
    if umap2d is not None:
        scatter_plot(
            umap2d, cluster_labels["kmeans"],
            f"{dataset.upper()} — Legal Argument Space (UMAP, K-Means K={n_clusters})",
            os.path.join(rdir, "las_umap.png")
        )

    print(f"[{dataset.upper()}] Legal Argument Space complete.")


def main():
    parser = argparse.ArgumentParser(
        description="Embed and cluster lifted hyperedges in Legal Argument Space."
    )
    parser.add_argument("--dataset",    choices=["scotus", "echr", "both"], default="both")
    parser.add_argument("--n-clusters", type=int, default=6,
                        help="Number of clusters for K-Means and Agglomerative (default: 6)")
    args = parser.parse_args()

    if args.dataset in ("scotus", "both"):
        process("scotus", args.n_clusters)
    if args.dataset in ("echr", "both"):
        process("echr", args.n_clusters)


if __name__ == "__main__":
    main()
