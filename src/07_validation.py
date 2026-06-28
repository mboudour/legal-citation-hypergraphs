"""
07_validation.py
================
Step 6 of the pipeline — Validation: Missing Precedent Task.

This script validates that the structural profile of a lifted hyperedge
contains useful information by testing a Missing Precedent recommendation
task:

  Given a lifted hyperedge ê(H) = (tail, head) and a head with one
  opinion masked out (v* removed), can we recover v* by ranking
  candidate opinions using the structural profile?

The task is framed as a ranking problem:
  - For each test hyperedge, mask one head node v*.
  - Rank all candidate opinions by their structural similarity to the
    partial hyperedge.
  - Evaluate whether v* appears in the top-K ranked candidates.

Evaluation metrics (per the project standard):
  F1, Recall, Precision, Accuracy, ROC-AUC, Log-Loss, Matthews Correlation
  Coefficient (MCC) — computed at a fixed decision threshold (top-K).

Baselines compared:
  1. Random ranking
  2. In-degree ranking (most-cited opinion first)
  3. Structural profile cosine similarity (our method)

Inputs:
  data/processed/{dataset}/E0.pkl
  data/processed/{dataset}/G_hat.pkl
  data/processed/{dataset}/profiles.pkl
  data/processed/{dataset}/las_features.pkl

Outputs:
  results/{dataset}/validation_results.csv   — per-method metric table
  results/{dataset}/validation_topk.png      — Recall@K curve

Usage:
  env/bin/python src/07_validation.py [--dataset scotus|echr|both]
                                       [--top-k K]
                                       [--n-test N]
"""

import argparse
import os
import pickle
import random

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.metrics import (
    accuracy_score, f1_score, log_loss, matthews_corrcoef,
    precision_score, recall_score, roc_auc_score,
)
from sklearn.metrics.pairwise import cosine_similarity

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
# Load helpers
# ---------------------------------------------------------------------------
def load_all(dataset: str):
    pdir = processed_dir(dataset)
    with open(os.path.join(pdir, "E0.pkl"),          "rb") as f: E0       = pickle.load(f)
    with open(os.path.join(pdir, "G_hat.pkl"),        "rb") as f: G_hat    = pickle.load(f)
    with open(os.path.join(pdir, "profiles.pkl"),     "rb") as f: profiles = pickle.load(f)
    with open(os.path.join(pdir, "las_features.pkl"), "rb") as f: las      = pickle.load(f)
    return E0, G_hat, profiles, las


# ---------------------------------------------------------------------------
# Build candidate pool and feature index
# ---------------------------------------------------------------------------
def build_feature_index(las: dict) -> tuple:
    """
    Returns:
      key_list  : list of H_key frozensets
      X_scaled  : numpy array (n_hyperedges, n_features)
    """
    return las["keys"], las["X_scaled"]


# ---------------------------------------------------------------------------
# Ranking methods
# ---------------------------------------------------------------------------
def rank_random(candidates: list, rng: random.Random) -> list:
    shuffled = candidates[:]
    rng.shuffle(shuffled)
    return shuffled


def rank_indegree(candidates: list, indegree: dict) -> list:
    return sorted(candidates, key=lambda v: indegree.get(v, 0), reverse=True)


def rank_profile_similarity(
    partial_head: frozenset,
    candidates: list,
    profiles: dict,
    G_hat: dict,
    E0: dict,
) -> list:
    """
    Rank candidates by cosine similarity of their structural profile
    to the profile of the partial hyperedge.

    The partial hyperedge profile is approximated by the mean profile
    of all lifted hyperedges whose head contains the partial_head.
    """
    # Find lifted hyperedges containing partial_head
    matching = [
        prof for H_key, prof in profiles.items()
        if partial_head <= H_key
    ]
    if not matching:
        return candidates  # fallback: no ordering

    feature_cols = [
        "closure", "brokerage", "authority_concentration",
        "temporal_span", "community_dispersion",
    ]

    def prof_vec(p):
        return np.array([float(p.get(c, 0) or 0) for c in feature_cols])

    mean_vec = np.mean([prof_vec(p) for p in matching], axis=0).reshape(1, -1)

    # Score each candidate by how much adding it changes the profile
    # (proxy: profile of the lifted hyperedge containing partial_head ∪ {v})
    scores = {}
    for v in candidates:
        probe = partial_head | {v}
        probe_matches = [
            prof for H_key, prof in profiles.items()
            if probe <= H_key
        ]
        if probe_matches:
            probe_vec = np.mean([prof_vec(p) for p in probe_matches], axis=0).reshape(1, -1)
            sim = cosine_similarity(mean_vec, probe_vec)[0, 0]
        else:
            sim = 0.0
        scores[v] = sim

    return sorted(candidates, key=lambda v: scores.get(v, 0), reverse=True)


# ---------------------------------------------------------------------------
# Evaluation at top-K
# ---------------------------------------------------------------------------
def evaluate_at_k(ranked: list, target: str, k: int, n_candidates: int) -> dict:
    """
    Binary classification: top-K as positive predictions.
    Returns dict of metrics.
    """
    top_k_set = set(ranked[:k])
    y_true = [1 if v == target else 0 for v in ranked]
    y_pred = [1 if v in top_k_set else 0 for v in ranked]
    y_score = [1.0 - (i / n_candidates) for i in range(n_candidates)]

    # Guard against degenerate cases
    if sum(y_true) == 0 or sum(y_true) == len(y_true):
        return {m: float("nan") for m in
                ["f1", "recall", "precision", "accuracy", "roc_auc", "log_loss", "mcc"]}

    try:
        roc = roc_auc_score(y_true, y_score)
    except Exception:
        roc = float("nan")
    try:
        ll = log_loss(y_true, [min(max(s, 1e-7), 1 - 1e-7) for s in y_score])
    except Exception:
        ll = float("nan")

    return {
        "f1":        f1_score(y_true, y_pred, zero_division=0),
        "recall":    recall_score(y_true, y_pred, zero_division=0),
        "precision": precision_score(y_true, y_pred, zero_division=0),
        "accuracy":  accuracy_score(y_true, y_pred),
        "roc_auc":   roc,
        "log_loss":  ll,
        "mcc":       matthews_corrcoef(y_true, y_pred),
    }


# ---------------------------------------------------------------------------
# Main validation loop
# ---------------------------------------------------------------------------
def run_validation(dataset: str, top_k: int, n_test: int):
    print(f"[{dataset.upper()}] Loading data …")
    E0, G_hat, profiles, las = load_all(dataset)

    # Build in-degree index from E0
    indegree = {}
    for u, meta in E0.items():
        for v in meta["F"]:
            indegree[v] = indegree.get(v, 0) + 1

    # Select test hyperedges: lifted hyperedges with |head| >= 3
    test_edges = [
        (H_key, edge) for H_key, edge in G_hat.items()
        if len(edge["head"]) >= 3
    ]
    rng = random.Random(42)
    rng.shuffle(test_edges)
    test_edges = test_edges[:n_test]
    print(f"[{dataset.upper()}] {len(test_edges)} test hyperedges (|head| ≥ 3).")

    # All candidate opinions (universe for ranking)
    all_opinions = list(set(E0.keys()))

    method_results = {"random": [], "indegree": [], "profile": []}

    for H_key, edge in test_edges:
        head = list(edge["head"])
        if len(head) < 3:
            continue
        # Mask one random head node
        target = rng.choice(head)
        partial_head = frozenset(h for h in head if h != target)

        # Candidate pool: all opinions not in partial_head
        candidates = [v for v in all_opinions if v not in partial_head]
        if target not in candidates:
            candidates.append(target)
        n_cand = len(candidates)

        # Rank by each method
        ranked_random  = rank_random(candidates, rng)
        ranked_indeg   = rank_indegree(candidates, indegree)
        ranked_profile = rank_profile_similarity(
            partial_head, candidates, profiles, G_hat, E0
        )

        for method, ranked in [
            ("random",  ranked_random),
            ("indegree", ranked_indeg),
            ("profile", ranked_profile),
        ]:
            metrics = evaluate_at_k(ranked, target, top_k, n_cand)
            method_results[method].append(metrics)

    # Aggregate
    metric_names = ["f1", "recall", "precision", "accuracy", "roc_auc", "log_loss", "mcc"]
    rows = []
    for method, results_list in method_results.items():
        if not results_list:
            continue
        df_m = pd.DataFrame(results_list)
        row  = {"method": method}
        for m in metric_names:
            row[m] = df_m[m].mean(skipna=True)
        rows.append(row)

    summary = pd.DataFrame(rows).set_index("method")
    rdir = results_dir(dataset)
    out_csv = os.path.join(rdir, "validation_results.csv")
    summary.to_csv(out_csv)
    print(f"[{dataset.upper()}] Validation results:\n{summary.to_string()}")
    print(f"[{dataset.upper()}] Saved {out_csv}")

    # Recall@K curve
    ks = list(range(1, min(top_k * 3, 50) + 1))
    fig, ax = plt.subplots(figsize=(8, 5))
    for method, results_list in method_results.items():
        recalls = []
        for k in ks:
            vals = []
            for H_key, edge in test_edges[:len(results_list)]:
                head = list(edge["head"])
                if len(head) < 3:
                    continue
                target = rng.choice(head)
                partial_head = frozenset(h for h in head if h != target)
                candidates = [v for v in all_opinions if v not in partial_head]
                if target not in candidates:
                    candidates.append(target)
                if method == "random":
                    ranked = rank_random(candidates, rng)
                elif method == "indegree":
                    ranked = rank_indegree(candidates, indegree)
                else:
                    ranked = rank_profile_similarity(partial_head, candidates, profiles, G_hat, E0)
                vals.append(1 if target in ranked[:k] else 0)
            recalls.append(np.mean(vals) if vals else 0)
        ax.plot(ks, recalls, marker=".", label=method)

    ax.set_xlabel("K")
    ax.set_ylabel("Recall@K")
    ax.set_title(f"{dataset.upper()} — Missing Precedent Recall@K")
    ax.legend()
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    out_png = os.path.join(rdir, "validation_topk.png")
    plt.savefig(out_png, dpi=150)
    plt.close()
    print(f"[{dataset.upper()}] Saved {out_png}")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main():
    parser = argparse.ArgumentParser(
        description="Missing Precedent validation task."
    )
    parser.add_argument("--dataset", choices=["scotus", "echr", "both"], default="both")
    parser.add_argument("--top-k",   type=int, default=10,
                        help="K for top-K evaluation (default: 10)")
    parser.add_argument("--n-test",  type=int, default=500,
                        help="Number of test hyperedges (default: 500)")
    args = parser.parse_args()

    if args.dataset in ("scotus", "both"):
        run_validation("scotus", args.top_k, args.n_test)
    if args.dataset in ("echr", "both"):
        run_validation("echr",   args.top_k, args.n_test)


if __name__ == "__main__":
    main()
