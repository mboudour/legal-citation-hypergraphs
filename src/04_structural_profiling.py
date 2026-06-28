"""
04_structural_profiling.py
==========================
Step 3 of the pipeline — Structural Profiling on the Lifted Hypergraph Ĝ.

For each lifted hyperedge ê(H) = (tail, head) in Ĝ, we compute a structural
profile consisting of two families of metrics, all computed on the subgraph
of the historical citation network induced by the head set.

Internal Structural Metrics (how the head nodes relate to one another):
  - closure   : density of edges within head(ê) in the citation graph
  - brokerage : fraction of pairs in head(ê) with no direct citation link
                (structural holes / bridging)
  - density   : same as closure (alias kept for clarity)

Contextual Structural Metrics (how the head sits in the full graph):
  - authority_concentration : mean in-degree of head nodes in the full graph
  - temporal_span           : max timestamp difference within head nodes (days)
  - community_dispersion    : number of distinct Louvain communities spanned

The full citation graph G used for contextual metrics is reconstructed from
E_0 (all citation edges).

Inputs:
  data/processed/{dataset}/E0.pkl
  data/processed/{dataset}/G_hat.pkl

Outputs:
  data/processed/{dataset}/profiles.pkl
    dict: { H_key (frozenset) -> profile_dict }
  data/processed/{dataset}/profiles.csv
    CSV with one row per lifted hyperedge

Usage:
  env/bin/python src/04_structural_profiling.py [--dataset scotus|echr|both]
"""

import argparse
import csv
import os
import pickle
from datetime import datetime

import networkx as nx
import numpy as np
from tqdm import tqdm

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

def processed_dir(dataset):
    return os.path.join(ROOT, "data", "processed", dataset)


# ---------------------------------------------------------------------------
# Build the full citation graph from E_0
# ---------------------------------------------------------------------------
def build_citation_graph(E0: dict) -> nx.DiGraph:
    """
    Build a directed citation graph G from E_0.
    Nodes are opinion IDs; edges are citation links u -> v for v in F(u).
    Node attribute 'tau' stores the timestamp string.
    """
    G = nx.DiGraph()
    for u, meta in E0.items():
        G.add_node(u, tau=meta.get("tau", ""))
        for v in meta["F"]:
            G.add_edge(u, v)
    return G


# ---------------------------------------------------------------------------
# Timestamp parsing
# ---------------------------------------------------------------------------
def parse_tau(tau_str: str):
    """Parse a timestamp string to a datetime object; return None on failure."""
    for fmt in ("%Y-%m-%d", "%Y-%m-%dT%H:%M:%S", "%Y/%m/%d", "%Y"):
        try:
            return datetime.strptime(tau_str[:len(fmt)], fmt)
        except (ValueError, TypeError):
            continue
    return None


# ---------------------------------------------------------------------------
# Internal Structural Metrics
# ---------------------------------------------------------------------------
def compute_closure(G: nx.DiGraph, head: frozenset) -> float:
    """
    Closure: density of the subgraph induced by head in G.
    Defined as |{ (i,j) in E : i,j in head }| / (|head| * (|head|-1))
    Returns 0.0 if |head| < 2.
    """
    if len(head) < 2:
        return 0.0
    sub = G.subgraph(head)
    n   = len(head)
    return sub.number_of_edges() / (n * (n - 1))


def compute_brokerage(G: nx.DiGraph, head: frozenset) -> float:
    """
    Brokerage: fraction of ordered pairs (i,j) in head with no direct edge.
    = 1 - closure
    """
    return 1.0 - compute_closure(G, head)


# ---------------------------------------------------------------------------
# Contextual Structural Metrics
# ---------------------------------------------------------------------------
def compute_authority_concentration(G: nx.DiGraph, head: frozenset) -> float:
    """Mean in-degree of head nodes in the full citation graph G."""
    if not head:
        return 0.0
    return np.mean([G.in_degree(v) for v in head if v in G])


def compute_temporal_span(E0: dict, head: frozenset) -> float:
    """
    Temporal span: number of days between the earliest and latest
    decision date among head nodes.  Returns -1 if timestamps unavailable.
    """
    dates = []
    for v in head:
        tau_str = E0.get(v, {}).get("tau", "")
        dt = parse_tau(tau_str)
        if dt:
            dates.append(dt)
    if len(dates) < 2:
        return -1.0
    return (max(dates) - min(dates)).days


def compute_community_dispersion(G: nx.DiGraph, head: frozenset,
                                  partition: dict) -> int:
    """
    Number of distinct communities (from a pre-computed Louvain partition)
    spanned by the head nodes.
    """
    communities = set()
    for v in head:
        if v in partition:
            communities.add(partition[v])
    return len(communities)


# ---------------------------------------------------------------------------
# Main profiling loop
# ---------------------------------------------------------------------------
def profile_hypergraph(E0: dict, G_hat: dict) -> dict:
    """
    Compute structural profiles for all lifted hyperedges in Ĝ.

    Returns
    -------
    profiles : dict { H_key -> profile_dict }
    """
    print("  Building full citation graph …")
    G = build_citation_graph(E0)

    # Louvain community detection on the undirected projection
    print("  Computing Louvain communities …")
    G_und = G.to_undirected()
    try:
        from networkx.algorithms.community import louvain_communities
        communities = louvain_communities(G_und, seed=42)
        partition   = {}
        for cid, comm in enumerate(communities):
            for node in comm:
                partition[node] = cid
    except Exception:
        partition = {}  # fallback: no community info

    profiles = {}
    for H_key, edge in tqdm(G_hat.items(), desc="  Profiling hyperedges"):
        head = edge["head"]
        tail = edge["tail"]

        cl  = compute_closure(G, head)
        br  = compute_brokerage(G, head)
        ac  = compute_authority_concentration(G, head)
        ts  = compute_temporal_span(E0, head)
        cd  = compute_community_dispersion(G, head, partition)

        profiles[H_key] = {
            # identity
            "H_key":         H_key,
            "tail_size":     len(tail),
            "head_size":     len(head),
            # internal metrics
            "closure":       cl,
            "brokerage":     br,
            "density":       cl,          # alias
            # contextual metrics
            "authority_concentration": ac,
            "temporal_span":           ts,
            "community_dispersion":    cd,
        }

    return profiles


# ---------------------------------------------------------------------------
# Save
# ---------------------------------------------------------------------------
def save_profiles(profiles: dict, pdir: str):
    pkl_path = os.path.join(pdir, "profiles.pkl")
    csv_path = os.path.join(pdir, "profiles.csv")

    with open(pkl_path, "wb") as f:
        pickle.dump(profiles, f)

    fieldnames = [
        "H_key", "tail_size", "head_size",
        "closure", "brokerage", "density",
        "authority_concentration", "temporal_span", "community_dispersion",
    ]
    with open(csv_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames, extrasaction="ignore")
        writer.writeheader()
        for prof in profiles.values():
            row = {k: prof[k] for k in fieldnames if k != "H_key"}
            row["H_key"] = str(sorted(prof["H_key"]))
            writer.writerow(row)

    print(f"  Saved profiles.pkl and profiles.csv to {pdir}/")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def process(dataset: str):
    pdir = processed_dir(dataset)

    e0_path    = os.path.join(pdir, "E0.pkl")
    ghat_path  = os.path.join(pdir, "G_hat.pkl")

    for path in (e0_path, ghat_path):
        if not os.path.exists(path):
            raise FileNotFoundError(
                f"[{dataset.upper()}] {os.path.basename(path)} not found. "
                f"Run previous pipeline steps first."
            )

    print(f"[{dataset.upper()}] Loading E_0 and Ĝ …")
    with open(e0_path,   "rb") as f:
        E0    = pickle.load(f)
    with open(ghat_path, "rb") as f:
        G_hat = pickle.load(f)

    print(f"[{dataset.upper()}] |E_0| = {len(E0):,}  |Ĝ| = {len(G_hat):,}")
    profiles = profile_hypergraph(E0, G_hat)
    save_profiles(profiles, pdir)
    print(f"[{dataset.upper()}] Profiling complete. {len(profiles):,} hyperedges profiled.")


def main():
    parser = argparse.ArgumentParser(
        description="Compute structural profiles on the lifted hypergraph Ĝ."
    )
    parser.add_argument("--dataset", choices=["scotus", "echr", "both"], default="both")
    args = parser.parse_args()

    if args.dataset in ("scotus", "both"):
        process("scotus")
    if args.dataset in ("echr", "both"):
        process("echr")


if __name__ == "__main__":
    main()
