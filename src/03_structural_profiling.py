"""
03_structural_profiling.py
==========================
Step 3 of the pipeline — Structural Profiling on the Directed Citation F-Hypergraph.

For each F-arc e_u = ({u}, F(u)) in the citation hypergraph G, we compute a
structural profile consisting of two families of metrics, all computed on the
subgraph of the historical citation network induced by the head set F(u).

Internal Structural Metrics (how the cited nodes relate to one another):
  - closure   : density of edges within F(u) in the citation graph
  - brokerage : fraction of pairs in F(u) with no direct citation link
                (structural holes / bridging)
  - density   : same as closure (alias kept for clarity)

Contextual Structural Metrics (how the cited nodes sit in the full graph):
  - authority_concentration : mean in-degree of nodes in F(u) in the full graph
  - temporal_span           : max timestamp difference within F(u) (days)
  - community_dispersion    : number of distinct Louvain communities spanned by F(u)

The full citation graph used for contextual metrics is the standard directed
graph projection of G.

Inputs:
  data/processed/{dataset}/G.pkl

Outputs:
  data/processed/{dataset}/profiles.pkl
    dict: { u (str) -> profile_dict }
  data/processed/{dataset}/profiles.csv
    CSV with one row per citing opinion (F-arc)

Usage:
  env/bin/python src/03_structural_profiling.py [--dataset scotus|echr|both]
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
# Build the full citation graph from G
# ---------------------------------------------------------------------------
def build_citation_graph(G_hyper: dict) -> nx.DiGraph:
    """
    Build a standard directed citation graph from the F-hypergraph G.
    Nodes are opinion IDs; edges are citation links u -> v for v in F(u).
    Node attribute 'tau' stores the timestamp string.
    """
    G_nx = nx.DiGraph()
    for u, meta in G_hyper.items():
        G_nx.add_node(u, tau=meta.get("tau", ""))
        for v in meta["F"]:
            G_nx.add_edge(u, v)
    return G_nx


# ---------------------------------------------------------------------------
# Timestamp parsing
# ---------------------------------------------------------------------------
def parse_tau(tau_str: str):
    """Parse a timestamp string to a datetime object; return None on failure."""
    if not tau_str:
        return None
    for fmt in ("%Y-%m-%d", "%Y-%m-%dT%H:%M:%S", "%Y/%m/%d", "%Y"):
        try:
            # Use only the prefix of the string that matches the format length
            # BUT use the actual date string length, not the format string length.
            # Simply try parsing the full string (or a reasonable prefix).
            candidate = tau_str[:19] if len(tau_str) > 10 else tau_str[:10]
            return datetime.strptime(candidate, fmt)
        except (ValueError, TypeError):
            continue
    return None


# ---------------------------------------------------------------------------
# Internal Structural Metrics
# ---------------------------------------------------------------------------
def compute_closure(G_nx: nx.DiGraph, head: frozenset) -> float:
    """
    Closure: density of the subgraph induced by head in G_nx.
    Defined as |{ (i,j) in E : i,j in head }| / (|head| * (|head|-1))
    Returns 0.0 if |head| < 2.
    """
    if len(head) < 2:
        return 0.0
    sub = G_nx.subgraph(head)
    n   = len(head)
    return sub.number_of_edges() / (n * (n - 1))


def compute_brokerage(G_nx: nx.DiGraph, head: frozenset) -> float:
    """
    Brokerage: fraction of ordered pairs (i,j) in head with no direct edge.
    = 1 - closure
    """
    return 1.0 - compute_closure(G_nx, head)


# ---------------------------------------------------------------------------
# Contextual Structural Metrics
# ---------------------------------------------------------------------------
def compute_authority_concentration(G_nx: nx.DiGraph, head: frozenset) -> float:
    """Mean in-degree of head nodes in the full citation graph."""
    if not head:
        return 0.0
    return np.mean([G_nx.in_degree(v) for v in head if v in G_nx])


def compute_temporal_span(tau_map: dict, head: frozenset) -> float:
    """
    Temporal span: number of days between the earliest and latest
    decision date among head nodes.  Returns -1 if timestamps unavailable.

    Parameters
    ----------
    tau_map : dict { opinion_id -> tau_str }
        Covers ALL opinion IDs (both citing and cited).
    """
    dates = []
    for v in head:
        tau_str = tau_map.get(v, "")
        dt = parse_tau(tau_str)
        if dt:
            dates.append(dt)
    if len(dates) < 2:
        return -1.0
    return (max(dates) - min(dates)).days


def compute_community_dispersion(G_nx: nx.DiGraph, head: frozenset,
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
def profile_hypergraph(G_hyper: dict, dataset: str) -> dict:
    """
    Compute structural profiles for all F-arcs in G.

    Returns
    -------
    profiles : dict { u (str) -> profile_dict }
    """
    print("  Building full citation graph …")
    G_nx = build_citation_graph(G_hyper)

    # Build a tau_map covering ALL opinion IDs (citing + cited) from opinions_meta.pkl.
    # This is essential because G_hyper only has tau for citing opinions; cited-only
    # opinions would otherwise have no timestamp, causing temporal_span to always be -1.
    opinions_meta_path = os.path.join(
        os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
        "data", "processed", dataset, "opinions_meta.pkl"
    )
    tau_map = {}
    if os.path.exists(opinions_meta_path):
        with open(opinions_meta_path, "rb") as _f:
            _om = pickle.load(_f)
        tau_map = {oid: m.get("tau", "") for oid, m in _om.items()}
    # Fallback: use tau from G_hyper for citing opinions not covered above
    for u, meta in G_hyper.items():
        if u not in tau_map:
            tau_map[u] = meta.get("tau", "")

    # Louvain community detection on the undirected projection
    print("  Computing Louvain communities …")
    G_und = G_nx.to_undirected()
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
    for u, meta in tqdm(G_hyper.items(), desc="  Profiling F-arcs"):
        head = meta["F"]

        cl  = compute_closure(G_nx, head)
        br  = compute_brokerage(G_nx, head)
        ac  = compute_authority_concentration(G_nx, head)
        ts  = compute_temporal_span(tau_map, head)
        cd  = compute_community_dispersion(G_nx, head, partition)

        profiles[u] = {
            # identity
            "citing_id":     u,
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
        "citing_id", "head_size",
        "closure", "brokerage", "density",
        "authority_concentration", "temporal_span", "community_dispersion",
    ]
    with open(csv_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames, extrasaction="ignore")
        writer.writeheader()
        for prof in profiles.values():
            writer.writerow(prof)

    print(f"  Saved profiles.pkl and profiles.csv to {pdir}/")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def process(dataset: str):
    pdir = processed_dir(dataset)
    g_path = os.path.join(pdir, "G.pkl")

    if not os.path.exists(g_path):
        raise FileNotFoundError(
            f"[{dataset.upper()}] G.pkl not found. Run 02_build_citation_hypergraph.py first."
        )

    print(f"[{dataset.upper()}] Loading F-hypergraph G …")
    with open(g_path, "rb") as f:
        G_hyper = pickle.load(f)

    print(f"[{dataset.upper()}] |G| = {len(G_hyper):,} F-arcs")
    profiles = profile_hypergraph(G_hyper, dataset)
    save_profiles(profiles, pdir)
    print(f"[{dataset.upper()}] Profiling complete. {len(profiles):,} F-arcs profiled.")


def main():
    parser = argparse.ArgumentParser(
        description="Compute structural profiles on the F-hypergraph G."
    )
    parser.add_argument("--dataset", choices=["scotus", "echr", "both"], default="scotus")
    args = parser.parse_args()

    if args.dataset in ("scotus", "both"):
        process("scotus")
    if args.dataset in ("echr", "both"):
        import os
        if not os.path.exists(os.path.join(processed_dir("echr"), "G.pkl")):
            print("[ECHR] Skipped — G.pkl not found. Run 02_build_citation_hypergraph.py --dataset echr first.")
        else:
            process("echr")


if __name__ == "__main__":
    main()
