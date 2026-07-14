"""
02_build_citation_hypergraph.py
===============================
Step 2 of the pipeline — Build the Directed Citation Hypergraph (F-hypergraph).

Given an opinion-level legal dataset in which each opinion u has:
  - a timestamp  tau(u)
  - a citation set F(u)  (the set of opinions cited by u)

we construct the citation hypergraph

    G = { e_u = ({u}, F(u)) : u in V }

where V is the universal vertex set of all opinions appearing in the corpus
(whether as citing or cited).

Following Gallo et al. (1993), because every hyperedge e_u has exactly one
tail node (the citing opinion), this structure is precisely an F-hypergraph.
Since citations strictly point backward in time, it is a directed acyclic
F-hypergraph.

Inputs (from data/raw/):
  SCOTUS:
    data/raw/scotus/clusters.csv     — opinion clusters with date_filed, case_name, docket_id
    data/raw/scotus/citations.csv    — citing_opinion_id, cited_opinion_id (from scotus_network.csv)

  ECHR:
    data/raw/echr/judgments.jsonl    — one judgment record per line (JSON)
    data/raw/echr/citations.csv      — citing_id, cited_id

Outputs (written to data/processed/):
  data/processed/scotus/G.pkl    — dict: { u: {"tau": timestamp_str, "F": frozenset} }
  data/processed/scotus/V.pkl    — set: universal vertex set
  data/processed/echr/G.pkl
  data/processed/echr/V.pkl

Usage:
  python src/02_build_citation_hypergraph.py [--dataset scotus|echr|both]
"""

import argparse
import csv
import json
import os
import pickle

from tqdm import tqdm

# Increase CSV field size limit for large text fields
csv.field_size_limit(min(os.sys.maxsize, 2 ** 31 - 1))

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def processed_dir(dataset):
    p = os.path.join(ROOT, "data", "processed", dataset)
    os.makedirs(p, exist_ok=True)
    return p


def raw_dir(dataset):
    return os.path.join(ROOT, "data", "raw", dataset)


# ---------------------------------------------------------------------------
# Core construction
# ---------------------------------------------------------------------------
def build_G(opinions_meta: dict, citations: dict) -> tuple:
    """
    Build the directed citation F-hypergraph G.

    Parameters
    ----------
    opinions_meta : dict
        { opinion_id (str) -> {"tau": timestamp_str, "case_name": str, ...} }
    citations : dict
        { citing_id (str) -> set of cited_ids (str) }

    Returns
    -------
    V : set of all vertex IDs (union of all citing and cited opinions)
    G : dict { u -> {"tau": tau(u), "case_name": str, "F": frozenset(F(u))} }
        Only entries where F(u) is non-empty are included.
    """
    V = set(opinions_meta.keys())
    G = {}

    for u, meta in opinions_meta.items():
        F_u = frozenset(citations.get(u, set()))
        if not F_u:
            continue  # skip opinions that cite nothing
        V.update(F_u)  # cited opinions not in opinions_meta still belong to V
        G[u] = {
            "tau":       meta.get("tau", ""),
            "case_name": meta.get("case_name", ""),
            "F":         F_u,
        }

    return V, G


# ---------------------------------------------------------------------------
# SCOTUS loader
# ---------------------------------------------------------------------------
def load_scotus():
    rdir = raw_dir("scotus")
    clusters_path  = os.path.join(rdir, "clusters.csv")
    citations_path = os.path.join(rdir, "citations.csv")

    for p in [clusters_path, citations_path]:
        if not os.path.exists(p):
            raise FileNotFoundError(
                f"[SCOTUS] File not found: {p}\n"
                "  Run 01_fetch_data.py --dataset scotus first."
            )

    # Load clusters: cluster_id -> {date_filed, case_name}
    # clusters.csv provides metadata keyed by cluster_id.
    # citations.csv (from scotus_network.csv) uses opinion IDs as node identifiers.
    # We build an opinion_id -> metadata map by loading all opinion IDs from citations.csv
    # and attaching cluster metadata where available via a best-effort join.
    # Since each cluster typically has one main opinion, we use cluster_id as a proxy
    # opinion_id for metadata purposes (most cluster_ids appear as opinion_ids in the network).
    print("[SCOTUS] Loading clusters (metadata) …")
    cluster_meta = {}
    with open(clusters_path, encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in tqdm(reader, desc="  clusters", leave=False):
            cid = row.get("id", "").strip()
            if cid:
                cluster_meta[cid] = {
                    "tau":       row.get("date_filed", ""),
                    "case_name": row.get("case_name", ""),
                }

    # Load citation edges (opinion-level from scotus_network.csv)
    print("[SCOTUS] Loading citations (opinion-level) …")
    citations = {}
    all_opinion_ids = set()
    with open(citations_path, encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in tqdm(reader, desc="  citations", leave=False):
            u = row.get("citing_opinion_id", "").strip()
            v = row.get("cited_opinion_id", "").strip()
            if u and v:
                citations.setdefault(u, set()).add(v)
                all_opinion_ids.add(u)
                all_opinion_ids.add(v)

    # Build opinions_meta for all opinion IDs seen in the citation network.
    # Use cluster metadata where the opinion_id matches a cluster_id (common case),
    # otherwise use empty metadata (tau='', case_name='').
    print("[SCOTUS] Joining opinion IDs to cluster metadata …")
    opinions_meta = {}
    for oid in all_opinion_ids:
        meta = cluster_meta.get(oid, {"tau": "", "case_name": ""})
        opinions_meta[oid] = meta

    matched = sum(1 for oid in all_opinion_ids if oid in cluster_meta)
    print(f"[SCOTUS] {matched:,} / {len(all_opinion_ids):,} opinion IDs matched to cluster metadata.")

    return opinions_meta, citations


# ---------------------------------------------------------------------------
# ECHR loader
# ---------------------------------------------------------------------------
def load_echr():
    rdir = raw_dir("echr")
    judgments_path = os.path.join(rdir, "judgments.jsonl")
    citations_path = os.path.join(rdir, "citations.csv")

    for p in [judgments_path, citations_path]:
        if not os.path.exists(p):
            raise FileNotFoundError(
                f"[ECHR] File not found: {p}\n"
                "  Run 01_fetch_data.py --dataset echr first."
            )

    # Load judgment metadata
    print("[ECHR] Loading judgments …")
    opinions_meta = {}
    with open(judgments_path, encoding="utf-8") as f:
        for line in tqdm(f, desc="  judgments", leave=False):
            jud = json.loads(line)
            jid = str(jud.get("itemid", "")).strip()
            if not jid:
                continue
            opinions_meta[jid] = {
                "tau":       jud.get("judgementdate", ""),
                "case_name": jud.get("docname", ""),
            }

    # Load citation edges
    print("[ECHR] Loading citations …")
    citations = {}
    with open(citations_path, encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in tqdm(reader, desc="  citations", leave=False):
            u = row.get("citing_id", "").strip()
            v = row.get("cited_id", "").strip()
            if u and v:
                citations.setdefault(u, set()).add(v)

    return opinions_meta, citations


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def process(dataset: str):
    print(f"\n[{dataset.upper()}] Building directed citation F-hypergraph G …")

    if dataset == "scotus":
        opinions_meta, citations = load_scotus()
    elif dataset == "echr":
        opinions_meta, citations = load_echr()
    else:
        raise ValueError(f"Unknown dataset: {dataset}")

    V, G = build_G(opinions_meta, citations)

    pdir = processed_dir(dataset)
    with open(os.path.join(pdir, "V.pkl"), "wb") as f:
        pickle.dump(V, f)
    with open(os.path.join(pdir, "G.pkl"), "wb") as f:
        pickle.dump(G, f)
    # Save full opinions_meta so downstream scripts can look up timestamps
    # for cited-only opinions (those not present as keys in G).
    with open(os.path.join(pdir, "opinions_meta.pkl"), "wb") as f:
        pickle.dump(opinions_meta, f)

    # Summary statistics
    head_sizes = [len(e["F"]) for e in G.values()]
    avg_head   = sum(head_sizes) / len(head_sizes) if head_sizes else 0
    max_head   = max(head_sizes) if head_sizes else 0

    print(f"[{dataset.upper()}] |V| = {len(V):,}")
    print(f"[{dataset.upper()}] |G| = {len(G):,} F-arcs")
    print(f"[{dataset.upper()}] avg |F(u)| = {avg_head:.1f}  max |F(u)| = {max_head}")
    print(f"[{dataset.upper()}] Saved to {pdir}/")


def main():
    parser = argparse.ArgumentParser(
        description="Construct the directed citation F-hypergraph G."
    )
    parser.add_argument(
        "--dataset", choices=["scotus", "echr", "both"], default="scotus"  # ECHR disabled until data is fetched
    )
    args = parser.parse_args()

    if args.dataset in ("scotus", "both"):
        process("scotus")
    if args.dataset in ("echr", "both"):
        echr_path = os.path.join(raw_dir("echr"), "judgments.jsonl")
        if not os.path.exists(echr_path):
            print("[ECHR] Skipped — run 01_fetch_data.py --dataset echr first.")
        else:
            process("echr")


if __name__ == "__main__":
    main()
