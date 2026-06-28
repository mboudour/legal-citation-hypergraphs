"""
02_induce_singleton_hypergraph.py
==================================
Step 1 of the pipeline — Induced Singleton-Tail Citation Hypergraph.

Given an opinion-level legal dataset in which each opinion u has:
  - a timestamp  tau(u)
  - a citation set F(u)  (the set of opinions cited by u)

we construct the singleton-tail citation hypergraph

    E_0 = { ({u}, F(u)) : u in V }

where V is the universal vertex set of all opinions appearing in the corpus
(whether as citing or cited).

This is the natural data model for opinion-level legal datasets (SCOTUS,
ECHR, CourtListener, Caselaw Access Project, …).  It is NOT claimed to be
a general directed hypergraph; it is the starting point for the canonical
lifting in Step 2.

Inputs (from data/raw/):
  data/raw/scotus/opinions.jsonl
  data/raw/scotus/citations.csv
  data/raw/echr/judgments.jsonl
  data/raw/echr/citations.csv

Outputs (written to data/processed/):
  data/processed/scotus/E0.pkl   — dict: { u: {"tau": timestamp, "F": frozenset} }
  data/processed/scotus/V.pkl    — set: universal vertex set
  data/processed/echr/E0.pkl
  data/processed/echr/V.pkl

Usage:
  env/bin/python src/02_induce_singleton_hypergraph.py [--dataset scotus|echr|both]
"""

import argparse
import csv
import json
import os
import pickle

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
def build_E0(opinions_meta: dict, citations: dict) -> tuple:
    """
    Build the singleton-tail citation hypergraph E_0.

    Parameters
    ----------
    opinions_meta : dict
        { opinion_id (str) -> {"tau": timestamp_str, ...} }
    citations : dict
        { citing_id (str) -> set of cited_ids (str) }

    Returns
    -------
    V  : set of all vertex IDs (union of all citing and cited opinions)
    E0 : dict { u -> {"tau": tau(u), "F": frozenset(F(u))} }
         Only entries where F(u) is non-empty are included.
    """
    V  = set(opinions_meta.keys())
    E0 = {}

    for u, meta in opinions_meta.items():
        F_u = frozenset(citations.get(u, set()))
        if not F_u:
            continue  # skip opinions that cite nothing
        V.update(F_u)  # cited opinions not in opinions_meta still belong to V
        E0[u] = {
            "tau": meta.get("tau", ""),
            "F":   F_u,
        }

    return V, E0


# ---------------------------------------------------------------------------
# SCOTUS loader
# ---------------------------------------------------------------------------
def load_scotus():
    rdir = raw_dir("scotus")
    opinions_path  = os.path.join(rdir, "opinions.jsonl")
    citations_path = os.path.join(rdir, "citations.csv")

    if not os.path.exists(opinions_path) or not os.path.exists(citations_path):
        raise FileNotFoundError(
            "[SCOTUS] Raw files not found. Run 01_fetch_data.py first."
        )

    # Load opinion metadata.
    # In CourtListener v4, date_filed and case_name live on the Cluster,
    # not the Opinion.  The Opinion has a 'cluster' field with a URL like
    # https://www.courtlistener.com/api/rest/v4/clusters/12345/
    # We extract the cluster_id from that URL and use it as a proxy key.
    # Dates are resolved by fetching clusters lazily (cached in clusters.jsonl).
    clusters_path = os.path.join(rdir, "clusters.jsonl")
    cluster_cache = {}
    if os.path.exists(clusters_path):
        with open(clusters_path, encoding="utf-8") as fc:
            for line in fc:
                cl = json.loads(line)
                cluster_cache[str(cl.get("id", ""))] = cl

    opinions_meta = {}
    with open(opinions_path, encoding="utf-8") as f:
        for line in f:
            op = json.loads(line)
            op_id = str(op.get("id", ""))
            if not op_id:
                continue
            # Extract cluster_id from cluster URL
            cluster_url = op.get("cluster", "")
            cluster_id  = cluster_url.rstrip("/").split("/")[-1] if cluster_url else ""
            # Resolve date_filed from cache
            tau = ""
            case_name = ""
            if cluster_id and cluster_id in cluster_cache:
                tau       = cluster_cache[cluster_id].get("date_filed", "")
                case_name = cluster_cache[cluster_id].get("case_name", "")
            opinions_meta[op_id] = {
                "tau":        tau,
                "case_name":  case_name,
                "cluster_id": cluster_id,
            }

    # Load citation edges
    citations = {}
    with open(citations_path, encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            u = row["citing_id"].strip()
            v = row["cited_id"].strip()
            if u and v:
                citations.setdefault(u, set()).add(v)

    return opinions_meta, citations


# ---------------------------------------------------------------------------
# ECHR loader
# ---------------------------------------------------------------------------
def load_echr():
    rdir = raw_dir("echr")
    judgments_path = os.path.join(rdir, "judgments.jsonl")
    citations_path = os.path.join(rdir, "citations.csv")

    if not os.path.exists(judgments_path) or not os.path.exists(citations_path):
        raise FileNotFoundError(
            "[ECHR] Raw files not found. Run 01_fetch_data.py first."
        )

    # Load judgment metadata
    opinions_meta = {}
    with open(judgments_path, encoding="utf-8") as f:
        for line in f:
            jud = json.loads(line)
            jid = str(jud.get("itemid", ""))
            if not jid:
                continue
            opinions_meta[jid] = {
                "tau":       jud.get("judgementdate", ""),
                "case_name": jud.get("docname", ""),
            }

    # Load citation edges
    citations = {}
    with open(citations_path, encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            u = row["citing_id"].strip()
            v = row["cited_id"].strip()
            if u and v:
                citations.setdefault(u, set()).add(v)

    return opinions_meta, citations


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def process(dataset: str):
    print(f"[{dataset.upper()}] Building singleton-tail citation hypergraph E_0 …")

    if dataset == "scotus":
        opinions_meta, citations = load_scotus()
    elif dataset == "echr":
        opinions_meta, citations = load_echr()
    else:
        raise ValueError(f"Unknown dataset: {dataset}")

    V, E0 = build_E0(opinions_meta, citations)

    pdir = processed_dir(dataset)
    with open(os.path.join(pdir, "V.pkl"),  "wb") as f:
        pickle.dump(V, f)
    with open(os.path.join(pdir, "E0.pkl"), "wb") as f:
        pickle.dump(E0, f)

    print(f"[{dataset.upper()}] |V| = {len(V):,}  |E_0| = {len(E0):,}")
    print(f"[{dataset.upper()}] Saved to {pdir}/")


def main():
    parser = argparse.ArgumentParser(
        description="Construct the singleton-tail citation hypergraph E_0."
    )
    parser.add_argument("--dataset", choices=["scotus", "echr", "both"], default="both")
    args = parser.parse_args()

    if args.dataset in ("scotus", "both"):
        process("scotus")
    if args.dataset in ("echr", "both"):
        process("echr")


if __name__ == "__main__":
    main()
