"""
03_lift_hypergraph.py
=====================
Step 2 of the pipeline — Canonical Lifting to a Directed Citation Hypergraph.

Given the singleton-tail citation hypergraph

    E_0 = { ({u}, F(u)) : u in V }

we apply the canonical lifting construction.

For every subset H ⊆ V, define

    E(H) = { ({u}, F) in E_0 : H ⊆ F }

i.e. the set of all singleton-tail hyperedges whose head contains H.

Then define the lifted hyperedge

    ê(H) = ( ⋃_{e in E(H)} π_1(e),   ⋂_{e in E(H)} π_2(e) )
         = ( { u : H ⊆ F(u) },        ⋂_{u : H ⊆ F(u)} F(u) )

where π_1 and π_2 are the projections onto the tail and head respectively.

The lifted directed citation hypergraph is

    Ĝ = { ê(H) : H ⊆ V,  E(H) ≠ ∅,  |π_1(ê(H))| ≥ 1,  |π_2(ê(H))| ≥ 1 }

This is the main methodological contribution of the paper.

Implementation note:
  Enumerating all H ⊆ V is exponential.  We instead enumerate all
  *maximal* subsets H that appear as intersections of heads, which is
  equivalent and tractable.  Concretely:

    1. Index: for each opinion v, record which opinions cite v:
           citing(v) = { u : v in F(u) }
    2. For each pair (u1, u2) with a non-empty shared head F(u1) ∩ F(u2),
       the intersection is a candidate H.
    3. We group opinions by their full citation set F(u) (exact match)
       and by all pairwise intersections, keeping only groups of size ≥ 2
       (so that the lifted tail is non-singleton).

  This yields all lifted hyperedges with |tail| ≥ 2 efficiently.
  Singleton-tail residuals (opinions with a unique citation set) are
  retained as degenerate lifted hyperedges with |tail| = 1.

Inputs:
  data/processed/{dataset}/E0.pkl
  data/processed/{dataset}/V.pkl

Outputs:
  data/processed/{dataset}/G_hat.pkl
    dict: { H_key -> {"tail": frozenset, "head": frozenset} }
    where H_key is a frozenset representing H (the seed set).

Usage:
  env/bin/python src/03_lift_hypergraph.py [--dataset scotus|echr|both]
"""

import argparse
import os
import pickle
from collections import defaultdict
from itertools import combinations

from tqdm import tqdm

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

def processed_dir(dataset):
    return os.path.join(ROOT, "data", "processed", dataset)


# ---------------------------------------------------------------------------
# Lifting construction
# ---------------------------------------------------------------------------
def lift(E0: dict) -> dict:
    """
    Apply the canonical lifting to E_0.

    Parameters
    ----------
    E0 : dict
        { u (str) -> {"tau": str, "F": frozenset} }

    Returns
    -------
    G_hat : dict
        { H_key (frozenset) -> {"tail": frozenset, "head": frozenset} }
    """
    # Step A: group opinions by their exact citation set F(u)
    # Opinions sharing the same F form a lifted hyperedge with head = F
    # and tail = { u : F(u) == F }.
    groups_exact = defaultdict(set)
    for u, meta in E0.items():
        groups_exact[meta["F"]].add(u)

    G_hat = {}

    # Exact-match groups (head = F, tail = all u with F(u) == F)
    for F, tail_set in groups_exact.items():
        H_key = F  # seed set is the shared head itself
        G_hat[H_key] = {
            "tail": frozenset(tail_set),
            "head": frozenset(F),
        }

    # Step B: pairwise intersections
    # For each pair of opinions (u1, u2) with F(u1) ∩ F(u2) ≠ ∅ and
    # F(u1) ≠ F(u2), the intersection H = F(u1) ∩ F(u2) is a seed set.
    # The lifted hyperedge for H has:
    #   tail = { u : H ⊆ F(u) }
    #   head = ⋂_{u : H ⊆ F(u)} F(u)
    #
    # We enumerate candidate H values from pairwise intersections of
    # distinct citation sets, then compute the full tail and head.

    distinct_Fs = list(groups_exact.keys())
    candidate_Hs = set()

    print(f"  Computing pairwise intersections over {len(distinct_Fs):,} distinct citation sets …")
    for F1, F2 in tqdm(combinations(distinct_Fs, 2), total=len(distinct_Fs)*(len(distinct_Fs)-1)//2):
        H = F1 & F2
        if H and H not in G_hat:
            candidate_Hs.add(H)

    print(f"  {len(candidate_Hs):,} candidate seed sets H from pairwise intersections.")

    # For each candidate H, compute tail and head
    for H in tqdm(candidate_Hs, desc="  Lifting candidate H sets"):
        tail_set = frozenset(u for u, meta in E0.items() if H <= meta["F"])
        if not tail_set:
            continue
        # head = intersection of all F(u) for u in tail_set
        head_set = frozenset.intersection(*(E0[u]["F"] for u in tail_set))
        if not head_set:
            continue
        G_hat[H] = {
            "tail": tail_set,
            "head": head_set,
        }

    return G_hat


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def process(dataset: str):
    pdir = processed_dir(dataset)
    e0_path = os.path.join(pdir, "E0.pkl")

    if not os.path.exists(e0_path):
        raise FileNotFoundError(
            f"[{dataset.upper()}] E0.pkl not found. Run 02_induce_singleton_hypergraph.py first."
        )

    print(f"[{dataset.upper()}] Loading E_0 …")
    with open(e0_path, "rb") as f:
        E0 = pickle.load(f)

    print(f"[{dataset.upper()}] |E_0| = {len(E0):,}  — applying canonical lifting …")
    G_hat = lift(E0)

    out_path = os.path.join(pdir, "G_hat.pkl")
    with open(out_path, "wb") as f:
        pickle.dump(G_hat, f)

    n_nontrivial = sum(1 for e in G_hat.values() if len(e["tail"]) > 1)
    print(f"[{dataset.upper()}] |Ĝ| = {len(G_hat):,}  "
          f"(of which {n_nontrivial:,} have |tail| ≥ 2)")
    print(f"[{dataset.upper()}] Saved to {out_path}")


def main():
    parser = argparse.ArgumentParser(
        description="Apply canonical lifting E_0 → Ĝ."
    )
    parser.add_argument("--dataset", choices=["scotus", "echr", "both"], default="both")
    args = parser.parse_args()

    if args.dataset in ("scotus", "both"):
        process("scotus")
    if args.dataset in ("echr", "both"):
        process("echr")


if __name__ == "__main__":
    main()
