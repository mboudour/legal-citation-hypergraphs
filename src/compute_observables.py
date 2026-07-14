"""
compute_observables.py
Compute new observables enabled by the F-hypergraph representation:
  1. Head cardinality distribution (with decade breakdown)
  2. Head overlap (pairwise Jaccard on a sample)
  3. Recurrent precedent bundles (most frequent head sets)
  4. Higher-order citation motifs
  5. Summary statistics JSON for embedding in the manuscript
"""
import pickle, json, collections, itertools
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
from pathlib import Path

# ── paths ─────────────────────────────────────────────────────────────────────
DATA = Path("/home/ubuntu/upload/scotus_results/scotus")
FIGS = Path("/home/ubuntu/jcllt_manuscript/figures")
FIGS.mkdir(exist_ok=True)

STYLE = {
    "axes.spines.top":    False,
    "axes.spines.right":  False,
    "axes.grid":          True,
    "grid.alpha":         0.35,
    "font.family":        "serif",
    "font.size":          11,
}
plt.rcParams.update(STYLE)

# ── load data ─────────────────────────────────────────────────────────────────
print("Loading …")
with open(DATA / "G.pkl",        "rb") as f: G        = pickle.load(f)
with open(DATA / "profiles.pkl", "rb") as f: profiles = pickle.load(f)

# G: dict  opinion_id -> {'tau': date_str, 'case_name': str, 'F': frozenset(cited_ids)}
# profiles: dict  opinion_id -> {citing_id, head_size, closure, brokerage, …}

# ── build working dataframe ───────────────────────────────────────────────────
rows = []
for oid, rec in G.items():
    tau = rec.get("tau", "")
    year = int(tau[:4]) if tau and len(tau) >= 4 else None
    head = rec.get("F", frozenset())
    rows.append({"opinion_id": oid, "year": year, "head_size": len(head), "head": head})

df = pd.DataFrame(rows).dropna(subset=["year"])
df["year"]   = df["year"].astype(int)
df["decade"] = (df["year"] // 10) * 10

# Keep only F-arcs with head_size >= 1 (citing at least one precedent)
df_arc = df[df["head_size"] >= 1].copy()
print(f"Total F-arcs with ≥1 cited precedent: {len(df_arc):,}")
print(f"Head size: mean={df_arc.head_size.mean():.2f}, "
      f"median={df_arc.head_size.median():.0f}, "
      f"max={df_arc.head_size.max()}")

# ── 1. HEAD CARDINALITY DISTRIBUTION ─────────────────────────────────────────
fig, axes = plt.subplots(1, 2, figsize=(11, 4.2))

# 1a. Overall histogram (log-scale y)
ax = axes[0]
counts = df_arc["head_size"].value_counts().sort_index()
ax.bar(counts.index, counts.values, color="#2c5f8a", alpha=0.85, width=0.8)
ax.set_yscale("log")
ax.set_xlabel("Head cardinality $|H|$")
ax.set_ylabel("Number of F-arcs (log scale)")
ax.set_title("(a) Head Cardinality Distribution")
ax.set_xlim(0, min(counts.index.max() + 1, 80))

# 1b. Median head size per decade
med_decade = df_arc.groupby("decade")["head_size"].median()
ax2 = axes[1]
ax2.plot(med_decade.index, med_decade.values, color="#2c5f8a",
         linewidth=2, marker="o", markersize=4)
ax2.set_xlabel("Decade")
ax2.set_ylabel("Median $|H|$")
ax2.set_title("(b) Median Head Cardinality per Decade")
ax2.xaxis.set_major_locator(mticker.MultipleLocator(20))
ax2.tick_params(axis="x", rotation=45)

fig.tight_layout()
fig.savefig(FIGS / "head_cardinality.pdf", bbox_inches="tight")
plt.close(fig)
print("Saved head_cardinality.pdf")

# ── 2. HEAD OVERLAP (pairwise Jaccard on a stratified sample) ─────────────────
# Sample up to 3000 F-arcs with head_size >= 2 for tractability
sample = df_arc[df_arc["head_size"] >= 2].sample(
    min(3000, len(df_arc[df_arc["head_size"] >= 2])), random_state=42
)
heads = list(sample["head"])

def jaccard(a, b):
    if not a and not b: return 1.0
    return len(a & b) / len(a | b)

print(f"Computing pairwise Jaccard on {len(heads)} F-arcs …")
# Compute a random subsample of 50k pairs
rng = np.random.default_rng(42)
n = len(heads)
idx_i = rng.integers(0, n, 50_000)
idx_j = rng.integers(0, n, 50_000)
mask  = idx_i != idx_j
idx_i, idx_j = idx_i[mask], idx_j[mask]
jacs = np.array([jaccard(heads[i], heads[j]) for i, j in zip(idx_i, idx_j)])

fig, ax = plt.subplots(figsize=(6, 4))
ax.hist(jacs, bins=50, color="#2c5f8a", alpha=0.85, edgecolor="white")
ax.set_xlabel("Pairwise Jaccard similarity of F-arc heads")
ax.set_ylabel("Count")
ax.set_title("Head Overlap Between F-arcs")
med_jac = float(np.median(jacs))
ax.axvline(med_jac, color="#c0392b", linewidth=1.8, linestyle="--",
           label=f"Median = {med_jac:.3f}")
ax.legend(fontsize=10)
fig.tight_layout()
fig.savefig(FIGS / "head_overlap.pdf", bbox_inches="tight")
plt.close(fig)
print(f"Saved head_overlap.pdf  (median Jaccard = {med_jac:.4f})")

# ── 3. RECURRENT PRECEDENT BUNDLES ────────────────────────────────────────────
# Count how often each frozen head set appears
bundle_counts = collections.Counter(
    row["head"] for _, row in df_arc[df_arc["head_size"] >= 2].iterrows()
)
top_bundles = bundle_counts.most_common(20)
print("\nTop-10 recurrent precedent bundles (by head set):")
for i, (bundle, cnt) in enumerate(top_bundles[:10]):
    print(f"  {i+1:2d}. count={cnt:4d}  size={len(bundle)}  ids={sorted(bundle)[:3]}…")

# Fraction of F-arcs that share a bundle with at least one other F-arc
repeated = sum(1 for _, c in bundle_counts.items() if c > 1)
total_in_repeated = sum(c for _, c in bundle_counts.items() if c > 1)
print(f"\nDistinct bundles appearing >1 time: {repeated:,}")
print(f"F-arcs sharing a bundle: {total_in_repeated:,} "
      f"({100*total_in_repeated/len(df_arc[df_arc['head_size']>=2]):.1f}%)")

# ── 4. HIGHER-ORDER CITATION MOTIFS ───────────────────────────────────────────
# Build a lookup: cited_id -> list of F-arc opinion_ids that cite it
cited_by = collections.defaultdict(set)
for oid, rec in G.items():
    for cited in rec.get("F", frozenset()):
        cited_by[cited].add(oid)

# Motif counts
chain       = 0   # A cites {B}, B cites {C}  (simple chain)
fan_out     = 0   # A cites {B, C, D, …}  (|head|>=3)
fan_in      = 0   # multiple opinions all cite B (|cited_by[B]|>=3)
convergence = 0   # A and B both cite {C, D} (shared bundle of size>=2)

# fan_out: F-arcs with head_size >= 3
fan_out = int((df_arc["head_size"] >= 3).sum())

# fan_in: precedents cited by >= 3 different F-arcs
fan_in = sum(1 for v in cited_by.values() if len(v) >= 3)

# chain: opinion A cites B, and B itself has an F-arc (i.e., B is also a citing opinion)
citing_set = set(G.keys())
chain = sum(
    1 for oid, rec in G.items()
    for cited in rec.get("F", frozenset())
    if cited in citing_set
)

# convergence: pairs of F-arcs sharing >=2 cited precedents
# (approximate via bundle_counts: bundles of size>=2 appearing >=2 times)
convergence = sum(c * (c-1) // 2
                  for bundle, c in bundle_counts.items()
                  if len(bundle) >= 2 and c >= 2)

print(f"\nMotif counts:")
print(f"  Chain       (A→B→C):          {chain:>10,}")
print(f"  Fan-out     (|head|≥3):        {fan_out:>10,}")
print(f"  Fan-in      (cited by ≥3):     {fan_in:>10,}")
print(f"  Convergence (shared bundle≥2): {convergence:>10,}")

# Plot motif bar chart
motif_labels = ["Chain\n(A->B->C)", "Fan-out\n(|H|>=3)", "Fan-in\n(cited>=3)", "Convergence\n(shared bundle)"]
motif_vals   = [chain, fan_out, fan_in, convergence]

fig, ax = plt.subplots(figsize=(7, 4))
bars = ax.bar(motif_labels, motif_vals, color=["#2c5f8a","#e67e22","#27ae60","#8e44ad"],
              alpha=0.88, edgecolor="white")
ax.set_ylabel("Count")
ax.set_title("Higher-Order Citation Motifs in the SCOTUS F-Hypergraph")
for bar, val in zip(bars, motif_vals):
    ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() * 1.02,
            f"{val:,}", ha="center", va="bottom", fontsize=9)
fig.tight_layout()
fig.savefig(FIGS / "citation_motifs.pdf", bbox_inches="tight")
plt.close(fig)
print("Saved citation_motifs.pdf")

# ── 5. SUMMARY STATS JSON ─────────────────────────────────────────────────────
stats = {
    "total_farcs":              int(len(df_arc)),
    "total_opinions":           int(len(df)),
    "mean_head_size":           round(float(df_arc.head_size.mean()), 2),
    "median_head_size":         int(df_arc.head_size.median()),
    "max_head_size":            int(df_arc.head_size.max()),
    "farcs_head_ge2":           int((df_arc.head_size >= 2).sum()),
    "farcs_head_ge3":           int((df_arc.head_size >= 3).sum()),
    "pct_head_ge2":             round(100*(df_arc.head_size >= 2).mean(), 1),
    "pct_head_ge3":             round(100*(df_arc.head_size >= 3).mean(), 1),
    "median_jaccard_overlap":   round(med_jac, 4),
    "distinct_bundles_gt1":     int(repeated),
    "farcs_in_repeated_bundle": int(total_in_repeated),
    "motif_chain":              int(chain),
    "motif_fan_out":            int(fan_out),
    "motif_fan_in":             int(fan_in),
    "motif_convergence":        int(convergence),
}

with open("/home/ubuntu/jcllt_manuscript/observables_stats.json", "w") as f:
    json.dump(stats, f, indent=2)
print("\nSaved observables_stats.json")
print(json.dumps(stats, indent=2))
