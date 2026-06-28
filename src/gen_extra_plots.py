"""
Generate additional time-evolution figures for the PTRS-A manuscript.

Data structures (from probe):
  profiles.pkl  : dict {citing_id -> {citing_id, head_size, closure, brokerage,
                                       density, authority_concentration,
                                       temporal_span, community_dispersion}}
  opinions_meta : dict {opinion_id -> {tau: 'YYYY-MM-DD', case_name: str}}
  las_clusters  : dict {'kmeans': ndarray, 'agglomerative': ndarray}
  las_features  : dict {'keys': list[str], 'X_raw': ..., 'X_scaled': ...}
  G.pkl         : dict {citing_id -> {tau, case_name, F: frozenset(head_ids)}}

Figures produced:
  1. closure_brokerage_combined.pdf
  2. head_size_violin.pdf
  3. cluster_fraction_decade.pdf
  4. authority_span_decade.pdf
  5. temporal_trajectory_full.pdf
"""

import pickle, warnings
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
from matplotlib.gridspec import GridSpec

warnings.filterwarnings("ignore")

DATA_DIR = "/home/ubuntu/upload/scotus_results/scotus"
OUT_DIR  = "/home/ubuntu/jcllt_manuscript/figures"

def load(name):
    with open(f"{DATA_DIR}/{name}", "rb") as f:
        return pickle.load(f)

profiles      = load("profiles.pkl")
opinions_meta = load("opinions_meta.pkl")
las_clusters  = load("las_clusters.pkl")
las_features  = load("las_features.pkl")
G             = load("G.pkl")

# ── build master DataFrame ────────────────────────────────────────────────────
rows = list(profiles.values())
df = pd.DataFrame(rows)
print("profiles columns:", df.columns.tolist())
print("profiles shape:  ", df.shape)

# attach year from G (which has tau per citing opinion)
def get_year(cid):
    meta = G.get(str(cid), {})
    tau  = meta.get("tau", "")
    if tau and len(tau) >= 4:
        try:
            return int(tau[:4])
        except Exception:
            pass
    # fallback: opinions_meta
    meta2 = opinions_meta.get(str(cid), {})
    tau2  = meta2.get("tau", "")
    if tau2 and len(tau2) >= 4:
        try:
            return int(tau2[:4])
        except Exception:
            pass
    return None

df["year"] = df["citing_id"].apply(get_year)
df = df.dropna(subset=["year"])
df["year"] = df["year"].astype(int)
df = df[df["year"].between(1791, 2024)]
df["decade"] = (df["year"] // 10) * 10

print(f"Rows with year: {len(df)}, decade range: {df['decade'].min()}–{df['decade'].max()}")

# ── attach cluster labels (use kmeans) ────────────────────────────────────────
keys_order = las_features["keys"]   # list of citing_ids in feature-matrix order
kmeans_labels = las_clusters["kmeans"]
key_to_cluster = {k: int(c) for k, c in zip(keys_order, kmeans_labels)}
df["cluster"] = df["citing_id"].map(key_to_cluster)

# ── decade aggregation ────────────────────────────────────────────────────────
metric_cols = ["closure", "brokerage", "authority_concentration",
               "temporal_span", "community_dispersion", "head_size"]
metric_cols = [c for c in metric_cols if c in df.columns]

decade_agg = df.groupby("decade")[metric_cols].agg(["mean", "sem"]).reset_index()
decades = decade_agg["decade"].values

def get_ms(col):
    if col not in metric_cols:
        return None, None
    try:
        m = decade_agg[(col, "mean")].values.astype(float)
        s = decade_agg[(col, "sem")].values.astype(float)
        return m, s
    except Exception:
        return None, None

# ── matplotlib style ──────────────────────────────────────────────────────────
plt.rcParams.update({
    "font.family":       "serif",
    "font.serif":        ["Times New Roman", "DejaVu Serif"],
    "font.size":         10,
    "axes.titlesize":    11,
    "axes.labelsize":    10,
    "legend.fontsize":   9,
    "xtick.labelsize":   9,
    "ytick.labelsize":   9,
    "figure.dpi":        150,
    "axes.spines.top":   False,
    "axes.spines.right": False,
})

BLUE   = "#1f77b4"
RED    = "#d62728"
GREEN  = "#2ca02c"
ORANGE = "#ff7f0e"
PURPLE = "#9467bd"
GREY   = "#7f7f7f"

# ─────────────────────────────────────────────────────────────────────────────
# FIGURE 1: Closure + Brokerage dual-axis combined
# ─────────────────────────────────────────────────────────────────────────────
cl_m, cl_s = get_ms("closure")
br_m, br_s = get_ms("brokerage")

if cl_m is not None and br_m is not None:
    fig, ax1 = plt.subplots(figsize=(7, 3.5))
    ax2 = ax1.twinx()
    ax2.spines["right"].set_visible(True)

    ax1.plot(decades, cl_m, color=BLUE, lw=2, marker="o", ms=4, label="Closure")
    ax1.fill_between(decades, np.maximum(0, cl_m - cl_s), cl_m + cl_s,
                     color=BLUE, alpha=0.18)
    ax1.set_ylabel("Mean Closure", color=BLUE)
    ax1.tick_params(axis="y", labelcolor=BLUE)
    ax1.set_ylim(bottom=0)

    ax2.plot(decades, br_m, color=RED, lw=2, marker="s", ms=4,
             linestyle="--", label="Brokerage")
    ax2.fill_between(decades, np.maximum(0, br_m - br_s), br_m + br_s,
                     color=RED, alpha=0.18)
    ax2.set_ylabel("Mean Brokerage", color=RED)
    ax2.tick_params(axis="y", labelcolor=RED)

    ax1.set_xlabel("Decade")
    ax1.set_title("Closure and Brokerage over SCOTUS History")
    lines1, l1 = ax1.get_legend_handles_labels()
    lines2, l2 = ax2.get_legend_handles_labels()
    ax1.legend(lines1 + lines2, l1 + l2, loc="center right", framealpha=0.9)
    plt.tight_layout()
    plt.savefig(f"{OUT_DIR}/closure_brokerage_combined.pdf", bbox_inches="tight")
    plt.close()
    print("✓ closure_brokerage_combined.pdf")

# ─────────────────────────────────────────────────────────────────────────────
# FIGURE 2: Head-size distribution per decade (box plot)
# ─────────────────────────────────────────────────────────────────────────────
if "head_size" in df.columns:
    all_dec = sorted(df["decade"].unique())
    # keep at most 14 decades
    if len(all_dec) > 14:
        idx = np.round(np.linspace(0, len(all_dec)-1, 14)).astype(int)
        sel = [all_dec[i] for i in idx]
    else:
        sel = all_dec

    plot_data, used_dec = [], []
    for d in sel:
        vals = df.loc[df["decade"] == d, "head_size"].dropna().values
        if len(vals) >= 3:
            plot_data.append(vals)
            used_dec.append(d)

    if plot_data:
        fig, ax = plt.subplots(figsize=(9, 3.8))
        bp = ax.boxplot(plot_data, patch_artist=True, notch=False,
                        medianprops=dict(color="black", lw=1.5),
                        whiskerprops=dict(color=GREY),
                        capprops=dict(color=GREY),
                        flierprops=dict(marker=".", color=GREY, alpha=0.3, ms=2),
                        showfliers=False)
        for patch in bp["boxes"]:
            patch.set_facecolor(BLUE)
            patch.set_alpha(0.55)
        ax.set_xticks(range(1, len(used_dec)+1))
        ax.set_xticklabels([str(d) for d in used_dec], rotation=45, ha="right")
        ax.set_xlabel("Decade")
        ax.set_ylabel("Head size $|H|$")
        ax.set_title("Distribution of F-arc Head Size per Decade")
        plt.tight_layout()
        plt.savefig(f"{OUT_DIR}/head_size_violin.pdf", bbox_inches="tight")
        plt.close()
        print("✓ head_size_violin.pdf")

# ─────────────────────────────────────────────────────────────────────────────
# FIGURE 3: Cluster fraction per decade (stacked area)
# ─────────────────────────────────────────────────────────────────────────────
if "cluster" in df.columns and df["cluster"].notna().sum() > 0:
    ct = df.dropna(subset=["cluster"]).groupby(["decade", "cluster"]).size().unstack(fill_value=0)
    ct_frac = ct.div(ct.sum(axis=1), axis=0)
    n_cl = ct_frac.shape[1]
    palette = plt.cm.tab10(np.linspace(0, 0.9, n_cl))

    fig, ax = plt.subplots(figsize=(7.5, 3.8))
    ax.stackplot(ct_frac.index, ct_frac.T.values,
                 labels=[f"Cluster {c}" for c in ct_frac.columns],
                 colors=palette, alpha=0.82)
    ax.set_xlabel("Decade")
    ax.set_ylabel("Fraction of F-arcs")
    ax.set_title("Typology Mix of Legal Synthesis per Decade")
    ax.legend(loc="upper left", ncol=2, fontsize=8, framealpha=0.85)
    ax.set_ylim(0, 1)
    plt.tight_layout()
    plt.savefig(f"{OUT_DIR}/cluster_fraction_decade.pdf", bbox_inches="tight")
    plt.close()
    print("✓ cluster_fraction_decade.pdf")

# ─────────────────────────────────────────────────────────────────────────────
# FIGURE 4: Authority concentration + Temporal span dual-axis
# ─────────────────────────────────────────────────────────────────────────────
au_m, au_s = get_ms("authority_concentration")
sp_m, sp_s = get_ms("temporal_span")

if au_m is not None and sp_m is not None:
    # filter out negative temporal span (missing data)
    valid = sp_m >= 0
    fig, ax1 = plt.subplots(figsize=(7, 3.5))
    ax2 = ax1.twinx()
    ax2.spines["right"].set_visible(True)

    ax1.plot(decades[valid], au_m[valid], color=ORANGE, lw=2, marker="^", ms=4,
             label="Authority Conc.")
    ax1.fill_between(decades[valid],
                     np.maximum(0, au_m[valid] - au_s[valid]),
                     au_m[valid] + au_s[valid],
                     color=ORANGE, alpha=0.18)
    ax1.set_ylabel("Mean Authority Concentration", color=ORANGE)
    ax1.tick_params(axis="y", labelcolor=ORANGE)
    ax1.set_ylim(bottom=0)

    ax2.plot(decades[valid], sp_m[valid], color=PURPLE, lw=2, marker="D", ms=4,
             linestyle="--", label="Temporal Span (yrs)")
    ax2.fill_between(decades[valid],
                     np.maximum(0, sp_m[valid] - sp_s[valid]),
                     sp_m[valid] + sp_s[valid],
                     color=PURPLE, alpha=0.18)
    ax2.set_ylabel("Mean Temporal Span (years)", color=PURPLE)
    ax2.tick_params(axis="y", labelcolor=PURPLE)

    ax1.set_xlabel("Decade")
    ax1.set_title("Authority Concentration and Temporal Span over SCOTUS History")
    lines1, l1 = ax1.get_legend_handles_labels()
    lines2, l2 = ax2.get_legend_handles_labels()
    ax1.legend(lines1 + lines2, l1 + l2, loc="upper left", framealpha=0.9)
    plt.tight_layout()
    plt.savefig(f"{OUT_DIR}/authority_span_decade.pdf", bbox_inches="tight")
    plt.close()
    print("✓ authority_span_decade.pdf")

# ─────────────────────────────────────────────────────────────────────────────
# FIGURE 5: Full 5-panel temporal trajectory
# ─────────────────────────────────────────────────────────────────────────────
metrics_spec = [
    ("closure",                 "Closure",                  BLUE),
    ("brokerage",               "Brokerage",                RED),
    ("authority_concentration", "Authority Concentration",  ORANGE),
    ("temporal_span",           "Temporal Span (years)",    PURPLE),
    ("community_dispersion",    "Community Dispersion",     GREEN),
]

available = []
for col, label, color in metrics_spec:
    m, s = get_ms(col)
    if m is not None:
        available.append((col, label, color, m, s))

if available:
    n = len(available)
    ncols = 3
    nrows = (n + ncols - 1) // ncols
    fig = plt.figure(figsize=(12, 3.5 * nrows))
    gs  = GridSpec(nrows, ncols, figure=fig, hspace=0.5, wspace=0.38)

    for idx, (col, label, color, m, s) in enumerate(available):
        row, col_idx = divmod(idx, ncols)
        ax = fig.add_subplot(gs[row, col_idx])
        # filter valid (non-negative for span)
        valid = m >= 0
        ax.plot(decades[valid], m[valid], color=color, lw=2, marker="o", ms=3)
        ax.fill_between(decades[valid],
                        np.maximum(0, m[valid] - s[valid]),
                        m[valid] + s[valid],
                        color=color, alpha=0.2)
        ax.set_title(label)
        ax.set_xlabel("Decade")
        ax.set_ylabel("Mean value")
        ax.set_ylim(bottom=0)
        ax.xaxis.set_major_locator(mticker.MultipleLocator(50))

    for idx in range(len(available), nrows * ncols):
        row, col_idx = divmod(idx, ncols)
        fig.add_subplot(gs[row, col_idx]).set_visible(False)

    fig.suptitle(
        "Temporal Evolution of Structural Profile Metrics — SCOTUS (1791–2024)",
        fontsize=12, y=1.01)
    plt.savefig(f"{OUT_DIR}/temporal_trajectory_full.pdf", bbox_inches="tight")
    plt.close()
    print("✓ temporal_trajectory_full.pdf")

print("\nAll extra plots done.")
