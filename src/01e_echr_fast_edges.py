#!/usr/bin/env python3
"""
01e_echr_fast_edges.py
----------------------
Fast vectorised extraction of ECHR citation edges from judgments_full.csv.
Uses regex on the 'scl' field to extract appnos (format: no. NNNNN/YY),
then resolves them against the 'extractedappno' index to get itemids.

Saves:
  data/processed/echr/G.pkl
  data/processed/echr/opinions_meta.pkl
  data/processed/echr/edges.csv

Usage:
    python3 src/01e_echr_fast_edges.py
"""

import logging, os, pickle, re
from datetime import datetime
from collections import defaultdict

import networkx as nx
import pandas as pd

logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
log = logging.getLogger(__name__)

INPUT_CSV  = "data/raw/echr/judgments_full.csv"
OUTPUT_DIR = "data/processed/echr/"
os.makedirs(OUTPUT_DIR, exist_ok=True)

# ------------------------------------------------------------------
# 1. Load
# ------------------------------------------------------------------
log.info("Loading %s …", INPUT_CSV)
df = pd.read_csv(INPUT_CSV)
log.info("Loaded %d rows", len(df))

# ------------------------------------------------------------------
# 2. Build appno -> itemid lookup from extractedappno field
# ------------------------------------------------------------------
# extractedappno is semicolon-separated list of appnos for this case
appno_to_itemid = {}
for _, row in df.iterrows():
    iid = str(row.get("itemid", "")).strip()
    raw = str(row.get("extractedappno", "") or "").strip()
    if not raw or raw == "nan":
        continue
    for appno in raw.split(";"):
        appno = appno.strip()
        if appno:
            appno_to_itemid[appno] = iid

log.info("Built appno->itemid lookup: %d entries", len(appno_to_itemid))

# ------------------------------------------------------------------
# 3. Extract citation edges from scl field using vectorised regex
# ------------------------------------------------------------------
# Pattern: "no. 12345/67" or "no. 12345/678"
APPNO_RE = re.compile(r'no\.\s*(\d{3,6}/\d{2,4})')

edges = []
unresolved = 0

for _, row in df.iterrows():
    citing_iid = str(row.get("itemid", "")).strip()
    if not citing_iid or citing_iid == "nan":
        continue
    scl = str(row.get("scl", "") or "").strip()
    if not scl or scl == "nan":
        continue
    cited_appnos = APPNO_RE.findall(scl)
    for ca in cited_appnos:
        cited_iid = appno_to_itemid.get(ca)
        if cited_iid and cited_iid != citing_iid:
            edges.append((citing_iid, cited_iid))
        else:
            unresolved += 1

log.info("Extracted %d citation edges (%d unresolved)", len(edges), unresolved)

# Save edges CSV
edges_df = pd.DataFrame(edges, columns=["citing_itemid", "cited_itemid"])
edges_df.to_csv(os.path.join(OUTPUT_DIR, "edges.csv"), index=False)
log.info("Saved edges.csv")

# ------------------------------------------------------------------
# 4. Build opinions_meta
# ------------------------------------------------------------------
def parse_date(s):
    if not s or str(s) == "nan":
        return None
    for fmt in ("%d/%m/%Y", "%Y-%m-%d", "%d/%m/%Y %H:%M:%S"):
        try:
            return datetime.strptime(str(s).strip(), fmt)
        except ValueError:
            pass
    return None

meta = {}
for _, row in df.iterrows():
    iid = str(row.get("itemid", "")).strip()
    if not iid or iid == "nan":
        continue
    date_str = str(row.get("judgementdate", "") or row.get("referencedate", "") or "").strip()
    dt = parse_date(date_str)
    name = str(row.get("docname", "") or iid).strip()
    meta[iid] = {
        "name":       name,
        "date":       dt,
        "date_str":   date_str,
        "appno":      str(row.get("extractedappno", "") or "").strip(),
        "importance": str(row.get("importance", "") or "").strip(),
        "violation":  str(row.get("violation", "") or "").strip(),
        "article":    str(row.get("article", "") or "").strip(),
        "respondent": str(row.get("respondent", "") or "").strip(),
    }

log.info("Built meta for %d opinions", len(meta))

# ------------------------------------------------------------------
# 5. Build NetworkX DiGraph
# ------------------------------------------------------------------
G = nx.DiGraph()
for iid, m in meta.items():
    G.add_node(iid, **m)
for u, v in edges:
    if u in G and v in G:
        G.add_edge(u, v)

log.info("Graph: %d nodes, %d edges", G.number_of_nodes(), G.number_of_edges())

# ------------------------------------------------------------------
# 6. Save
# ------------------------------------------------------------------
with open(os.path.join(OUTPUT_DIR, "G.pkl"), "wb") as fh:
    pickle.dump(G, fh)
with open(os.path.join(OUTPUT_DIR, "opinions_meta.pkl"), "wb") as fh:
    pickle.dump(meta, fh)

log.info("Saved G.pkl and opinions_meta.pkl → %s", OUTPUT_DIR)

# Quick stats
in_deg  = [d for _, d in G.in_degree()]
out_deg = [d for _, d in G.out_degree()]
log.info("In-degree  mean=%.2f  max=%d",
         sum(in_deg)/max(len(in_deg),1), max(in_deg, default=0))
log.info("Out-degree mean=%.2f  max=%d",
         sum(out_deg)/max(len(out_deg),1), max(out_deg, default=0))
log.info("Opinions with >=1 outgoing citation: %d / %d (%.1f%%)",
         sum(1 for d in out_deg if d > 0), len(out_deg),
         100*sum(1 for d in out_deg if d > 0)/max(len(out_deg), 1))

# Head size distribution (for F-hypergraph context)
out_deg_nonzero = [d for d in out_deg if d > 0]
if out_deg_nonzero:
    import statistics
    log.info("Out-degree (citing opinions) — median=%.1f  mean=%.2f  max=%d",
             statistics.median(out_deg_nonzero),
             sum(out_deg_nonzero)/len(out_deg_nonzero),
             max(out_deg_nonzero))
