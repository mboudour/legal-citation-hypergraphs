#!/usr/bin/env python3
"""
01f_echr_prepare_pipeline.py
-----------------------------
Convert the ECHR processed data (G.pkl, opinions_meta.pkl) into the
raw format expected by the pipeline (judgments.jsonl + citations.csv),
then run scripts 02-06 on the ECHR dataset.

Usage:
    python3 src/01f_echr_prepare_pipeline.py
"""

import csv, json, logging, os, pickle, subprocess, sys

logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
log = logging.getLogger(__name__)

ROOT       = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
PROC_DIR   = os.path.join(ROOT, "data", "processed", "echr")
RAW_DIR    = os.path.join(ROOT, "data", "raw", "echr")
os.makedirs(RAW_DIR, exist_ok=True)

# ------------------------------------------------------------------
# 1. Load processed data
# ------------------------------------------------------------------
log.info("Loading G.pkl and opinions_meta.pkl …")
with open(os.path.join(PROC_DIR, "G.pkl"), "rb") as fh:
    G = pickle.load(fh)
with open(os.path.join(PROC_DIR, "opinions_meta.pkl"), "rb") as fh:
    meta = pickle.load(fh)

log.info("G: %d nodes, %d edges", G.number_of_nodes(), G.number_of_edges())
log.info("Meta: %d entries", len(meta))

# ------------------------------------------------------------------
# 2. Write judgments.jsonl
# ------------------------------------------------------------------
jpath = os.path.join(RAW_DIR, "judgments.jsonl")
log.info("Writing %s …", jpath)
with open(jpath, "w") as fh:
    for iid, m in meta.items():
        # Convert date to YYYY-MM-DD for pipeline compatibility
        dt = m.get("date")
        if dt is not None:
            tau_str = dt.strftime("%Y-%m-%d")
        else:
            tau_str = ""
        rec = {
            "itemid":        iid,
            "docname":       m.get("name", iid),
            "judgementdate": tau_str,
        }
        fh.write(json.dumps(rec) + "\n")
log.info("Wrote %d records to judgments.jsonl", len(meta))

# ------------------------------------------------------------------
# 3. Write citations.csv
# ------------------------------------------------------------------
cpath = os.path.join(RAW_DIR, "citations.csv")
log.info("Writing %s …", cpath)
with open(cpath, "w", newline="") as fh:
    writer = csv.DictWriter(fh, fieldnames=["citing_id", "cited_id"])
    writer.writeheader()
    n = 0
    for u, v in G.edges():
        writer.writerow({"citing_id": u, "cited_id": v})
        n += 1
log.info("Wrote %d citation edges to citations.csv", n)

# ------------------------------------------------------------------
# 4. Run pipeline scripts 02-06 on ECHR
# ------------------------------------------------------------------
scripts = [
    ("02_induce_singleton_hypergraph.py", ["--dataset", "echr"]),
    ("03_lift_hypergraph.py",             ["--dataset", "echr"]),
    ("04_structural_profiling.py",        ["--dataset", "echr"]),
    ("05_legal_argument_space.py",        ["--dataset", "echr"]),
    ("06_empirical_analysis.py",          ["--dataset", "echr"]),
]

src_dir = os.path.join(ROOT, "src")
python  = sys.executable

for script, args in scripts:
    script_path = os.path.join(src_dir, script)
    if not os.path.exists(script_path):
        log.warning("Script not found: %s — skipping", script_path)
        continue
    cmd = [python, script_path] + args
    log.info("Running: %s", " ".join(cmd))
    result = subprocess.run(cmd, cwd=ROOT, capture_output=False, text=True)
    if result.returncode != 0:
        log.error("Script %s failed with return code %d", script, result.returncode)
        sys.exit(result.returncode)
    log.info("Script %s completed successfully", script)

log.info("All ECHR pipeline scripts completed.")
