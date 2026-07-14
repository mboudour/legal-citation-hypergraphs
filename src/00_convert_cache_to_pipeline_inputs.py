"""
00_convert_cache_to_pipeline_inputs.py
=======================================
One-time conversion: transforms network_from_cache.json (produced by the
jorgecarleitao/echr_network HTML-cache extractor) into the two files that
the pipeline expects in data/raw/echr/:

  judgments.jsonl   — one JSON object per line, fields:
                        itemid, docname, judgementdate, appno
  citations.csv     — two columns: citing_id, cited_id

Run this script ONCE before running the pipeline steps 02–06.

Usage (from the computations/ directory):
  python src/00_convert_cache_to_pipeline_inputs.py

Input:
  data/raw/echr/network_from_cache.json

Outputs:
  data/raw/echr/judgments.jsonl
  data/raw/echr/citations.csv
"""

import csv
import json
import os
import re
import sys

# ---------------------------------------------------------------------------
# Paths — resolve relative to this script's location
# ---------------------------------------------------------------------------
ROOT        = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
RAW_ECHR    = os.path.join(ROOT, "data", "raw", "echr")
INPUT_JSON  = os.path.join(RAW_ECHR, "network_from_cache.json")
JUDGMENTS   = os.path.join(RAW_ECHR, "judgments.jsonl")
CITATIONS   = os.path.join(RAW_ECHR, "citations.csv")


def convert(force: bool = False):
    # Guard: skip if outputs already exist
    if os.path.exists(JUDGMENTS) and os.path.exists(CITATIONS) and not force:
        print("[ECHR] judgments.jsonl and citations.csv already exist.")
        print("       Pass --force to regenerate them.")
        return

    if not os.path.exists(INPUT_JSON):
        print(f"[ERROR] Input file not found: {INPUT_JSON}")
        print("        Run the cache extraction script first to produce network_from_cache.json")
        sys.exit(1)

    print(f"[ECHR] Reading {INPUT_JSON} ...")
    with open(INPUT_JSON, encoding="utf-8") as f:
        network = json.load(f)

    print(f"[ECHR] {len(network):,} cases loaded.")

    # ------------------------------------------------------------------
    # Write judgments.jsonl
    # ------------------------------------------------------------------
    n_judgments = 0
    with open(JUDGMENTS, "w", encoding="utf-8") as fout:
        for doc_id, meta in network.items():
            year = meta.get("year")
            # Produce an ISO-style date string from the year (day/month unknown)
            judgementdate = f"{year}-01-01" if year else ""

            record = {
                "itemid":        doc_id,
                "docname":       meta.get("case_name", ""),
                "judgementdate": judgementdate,
                "appno":         "",   # not available from HTML extraction
            }
            fout.write(json.dumps(record) + "\n")
            n_judgments += 1

    print(f"[ECHR] {n_judgments:,} records written to {JUDGMENTS}")

    # ------------------------------------------------------------------
    # Write citations.csv
    # ------------------------------------------------------------------
    n_edges = 0
    with open(CITATIONS, "w", newline="", encoding="utf-8") as fout:
        writer = csv.writer(fout)
        writer.writerow(["citing_id", "cited_id"])
        for doc_id, meta in network.items():
            refs = meta.get("references", [])
            for cited_id in refs:
                if cited_id and cited_id != doc_id:
                    writer.writerow([doc_id, cited_id])
                    n_edges += 1

    citing_cases = sum(1 for v in network.values() if v.get("references"))
    print(f"[ECHR] {n_edges:,} citation edges written to {CITATIONS}")
    print(f"[ECHR] ({citing_cases:,} citing cases, "
          f"{n_edges:,} total links, "
          f"mean {n_edges/citing_cases:.1f} citations per citing case)")
    print("[ECHR] Conversion complete.")
    print()
    print("Next steps:")
    print("  env/bin/python src/02_build_citation_hypergraph.py --dataset echr")
    print("  env/bin/python src/03_structural_profiling.py      --dataset echr")
    print("  env/bin/python src/04_legal_argument_space.py      --dataset echr")
    print("  env/bin/python src/05_empirical_analysis.py        --dataset echr")
    print("  env/bin/python src/06_validation.py                --dataset echr")


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(
        description="Convert network_from_cache.json to pipeline inputs."
    )
    parser.add_argument(
        "--force", action="store_true",
        help="Overwrite existing judgments.jsonl and citations.csv"
    )
    args = parser.parse_args()
    convert(force=args.force)
