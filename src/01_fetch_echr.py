"""
01_fetch_echr.py
================
Download ECHR/HUDOC judgment metadata and citation links using the
echr-extractor library (pip install echr-extractor).

Outputs (written to data/raw/echr/):
  judgments.jsonl   — one JSON object per judgment
  citations.csv     — citing_id, cited_id (one row per citation link)

No API key required.

Usage:
  env/bin/python src/01_fetch_echr.py [--count N]
"""

import argparse
import csv
import json
import os
import sys
import re

ROOT    = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
RAW_DIR = os.path.join(ROOT, "data", "raw", "echr")
os.makedirs(RAW_DIR, exist_ok=True)


def parse_scl(scl_value) -> list:
    """
    Parse the 'scl' field from HUDOC metadata.
    The scl field is a string containing ECHR case itemids like '001-XXXXXX'.
    Returns a list of itemid strings.
    """
    if not scl_value:
        return []
    if isinstance(scl_value, str):
        return re.findall(r'\b\d{3}-\d+\b', scl_value)
    if isinstance(scl_value, list):
        refs = []
        for item in scl_value:
            refs.extend(re.findall(r'\b\d{3}-\d+\b', str(item)))
        return refs
    return []


def main():
    parser = argparse.ArgumentParser(description="Fetch ECHR/HUDOC data.")
    parser.add_argument("--count", type=int, default=25000,
                        help="Maximum number of judgments to fetch (default: 25000)")
    args = parser.parse_args()

    from echr_extractor import get_echr, get_nodes_edges

    judgments_path = os.path.join(RAW_DIR, "judgments.jsonl")
    citations_path = os.path.join(RAW_DIR, "citations.csv")

    # -----------------------------------------------------------------------
    # Step 1: Download metadata — fetch ALL fields (needed for get_nodes_edges)
    # -----------------------------------------------------------------------
    print(f"Fetching up to {args.count:,} ECHR judgments (English) from HUDOC …")
    df = get_echr(
        count=args.count,
        language=["ENG"],
        save_file="n",
        verbose=True,
        memory_efficient=True,
        progress_bar=True,
    )

    if df is False or df is None or len(df) == 0:
        print("[ERROR] No data returned from echr-extractor. Aborting.")
        sys.exit(1)

    print(f"Downloaded {len(df):,} judgments.")
    print(f"Columns: {df.columns.tolist()}")

    # -----------------------------------------------------------------------
    # Step 2: Write judgments.jsonl
    # -----------------------------------------------------------------------
    print(f"Writing {judgments_path} …")
    with open(judgments_path, "w", encoding="utf-8") as jf:
        for _, row in df.iterrows():
            obj = {
                "itemid":        str(row.get("itemid", "")),
                "docname":       str(row.get("docname", "")),
                "judgementdate": str(row.get("judgementdate", "")),
                "appno":         str(row.get("appno", "")),
                "respondent":    str(row.get("respondent", "")),
                "importance":    str(row.get("importance", "")),
                "scl":           str(row.get("scl", "")),
            }
            jf.write(json.dumps(obj) + "\n")
    print(f"Written {len(df):,} records to {judgments_path}")

    # -----------------------------------------------------------------------
    # Step 3: Generate citation edges
    # Strategy A: use get_nodes_edges (uses extractedappno cross-referencing)
    # Strategy B: parse scl field directly (itemid-based)
    # We use Strategy A first, fall back to B if it fails.
    # -----------------------------------------------------------------------
    print("Generating citation network edges …")

    citations_written = 0
    try:
        nodes, edges, missing = get_nodes_edges(df=df, save_file="n")
        print(f"Nodes: {len(nodes):,}  Edges: {len(edges):,}  Missing refs: {len(missing):,}")
        print(f"Edge columns: {edges.columns.tolist()}")

        # Identify the citing/cited column names
        col_map = {c.lower(): c for c in edges.columns}
        citing_col = col_map.get("citing_id", col_map.get("source", col_map.get("from", None)))
        cited_col  = col_map.get("cited_id",  col_map.get("target", col_map.get("to",   None)))

        if citing_col is None or cited_col is None:
            # Try first two columns
            citing_col, cited_col = edges.columns[0], edges.columns[1]

        print(f"Writing {citations_path} (using columns: {citing_col}, {cited_col}) …")
        with open(citations_path, "w", encoding="utf-8", newline="") as cf:
            writer = csv.writer(cf)
            writer.writerow(["citing_id", "cited_id"])
            for _, row in edges.iterrows():
                citing = str(row[citing_col])
                cited  = str(row[cited_col])
                if citing and cited and citing != "nan" and cited != "nan":
                    writer.writerow([citing, cited])
                    citations_written += 1

    except Exception as e:
        print(f"[WARN] get_nodes_edges failed: {e}")
        print("Falling back to direct scl field parsing …")

        # Strategy B: parse scl itemids directly
        with open(citations_path, "w", encoding="utf-8", newline="") as cf:
            writer = csv.writer(cf)
            writer.writerow(["citing_id", "cited_id"])
            for _, row in df.iterrows():
                citing_id = str(row.get("itemid", ""))
                scl_refs  = parse_scl(row.get("scl", ""))
                for cited_id in scl_refs:
                    writer.writerow([citing_id, cited_id])
                    citations_written += 1

    print(f"Done. {citations_written:,} citation edges written to {citations_path}")


if __name__ == "__main__":
    main()
