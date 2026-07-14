"""
01_fetch_data.py
================
Download raw data for both datasets.

SCOTUS strategy (PURE BULK CSV, NO REST API):
  Step A: Download dockets bulk CSV (~4.7 GB). Filter to court_id='scotus'. Extract docket IDs.
          CACHED: if dockets.csv already exists, reads from cache.
  Step B: Download opinion-clusters bulk CSV (~2.3 GB). Filter to rows where docket_id is in
          our SCOTUS docket IDs (primary) OR scdb_id is non-empty (secondary fallback, since
          scdb_id is populated only for SCOTUS cases). Extract cluster IDs, case names, dates.
          CACHED: if clusters.csv already exists, reads from cache.
  Step C: Download scotus_network.csv (6.6 MB, pre-built SCOTUS citation network).
          Columns: id, citing_opinion_id, cited_opinion_id, depth.
          This replaces the 54 GB opinions file + 498 MB citation-map approach entirely.
          Nodes are opinion IDs (not cluster IDs). Metadata is joined from clusters.csv
          via the docket_id → cluster_id relationship.
          Output: citations.csv with columns citing_opinion_id, cited_opinion_id.

NOTE: We use opinion IDs as node identifiers throughout the F-hypergraph. The clusters.csv
provides case_name and date_filed metadata keyed by cluster_id. The join between opinion IDs
and cluster IDs is deferred to 02_build_citation_hypergraph.py, which uses the clusters.csv
to enrich nodes with metadata.

ECHR strategy:
  Use HUDOC public API to fetch judgment metadata and citation links.

Outputs:
  data/raw/scotus/dockets.csv      — id, court_id (filtered to scotus)
  data/raw/scotus/clusters.csv     — id, case_name, date_filed, docket_id
  data/raw/scotus/citations.csv    — citing_opinion_id, cited_opinion_id (SCOTUS only)
  data/raw/echr/judgments.jsonl    — one judgment record per line
  data/raw/echr/citations.csv      — citing_id, cited_id

Usage:
  python src/01_fetch_data.py [--dataset scotus|echr|both] [--force]
"""

import argparse
import bz2
import csv
import json
import os
import sys
import time

import requests
from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry
from tqdm import tqdm

# Increase CSV field size limit for large text fields
csv.field_size_limit(min(sys.maxsize, 2 ** 31 - 1))

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
RAW_SCOTUS = os.path.join(ROOT, "data", "raw", "scotus")
RAW_ECHR   = os.path.join(ROOT, "data", "raw", "echr")
os.makedirs(RAW_SCOTUS, exist_ok=True)
os.makedirs(RAW_ECHR,   exist_ok=True)

# ---------------------------------------------------------------------------
# Shared HTTP session with retry + long timeout
# ---------------------------------------------------------------------------
TIMEOUT = 90

def make_session():
    session = requests.Session()
    retry = Retry(
        total=5,
        backoff_factor=2,
        status_forcelist=[429, 500, 502, 503, 504],
        allowed_methods=["GET"],
    )
    adapter = HTTPAdapter(max_retries=retry)
    session.mount("https://", adapter)
    session.mount("http://",  adapter)
    return session

# ---------------------------------------------------------------------------
# Bulk CSV Streamer (for large bz2 files)
# ---------------------------------------------------------------------------
def stream_bz2_csv(url, desc):
    """
    Stream a .bz2 compressed CSV from a URL, yielding rows as dicts.

    FIXES applied in this version:
    1. Header line is parsed with csv.reader (not text.split(",")).
       The old split(",") caused column misalignment when any header field
       was quoted, silently breaking all dict lookups for later columns
       (e.g. court_id at col 42 in dockets, docket_id at col 33 in clusters).
    2. Row dicts are only yielded when len(row) == len(header), not >=,
       to avoid misaligned short rows from quoted-newline fragments.

    KNOWN LIMITATION: This streamer splits on raw b"\\n" bytes. Rows that
    contain quoted newlines (e.g. syllabus/headnotes in opinion-clusters) will
    be split and silently skipped. This affects a small minority of rows and
    is acceptable for dockets and clusters.
    """
    session = make_session()
    resp = session.get(url, stream=True, timeout=TIMEOUT)
    resp.raise_for_status()
    total = int(resp.headers.get("content-length", 0))
    decompressor = bz2.BZ2Decompressor()
    buf = b""
    header = None
    pbar = tqdm(total=total, unit="B", unit_scale=True, desc=f"  {desc}")
    for chunk in resp.iter_content(chunk_size=1024 * 1024):  # 1 MB chunks
        pbar.update(len(chunk))
        buf += decompressor.decompress(chunk)
        while b"\n" in buf:
            line, buf = buf.split(b"\n", 1)
            try:
                text = line.decode("utf-8", errors="replace")
            except Exception:
                continue
            if header is None:
                # FIX: use csv.reader to parse header, not split(",")
                try:
                    header = next(csv.reader([text]))
                except Exception:
                    header = [h.strip() for h in text.split(",")]
                continue
            try:
                row = next(csv.reader([text]))
                if len(row) == len(header):
                    yield dict(zip(header, row))
            except Exception:
                continue
    pbar.close()

# ---------------------------------------------------------------------------
# SCOTUS — PURE BULK CSV PIPELINE
# ---------------------------------------------------------------------------
DOCKETS_URL      = "https://com-courtlistener-storage.s3-us-west-2.amazonaws.com/bulk-data/dockets-2026-03-31.csv.bz2"
CLUSTERS_URL     = "https://com-courtlistener-storage.s3-us-west-2.amazonaws.com/bulk-data/opinion-clusters-2026-03-31.csv.bz2"
SCOTUS_NETWORK_URL = "https://com-courtlistener-storage.s3-us-west-2.amazonaws.com/bulk-data/randoms/scotus_network.csv"


def fetch_scotus_dockets(force=False):
    out_path = os.path.join(RAW_SCOTUS, "dockets.csv")
    docket_ids = set()
    if os.path.exists(out_path) and not force:
        print("[SCOTUS] dockets.csv already exists. Reading cached docket IDs...")
        with open(out_path, "r") as f:
            reader = csv.DictReader(f)
            for row in reader:
                docket_ids.add(row["id"])
        print(f"[SCOTUS] Loaded {len(docket_ids):,} SCOTUS docket IDs.")
        return docket_ids

    print("[SCOTUS] Downloading dockets (~4.7 GB) and filtering to 'scotus' ...")
    written = 0
    with open(out_path, "w", newline="") as fout:
        writer = csv.writer(fout)
        writer.writerow(["id", "court_id"])
        for row in stream_bz2_csv(DOCKETS_URL, "dockets"):
            if row.get("court_id", "").lower() == "scotus":
                did = row.get("id", "")
                if did:
                    docket_ids.add(did)
                    writer.writerow([did, "scotus"])
                    written += 1
    print(f"[SCOTUS] {written:,} SCOTUS dockets found → {out_path}")
    return docket_ids


def fetch_scotus_clusters(docket_ids, force=False):
    out_path = os.path.join(RAW_SCOTUS, "clusters.csv")
    cluster_ids = set()
    if os.path.exists(out_path) and not force:
        print("[SCOTUS] clusters.csv already exists. Reading cached cluster IDs...")
        with open(out_path, "r") as f:
            reader = csv.DictReader(f)
            for row in reader:
                if row.get("id"):
                    cluster_ids.add(row["id"])
        print(f"[SCOTUS] Loaded {len(cluster_ids):,} SCOTUS cluster IDs.")
        return cluster_ids

    print("[SCOTUS] Downloading opinion-clusters (~2.3 GB) and filtering by SCOTUS docket IDs ...")
    written = 0
    with open(out_path, "w", newline="") as fout:
        writer = csv.writer(fout)
        writer.writerow(["id", "case_name", "date_filed", "docket_id"])
        for row in stream_bz2_csv(CLUSTERS_URL, "clusters"):
            did = row.get("docket_id", "")
            # Primary filter: docket_id in SCOTUS dockets
            is_scotus = (did in docket_ids)
            # Secondary fallback: scdb_id is non-empty (populated only for SCOTUS cases)
            if not is_scotus:
                scdb = row.get("scdb_id", "").strip().strip('"')
                is_scotus = bool(scdb)
            if is_scotus:
                cid = row.get("id", "")
                if cid:
                    cluster_ids.add(cid)
                    writer.writerow([
                        cid,
                        row.get("case_name", ""),
                        row.get("date_filed", ""),
                        did
                    ])
                    written += 1
    print(f"[SCOTUS] {written:,} SCOTUS clusters found → {out_path}")
    return cluster_ids


def fetch_scotus_citations(force=False):
    """
    Download scotus_network.csv (6.6 MB) — a pre-built SCOTUS-only citation network.
    Columns: id, citing_opinion_id, cited_opinion_id, depth.
    We keep only citing_opinion_id and cited_opinion_id.

    This replaces the previous approach of downloading the 54 GB opinions file
    to build an opinion→cluster map, then filtering the 498 MB citation-map.
    The scotus_network.csv is already filtered to SCOTUS and downloads in seconds.
    """
    out_path = os.path.join(RAW_SCOTUS, "citations.csv")
    if os.path.exists(out_path) and not force:
        print("[SCOTUS] citations.csv already exists. Use --force to re-fetch.")
        return

    print("[SCOTUS] Downloading scotus_network.csv (6.6 MB, pre-built SCOTUS citation network) ...")
    session = make_session()
    resp = session.get(SCOTUS_NETWORK_URL, stream=True, timeout=TIMEOUT)
    resp.raise_for_status()
    total = int(resp.headers.get("content-length", 0))
    written = 0
    with open(out_path, "w", newline="") as fout:
        writer = csv.writer(fout)
        writer.writerow(["citing_opinion_id", "cited_opinion_id"])
        pbar = tqdm(total=total, unit="B", unit_scale=True, desc="  scotus_network")
        reader = csv.DictReader(resp.iter_lines(decode_unicode=True))
        for row in reader:
            pbar.update(len(",".join(row.values())))
            citing = row.get("citing_opinion_id", "")
            cited  = row.get("cited_opinion_id", "")
            if citing and cited:
                writer.writerow([citing, cited])
                written += 1
        pbar.close()
    print(f"[SCOTUS] {written:,} SCOTUS citation edges → {out_path}")


def fetch_scotus(force=False):
    print("[SCOTUS] Starting bulk-CSV pipeline. NO REST API calls.")
    docket_ids  = fetch_scotus_dockets(force=force)
    cluster_ids = fetch_scotus_clusters(docket_ids, force=force)
    fetch_scotus_citations(force=force)
    print("[SCOTUS] Done.")


# ---------------------------------------------------------------------------
# ECHR — fetch judgments and citations from HUDOC
# ---------------------------------------------------------------------------
HUDOC_SEARCH = "https://hudoc.echr.coe.int/app/query/results"

def fetch_echr(max_pages=None, force=False):
    judgments_path = os.path.join(RAW_ECHR, "judgments.jsonl")
    citations_path = os.path.join(RAW_ECHR, "citations.csv")

    if os.path.exists(judgments_path) and os.path.exists(citations_path) and not force:
        print("[ECHR] Raw files already exist. Use --force to re-download.")
        return

    session = make_session()
    params_base = {
        "query": "contentsitename=ECHR AND (doctype=GRANDCHAMBER OR doctype=CHAMBER)",
        "select": "itemid,docname,kpdate,appno,article,respondent,importance",
        "sort": "kpdate Ascending",
        "rankingModelId": "11111111-0000-0000-0000-000000000000",
    }
    page_size = 500
    start = 0
    page = 0
    judgments = []

    print("[ECHR] Fetching judgment metadata from HUDOC ...")
    with tqdm(desc="  judgments", unit=" pages") as pbar:
        while True:
            params = {**params_base, "start": start, "length": page_size}
            resp = session.get(HUDOC_SEARCH, params=params, timeout=TIMEOUT)
            resp.raise_for_status()
            data = resp.json()
            results = data.get("results", {}).get("Result", [])
            if not results:
                break
            for r in results:
                judgments.append({
                    "id":         r.get("columns", {}).get("itemid", ""),
                    "case_name":  r.get("columns", {}).get("docname", ""),
                    "date":       r.get("columns", {}).get("kpdate", ""),
                    "appno":      r.get("columns", {}).get("appno", ""),
                    "articles":   r.get("columns", {}).get("article", ""),
                    "respondent": r.get("columns", {}).get("respondent", ""),
                })
            start += page_size
            page += 1
            pbar.update(1)
            pbar.set_postfix({"total": len(judgments)})
            if max_pages and page >= max_pages:
                break
            time.sleep(0.05)

    print(f"[ECHR] {len(judgments):,} judgments fetched.")

    with open(judgments_path, "w") as f:
        for j in judgments:
            f.write(json.dumps(j) + "\n")
    print(f"[ECHR] Judgments written to {judgments_path}")

    print("[ECHR] Fetching citations for each judgment ...")
    id_set = {j["id"] for j in judgments}
    written = 0
    with open(citations_path, "w", newline="") as fout:
        writer = csv.writer(fout)
        writer.writerow(["citing_id", "cited_id"])
        for j in tqdm(judgments, desc="  citations"):
            jid = j["id"]
            try:
                params = {
                    "query": f"contentsitename=ECHR AND citedin={jid}",
                    "select": "itemid",
                    "start": 0,
                    "length": 500,
                    "rankingModelId": "11111111-0000-0000-0000-000000000000",
                }
                resp = session.get(HUDOC_SEARCH, params=params, timeout=TIMEOUT)
                resp.raise_for_status()
                data = resp.json()
                for r in data.get("results", {}).get("Result", []):
                    cited_id = r.get("columns", {}).get("itemid", "")
                    if cited_id and cited_id in id_set:
                        writer.writerow([jid, cited_id])
                        written += 1
            except Exception:
                pass
            time.sleep(0.05)

    print(f"[ECHR] {written:,} citation edges → {citations_path}")
    print("[ECHR] Done.")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main():
    parser = argparse.ArgumentParser(description="Fetch raw data for legal hypergraph pipeline.")
    parser.add_argument("--dataset",   choices=["scotus", "echr", "both"], default="scotus")  # ECHR disabled until citation source is resolved
    parser.add_argument("--force",     action="store_true", help="Re-download even if files exist")
    parser.add_argument("--max-pages", type=int, default=None,
                        help="Limit number of ECHR API pages (for testing)")
    args = parser.parse_args()

    if args.dataset in ("scotus", "both"):
        fetch_scotus(force=args.force)
    if args.dataset in ("echr", "both"):
        print("[ECHR] Skipped — ECHR citation source not yet resolved. Run with --dataset echr when ready.")
        # fetch_echr(max_pages=args.max_pages, force=args.force)


if __name__ == "__main__":
    main()
