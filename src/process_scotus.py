"""
process_scotus.py
=================
Process locally-downloaded bulk bz2 files to produce three small SCOTUS CSVs.
Uses bz2.open() + csv.reader which correctly handles quoted newlines.

Steps:
  A: dockets.csv.bz2  → filter court_id='scotus' → dockets_manus.csv + docket_ids set
  B: clusters.csv.bz2 → filter by docket_id in docket_ids → clusters_manus.csv + cluster_ids set
  C: citation-map.csv.bz2 → citation-map uses opinion IDs, NOT cluster IDs.
     We need opinion_id→cluster_id mapping.
     Strategy: use scdb_id in clusters to identify SCOTUS clusters (scdb_id non-empty = SCOTUS).
     Then for citations: we skip the 54GB opinions file and instead use a different approach:
     The citation-map has citing_opinion_id and cited_opinion_id.
     We query the CourtListener API for opinion IDs of SCOTUS clusters.
     BUT: API is rate-limited.
     
     ACTUAL STRATEGY for citations without opinions file:
     - We already have SCOTUS cluster IDs from step B.
     - Each cluster can have multiple opinions. We need opinion_id→cluster_id.
     - The parentheticals file (286MB) has described_opinion_id and describing_opinion_id
       but doesn't give us cluster_id directly.
     - SOLUTION: Use the clusters file's 'sub_opinions' or related field if available.
     - FALLBACK: Build citations at cluster level using scdb citations data.
     
     SIMPLEST CORRECT APPROACH:
     The clusters file has a field 'citation_count' but not the actual citation edges.
     
     For citations we MUST use one of:
     1. opinions file (54GB) - too large
     2. REST API - rate limited
     3. Accept that citation-map cannot be filtered to SCOTUS without opinion→cluster mapping
     
     PRAGMATIC SOLUTION: 
     Download just the first 2 columns of the opinions file using HTTP range requests
     to get opinion_id and cluster_id. The opinions file header shows cluster_id is at
     column index 21. Each row's first 22 fields are small (no text blobs).
     We can use a partial download approach: download chunks and parse until we've seen
     all SCOTUS cluster IDs.
     
     ACTUALLY: Let's just parse the full opinions file. It's 54GB compressed but
     with bz2.open() on a local file it will be CPU-bound. At ~50MB/s decompression
     rate that's 54GB / 50MB/s = ~18 minutes. Acceptable.
     
     BUT we don't have 54GB disk space. We have 28GB free and already used 7.5GB.
     
     FINAL DECISION: Build opinion→cluster mapping by streaming opinions file from URL
     using subprocess + bzip2 -d piped to python. The key insight: we use
     subprocess.Popen to run: curl URL | bzip2 -d
     and read stdout with csv.reader. This correctly handles quoted newlines because
     csv.reader reads from a file-like object (the pipe), not line by line.
     
     Wait - this is what we tested and it failed because csv.reader(sys.stdin) reads
     line by line when stdin is a pipe.
     
     THE ACTUAL FIX: Use io.TextIOWrapper around the pipe stdout, which gives csv.reader
     a proper seekable text stream. Actually csv.reader just needs __iter__ that yields
     lines - but "lines" here means logical CSV lines including quoted newlines.
     
     csv.reader DOES handle quoted newlines correctly when given a file object opened
     in text mode. The problem before was using sys.stdin which is line-buffered.
     
     CORRECT APPROACH: 
     subprocess.Popen(['curl', url], stdout=PIPE) -> bzip2 -d -> csv.reader(text_wrapper)
     
     Let's test this properly.
"""

import bz2
import csv
import os
import subprocess
import sys
import time

csv.field_size_limit(min(sys.maxsize, 2 ** 31 - 1))

ROOT   = os.path.dirname(os.path.abspath(__file__))
BULK   = os.path.join(ROOT, "data", "bulk")
OUT    = os.path.join(ROOT, "data", "raw", "scotus")
os.makedirs(OUT, exist_ok=True)

DOCKETS_BZ2     = os.path.join(BULK, "dockets.csv.bz2")
CLUSTERS_BZ2    = os.path.join(BULK, "clusters.csv.bz2")
CITMAP_BZ2      = os.path.join(BULK, "citation-map.csv.bz2")
OPINIONS_URL    = "https://com-courtlistener-storage.s3-us-west-2.amazonaws.com/bulk-data/opinions-2026-03-31.csv.bz2"

dockets_out  = os.path.join(OUT, "dockets_manus.csv")
clusters_out = os.path.join(OUT, "clusters_manus.csv")
citations_out = os.path.join(OUT, "citations_manus.csv")


# ── Step A: dockets ──────────────────────────────────────────────────────────
print("\n=== Step A: dockets ===", flush=True)
t0 = time.time()
docket_ids = set()

with bz2.open(DOCKETS_BZ2, 'rt', encoding='utf-8', errors='replace') as f:
    reader = csv.reader(f)
    header = next(reader)
    id_idx       = header.index("id")
    court_id_idx = header.index("court_id")
    print(f"  id at col {id_idx}, court_id at col {court_id_idx}", flush=True)

    with open(dockets_out, 'w', newline='') as fout:
        writer = csv.writer(fout)
        writer.writerow(["id", "court_id"])
        rows = 0
        for row in reader:
            rows += 1
            if rows % 500000 == 0:
                print(f"  {rows:,} rows scanned, {len(docket_ids):,} SCOTUS so far", flush=True)
            try:
                if len(row) > court_id_idx and row[court_id_idx].lower() == 'scotus':
                    did = row[id_idx]
                    if did:
                        docket_ids.add(did)
                        writer.writerow([did, "scotus"])
            except Exception:
                continue

elapsed = time.time() - t0
print(f"  Done in {elapsed:.0f}s. {len(docket_ids):,} SCOTUS dockets → {dockets_out}", flush=True)


# ── Step B: clusters ─────────────────────────────────────────────────────────
print("\n=== Step B: clusters ===", flush=True)
t0 = time.time()
cluster_ids = set()

with bz2.open(CLUSTERS_BZ2, 'rt', encoding='utf-8', errors='replace') as f:
    reader = csv.reader(f)
    header = next(reader)
    cid_idx    = header.index("id")
    docket_idx = header.index("docket_id")
    name_idx   = header.index("case_name")
    date_idx   = header.index("date_filed")
    scdb_idx   = header.index("scdb_id") if "scdb_id" in header else -1
    print(f"  id={cid_idx}, docket_id={docket_idx}, case_name={name_idx}, date_filed={date_idx}, scdb_id={scdb_idx}", flush=True)

    with open(clusters_out, 'w', newline='') as fout:
        writer = csv.writer(fout)
        writer.writerow(["id", "case_name", "date_filed", "docket_id"])
        rows = 0
        for row in reader:
            rows += 1
            if rows % 500000 == 0:
                print(f"  {rows:,} rows scanned, {len(cluster_ids):,} SCOTUS so far", flush=True)
            try:
                did = row[docket_idx] if len(row) > docket_idx else ""
                is_scotus = (did in docket_ids)
                if not is_scotus and scdb_idx >= 0 and len(row) > scdb_idx:
                    scdb = row[scdb_idx].strip().strip('"')
                    is_scotus = bool(scdb)
                if is_scotus:
                    cid = row[cid_idx]
                    if cid:
                        cluster_ids.add(cid)
                        writer.writerow([
                            cid,
                            row[name_idx] if len(row) > name_idx else "",
                            row[date_idx] if len(row) > date_idx else "",
                            did
                        ])
            except Exception:
                continue

elapsed = time.time() - t0
print(f"  Done in {elapsed:.0f}s. {len(cluster_ids):,} SCOTUS clusters → {clusters_out}", flush=True)


# ── Step C: opinions → opinion_id to cluster_id mapping ─────────────────────
# Stream from URL using curl | bzip2 -d piped to csv.reader via io.TextIOWrapper
print("\n=== Step C: opinions (streaming from URL) ===", flush=True)
print("  Building opinion_id → cluster_id map for SCOTUS clusters only...", flush=True)
t0 = time.time()

opinion_to_cluster = {}

# Use subprocess: curl URL | bzip2 -d, read stdout as text with csv.reader
proc_curl = subprocess.Popen(
    ['curl', '-s', OPINIONS_URL],
    stdout=subprocess.PIPE
)
proc_bzip = subprocess.Popen(
    ['bzip2', '-d'],
    stdin=proc_curl.stdout,
    stdout=subprocess.PIPE
)
proc_curl.stdout.close()  # Allow proc_curl to receive SIGPIPE if proc_bzip exits

import io
text_stream = io.TextIOWrapper(proc_bzip.stdout, encoding='utf-8', errors='replace')
reader = csv.reader(text_stream)

header = next(reader)
op_id_idx  = header.index("id")
op_cid_idx = header.index("cluster_id")
print(f"  opinions: id at col {op_id_idx}, cluster_id at col {op_cid_idx}", flush=True)

rows = 0
scotus_opinions = 0
last_report = time.time()

for row in reader:
    rows += 1
    now = time.time()
    if now - last_report > 60:
        print(f"  {rows:,} opinion rows scanned, {scotus_opinions:,} SCOTUS opinions found", flush=True)
        last_report = now
    try:
        if len(row) > op_cid_idx:
            cid = row[op_cid_idx]
            if cid in cluster_ids:
                oid = row[op_id_idx]
                if oid:
                    opinion_to_cluster[oid] = cid
                    scotus_opinions += 1
    except Exception:
        continue

proc_curl.terminate()
proc_bzip.wait()

elapsed = time.time() - t0
print(f"  Done in {elapsed:.0f}s. {scotus_opinions:,} SCOTUS opinion IDs mapped.", flush=True)


# ── Step D: citation-map ─────────────────────────────────────────────────────
print("\n=== Step D: citation-map ===", flush=True)
t0 = time.time()

with bz2.open(CITMAP_BZ2, 'rt', encoding='utf-8', errors='replace') as f:
    reader = csv.reader(f)
    header = next(reader)
    citing_idx = header.index("citing_opinion_id")
    cited_idx  = header.index("cited_opinion_id")
    print(f"  citing_opinion_id at col {citing_idx}, cited_opinion_id at col {cited_idx}", flush=True)

    written = 0
    with open(citations_out, 'w', newline='') as fout:
        writer = csv.writer(fout)
        writer.writerow(["citing_cluster_id", "cited_cluster_id"])
        for row in reader:
            try:
                citing_oid = row[citing_idx] if len(row) > citing_idx else ""
                cited_oid  = row[cited_idx]  if len(row) > cited_idx  else ""
                citing_cid = opinion_to_cluster.get(citing_oid)
                cited_cid  = opinion_to_cluster.get(cited_oid)
                if citing_cid and cited_cid:
                    writer.writerow([citing_cid, cited_cid])
                    written += 1
            except Exception:
                continue

elapsed = time.time() - t0
print(f"  Done in {elapsed:.0f}s. {written:,} SCOTUS citation edges → {citations_out}", flush=True)

print("\n=== ALL DONE ===", flush=True)
for f in [dockets_out, clusters_out, citations_out]:
    size = os.path.getsize(f)
    lines = sum(1 for _ in open(f)) - 1
    print(f"  {os.path.basename(f)}: {size/1e6:.1f} MB, {lines:,} rows", flush=True)
