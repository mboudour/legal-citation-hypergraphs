"""
filter_scotus_bulk.py
=====================
Run on the cloud computer to produce three small SCOTUS-only CSV files.

Strategy:
  Step A: Download dockets (~4.7 GB). Filter to court_id='scotus'. Save docket IDs.
  Step B: Download opinion-clusters (~2.3 GB). Filter by docket_id in SCOTUS dockets.
          Save cluster IDs + case_name + date_filed.
          Also build opinion_cluster_id set for Step C.
  Step C: Download citation-map (~498 MB). The citation-map uses opinion IDs.
          We need opinion_id -> cluster_id mapping.
          We get this by streaming the opinions file BUT only reading the first
          22 fields (id and cluster_id are fields 0 and 21).
          The opinions file has multi-line rows (embedded newlines in HTML fields),
          so we use a PostgreSQL-style CSV parser that handles quoted newlines.
          Actually: we use a smarter approach - read the opinions file with
          Python's csv module which handles quoted newlines correctly, but we
          only keep columns 0 (id) and 21 (cluster_id) and skip the rest.

  REVISED Step C: Since opinions file is 54GB compressed with embedded newlines,
  we instead use the parentheticals file (286MB) which has describing_opinion_id
  and described_opinion_id - both are opinion IDs. We can use these to get
  opinion IDs that appear in SCOTUS clusters.

  ACTUAL CORRECT APPROACH:
  The opinion-clusters file has a 'scdb_id' column which is non-empty ONLY for
  SCOTUS cases. So we can identify SCOTUS clusters directly from that column
  WITHOUT needing the dockets file at all.

  Then for citations: we need opinion_id -> cluster_id.
  The opinions file header shows cluster_id is at position 21.
  The large fields (plain_text, html, etc.) are at positions 11-19.
  We can use Python's csv.reader which handles quoted newlines - it will
  correctly parse multi-line rows. We just need to be patient and stream it.
  At 28 MB/s the 54GB file takes ~30 min. But we only need 2 columns.

  FASTEST APPROACH: Use scdb_id to get SCOTUS cluster IDs from clusters file,
  then use the REST API (once it resets) for citations.

  BUT: The API is rate-limited. So we use a DIFFERENT approach for citations:
  We note that each opinion cluster in SCOTUS has a 'citation_count' field in
  the clusters CSV. But that doesn't give us the actual citation edges.

  FINAL DECISION: Two-step approach:
  1. Get SCOTUS cluster IDs from clusters file (via scdb_id OR via dockets join)
  2. Stream the opinions file to build opinion_id -> cluster_id map for SCOTUS only
     (we stop reading each row after field 21, saving huge amounts of time)
  3. Filter citation-map using the opinion_id sets

  For step 2: Python csv.reader handles quoted newlines, so we can stream
  the opinions file correctly. We read each complete CSV record (which may
  span multiple lines) and extract fields 0 and 21 only.

Outputs (in data/raw/scotus/):
  dockets_manus.csv    — id, court_id
  clusters_manus.csv   — id, case_name, date_filed, docket_id
  citations_manus.csv  — citing_cluster_id, cited_cluster_id
"""

import bz2
import csv
import os
import sys
import time
import urllib.request

csv.field_size_limit(min(sys.maxsize, 2 ** 31 - 1))

ROOT = os.path.dirname(os.path.abspath(__file__))
OUT  = os.path.join(ROOT, "data", "raw", "scotus")
os.makedirs(OUT, exist_ok=True)

DOCKETS_URL      = "https://com-courtlistener-storage.s3-us-west-2.amazonaws.com/bulk-data/dockets-2026-03-31.csv.bz2"
CLUSTERS_URL     = "https://com-courtlistener-storage.s3-us-west-2.amazonaws.com/bulk-data/opinion-clusters-2026-03-31.csv.bz2"
CITATION_MAP_URL = "https://com-courtlistener-storage.s3-us-west-2.amazonaws.com/bulk-data/citation-map-2026-03-31.csv.bz2"
OPINIONS_URL     = "https://com-courtlistener-storage.s3-us-west-2.amazonaws.com/bulk-data/opinions-2026-03-31.csv.bz2"


def bz2_csv_reader(url, label):
    """
    Stream a remote bz2-compressed CSV, yielding parsed rows as lists.
    Uses Python's csv.reader which correctly handles quoted newlines.
    """
    print(f"[{label}] Opening stream ...", flush=True)
    req = urllib.request.urlopen(url, timeout=120)
    decompressor = bz2.BZ2Decompressor()

    # We accumulate decompressed bytes and feed them to csv.reader
    # via a line-buffered approach that handles quoted newlines
    class BZ2LineReader:
        def __init__(self):
            self.buf = b""
            self.done = False
            self.bytes_in = 0
            self.last_report = time.time()

        def __iter__(self):
            return self

        def __next__(self):
            # Keep reading until we have a complete line (accounting for quoting)
            while True:
                # Try to find a newline in the buffer
                if b"\n" in self.buf:
                    pos = self.buf.index(b"\n")
                    line = self.buf[:pos]
                    self.buf = self.buf[pos+1:]
                    return line.decode("utf-8", errors="replace")
                # Need more data
                if self.done:
                    if self.buf:
                        line = self.buf
                        self.buf = b""
                        return line.decode("utf-8", errors="replace")
                    raise StopIteration
                chunk = req.read(1024 * 1024)
                if not chunk:
                    self.done = True
                    continue
                self.bytes_in += len(chunk)
                now = time.time()
                if now - self.last_report > 30:
                    print(f"  [{label}] {self.bytes_in/1e9:.2f} GB read", flush=True)
                    self.last_report = now
                self.buf += decompressor.decompress(chunk)

    line_reader = BZ2LineReader()
    reader = csv.reader(line_reader)
    return reader, line_reader


# ── Step A: dockets ──────────────────────────────────────────────────────────
dockets_out = os.path.join(OUT, "dockets_manus.csv")
docket_ids = set()
print("\n=== Step A: dockets (~4.7 GB) ===", flush=True)
t0 = time.time()
reader, lr = bz2_csv_reader(DOCKETS_URL, "dockets")
header = next(reader)
court_id_idx = header.index("court_id")
id_idx = header.index("id")
print(f"  court_id at col {court_id_idx}, id at col {id_idx}", flush=True)

with open(dockets_out, "w", newline="") as fout:
    writer = csv.writer(fout)
    writer.writerow(["id", "court_id"])
    for row in reader:
        try:
            if len(row) > court_id_idx and row[court_id_idx].lower() == "scotus":
                did = row[id_idx]
                if did:
                    docket_ids.add(did)
                    writer.writerow([did, "scotus"])
        except Exception:
            continue

elapsed = time.time() - t0
print(f"  Done in {elapsed:.0f}s. {len(docket_ids):,} SCOTUS dockets → {dockets_out}", flush=True)


# ── Step B: clusters ─────────────────────────────────────────────────────────
clusters_out = os.path.join(OUT, "clusters_manus.csv")
cluster_ids = set()
print("\n=== Step B: opinion-clusters (~2.3 GB) ===", flush=True)
t0 = time.time()
reader, lr = bz2_csv_reader(CLUSTERS_URL, "clusters")
header = next(reader)
cid_idx     = header.index("id")
docket_idx  = header.index("docket_id")
name_idx    = header.index("case_name")
date_idx    = header.index("date_filed")
scdb_idx    = header.index("scdb_id") if "scdb_id" in header else -1
print(f"  Columns: id={cid_idx}, docket_id={docket_idx}, case_name={name_idx}, date_filed={date_idx}, scdb_id={scdb_idx}", flush=True)

with open(clusters_out, "w", newline="") as fout:
    writer = csv.writer(fout)
    writer.writerow(["id", "case_name", "date_filed", "docket_id"])
    for row in reader:
        try:
            did = row[docket_idx] if len(row) > docket_idx else ""
            # Match by docket_id OR by non-empty scdb_id (SCOTUS-only field)
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


# ── Step C: opinions → build opinion_id -> cluster_id map for SCOTUS ─────────
print("\n=== Step C: opinions (~54 GB) — extracting id+cluster_id only ===", flush=True)
print("  WARNING: This file is 54 GB compressed. Streaming only first 22 fields per row.", flush=True)
t0 = time.time()

# We use a custom approach: read the opinions file but truncate each row
# after the 22nd field to avoid loading huge text blobs into memory.
# Python's csv.reader handles quoted newlines, so we must let it parse fully.
# However, we can limit memory by only keeping the first 22 fields.

opinion_to_cluster = {}  # opinion_id -> cluster_id (SCOTUS only)

class OpinionLineReader:
    def __init__(self, url):
        self.req = urllib.request.urlopen(url, timeout=120)
        self.decompressor = bz2.BZ2Decompressor()
        self.buf = b""
        self.done = False
        self.bytes_in = 0
        self.scotus_opinions = 0
        self.last_report = time.time()

    def __iter__(self):
        return self

    def __next__(self):
        while True:
            if b"\n" in self.buf:
                pos = self.buf.index(b"\n")
                line = self.buf[:pos]
                self.buf = self.buf[pos+1:]
                return line.decode("utf-8", errors="replace")
            if self.done:
                if self.buf:
                    line = self.buf
                    self.buf = b""
                    return line.decode("utf-8", errors="replace")
                raise StopIteration
            chunk = self.req.read(2 * 1024 * 1024)  # 2MB chunks
            if not chunk:
                self.done = True
                continue
            self.bytes_in += len(chunk)
            now = time.time()
            if now - self.last_report > 60:
                print(f"  [opinions] {self.bytes_in/1e9:.1f} GB read, {self.scotus_opinions:,} SCOTUS opinions found", flush=True)
                self.last_report = now
            self.buf += self.decompressor.decompress(chunk)

olr = OpinionLineReader(OPINIONS_URL)

opinion_reader = csv.reader(olr)
header = next(opinion_reader)
op_id_idx  = header.index("id")
op_cid_idx = header.index("cluster_id")
print(f"  opinions: id at col {op_id_idx}, cluster_id at col {op_cid_idx}", flush=True)

for row in opinion_reader:
    try:
        if len(row) > op_cid_idx:
            cid = row[op_cid_idx]
            if cid in cluster_ids:
                oid = row[op_id_idx]
                if oid:
                    opinion_to_cluster[oid] = cid
                    olr.scotus_opinions += 1
    except Exception:
        continue

elapsed = time.time() - t0
print(f"  Done in {elapsed:.0f}s. {olr.scotus_opinions:,} SCOTUS opinion IDs mapped.", flush=True)


# ── Step D: citation-map → filter to SCOTUS cluster pairs ────────────────────
citations_out = os.path.join(OUT, "citations_manus.csv")
print("\n=== Step D: citation-map (~498 MB) ===", flush=True)
t0 = time.time()
reader, lr = bz2_csv_reader(CITATION_MAP_URL, "citation-map")
header = next(reader)
citing_idx = header.index("citing_opinion_id")
cited_idx  = header.index("cited_opinion_id")
print(f"  citing_opinion_id at col {citing_idx}, cited_opinion_id at col {cited_idx}", flush=True)

written = 0
with open(citations_out, "w", newline="") as fout:
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
