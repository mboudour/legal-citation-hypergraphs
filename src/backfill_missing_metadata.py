#!/usr/bin/env python3
"""
backfill_missing_metadata.py
============================
Recovers missing case_name and tau (date_filed) for opinion nodes that have
empty metadata in opinions_meta.pkl and G.pkl.

These are opinion IDs that appear in the head sets H of F-arcs but whose
metadata was not fetched during the original pipeline run.

Strategy:
  For each missing opinion ID, call:
    GET https://www.courtlistener.com/api/rest/v4/opinions/{id}/
  to get the cluster URL, then:
    GET {cluster_url}
  to get case_name and date_filed.

  Rate limit: ~5 requests per 50 seconds. The script respects this with
  automatic retry on 429 responses.

Usage:
  python3 backfill_missing_metadata.py \\
      --opinions_meta path/to/opinions_meta.pkl \\
      --G             path/to/G.pkl \\
      --token         YOUR_COURTLISTENER_TOKEN \\
      --output_dir    path/to/output/dir/

  The script writes updated opinions_meta.pkl and G.pkl to --output_dir.
  It also writes a progress file backfill_progress.json so it can be
  resumed if interrupted.

Estimated runtime: ~38 hours for 5,714 IDs at the API rate limit.
The script can be safely interrupted and resumed at any time.
"""

import argparse
import json
import os
import pickle
import time

import requests

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------
BASE_URL = "https://www.courtlistener.com/api/rest/v4"
BURST_SIZE = 5          # requests before hitting 429
BURST_WINDOW = 55       # seconds to wait after hitting 429
INTER_REQUEST_SLEEP = 0.5  # seconds between requests within a burst


def parse_args():
    p = argparse.ArgumentParser(description="Backfill missing opinion metadata from CourtListener API")
    p.add_argument("--opinions_meta", required=True, help="Path to opinions_meta.pkl")
    p.add_argument("--G",             required=True, help="Path to G.pkl")
    p.add_argument("--token",         required=True, help="CourtListener API token")
    p.add_argument("--output_dir",    required=True, help="Directory to write updated pkl files")
    return p.parse_args()


def api_get(url, headers, max_retries=10):
    """GET url with automatic retry on 429."""
    for attempt in range(max_retries):
        r = requests.get(url, headers=headers, timeout=15)
        if r.status_code == 200:
            return r.json()
        elif r.status_code == 429:
            retry_after = int(r.headers.get("retry-after", BURST_WINDOW))
            print(f"    [429] Rate limited. Waiting {retry_after}s ...", flush=True)
            time.sleep(retry_after + 2)
        elif r.status_code == 404:
            return None  # opinion genuinely not found
        else:
            print(f"    [HTTP {r.status_code}] {url}", flush=True)
            time.sleep(5)
    return None


def fetch_opinion_meta(opinion_id, headers):
    """
    Returns (case_name, date_filed) for a CourtListener opinion ID,
    or (None, None) if not found.
    """
    opinion_data = api_get(f"{BASE_URL}/opinions/{opinion_id}/", headers)
    if opinion_data is None:
        return None, None

    cluster_url = opinion_data.get("cluster", "")
    if not cluster_url:
        return None, None

    time.sleep(INTER_REQUEST_SLEEP)
    cluster_data = api_get(cluster_url, headers)
    if cluster_data is None:
        return None, None

    case_name  = cluster_data.get("case_name", "") or cluster_data.get("case_name_short", "")
    date_filed = cluster_data.get("date_filed", "")
    return case_name, date_filed


def main():
    args = parse_args()
    os.makedirs(args.output_dir, exist_ok=True)

    headers = {"Authorization": f"Token {args.token}"}
    progress_file = os.path.join(args.output_dir, "backfill_progress.json")

    # ---- Load data ----
    print("Loading opinions_meta.pkl ...", flush=True)
    with open(args.opinions_meta, "rb") as f:
        om = pickle.load(f)

    print("Loading G.pkl ...", flush=True)
    with open(args.G, "rb") as f:
        G = pickle.load(f)

    # ---- Identify missing IDs ----
    missing_ids = sorted([k for k, v in om.items() if not v.get("case_name", "")])
    print(f"Total opinion IDs with missing metadata: {len(missing_ids)}", flush=True)

    # ---- Load progress (resume support) ----
    recovered = {}
    if os.path.exists(progress_file):
        with open(progress_file) as f:
            recovered = json.load(f)
        print(f"Resuming: {len(recovered)} IDs already recovered.", flush=True)

    remaining = [oid for oid in missing_ids if oid not in recovered]
    print(f"Remaining to fetch: {len(remaining)}", flush=True)

    # ---- Fetch loop ----
    for i, oid in enumerate(remaining):
        case_name, date_filed = fetch_opinion_meta(oid, headers)

        if case_name or date_filed:
            recovered[oid] = {"case_name": case_name or "", "tau": date_filed or ""}
        else:
            recovered[oid] = {"case_name": "", "tau": ""}  # mark as attempted

        # Progress report every 50 IDs
        if (i + 1) % 50 == 0 or (i + 1) == len(remaining):
            pct = 100 * (i + 1) / len(remaining)
            named = sum(1 for v in recovered.values() if v.get("case_name", ""))
            print(f"  [{i+1}/{len(remaining)} ({pct:.1f}%)] recovered names: {named}", flush=True)
            # Save progress checkpoint
            with open(progress_file, "w") as f:
                json.dump(recovered, f)

        time.sleep(INTER_REQUEST_SLEEP)

    # ---- Patch opinions_meta ----
    print("\nPatching opinions_meta ...", flush=True)
    patched_om = 0
    for oid, meta in recovered.items():
        if oid in om:
            if meta.get("case_name") and not om[oid].get("case_name"):
                om[oid]["case_name"] = meta["case_name"]
                patched_om += 1
            if meta.get("tau") and not om[oid].get("tau"):
                om[oid]["tau"] = meta["tau"]
    print(f"  Patched {patched_om} entries in opinions_meta.", flush=True)

    # ---- Patch G ----
    print("Patching G ...", flush=True)
    patched_G = 0
    for oid, meta in recovered.items():
        if oid in G:
            if meta.get("case_name") and not G[oid].get("case_name"):
                G[oid]["case_name"] = meta["case_name"]
                patched_G += 1
            if meta.get("tau") and not G[oid].get("tau"):
                G[oid]["tau"] = meta["tau"]
    print(f"  Patched {patched_G} entries in G.", flush=True)

    # ---- Save updated files ----
    om_out = os.path.join(args.output_dir, "opinions_meta.pkl")
    G_out  = os.path.join(args.output_dir, "G.pkl")

    print(f"\nSaving updated opinions_meta.pkl to {om_out} ...", flush=True)
    with open(om_out, "wb") as f:
        pickle.dump(om, f)

    print(f"Saving updated G.pkl to {G_out} ...", flush=True)
    with open(G_out, "wb") as f:
        pickle.dump(G, f)

    # ---- Summary ----
    total_named = sum(1 for v in om.values() if v.get("case_name", ""))
    total_dated = sum(1 for v in om.values() if v.get("tau", ""))
    print(f"\n=== Backfill complete ===")
    print(f"  Opinions with case_name: {total_named} / {len(om)}")
    print(f"  Opinions with tau:       {total_dated} / {len(om)}")
    print(f"  Updated files written to: {args.output_dir}")


if __name__ == "__main__":
    main()
