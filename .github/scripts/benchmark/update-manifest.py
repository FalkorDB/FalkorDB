#!/usr/bin/env python3
"""Update a FalkorDB/benchmark-compatible ui/public/summaries/manifest.json.

The upstream manifest.json shape is a dict keyed by the "latest pointer"
filename (e.g. "falkordb_vs_falkordb.json"), each value a list of
{"filename": ..., "timestamp": ...} entries describing every historical
snapshot available for the run-history dropdown in the dashboard.

This repo only ever publishes the falkordb_vs_falkordb comparison (not the
upstream repo's neo4j/memgraph/aws-tests comparisons), so this script always
rewrites the manifest to contain *only* the given --key. That keeps a
freshly-seeded manifest (copied from a prior publish of the same view) from
ever carrying stale unrelated keys forward.

For non-canonical (per-branch preview) views, --retention caps how much
history accumulates; entries beyond the cap are dropped from the manifest
*and* their JSON files are deleted from --summaries-dir so they don't keep
being published forever.
"""
import argparse
import json
import os
import sys


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", required=True, help="path to manifest.json (read + overwritten)")
    parser.add_argument("--key", required=True, help='manifest key, e.g. "falkordb_vs_falkordb.json"')
    parser.add_argument("--add-filename", required=True, help="filename of the new timestamped snapshot")
    parser.add_argument("--add-timestamp", required=True, type=int, help="epoch seconds of the new snapshot")
    parser.add_argument("--summaries-dir", required=True, help="directory containing the snapshot JSON files")
    parser.add_argument("--retention", type=int, default=0, help="max history entries to keep (0 = unlimited)")
    args = parser.parse_args()

    entries = []
    if os.path.exists(args.manifest):
        try:
            with open(args.manifest, encoding="utf-8") as f:
                existing = json.load(f)
            entries = list(existing.get(args.key, []))
        except (json.JSONDecodeError, OSError) as exc:
            print(f"::warning::could not read existing manifest ({exc}); starting fresh", file=sys.stderr)

    by_filename = {e["filename"]: e for e in entries if "filename" in e and "timestamp" in e}
    by_filename[args.add_filename] = {"filename": args.add_filename, "timestamp": args.add_timestamp}
    merged = sorted(by_filename.values(), key=lambda e: e["timestamp"], reverse=True)

    if args.retention > 0 and len(merged) > args.retention:
        kept, pruned = merged[: args.retention], merged[args.retention :]
        for entry in pruned:
            stale_path = os.path.join(args.summaries_dir, entry["filename"])
            try:
                os.remove(stale_path)
                print(f"pruned stale snapshot {stale_path}")
            except FileNotFoundError:
                # Already gone (a prior run pruned it, or the manifest listed a
                # file that was never published) — nothing to reclaim, so ignore.
                pass
        merged = kept

    with open(args.manifest, "w", encoding="utf-8") as f:
        json.dump({args.key: merged}, f, indent=2)
        f.write("\n")

    print(f"manifest updated: {args.key} now has {len(merged)} entr{'y' if len(merged) == 1 else 'ies'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
