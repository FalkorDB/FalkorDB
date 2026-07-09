#!/usr/bin/env python3
"""Aggregate every published benchmark snapshot into one per-query `trend.json`
for the main-branch trend page (benchmark/trend/).

Reads the canonical view's manifest + snapshot JSONs (the same files the
dashboard's run-history dropdown uses) and, for each metric and query, emits the
two engines' values over time. The trend page then renders one sparkline per
query with a drill-down — no per-snapshot fetching in the browser.

Output shape (compact; points are [timestamp, a_value_or_null, b_value_or_null]):

  {
    "engines": {"a": "falkordb-c", "b": "falkordb-rs"},
    "metrics": ["exec", "p50", "p95", "p99"],
    "queries": ["aggregate_age", ...],
    "series": { "exec": { "aggregate_age": [[172..., 81.1, 50.2], ...] }, ... }
  }
"""
import argparse
import json
import sys

from query_metrics import METRICS, per_query


def _run(summary, vendor):
    return next((r for r in summary.get("runs", []) if r.get("vendor") == vendor), None)


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--summaries-dir", required=True)
    p.add_argument("--manifest", required=True)
    p.add_argument("--out", required=True)
    p.add_argument("--key", default="falkordb_vs_falkordb.json",
                   help="manifest key whose history to trend")
    p.add_argument("--name-a", default="falkordb-c")
    p.add_argument("--name-b", default="falkordb-rs")
    args = p.parse_args()

    try:
        with open(args.manifest, encoding="utf-8") as f:
            entries = json.load(f).get(args.key, [])
    except (OSError, json.JSONDecodeError) as exc:
        print(f"could not read manifest {args.manifest}: {exc}", file=sys.stderr)
        return 1

    # Oldest → newest so each series reads left-to-right in time.
    entries = sorted(
        (e for e in entries if "filename" in e and "timestamp" in e),
        key=lambda e: e["timestamp"],
    )

    # series[metric][query] = [[t, a, b], ...]
    series = {m: {} for m in METRICS}
    queries = set()

    for e in entries:
        t = int(e["timestamp"])
        try:
            with open(f"{args.summaries_dir}/{e['filename']}", encoding="utf-8") as f:
                summ = json.load(f)
        except (OSError, json.JSONDecodeError):
            continue  # a pruned/corrupt snapshot shouldn't sink the whole trend
        run_a, run_b = _run(summ, args.name_a), _run(summ, args.name_b)
        for m in METRICS:
            qa, qb = per_query(run_a, m), per_query(run_b, m)
            for q in set(qa) | set(qb):
                queries.add(q)
                series[m].setdefault(q, []).append([t, qa.get(q), qb.get(q)])

    out = {
        "engines": {"a": args.name_a, "b": args.name_b},
        "metrics": list(METRICS),
        "queries": sorted(queries),
        "series": series,
    }
    with open(args.out, "w", encoding="utf-8") as f:
        json.dump(out, f, separators=(",", ":"))
        f.write("\n")

    print(f"trend.json: {len(queries)} queries × {len(entries)} snapshots → {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
