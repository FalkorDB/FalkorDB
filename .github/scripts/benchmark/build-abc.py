#!/usr/bin/env python3
"""Fold the per-size A/B/C summaries into one `abc.json` for the PR impact page
(benchmark/branch/pr-<N>/impact/), which has an in-page size switcher.

Reads summary-<size>.json for each size present in --summaries-dir and emits
throughput, aggregate latency, and per-query metrics for all three variants,
so the page can flip sizes/metrics without re-fetching. B→C (variant B vs C) is
computed in the page.
"""
import argparse
import json
import sys

from query_metrics import METRICS, parse_latency_ms, per_query

SIZE_ORDER = ["small", "medium", "large"]


def _run(summ, vendor):
    return next((r for r in summ.get("runs", []) if r.get("vendor") == vendor), None)


def _mps(run):
    v = (run or {}).get("result", {}).get("actual-messages-per-second")
    return v if isinstance(v, (int, float)) else None


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--summaries-dir", required=True)
    p.add_argument("--out", required=True)
    p.add_argument("--name-a", default="falkordb-c")
    p.add_argument("--name-b", default="falkordb-rs")
    p.add_argument("--name-c", default="falkordb-pr")
    args = p.parse_args()
    names = {"a": args.name_a, "b": args.name_b, "c": args.name_c}

    sizes, throughput, latency, perquery = [], {}, {}, {}
    for sz in SIZE_ORDER:
        try:
            with open(f"{args.summaries_dir}/summary-{sz}.json", encoding="utf-8") as f:
                summ = json.load(f)
        except (OSError, json.JSONDecodeError):
            continue
        runs = {k: _run(summ, v) for k, v in names.items()}
        if not runs["b"] or not runs["c"]:
            continue  # need at least B and C to be an A/B/C impact view

        sizes.append(sz)
        throughput[sz] = {k: _mps(runs[k]) for k in names}
        latency[sz] = {
            k: {pct: parse_latency_ms((runs[k] or {}).get("result", {}).get("latency", {}).get(pct))
                for pct in ("p50", "p95", "p99")}
            for k in names
        }
        perquery[sz] = {}
        for m in METRICS:
            pq = {k: per_query(runs[k], m) for k in names}
            all_q = set().union(*(set(pq[k]) for k in names))
            perquery[sz][m] = {q: {k: pq[k].get(q) for k in names} for q in all_q}

    if not sizes:
        print("no A/B/C summaries found — impact page needs variant C", file=sys.stderr)
        return 1

    out = {
        "variants": names,
        "sizes": sizes,
        "metrics": list(METRICS),
        "throughput": throughput,
        "latency": latency,
        "perquery": perquery,
    }
    with open(args.out, "w", encoding="utf-8") as f:
        json.dump(out, f, separators=(",", ":"))
        f.write("\n")
    print(f"abc.json: {len(sizes)} size(s) → {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
