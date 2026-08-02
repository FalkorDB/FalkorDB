#!/usr/bin/env python3
"""Regression gate: compare a benchmark CSV against a baseline CSV.

Usage:
  python3 bench/compare.py [current.csv] [baseline.csv] [--threshold 1.10]
                           [--metrics cycles,instr,...]

Defaults: bench/results/current.csv vs bench/baseline/rust.csv. Every metric
present in both CSVs (instr, cycles, branches, br_miss, l1d_miss, alloc_bytes,
dealloc_bytes, ms) is compared and gated; noisy counters get a looser default
threshold (see METRICS). --threshold overrides the threshold for all metrics
uniformly. Exits 1 if any query/metric exceeds its threshold.
Baselines are NOT committed (bench/.gitignore ignores baseline/): a checked-in
baseline goes stale the moment anything lands and then reports phantom
regressions for everyone. Produce your own from a `main` build:

    git stash && cargo build --release
    python3 bench/run_bench.py --out bench/baseline/rust.csv

CI does not use a stored baseline at all — benchmark-cov.yml measures main, the
PR and the C engine in the same run and compares those.
"""
import argparse, csv, os, sys

HERE = os.path.dirname(os.path.abspath(__file__))

# metric -> default regression threshold. Deterministic counters are gated
# tightly; wall-clock and miss counters are noisy run-to-run, so they get slack.
METRICS = {
    "cycles": 1.10,
    "instr": 1.10,
    "branches": 1.10,
    "br_miss": 1.25,
    "l1d_miss": 1.25,
    "alloc_bytes": 1.10,
    "dealloc_bytes": 1.10,
    "ms": 1.25,
}

ap = argparse.ArgumentParser()
ap.add_argument("current", nargs="?", default=os.path.join(HERE, "results/current.csv"))
ap.add_argument("baseline", nargs="?", default=os.path.join(HERE, "baseline/rust.csv"))
ap.add_argument("--threshold", type=float, default=None,
                help="override the regression threshold for every metric")
ap.add_argument("--metrics", default=None,
                help="comma-separated subset of metrics to compare/gate")
args = ap.parse_args()

metrics = list(METRICS)
if args.metrics:
    metrics = [m.strip() for m in args.metrics.split(",") if m.strip()]
    unknown = [m for m in metrics if m not in METRICS]
    if unknown:
        sys.exit(f"unknown metric(s): {', '.join(unknown)}")
thresholds = {m: (args.threshold if args.threshold is not None else METRICS[m]) for m in metrics}

with open(args.current, newline="") as cur_f, open(args.baseline, newline="") as base_f:
    cur = {r["query"]: r for r in csv.DictReader(cur_f)}
    base = {r["query"]: r for r in csv.DictReader(base_f)}

def val(row, key):
    v = row.get(key, "")
    try:
        return float(v)
    except (TypeError, ValueError):
        return None


def has_data(rows, key):
    return any(val(r, key) is not None for r in rows)


present = [m for m in metrics
           if has_data(list(base.values()), m) and has_data(list(cur.values()), m)]
skipped = [m for m in metrics if m not in present]


def ratio(b, c, key):
    bv, cv = val(b, key), val(c, key)
    # A non-positive baseline makes the ratio meaningless (and sign-flipped),
    # so leave it uncompared rather than reporting a bogus improvement.
    if bv is None or cv is None or bv <= 0:
        return None
    return cv / bv


WIDTHS = {m: max(8, len(m) + 2) for m in metrics}


def fmt(r, w):
    return f"{r:>{w}.2f}" if r is not None else f"{'-':>{w}}"


header = (f"{'query':<20} {'base cyc':>13} {'cur cyc':>13} "
          + "".join(f"{m:>{WIDTHS[m]}}" for m in present))
print(header)
print("-" * len(header))

regressions = []
for name, b in base.items():
    c = cur.get(name)
    if not c:
        print(f"{name:<20} MISSING from current")
        continue
    ratios = {m: ratio(b, c, m) for m in present}
    hits = [m for m in present if ratios[m] is not None and ratios[m] > thresholds[m]]
    for m in hits:
        regressions.append((name, m, ratios[m], val(b, m), val(c, m)))
    flag = "  <-- REGRESSION: " + ",".join(hits) if hits else ""
    bc, cc = val(b, "cycles"), val(c, "cycles")
    cyc = f" {bc:>13,.0f} {cc:>13,.0f}" if bc is not None and cc is not None else f" {'-':>13} {'-':>13}"
    print(f"{name:<20}{cyc} "
          + "".join(fmt(ratios[m], WIDTHS[m]) for m in present) + flag)

for name in cur:
    if name not in base:
        print(f"{name:<20} NEW (not in baseline)")

print("\nthresholds: " + ", ".join(f"{m} {thresholds[m]:.0%}" for m in present))
if skipped:
    print("no data in both CSVs (skipped): " + ", ".join(skipped))

if regressions:
    print(f"\n{len(regressions)} regression(s):")
    for name, m, r, bv, cv in sorted(regressions, key=lambda x: -x[2]):
        print(f"  {name} [{m}]: {r:.2f}x  ({bv:,.0f} -> {cv:,.0f}, "
              f"threshold {thresholds[m]:.2f}x)")
    sys.exit(1)
print("\nno regressions")
