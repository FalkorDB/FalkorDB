#!/usr/bin/env python3
"""Render a short Markdown summary of a falkordb_vs_falkordb comparison JSON
(as produced by `benchmark aggregate-aws-tests`) for posting as a PR comment.

The aggregate stamps each run's originating subfolder name into its `vendor`
field (e.g. "falkordb-c" / "falkordb-rs"), so the per-vendor rows are already
unambiguous. On top of that we surface the actual A/B *gap* — variant B's
throughput and latency relative to variant A — since publishing that gap is
the whole point of the comparison.
"""
import argparse
import json
import sys


def _latency_ms(value):
    """Parse a latency string like "20.61ms" into float milliseconds.

    Returns None for anything we don't recognise (a missing value, or a unit
    the upstream tool didn't emit before) so the caller can skip that row
    rather than crash the whole comment.
    """
    if isinstance(value, (int, float)):
        return float(value)
    if isinstance(value, str) and value.endswith("ms"):
        try:
            return float(value[:-2])
        except ValueError:
            return None
    return None


def fmt_run(run: dict) -> str:
    result = run.get("result", {})
    latency = result.get("latency", {})
    mps = result.get("actual-messages-per-second")
    mps_str = f"{mps:,.0f}" if isinstance(mps, (int, float)) else "n/a"
    errors = result.get("errors", "n/a")
    return (
        f"| {run.get('vendor', 'unknown')} "
        f"| {latency.get('p50', 'n/a')} "
        f"| {latency.get('p95', 'n/a')} "
        f"| {latency.get('p99', 'n/a')} "
        f"| {mps_str} "
        f"| {errors} |"
    )


def _find_run(runs: list, vendor: str):
    return next((r for r in runs if r.get("vendor") == vendor), None)


def gap_lines(runs: list, name_a: str, name_b: str) -> list:
    """A small "gap" table: variant B relative to variant A.

    Aligns the two runs by their `vendor` label (not array position) so a
    reordered aggregate can't silently swap which side is which. If either
    side is missing we omit the section rather than guess.
    """
    a, b = _find_run(runs, name_a), _find_run(runs, name_b)
    if a is None or b is None:
        return []

    ra, rb = a.get("result", {}), b.get("result", {})
    rows = []

    mps_a = ra.get("actual-messages-per-second")
    mps_b = rb.get("actual-messages-per-second")
    if isinstance(mps_a, (int, float)) and isinstance(mps_b, (int, float)) and mps_a:
        rows.append(f"| msg/s (actual) | {mps_a:,.0f} | {mps_b:,.0f} | {mps_b / mps_a:.2f}× |")

    la, lb = ra.get("latency", {}), rb.get("latency", {})
    for pct in ("p50", "p95", "p99"):
        ms_a, ms_b = _latency_ms(la.get(pct)), _latency_ms(lb.get(pct))
        if ms_a and ms_b is not None:
            rows.append(f"| {pct} latency | {la.get(pct)} | {lb.get(pct)} | {ms_b / ms_a:.2f}× |")

    if not rows:
        return []

    return [
        "",
        f"**Gap — `{name_b}` relative to `{name_a}`** "
        f"(msg/s ×>1 ⇒ B higher throughput; latency ×>1 ⇒ B slower)",
        "",
        f"| metric | {name_a} | {name_b} | B ÷ A |",
        "| --- | --- | --- | --- |",
        *rows,
    ]


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("summary_json")
    parser.add_argument("--view", required=True)
    parser.add_argument("--published-url", required=True)
    parser.add_argument("--name-a", default="falkordb-c", help="vendor label of variant A (the C baseline)")
    parser.add_argument("--name-b", default="falkordb-rs", help="vendor label of variant B (this repo)")
    args = parser.parse_args()

    try:
        with open(args.summary_json, encoding="utf-8") as f:
            data = json.load(f)
    except (OSError, json.JSONDecodeError) as exc:
        print(f"Could not read {args.summary_json}: {exc}", file=sys.stderr)
        return 1

    runs = data.get("runs", [])
    lines = [
        f"### FalkorDB A/B benchmark — `{args.view}`",
        "",
        "| vendor | p50 | p95 | p99 | msg/s (actual) | errors |",
        "| --- | --- | --- | --- | --- | --- |",
    ]
    lines.extend(fmt_run(run) for run in runs)
    lines.extend(gap_lines(runs, args.name_a, args.name_b))
    lines.append("")
    lines.append(f"📊 [Full dashboard]({args.published_url})")
    print("\n".join(lines))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
