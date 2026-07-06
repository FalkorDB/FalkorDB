#!/usr/bin/env python3
"""Render the Markdown PR comment for a FalkorDB benchmark run.

Two shapes, both produced by `benchmark aggregate-aws-tests` (which stamps each
run's originating subfolder name into its `vendor` field):

  A/B   — two variants (C engine `falkordb-c` vs Rust `falkordb-rs`).
  A/B/C — three variants, where C (`falkordb-pr`) is the PR's image. Then
          **B→C is exactly this PR's impact** and A/C compares it to the C engine.

Multiple dataset sizes are passed as repeated `--summary <label>:<path>` (one
aggregate JSON per size); the comment shows one row per size. A single
positional summary is still accepted for the plain two-variant case.
"""
import argparse
import json
import sys

from query_metrics import per_query

SIZE_ORDER = {"small": 0, "medium": 1, "large": 2}


def _load(path: str):
    with open(path, encoding="utf-8") as f:
        return json.load(f)


def _run(summary: dict, vendor: str):
    return next((r for r in summary.get("runs", []) if r.get("vendor") == vendor), None)


def _mps(run: dict):
    v = (run or {}).get("result", {}).get("actual-messages-per-second")
    return float(v) if isinstance(v, (int, float)) else None


def _latency_ms(value):
    """Parse a latency string like "20.61ms" into float ms (None if unparseable)."""
    if isinstance(value, (int, float)):
        return float(value)
    if isinstance(value, str) and value.endswith("ms"):
        try:
            return float(value[:-2])
        except ValueError:
            return None
    return None


def _rat(cur, base):
    """cur ÷ base as a rendered ratio cell, or 'n/a'. `good` if >=1 (higher better)."""
    if cur is None or base is None or base == 0:
        return "n/a", "neu"
    r = cur / base
    cls = "good" if r >= 1.03 else "bad" if r <= 0.97 else "neu"
    return (f"{r:.1f}×" if r >= 10 else f"{r:.2f}×"), cls


def _fmt(x):
    if x is None:
        return "n/a"
    return f"{x:,.0f}" if x >= 100 else f"{x:,.1f}" if x >= 10 else f"{x:.2f}"


def throughput_table(sizes, name_a, name_b, name_c, has_c):
    head = ["dataset", f"A · {name_a}", f"B · {name_b}"]
    if has_c:
        head += [f"C · this PR", "B→C", "vs C-engine"]
    else:
        head += ["B ÷ A"]
    lines = ["| " + " | ".join(head) + " |", "| " + " | ".join(["---"] * len(head)) + " |"]
    for label, summ in sizes:
        a, b = _mps(_run(summ, name_a)), _mps(_run(summ, name_b))
        row = [f"`{label}`", _fmt(a), _fmt(b)]
        if has_c:
            c = _mps(_run(summ, name_c))
            bc, _ = _rat(c, b)
            vc, _ = _rat(c, a)
            row += [_fmt(c), bc, vc]
        else:
            ba, _ = _rat(b, a)
            row += [ba]
        lines.append("| " + " | ".join(row) + " |")
    return lines


def latency_table(sizes, name_b, name_c):
    """Aggregate workload latency, this PR (C) relative to published Rust (B). ×<1 = PR faster."""
    lines = [
        "",
        "**Aggregate latency — C (this PR) ÷ B (published Rust)** · ×<1 = PR faster",
        "",
        "| dataset | p50 | p95 | p99 |",
        "| --- | --- | --- | --- |",
    ]
    any_row = False
    for label, summ in sizes:
        b, c = _run(summ, name_b), _run(summ, name_c)
        if not b or not c:
            continue
        lb, lc = b.get("result", {}).get("latency", {}), c.get("result", {}).get("latency", {})
        cells = []
        for pct in ("p50", "p95", "p99"):
            r, _ = _rat(_latency_ms(lc.get(pct)), _latency_ms(lb.get(pct)))
            cells.append(r)
        lines.append(f"| `{label}` | " + " | ".join(cells) + " |")
        any_row = True
    return lines if any_row else []


def _ratio(r):
    return (f"{r:.1f}×" if r >= 10 else f"{r:.2f}×")


def movers_lines(sizes, name_b, name_c, top=6, floor_ms=0.05):
    """Top per-query movers by server exec time, B→C, on the largest size present.

    >1× = the PR sped that query up. Queries where both sides are below
    `floor_ms` are dropped as noise.
    """
    if not sizes:
        return []
    label, summ = sizes[-1]  # sizes are sorted small→large; last = largest present
    b = per_query(_run(summ, name_b), "exec")
    c = per_query(_run(summ, name_c), "exec")

    rows = []
    for q in set(b) & set(c):
        if c[q] <= 0 or (b[q] < floor_ms and c[q] < floor_ms):
            continue
        rows.append((q, b[q] / c[q]))
    if not rows:
        return []

    rows.sort(key=lambda r: r[1], reverse=True)
    faster = [r for r in rows if r[1] >= 1.05][:top]
    slower = sorted((r for r in rows if r[1] <= 0.90), key=lambda r: r[1])[:top]

    out = ["", f"**Top per-query movers** — B→C server exec, `{label}` (>1× = PR faster)"]
    if faster:
        out.append("⚡ Faster: " + " · ".join(f"`{q}` {_ratio(r)}" for q, r in faster))
    if slower:
        out.append("⚠️ Slower: " + " · ".join(f"`{q}` {_ratio(r)}" for q, r in slower))
    if not faster and not slower:
        out.append("_no per-query change beyond noise._")
    return out


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("summary_json", nargs="?", help="single aggregate JSON (two-variant fallback)")
    p.add_argument("--summary", action="append", default=[], metavar="LABEL:PATH",
                   help="a size-labelled aggregate JSON; repeat for multiple sizes")
    p.add_argument("--view", required=True)
    p.add_argument("--published-url", required=True)
    p.add_argument("--name-a", default="falkordb-c")
    p.add_argument("--name-b", default="falkordb-rs")
    p.add_argument("--name-c", default="falkordb-pr")
    args = p.parse_args()

    # Collect (label, summary) pairs, in small→medium→large order.
    sizes = []
    try:
        for spec in args.summary:
            label, _, path = spec.partition(":")
            sizes.append((label or "default", _load(path)))
        if args.summary_json:
            sizes.append(("default", _load(args.summary_json)))
    except (OSError, json.JSONDecodeError) as exc:
        print(f"Could not read a summary: {exc}", file=sys.stderr)
        return 1
    if not sizes:
        print("No summaries given (need a positional JSON or --summary LABEL:PATH)", file=sys.stderr)
        return 2
    sizes.sort(key=lambda s: SIZE_ORDER.get(s[0], 99))

    has_c = any(_run(summ, args.name_c) for _, summ in sizes)
    title = "A/B/C" if has_c else "A/B"

    lines = [f"### FalkorDB {title} benchmark — `{args.view}`", ""]
    if has_c:
        lines += [
            f"**Variants:** A `{args.name_a}` (C engine) · "
            f"B `{args.name_b}` (Rust, published) · C this PR (`rc-pr`)",
            "_B→C = this PR's impact (>1× throughput = faster); vs-C-engine = C ÷ A._",
            "",
        ]
    lines += ["**Throughput — actual msg/s**", ""]
    lines += throughput_table(sizes, args.name_a, args.name_b, args.name_c, has_c)
    if has_c:
        lines += latency_table(sizes, args.name_b, args.name_c)
        lines += movers_lines(sizes, args.name_b, args.name_c)
    lines += ["", f"📊 [Full dashboard]({args.published_url})"]
    print("\n".join(lines))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
