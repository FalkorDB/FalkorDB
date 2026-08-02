#!/usr/bin/env python3
"""Build the PR comment from the CSVs the benchmark-cov jobs produce.

Each measurement job writes one CSV and uploads it; this merges them into a
single markdown report. Kept out of the workflow YAML so it can be run — and
tested — against real CSVs without a CI round trip:

    python3 bench/report.py \
        --measure pr=full-pr.csv base=full-base.csv c=full-c.csv \
        --callgrind pr=cg-pr.csv base=cg-base.csv c=cg-c.csv c2=cg-c2.csv \
        --coverage coverage.txt --provenance provenance.json

Every input is optional: a job that failed simply contributes no section, and
the rest of the report is still posted. That is deliberate — a C-side failure
must not cost the PR-vs-base reading, which is the part that gates the PR.
"""
import argparse, csv, json, os, sys

# A C row is reported as a range when the two passes disagree by more than
# this. The C engine cannot be pinned as tightly as the Rust module (see the
# note printed below), so its numbers are indicative rather than exact.
C_TOLERANCE = 0.02

# Ranking by allocated bytes needs a magnitude floor. Most rows allocate only a
# couple of KB, and a ratio over a ~2 KB denominator is noise dressed as
# signal. Rows below the floor are still measured and still in the CSVs; they
# are just not ranked.
ALLOC_FLOOR = 65536


def read_rows(path):
    """query -> row dict. Missing/unreadable file is an empty result."""
    if not path or not os.path.exists(path):
        return {}
    try:
        with open(path, newline="") as f:
            return {r["query"]: r for r in csv.DictReader(f) if r.get("query")}
    except OSError:
        return {}


def num(row, key):
    """float(row[key]) or None. Never raises: instr/cycles are legitimately
    empty on a runner with no PMU, and a bare float() on that would take the
    whole report down."""
    try:
        return float(row[key])
    except (KeyError, TypeError, ValueError):
        return None


def kv_paths(pairs):
    """['pr=a.csv', 'c=b.csv'] -> {'pr': 'a.csv', 'c': 'b.csv'}"""
    out = {}
    for p in pairs or []:
        if "=" not in p:
            sys.exit(f"expected name=path, got {p!r}")
        k, v = p.split("=", 1)
        out[k] = v
    return out


def measure_section(out, m):
    """Full 317-query set: PR vs base, and PR vs C, on allocated bytes."""
    pr, base, c = m.get("pr", {}), m.get("base", {}), m.get("c", {})
    if not pr:
        out += ["_No PR measurement — see the `measure (pr)` job._", ""]
        return

    # Prefer instructions when a backend delivered them; otherwise allocated
    # bytes, which is deterministic on any jemalloc-built redis. Never
    # wall-clock: on a shared runner that ranks noise.
    have_instr = any(num(r, "instr") is not None for r in pr.values())
    metric = "instr" if have_instr else "alloc_bytes"
    floor = 0 if have_instr else ALLOC_FLOOR

    out += ["## Benchmark", ""]
    out += [
        f"Full {len(pr)}-query set. Each engine is measured on its own runner, "
        f"in parallel, inside the same container image — so all three see an "
        f"identical redis and jemalloc build and the numbers are comparable.",
        "",
    ]
    if have_instr:
        out += ["Ranked by **instructions**.", ""]
    else:
        out += [
            "Ranked by **allocated bytes** — jemalloc merged-arena deltas, which "
            "are deterministic. Instructions are unavailable (hosted runners "
            "expose no PMU) and are left empty rather than filled with a "
            "wall-clock substitute, which would turn the gate into a noise "
            "detector. The deterministic instruction reading is the callgrind "
            "table below.",
            "",
        ]

    if base:
        rows, skipped = [], 0
        for q, p in pr.items():
            b = base.get(q)
            if not b:
                continue
            a, z = num(b, metric), num(p, metric)
            if a is None or z is None or a <= 0:
                continue
            if max(a, z) < floor:
                skipped += 1
                continue
            rows.append((z / a, q, a, z))
        rows.sort()
        improved = [r for r in rows if r[0] < 0.95][:15]
        regressed = [r for r in reversed(rows) if r[0] > 1.05][:15]
        for title, subset in (("Improved", improved), ("Regressed", regressed)):
            if subset:
                out += [f"### {title} vs base ({len(subset)} shown)", "",
                        "| query | base | PR | ratio |", "|---|---|---|---|"]
                out += [f"| {q} | {a:,.0f} | {b:,.0f} | {r:.2f}x |"
                        for r, q, a, b in subset]
                out.append("")
        if rows and not improved and not regressed:
            out += ["No row moved more than 5% either way vs base.", ""]
        if rows:
            geo = 1.0
            for r, *_ in rows:
                geo *= r ** (1.0 / len(rows))
            note = (f" {skipped} row{'s' if skipped != 1 else ''} below the "
                    f"{floor // 1024} KB floor {'were' if skipped != 1 else 'was'} "
                    f"measured but not ranked." if skipped else "")
            out += [f"**{len(rows)} row{'s' if len(rows) != 1 else ''} compared "
                    f"vs base, {metric} geomean {geo:.3f}.**{note}", ""]
    else:
        out += ["_No base measurement — see the `measure (base)` job._", ""]

    if c:
        rows = []
        for q in set(pr) & set(c):
            a, b = num(c[q], "alloc_bytes"), num(pr[q], "alloc_bytes")
            if a and b and a > 0:
                rows.append((b / a, q, b, a))
        rows.sort(reverse=True)
        worse = [r for r in rows if r[0] > 1.0]
        out += [
            "### vs the C engine — allocated bytes",
            "",
            f"{len(rows)} comparable quer{'ies' if len(rows) != 1 else 'y'} "
            f"reported allocation on both engines; **{len(worse)} "
            f"allocate{'s' if len(worse) == 1 else ''} more than C**.",
            "",
            "| query | PR bytes | C bytes | PR/C |",
            "|---|---|---|---|",
        ]
        out += [f"| {q} | {b:,.0f} | {a:,.0f} | {r:.2f}x |"
                for r, q, b, a in rows[:15]]
        if len(rows) > 15:
            out += ["", f"_Worst 15 of {len(rows)} shown._"]
        out.append("")


def callgrind_section(out, cg):
    """Curated subset, exact instruction counts under callgrind."""
    def instr(d):
        return {q: v for q, v in
                ((q, num(r, "instr")) for q, r in d.items()) if v is not None}

    pr, base = instr(cg.get("pr", {})), instr(cg.get("base", {}))
    c, c2 = instr(cg.get("c", {})), instr(cg.get("c2", {}))
    if not pr or not base:
        out += ["## Instruction counts (callgrind)", "",
                "_No output — see the `callgrind` jobs._", ""]
        return

    def c_cell(q):
        """(C display, PR/C display), as a range when the two passes disagree."""
        vals = [v for v in (c.get(q), c2.get(q)) if v]
        if not vals:
            return "—", "—"
        b = pr.get(q)
        if len(vals) == 1 or abs(max(vals) / min(vals) - 1) <= C_TOLERANCE:
            mid = sum(vals) / len(vals)
            tag = "" if len(vals) == 2 else " (1 pass)"
            return f"{mid:,.0f}{tag}", (f"{b / mid:.2f}x" if b else "—")
        lo, hi = min(vals), max(vals)
        return (f"{lo:,.0f}–{hi:,.0f}",
                (f"{b / hi:.2f}–{b / lo:.2f}x" if b else "—"))

    out += [
        "## Instruction counts (callgrind)",
        "",
        "Counted in software, so no PMU is needed — and no hosted runner has "
        "one. Each number is the per-execution cost isolated by differencing "
        "two complete runs at different repeat counts, so server startup, graph "
        "setup and plan compilation cancel out.",
        "",
        "Each query's repeat count is scaled so its error bar lands at "
        "**~0.2%**, so anything past ~0.5% is worth a look and anything past 1% "
        "is real. Curated subset on a smaller graph than the full set above, so "
        "these absolute numbers are not comparable to its rows — only the "
        "ratios are.",
        "",
        "| query | base | PR | PR/base | C | PR/C |",
        "|---|---|---|---|---|---|",
    ]
    for q in sorted(set(pr) & set(base), key=lambda q: pr[q] / base[q] if base[q] else 1):
        a, b = base[q], pr[q]
        cd, cr = c_cell(q)
        out.append(f"| {q} | {a:,.0f} | {b:,.0f} | "
                   f"{f'{b / a:.4f}x' if a else '—'} | {cd} | {cr} |")
    out += [
        "",
        "`RETURN 1` is the fixed per-query floor — it should read ~1.00x for "
        "PR/base, and if it does not, something changed in the request path "
        "rather than in the query being measured.",
        "",
        "**Read the C column as indicative, not exact.** It is measured twice "
        f"and shown as a range when the passes disagree by more than "
        f"{C_TOLERANCE * 100:.0f}%. The C engine cannot be pinned as tightly as "
        "this module: `THREAD_COUNT 0` refuses to start, so a worker thread "
        "always exists and valgrind schedules the handoff nondeterministically. "
        "PR/base in the same table is reproducible to ~0.07%, so a PR/C ratio "
        "is a direction, and PR/base is a gate.",
        "",
    ]
    spread = max((abs(c2[q] / c[q] - 1) for q in set(c) & set(c2) if c[q]),
                 default=None)
    if spread is not None:
        out += [f"_Worst disagreement between the two C passes: "
                f"{spread * 100:.1f}%._", ""]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--measure", nargs="*", metavar="NAME=CSV")
    ap.add_argument("--callgrind", nargs="*", metavar="NAME=CSV")
    ap.add_argument("--coverage", metavar="TXT")
    ap.add_argument("--provenance", metavar="JSON")
    ap.add_argument("--marker", default="<!-- benchmark-cov-report -->")
    args = ap.parse_args()

    measure = {k: read_rows(v) for k, v in kv_paths(args.measure).items()}
    cg = {k: read_rows(v) for k, v in kv_paths(args.callgrind).items()}

    out = []
    measure_section(out, measure)
    callgrind_section(out, cg)

    if args.provenance and os.path.exists(args.provenance):
        with open(args.provenance) as f:
            prov = json.load(f)
        out += ["<details><summary>What was measured</summary>", ""]
        out += [
            "| side | image | digest |",
            "|---|---|---|",
        ]
        for side, info in prov.items():
            out.append(f"| {side} | `{info.get('image', '?')}` | "
                       f"`{(info.get('digest') or '?')[:19]}` |")
        out += [
            "",
            "The **base** side is the `edge-rs` image — the tip of `main-rs` at "
            "the time the images were built, which is not necessarily this PR's "
            "merge base. Rebuilding both sides from source would be exact but "
            "costs ~10 minutes of compile per side; if a row looks surprising, "
            "re-measure it locally against a base build before acting on it.",
            "",
            "</details>",
            "",
        ]

    if args.coverage and os.path.exists(args.coverage):
        with open(args.coverage) as f:
            body = f.read()[-6000:]
        out += ["<details><summary>Query-set coverage of the graph crate</summary>",
                "", "```", body, "```", "", "</details>", ""]

    out.append(args.marker)
    print("\n".join(out))


if __name__ == "__main__":
    main()
