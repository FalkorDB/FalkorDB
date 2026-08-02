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
import argparse, csv, glob, json, os, sys

# A C row is reported as a range when the two passes disagree by more than
# this. The C engine cannot be pinned as tightly as the Rust module (see the
# note printed below), so its numbers are indicative rather than exact.
C_TOLERANCE = 0.02

# Ranking by allocated bytes needs a magnitude floor. Most rows allocate only a
# couple of KB, and a ratio over a ~2 KB denominator is noise dressed as
# signal. Rows below the floor are still measured and still in the CSVs; they
# are just not ranked.
ALLOC_FLOOR = 65536

# Below this, a callgrind instruction count for the C engine is an error reply
# rather than an executed query — see c_cell().
C_ERROR_FLOOR = 20000


def read_rows(pattern):
    """query -> row dict, merged over every file matching `pattern`.

    A glob rather than a path because the callgrind subset is sharded across
    parallel jobs and arrives as cg-<side>-1.csv .. cg-<side>-4.csv. A shard
    that failed contributes nothing and the rest still report.
    """
    out = {}
    for path in sorted(glob.glob(pattern)) if pattern else []:
        try:
            with open(path, newline="") as f:
                for r in csv.DictReader(f):
                    if r.get("query"):
                        out[r["query"]] = r
        except OSError:
            continue
    return out


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
        # A C value far below the cheapest real query is an error reply, not a
        # measurement: the C engine rejects a few things this set exercises
        # (`=~`, for one) and the error path costs ~500-2500 instructions,
        # against ~149k for `RETURN 1` on C. Reporting that as a ratio invents
        # a 60x win. The floor sits well under any real query and well over any
        # error path.
        vals = [v for v in vals if v >= C_ERROR_FLOOR]
        # Both passes required. A single pass is not a measurement: the whole
        # reason C is run twice is that it cannot be pinned as tightly as this
        # module, and one number with nothing to check it against is exactly
        # the case where the noise is invisible. An earlier revision printed
        # these as "(1 pass)" and 43 of 93 rows in one report were single-pass
        # — including `RETURN 1` at 4.0M instructions against a true cost near
        # 150k. Report nothing rather than that.
        if len(vals) < 2:
            return "n/a", "n/a"
        lo, hi = min(vals), max(vals)
        b = pr.get(q)
        if hi / lo - 1 <= C_TOLERANCE:
            mid = (lo + hi) / 2
            return f"{mid:,.0f}", (f"{b / mid:.2f}x" if b else "—")
        # The passes disagree. Show the range so the width is visible rather
        # than hiding it behind a midpoint.
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
    ]

    shared = sorted(set(pr) & set(base), key=lambda q: pr[q] / base[q] if base[q] else 1)

    # The C engine is not measured under callgrind any more — it busy-waits on
    # a worker thread that valgrind schedules arbitrarily, so its counts are
    # not reproducible (331M instructions of drift between two identical runs).
    # Keep the columns working for a local run that does supply C data, but
    # drop them entirely rather than print a column of n/a.
    have_c = bool(c or c2)

    def row(q):
        a, b = base[q], pr[q]
        cells = f"| {q} | {a:,.0f} | {b:,.0f} | {f'{b / a:.4f}x' if a else '—'} |"
        if have_c:
            cd, cr = c_cell(q)
            cells += f" {cd} | {cr} |"
        return cells

    HEAD = (["| query | base | PR | PR/base | C | PR/C |", "|---|---|---|---|---|---|"]
            if have_c else
            ["| query | base | PR | PR/base |", "|---|---|---|---|"])

    # 93 rows is too many to read inline, and the ones that matter are the ones
    # that moved. Anything past 0.5% is worth a look per the note above, so that
    # is the cut: movers up front, everything else one click away.
    movers = [q for q in shared if base[q] and abs(pr[q] / base[q] - 1) > 0.005]
    if movers:
        out += [f"### Moved more than 0.5% ({len(movers)} of {len(shared)})", ""]
        out += HEAD + [row(q) for q in movers] + [""]
    else:
        out += [f"**No query moved more than 0.5%** across {len(shared)} measured "
                f"— the PR is instruction-neutral against its base.", ""]

    out += ["<details><summary>All " + str(len(shared)) + " queries</summary>", ""]
    out += HEAD + [row(q) for q in shared]
    out += ["", "</details>", ""]

    out += [
        "`RETURN 1` is the fixed per-query floor — it should read ~1.00x for "
        "PR/base, and if it does not, something changed in the request path "
        "rather than in the query being measured.",
        "",
    ]
    if have_c:
        out += [
            "**Read the C column as indicative, not exact.** It is measured "
            f"twice and shown as a range when the passes disagree by more than "
            f"{C_TOLERANCE * 100:.0f}%; a row with only one usable pass is shown "
            "as n/a, because one number with nothing to check it against is "
            "exactly where the noise hides.",
            "",
        ]
    else:
        out += [
            "_The C engine is not in this table._ callgrind needs a "
            "deterministic process, and the C engine busy-waits on a worker "
            "thread that valgrind schedules arbitrarily — measured at "
            "331,579,187 instructions of drift between two identical runs, "
            "against ~100k for this module. The vs-C comparison lives on "
            "allocated bytes above, which thread scheduling does not affect.",
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
        head = prov.get("_head_sha")
        if head:
            out += [f"PR head `{head[:9]}` — the `rc-pr` image below was confirmed "
                    f"built from this commit before measuring.", ""]
        out += [
            "| side | image | digest |",
            "|---|---|---|",
        ]
        # `_`-prefixed keys are metadata, not sides.
        for side, info in prov.items():
            if side.startswith("_"):
                continue
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
