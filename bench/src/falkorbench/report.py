"""Build the PR comment from the CSVs the benchmark-cov jobs produce.

Each measurement job writes one CSV and uploads it; this merges them into a
single markdown report. Kept out of the workflow YAML so it can be run — and
tested — against real CSVs without a CI round trip.

Every input is optional: a job that failed contributes no section and the rest
of the report is still posted. That is deliberate, and it is also why the
run-level pass/fail decision lives *here* rather than in the workflow. GitHub
collapses a matrix into one `result`, so a YAML gate on `needs.measure.result`
fails the whole run when only the C leg failed — contradicting the design, and
guaranteed to happen eventually because the C image is a moving tag. This module
knows exactly which sides arrived, so it decides: missing pr/base is fatal, a
missing C side costs its column and nothing else.
"""

from __future__ import annotations

import json
import os
from dataclasses import dataclass
from dataclasses import field

from falkorbench.metrics import ALLOC_FLOOR
from falkorbench.metrics import C_ERROR_FLOOR
from falkorbench.metrics import C_TOLERANCE
from falkorbench.metrics import CONTROL_QUERY
from falkorbench.metrics import MS_THRESHOLD
from falkorbench.metrics import Row
from falkorbench.metrics import geomean
from falkorbench.metrics import normalise_ms

MARKER = "<!-- benchmark-cov-report -->"


@dataclass
class Report:
    lines: list[str] = field(default_factory=list)
    #: Reasons the *run* should fail, as opposed to a column simply being absent.
    fatal: list[str] = field(default_factory=list)

    def add(self, *lines: str) -> None:
        self.lines.extend(lines)

    def text(self) -> str:
        return "\n".join([*self.lines, MARKER])


def measure_section(rep: Report, sides: dict[str, dict[str, Row]]) -> None:
    """Full query set: PR vs base, and PR vs C, on allocated bytes."""
    pr, base, c = sides.get("pr", {}), sides.get("base", {}), sides.get("c", {})
    if not pr:
        rep.add("_No PR measurement — see the `measure (pr)` job._", "")
        rep.fatal.append("no PR measurement CSV")
        return

    # Prefer instructions when a backend delivered them; otherwise allocated
    # bytes, which is deterministic on any jemalloc-built redis. Never
    # wall-clock: on a shared runner that ranks noise.
    have_instr = any(r.get("instr") is not None for r in pr.values())
    metric = "instr" if have_instr else "alloc_bytes"
    floor = 0 if have_instr else ALLOC_FLOOR

    rep.add(
        "## Benchmark",
        "",
        f"Full {len(pr)}-query set. Each engine is measured on its own runner, in "
        f"parallel, inside the same container image — so all three see an identical "
        f"redis and jemalloc build and the numbers are comparable.",
        "",
    )
    if have_instr:
        rep.add("Ranked by **instructions**.", "")
    else:
        rep.add(
            "Ranked by **allocated bytes** — jemalloc merged-arena deltas, which are "
            "deterministic. Instructions are unavailable (hosted runners expose no PMU) "
            "and are left empty rather than filled with a wall-clock substitute, which "
            "would turn the gate into a noise detector. The deterministic instruction "
            "reading is the callgrind table below.",
            "",
        )

    if not base:
        rep.add("_No base measurement — see the `measure (base)` job._", "")
        rep.fatal.append("no base measurement CSV")
    else:
        _vs_base(rep, pr, base, metric, floor)
        _wall_clock(rep, pr, base)

    if c:
        _vs_c(rep, pr, c)


def _vs_base(
    rep: Report, pr: dict[str, Row], base: dict[str, Row], metric: str, floor: int
) -> None:
    rows, skipped = [], 0
    for name, p in pr.items():
        b = base.get(name)
        if b is None:
            continue
        a, z = b.get(metric), p.get(metric)
        if a is None or z is None or a <= 0:
            continue
        if max(a, z) < floor:
            skipped += 1
            continue
        rows.append((z / a, name, a, z))
    rows.sort()

    improved = [r for r in rows if r[0] < 0.95][:15]
    regressed = [r for r in reversed(rows) if r[0] > 1.05][:15]
    for title, subset in (("Improved", improved), ("Regressed", regressed)):
        if subset:
            rep.add(
                f"### {title} vs base ({len(subset)} shown)",
                "",
                "| query | base | PR | ratio |",
                "|---|---|---|---|",
                *[f"| {n} | {a:,.0f} | {z:,.0f} | {r:.2f}x |" for r, n, a, z in subset],
                "",
            )
    if rows and not improved and not regressed:
        rep.add("No row moved more than 5% either way vs base.", "")
    if rows:
        geo = geomean(r for r, *_ in rows)
        note = ""
        if skipped:
            plural = "s were" if skipped != 1 else " was"
            note = f" {skipped} row{plural} below the {floor // 1024} KB floor measured but not ranked."
        plural = "s" if len(rows) != 1 else ""
        rep.add(
            f"**{len(rows)} row{plural} compared vs base, {metric} geomean {geo:.3f}.**{note}",
            "",
        )


def _wall_clock(rep: Report, pr: dict[str, Row], base: dict[str, Row]) -> None:
    """A coarse net for a regression that neither allocates nor lands in the
    callgrind subset.

    Raw `ms` is useless across two hosts and the numbers say so: on a null
    comparison the median PR/base ratio was 1.464, with 316 of 317 rows off by
    more than 5%, purely because each side got its own runner. Dividing by the
    control row cancels that; the same data then read median 0.973, p99 1.110,
    max 1.17. Hence a ±50% threshold — comfortably outside measured noise, still
    tight enough to catch something gross. Anything finer belongs to the two
    deterministic metrics.
    """
    offset = normalise_ms(pr, base)
    if offset is None:
        return

    flagged = []
    for name in set(pr) & set(base):
        a, b = base[name].get("ms"), pr[name].get("ms")
        if a is None or b is None or a <= 0:
            continue
        r = (b / a) / offset
        if abs(r - 1) > MS_THRESHOLD:
            flagged.append((r, name, a, b))
    flagged.sort(reverse=True)

    rep.add("### Wall-clock outliers", "")
    if flagged:
        plural = "ies" if len(flagged) != 1 else "y"
        rep.add(
            f"{len(flagged)} quer{plural} moved more than {MS_THRESHOLD:.0%} after "
            f"cancelling the per-runner speed offset (this run: {offset:.2f}x, from "
            f"`{CONTROL_QUERY}`). Wall-clock is noisy — on identical engines nothing "
            f"exceeded 1.17x — so treat these as leads to re-measure, not results.",
            "",
            "| query | base ms | PR ms | normalised |",
            "|---|---|---|---|",
            *[f"| {n} | {a:.3f} | {b:.3f} | {r:.2f}x |" for r, n, a, b in flagged[:15]],
            "",
        )
    else:
        rep.add(
            f"None. No query moved more than {MS_THRESHOLD:.0%} in wall-clock after "
            f"cancelling the per-runner speed offset (this run: {offset:.2f}x, measured "
            f"from `{CONTROL_QUERY}`).",
            "",
        )


def _vs_c(rep: Report, pr: dict[str, Row], c: dict[str, Row]) -> None:
    rows = []
    for name in set(pr) & set(c):
        a, b = c[name].get("alloc_bytes"), pr[name].get("alloc_bytes")
        if a is None or b is None or a <= 0 or b <= 0:
            continue
        rows.append((b / a, name, b, a))
    rows.sort(reverse=True)
    if not rows:
        return
    worse = [r for r in rows if r[0] > 1.0]
    plural = "ies" if len(rows) != 1 else "y"
    rep.add(
        "### vs the C engine — allocated bytes",
        "",
        f"{len(rows)} comparable quer{plural} reported allocation on both engines; "
        f"**{len(worse)} allocate{'s' if len(worse) == 1 else ''} more than C**.",
        "",
        "| query | PR bytes | C bytes | PR/C |",
        "|---|---|---|---|",
        *[f"| {n} | {b:,.0f} | {a:,.0f} | {r:.2f}x |" for r, n, b, a in rows[:15]],
    )
    if len(rows) > 15:
        rep.add("", f"_Worst 15 of {len(rows)} shown._")
    rep.add("")


def callgrind_section(rep: Report, sides: dict[str, dict[str, Row]]) -> None:
    """Curated subset, exact instruction counts under callgrind."""

    def instrs(d: dict[str, Row]) -> dict[str, float]:
        return {k: v for k, v in ((k, r.get("instr")) for k, r in d.items()) if v is not None}

    pr = instrs(sides.get("pr", {}))
    base = instrs(sides.get("base", {}))
    c = instrs(sides.get("c", {}))
    c2 = instrs(sides.get("c2", {}))

    rep.add("## Instruction counts (callgrind)", "")
    if not pr or not base:
        rep.add("_No output — see the `callgrind` jobs._", "")
        rep.fatal.append("no callgrind output for pr and/or base")
        return

    rep.add(
        "Counted in software, so no PMU is needed — and no hosted runner has one. Each "
        "number is the per-execution cost isolated by differencing two complete runs at "
        "different repeat counts, so server startup, graph setup and plan compilation "
        "cancel out.",
        "",
        "Each query's repeat count is scaled so its error bar lands at **~0.2%**, so "
        "anything past ~0.5% is worth a look and anything past 1% is real. Curated subset "
        "on a smaller graph than the full set above, so these absolute numbers are not "
        "comparable to its rows — only the ratios are.",
        "",
    )

    have_c = bool(c or c2)
    shared = sorted(set(pr) & set(base), key=lambda q: pr[q] / base[q] if base[q] else 1.0)

    def c_cell(name: str) -> tuple[str, str]:
        """(C display, PR/C display), as a range when the two passes disagree."""
        vals = [v for v in (c.get(name), c2.get(name)) if v]
        # A C value far below the cheapest real query is an error reply, not a
        # measurement: the C engine rejects a few things this set exercises
        # (`=~`, for one) and the error path costs ~500-2500 instructions against
        # ~149k for `RETURN 1`. Reporting that as a ratio invents a 60x win.
        vals = [v for v in vals if v >= C_ERROR_FLOOR]
        # Both passes required. One number with nothing to check it against is
        # exactly where the noise hides — an earlier revision printed those as
        # "(1 pass)" and 43 of 93 rows in one report were single-pass, including
        # `RETURN 1` at 4.0M instructions against a true cost near 150k.
        if len(vals) < 2:
            return "n/a", "n/a"
        lo, hi = min(vals), max(vals)
        b = pr.get(name)
        if hi / lo - 1 <= C_TOLERANCE:
            mid = (lo + hi) / 2
            return f"{mid:,.0f}", (f"{b / mid:.2f}x" if b else "—")
        return f"{lo:,.0f}–{hi:,.0f}", (f"{b / hi:.2f}–{b / lo:.2f}x" if b else "—")

    def row(name: str) -> str:
        a, b = base[name], pr[name]
        cells = f"| {name} | {a:,.0f} | {b:,.0f} | {f'{b / a:.4f}x' if a else '—'} |"
        if have_c:
            display, rel = c_cell(name)
            cells += f" {display} | {rel} |"
        return cells

    head = (
        ["| query | base | PR | PR/base | C | PR/C |", "|---|---|---|---|---|---|"]
        if have_c
        else ["| query | base | PR | PR/base |", "|---|---|---|---|"]
    )

    # 93 rows is too many to read inline, and the ones that matter are the ones
    # that moved. Anything past 0.5% is worth a look per the note above.
    movers = [q for q in shared if base[q] and abs(pr[q] / base[q] - 1) > 0.005]
    if movers:
        rep.add(f"### Moved more than 0.5% ({len(movers)} of {len(shared)})", "")
        rep.add(*head, *[row(q) for q in movers], "")
    else:
        rep.add(
            f"**No query moved more than 0.5%** across {len(shared)} measured — the PR is "
            f"instruction-neutral against its base.",
            "",
        )

    rep.add(f"<details><summary>All {len(shared)} queries</summary>", "")
    rep.add(*head, *[row(q) for q in shared])
    rep.add("", "</details>", "")

    rep.add(
        f"`{CONTROL_QUERY}` is the fixed per-query floor — it should read ~1.00x for "
        f"PR/base, and if it does not, something changed in the request path rather than "
        f"in the query being measured.",
        "",
    )
    if have_c:
        rep.add(
            "**Read the C column as indicative, not exact.** It is measured twice and "
            f"shown as a range when the passes disagree by more than {C_TOLERANCE:.0%}; a "
            "row with only one usable pass is shown as n/a.",
            "",
        )
        spread = max((abs(c2[q] / c[q] - 1) for q in set(c) & set(c2) if c[q]), default=None)
        if spread is not None:
            rep.add(f"_Worst disagreement between the two C passes: {spread * 100:.1f}%._", "")
    else:
        rep.add(
            "_The C engine is not in this table._ callgrind needs a deterministic process, "
            "and the C engine busy-waits on a worker thread that valgrind schedules "
            "arbitrarily — measured at 331,579,187 instructions of drift between two "
            "identical runs, against ~100k for this module. The vs-C comparison lives on "
            "allocated bytes above, which thread scheduling does not affect.",
            "",
        )


def provenance_section(rep: Report, path: str | None) -> None:
    if not path or not os.path.exists(path):
        return
    with open(path) as f:
        prov = json.load(f)

    rep.add("<details><summary>What was measured</summary>", "")
    head = prov.get("_head_sha")
    if head:
        rep.add(
            f"PR head `{head[:9]}` — the `rc-pr` image below was confirmed built from "
            f"this commit before measuring.",
            "",
        )
    rep.add("| side | image | digest |", "|---|---|---|")
    for side, info in prov.items():
        # `_`-prefixed keys are metadata, not sides.
        if side.startswith("_"):
            continue
        digest = (info.get("digest") or "?")[:19]
        rep.add(f"| {side} | `{info.get('image', '?')}` | `{digest}` |")

    # The environment all three sides were measured in. Worth surfacing rather
    # than leaving in the JSON: it is content-addressed on the C image digest plus
    # the harness lockfile, so it identifies the exact toolchain behind the
    # numbers.
    bench_image = prov.get("_bench_image")
    if bench_image:
        rep.add("", f"Measured inside `{bench_image}`.")

    rep.add(
        "",
        "The **base** side is the `edge-rs` image — the tip of the trunk at the time the "
        "images were built, which is not necessarily this PR's merge base. Rebuilding both "
        "sides from source would be exact but costs ~10 minutes of compile per side; if a "
        "row looks surprising, re-measure it locally against a base build before acting "
        "on it.",
        "",
        "</details>",
        "",
    )


def coverage_section(rep: Report, path: str | None) -> None:
    if not path or not os.path.exists(path):
        return
    with open(path) as f:
        body = f.read()[-6000:]
    rep.add(
        "<details><summary>Query-set coverage of the graph crate</summary>",
        "",
        "```",
        body,
        "```",
        "",
        "</details>",
        "",
    )


def build(
    measure: dict[str, dict[str, Row]],
    callgrind: dict[str, dict[str, Row]],
    *,
    provenance: str | None = None,
    coverage: str | None = None,
) -> Report:
    rep = Report()
    measure_section(rep, measure)
    callgrind_section(rep, callgrind)
    provenance_section(rep, provenance)
    coverage_section(rep, coverage)
    return rep
