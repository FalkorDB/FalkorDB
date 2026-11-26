"""The perf dashboard: one static site built from every stored run.

`bench nightly dashboard --data <dir> --out <site>`. <dir> holds one folder per run,
`<date>-<run id>/`, as perf-dashboard.yml stores it on the gh-pages branch:
`nightly.json`; `run.json`, {"trigger": "nightly", "scope": "main"} (what started the run, and
which line it measures: main, a version branch, the C engine; without it the trigger is unknown
and the scope main); and per check `<check>/` with its result.json and the files its `artifacts`
list. Writes `index.html` (the runs), `run/<folder>.html` (one run) and `trends.html` (each row's
two values over time) into <site>, linking into <dir>. Plain HTML and inline SVG, readable
without JavaScript. Standard library only (and falkorbench.contract and nightly.report, which are
too).
"""

from __future__ import annotations

import csv
import html
import json
import math
import os
import re
import statistics
import urllib.parse
from dataclasses import dataclass
from pathlib import Path

from falkorbench import contract
from falkorbench.nightly import report

FOLDER_RE = re.compile(r"(\d{4}-\d{2}-\d{2})-(\d{1,20})")
CHECK_RE = re.compile(r"[a-z][a-z0-9-]{0,42}")
TREND_RUNS = 90  # the newest runs of a scope that a trend draws
CROSSED = ("regression", "improvement")
FAILING = ("regression", "error")
KNOWN_STATES = {"green", "red", "error", "skipped", "pass", "regression", "improvement"}

# What counts as a regression, per check, in plain words. Every number here is one of the check's
# own constants: bench/tests/test_dashboard.py imports them and fails when the text drifts.
SOAK_RULES = (
    "Each scenario runs once a side. An op's p50 regresses past 2x and by more than 2 ms, its "
    "throughput below 0.7x, a scenario's setup wall time past 1.5x and by more than 5 s; each only "
    "when a re-run of the scenario, order swapped, crosses again. An op is compared once the "
    "baseline has 200 samples; its latency once the candidate has 20 too. At once, no re-run: a "
    "correctness check broken on the candidate, whether or not the baseline breaks it too or an "
    "issue tracks it, a crash, a scenario that did not complete; a latency or resource check "
    "broken on the candidate counts only when every run pair repeats it. Info, not gated: a "
    "p99 shift (past 2x and 10 ms, "
    "with 1,000 baseline samples). Only the release gate leaves out a break the baseline shares, "
    "and a known flaky check while its issue is open (#3095, #3512) unless the candidate broke it "
    "every time it was checked (at least 10 times) in every run; a check recorded only when it "
    "fails cannot show that."
)
RULES = {
    "micro-hw": (
        "Each benchmark query runs 2 times a side on one C4 VM; a row is one query (reads or "
        "writes) and one metric, compared by medians. It regresses when it rises past the "
        "threshold by more than the floor (or past the steep ratio by more than the smaller "
        "floor), both candidate readings sit above both baseline readings, and a re-run repeats "
        "it; a query that errors on the candidate is a regression at once, on both sides too. Rust: "
        "instructions +10% and 250,000; cycles +10% and 20,000,000, or 1.8x and 250,000; branch "
        "misses +25% and 12,000, or 2.8x and 2,000; L2 misses +25% and "
        "225,000, or 3.0x and 15,000; allocated bytes +10% and 65,536, or 8.0x and 4,096. "
        "C (either side): instructions +10% and 1,300,000; cycles +10% and 270,000; branch misses "
        "+25% and 40,000; L2 misses +25% and 665,000, or 3.1x and 15,000; allocated bytes +10% and "
        "3,100,000, or 8.0x and 4,096. Info, not gated: a crossing the re-run did not repeat, "
        "each class's median ratio and sign count."
    ),
    "pokec": (
        "The Pokec dataset, read-only and 5% writes, 3 runs a side. A latency row (p50, p95, p99 "
        "of each query and of all queries) regresses past 1.3x and past a floor (p50 1 ms, p95 "
        "5 ms, p99 15 ms), in the first pass and again in a re-run of 2 runs a side; under writes "
        "only all queries and a read query's p50 gate. Throughput regresses when it drops past "
        "1.3x in both passes. At once, no re-run: a run that did not finish (it took 3x the other "
        "side's slowest run, or its server stopped), an error reply, whether or not the baseline "
        "fails too; a query that times out in 2 "
        "runs, or a PING over 1,000 ms in 2 runs that is past 1.3x the baseline's. Info, not "
        "gated: a crossing the re-run did not repeat, a freeze as long "
        "on the baseline."
    ),
    "ldbc": (
        "LDBC SNB interactive queries, each image its own server. A query's p50 or p95 latency "
        "regresses past 2x and by more than a floor (p50 0.5 ms, p95 2 ms), confirmed by a "
        "re-run of that query on both sides. At once: different rows returned, or a request the "
        "candidate failed (error, timeout, server down), whether or not the baseline failed it "
        "too. Info, not gated: a crossing the re-run did not repeat, a query the baseline cannot "
        "run at all."
    ),
    "soak-a": SOAK_RULES,
    "soak-b": SOAK_RULES,
    "fuzz": (
        "The control epoch (4 runs: candidate, baseline, baseline, candidate) checks invariants. "
        "A data invariant (writes add up, upsert edges complete) or a crash on the candidate "
        "is a regression at once, whether or not the baseline breaks it too; a timing invariant "
        "(no freeze: no PING later than 500 ms "
        "while a plain server answered within 100 ms; no timeouts while waiting; writers don't "
        "starve; quiet graphs stay fast; requests drain; no error replies) broken on the "
        "candidate counts only when one more pair repeats it. Graph build at scale 0.25: a node or edge phase regresses past "
        "1.5x and by more than 500 ms, a wiring query past 1.5x and by more than 2 ms, in the "
        "first runs and again in an extra pair. Info, not gated: a break that did not repeat."
    ),
}

STYLE = """
:root{--bg:#fff;--fg:#1b1d22;--muted:#5d6470;--line:#d9dce1;--card:#f6f7f9;--accent:#3a5bd9;
--base:#8a919c;--red:#c62828;--green:#2e7d32;--amber:#a15c00}
@media (prefers-color-scheme:dark){:root{--bg:#121417;--fg:#e6e8eb;--muted:#9aa1ab;--line:#2c3036;
--card:#1a1d21;--accent:#8ea6ff;--base:#7d848e;--red:#ef7070;--green:#6fcf75;--amber:#e0a84a}}
body{margin:0 auto;max-width:1200px;padding:0 16px 48px;background:var(--bg);color:var(--fg);
font:14px/1.45 system-ui,sans-serif}
header{display:flex;gap:16px;padding:12px 0;border-bottom:1px solid var(--line);margin-bottom:12px}
a{color:var(--accent)}h1{font-size:20px}h2{font-size:16px;margin:24px 0 6px}
table{border-collapse:collapse;width:100%;margin:6px 0}.wide{overflow-x:auto}
th,td{text-align:left;padding:4px 8px;border-bottom:1px solid var(--line);vertical-align:top}
td.n{text-align:right;font-variant-numeric:tabular-nums;white-space:nowrap}
.card{background:var(--card);border:1px solid var(--line);border-radius:6px;padding:4px 12px 10px;
margin:12px 0}.muted,.rules{color:var(--muted)}code{font-size:12px;word-break:break-all}
.green,.pass,.improvement{color:var(--green)}.red,.regression{color:var(--red)}
.error,.skipped,.info{color:var(--amber)}label{margin-right:12px}input[type=radio]{margin:0 4px 0 0}
svg .c{fill:none;stroke:var(--accent);stroke-width:1.5}svg .b{fill:none;stroke:var(--base);
stroke-width:1.5;stroke-dasharray:3 2}svg circle.c,svg circle.b{stroke:none}svg circle.c{fill:var(--accent)}
svg circle.b{fill:var(--base)}svg .m{stroke:var(--line)}svg circle.regression{fill:var(--red)}svg circle.improvement{fill:var(--green)}
"""


@dataclass
class Row:
    candidate: float | None
    baseline: float | None
    status: str
    rerun: str = ""


@dataclass
class Run:
    folder: str  # <date>-<run id>
    night: dict
    status: str
    problems: list[str]
    trigger: str
    scope: str
    rows: dict[str, dict[tuple[str, str], Row]]


def esc(value: object) -> str:
    return html.escape(str(value), quote=True)


def state(value: object) -> str:
    """A status or verdict as a CSS class: only the ones the stylesheet knows."""
    return value if isinstance(value, str) and value in KNOWN_STATES else "error"


def _number(text: object) -> float | None:
    try:
        value = float(text)  # type: ignore[arg-type]
    except (TypeError, ValueError):
        return None
    return value if math.isfinite(value) else None


def _reps(text: str | None) -> list[float]:
    return [v for v in map(_number, (text or "").split(";")) if v is not None]


def read_rows(path: Path) -> dict[tuple[str, str], Row]:
    """A check's raw/rows.csv as (row name, metric) -> its two values, in whichever layout the
    check writes: name + candidate/baseline (soak, pokec), query + class + reps (micro-hw),
    query + phase (ldbc, whose `rerun` rows are the re-run) or one value per side (fuzz)."""
    rows: dict[tuple[str, str], Row] = {}
    reruns: dict[tuple[str, str], Row] = {}
    try:
        with path.open(newline="", encoding="utf-8") as file:
            table = list(csv.DictReader(file))
    except (OSError, UnicodeError, csv.Error):
        return rows
    for r in table:
        metric = r.get("metric") or ""
        if "side" in r:  # fuzz: only its finding rows, "finding (<status>)", pair the sides
            phase = r.get("phase") or ""
            if phase.startswith("finding (") and r["side"] in ("candidate", "baseline"):
                row = rows.setdefault((r.get("name") or "", metric), Row(None, None, phase[9:-1]))
                setattr(row, r["side"], _number(r.get("value")))
            continue
        name = r.get("name") or r.get("query") or ""
        if r.get("class"):  # micro-hw names a finding "<query> [<class>]"
            name = f"{name} [{r['class']}]"
        elif "phase" in r:  # ldbc names one "<query> p50"
            name = f"{name} {metric.removesuffix('_ms')}"
        elif "kind" in r:  # soak renames an info row "<name> (<note>)" or "(p99 shift, not gated)"
            for suffix in (f" ({r.get('note')})", " (p99 shift, not gated)"):
                name = name.removesuffix(suffix)
        if "candidate" in r:
            cand, base = _number(r.get("candidate")), _number(r.get("baseline"))
        else:
            reps_c, reps_b = _reps(r.get("candidate_reps")), _reps(r.get("baseline_reps"))
            cand = statistics.median(reps_c) if reps_c else None
            base = statistics.median(reps_b) if reps_b else None
        if r.get("phase") == "rerun":
            reruns[name, metric] = Row(cand, base, r.get("status") or "")
            continue
        rerun = ""
        if r.get("confirm_candidate_reps"):
            rerun = " vs ".join(
                " · ".join(map(report.number, _reps(r.get(f"confirm_{which}_reps"))))
                for which in ("candidate", "baseline")
            )
        elif r.get("re_run"):
            rerun = {"1": "crossed again", "-1": "improved again"}.get(r["re_run"], "did not cross")
        rows[name, metric] = Row(cand, base, r.get("status") or "", rerun)
    for key, again in reruns.items():  # ldbc: a main row crossed; its re-run decides the status
        if key in rows:
            rows[key].rerun = f"{report.number(again.candidate)} vs {report.number(again.baseline)}"
            rows[key].status = again.status
    return rows


def _json(path: Path) -> dict:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, UnicodeError, ValueError):
        return {}
    return value if isinstance(value, dict) else {}


def load(data: Path) -> list[Run]:
    """Every run folder in <data>, oldest first. A night that is missing, unreadable or that the
    contract rejects reads as an error, never green."""
    runs = []
    folders = sorted(p for p in data.iterdir() if p.is_dir()) if data.is_dir() else []
    for folder in (p for p in folders if FOLDER_RE.fullmatch(p.name)):
        night, status, why = report.read_night(folder / "nightly.json")
        problems = [why] if why else sorted({p.code for p in contract.validate_nightly(night)})
        meta = _json(folder / "run.json")
        rows = {}
        for check in report.checks_of(night):
            name = check.get("check")
            if isinstance(name, str) and CHECK_RE.fullmatch(name):
                rows[name] = read_rows(folder / name / "raw" / "rows.csv")
        runs.append(
            Run(
                folder.name,
                night,
                "error" if problems else status,
                problems,
                report.one_line(meta.get("trigger"), "unknown", 40),
                report.one_line(meta.get("scope"), "main", 40),
                rows,
            )
        )
    return sorted(runs, key=lambda r: (r.folder[:10], int(r.folder[11:])))


def table(head: list[str], rows: list[str]) -> str:
    cells = "".join(f"<th>{esc(h)}</th>" for h in head)
    return f"<div class=wide><table><tr>{cells}</tr>{''.join(rows)}</table></div>"


def page(title: str, body: str, root: str = "") -> str:
    return (
        "<!doctype html><html lang=en><head><meta charset=utf-8>"
        "<meta name=viewport content='width=device-width,initial-scale=1'>"
        f"<title>{esc(title)}</title><style>{STYLE}</style></head><body><header>"
        f"<strong>FalkorDB perf</strong><a href='{root}index.html'>Runs</a>"
        f"<a href='{root}trends.html'>Trends</a></header>{body}</body></html>\n"
    )


def scope_filter(scopes: list[str]) -> str:
    """Radio buttons that show one scope's rows, by CSS alone. The page's <main> must follow them."""
    radios = ["<input type=radio name=scope id=scope-all checked><label for=scope-all>all</label>"]
    css = []
    for i, scope in enumerate(scopes):
        radios.append(
            f"<input type=radio name=scope id=scope-{i}><label for=scope-{i}>{esc(scope)}</label>"
        )
        css.append(f"#scope-{i}:checked~main [data-scope]:not([data-scope=s{i}]){{display:none}}")
    return "Scope: " + "".join(radios) + f"<style>{''.join(css)}</style>"


def side(night: dict, which: str) -> str:
    value = night.get(which) if isinstance(night.get(which), dict) else {}
    commit = value.get("commit") if isinstance(value.get("commit"), str) else ""
    return (
        f"<tr><th>{which}</th><td>{esc(report.label(value))}</td><td><code>"
        f"{esc(report.one_line(value.get('image'), 'unknown', 200))}</code></td>"
        f"<td><code>{esc(commit[:12])}</code></td></tr>"
    )


def runs_page(runs: list[Run], scopes: list[str]) -> str:
    named = [
        [c for c in report.checks_of(run.night) if isinstance(c.get("check"), str)] for run in runs
    ]
    checks = sorted({c["check"] for each in named for c in each})
    lines = []
    for run, each in reversed(list(zip(runs, named, strict=True))):
        verdicts = {c["check"]: c.get("verdict") for c in each}
        cells = "".join(
            f"<td class={state(verdicts[c])}>{esc(verdicts[c])}</td>"
            if c in verdicts
            else "<td></td>"
            for c in checks
        )
        lines.append(
            f"<tr data-scope=s{scopes.index(run.scope)}><td><a href='run/{run.folder}.html'>"
            f"{run.folder[:10]} · {run.folder[11:]}</a></td><td>{esc(run.scope)}</td><td>{esc(run.trigger)}</td>"
            f"<td class={state(run.status)}>{esc(run.status)}</td>"
            f"<td>{esc(report.label(run.night.get('candidate')))}</td>"
            f"<td>{esc(report.label(run.night.get('baseline')))}</td>{cells}</tr>"
        )
    return page(
        "Perf runs",
        (
            "<h1>Perf runs</h1><p class=muted>Each run compares one image with a baseline on the same "
            "machine, with every check. Newest first; a run's page says what failed and where to look."
            f"</p>{scope_filter(scopes)}<main>"
            + table(["run", "scope", "trigger", "status", "candidate", "baseline", *checks], lines)
            + "</main>"
        ),
    )


def link(href: str, text: str) -> str:
    return f"<a href='{esc(href)}'>{esc(text)}</a>"


def as_list(value) -> list:
    """A rejected document may hold a number where a list belongs: render it as empty."""
    return value if isinstance(value, list) else []


def files_of(run: Run, check: dict, data_href: str, data: Path) -> str:
    """Links to the check's own files as stored: result.json, the runner's record, its artifacts."""
    name = check.get("check")
    if not (isinstance(name, str) and CHECK_RE.fullmatch(name)):
        return ""
    root = (data / run.folder / name).resolve()
    paths = ["result.json", "perf-vm.json", "perf-hosted.json"]
    paths += [a for a in as_list(check.get("artifacts")) if isinstance(a, str) and "\0" not in a]
    found = [
        p
        for p in dict.fromkeys(paths)
        if (root / p).resolve().is_relative_to(root) and (root / p).is_file()
    ]
    if not found:
        return "<p class=muted>No files of this check are stored.</p>"
    href = f"../{data_href}/{run.folder}/{name}/"
    shown = [link(href + urllib.parse.quote(p), p) for p in found]
    rest = f" <details><summary>{len(shown) - 4} more</summary>{' '.join(shown[4:])}</details>"
    return "<p>Files: " + " ".join(shown[:4]) + (rest if len(shown) > 4 else "") + "</p>"


def regressions(run: Run, check: dict) -> str:
    """The check's regression rows, each with its re-run from raw/rows.csv when that holds one."""
    rows = run.rows.get(str(check.get("check")), {})
    lines = []
    for f in as_list(check.get("findings")):
        if not isinstance(f, dict) or f.get("status") != "regression":
            continue
        name, metric = f.get("name"), f.get("metric")
        ok = isinstance(name, str) and isinstance(metric, str)
        row = (rows.get((name, metric)) or rows.get((name, ""))) if ok else None  # fuzz: no metric
        again = row.rerun if row and row.rerun else ("confirmed" if check.get("confirmed") else "")
        lines.append(
            f"<tr><td>{esc(report.one_line(f.get('name'), 'unnamed'))}</td><td>{esc(f.get('metric'))}</td>"
            f"<td class=n>{esc(report.number(f.get('candidate')))}</td>"
            f"<td class=n>{esc(report.number(f.get('baseline')))}</td>"
            f"<td class='n {state(f.get('status'))}'>{esc(report.ratio(f.get('ratio')))}</td>"
            f"<td class=n>{esc(report.ratio(f.get('threshold')))}</td><td>{esc(again)}</td></tr>"
        )
    head = ["row", "metric", "candidate", "baseline", "ratio", "threshold", "re-run"]
    return table(head, lines) if lines else ""


def broken_html(check: dict) -> str:
    """The check's pass/fail checks broken on the candidate (#4065): baseline too, issue."""
    repo = f"https://github.com/{report.DEFAULT_REPOSITORY}"
    lines = []
    for entry in report.broken_of(check):
        issue = report.issue_of(entry)
        lines.append(
            f"<tr><td>{esc(report.one_line(entry.get('name'), 'unnamed'))}</td>"
            f"<td>{'yes' if entry.get('baseline_too') is True else 'no'}</td>"
            f"<td>{link(f'{repo}/issues/{issue}', f'#{issue}') if issue else 'none known'}</td></tr>"
        )
    if not lines:
        return ""
    head = ["pass/fail check broken on the candidate", "baseline too", "issue"]
    return table(head, lines)


def commits_html(night: dict) -> str:
    commits, why = report.commits_since_green(night)
    previous, candidate = night.get("previous_green_commit"), report.candidate_commit(night)
    repo = f"https://github.com/{report.DEFAULT_REPOSITORY}"
    compare = ""
    if all(isinstance(c, str) and contract.COMMIT_RE.fullmatch(c) for c in (previous, candidate)):
        compare = " " + link(f"{repo}/compare/{previous}...{candidate}", "compare")
    if why:
        return f"<p>Commits since the last green night: {esc(why)}.{compare}</p>"
    items = "".join(
        f"<li>{link(f'{repo}/pull/{pull}', f'#{pull}') if pull else f'<code>{sha[:7]}</code>'} "
        f"{esc(title)}</li>"
        for sha, title, pull in commits
    )
    return f"<p>Commits since the last green night ({len(commits)}):{compare}</p><ul>{items}</ul>"


def run_page(run: Run, data_href: str, data: Path) -> str:
    night = run.night
    url = report.run_url_of(night, {})
    links = [link(url, "GitHub run")] if url else []
    if (data / run.folder / "nightly.json").is_file():
        links.append(link(f"../{data_href}/{run.folder}/nightly.json", "nightly.json"))
    notes = []
    if run.problems:
        notes.append(
            f"<p class=error>nightly.json is not valid ({esc(', '.join(run.problems))}): read as an error.</p>"
        )
    if night.get("skip_reason"):
        notes.append(
            f"<p class=skipped>Skipped: {esc(report.one_line(night.get('skip_reason'), ''))}</p>"
        )
    if run.status in ("red", "error"):
        notes.append(commits_html(night))
    checks = report.checks_of(night)
    checks.sort(key=lambda c: c.get("verdict") not in FAILING)
    cards = []
    for check in checks:
        name, verdict = str(check.get("check")), check.get("verdict")
        rules = RULES.get(name)
        body = [
            f"<h2>{esc(name)} <span class={state(verdict)}>{esc(verdict)}</span>"
            + (" · confirmed" if check.get("confirmed") else "")
            + "</h2>"
        ]
        if check.get("summary"):
            body.append(f"<p>{esc(report.one_line(check.get('summary'), ''))}</p>")
        if check.get("error"):
            body.append(f"<p class=error>error: {esc(report.one_line(check.get('error'), ''))}</p>")
        if verdict in FAILING:
            body.append(broken_html(check))
            body.append(regressions(run, check))
            if rules:
                body.append(f"<p class=rules>What counts as a regression: {esc(rules)}</p>")
        body.append(files_of(run, check, data_href, data))
        cards.append(f"<section class=card>{''.join(body)}</section>")
    title = f"{run.folder[:10]} · {run.folder[11:]}"
    return page(
        f"Perf run {title}",
        (
            f"<p><a href='../index.html'>← Runs</a></p><h1><span class={state(run.status)}>"
            f"{esc(run.status.upper())}</span> {esc(run.scope)} · {esc(run.trigger)} · {title}</h1>"
            f"<table>{side(night, 'candidate')}{side(night, 'baseline')}</table>"
            f"<p>{' · '.join(links)}</p>{''.join(notes)}{''.join(cards) or '<p>No checks ran.</p>'}"
        ),
        root="../",
    )


def sparkline(points: list[tuple[int, Row]], width: int, changes: list[int]) -> str:
    """Both images' values over the scope's runs: candidate solid, baseline dashed, one point per
    run that measured the row, a dot where the check called it a regression or improvement, and a
    vertical line at each run whose baseline differs from the run before."""
    w, h = 160, 34
    values = [v for _, r in points for v in (r.candidate, r.baseline) if v is not None]
    if not values:
        return ""
    low, high = min(values), max(values)

    def xy(i: int, v: float) -> str:
        y = h / 2 if high == low else h - 3 - (h - 6) * (v - low) / (high - low)
        return f"{3 + (w - 6) * i / max(width - 1, 1):.1f},{y:.1f}"

    parts = [
        f"<line class=m x1={x} x2={x} y1=0 y2={h} />"
        for x in (xy(i, low).split(",")[0] for i in changes)
    ]
    for which in ("candidate", "baseline"):
        line = [(i, getattr(r, which)) for i, r in points if getattr(r, which) is not None]
        if line:
            parts.append(
                f"<polyline class={which[0]} points='{' '.join(xy(i, v) for i, v in line)}'/>"
            )
            x, y = xy(*line[-1]).split(",")
            parts.append(f"<circle class={which[0]} cx={x} cy={y} r=2 />")
    for i, r in points:
        if r.status in CROSSED and r.candidate is not None:
            x, y = xy(i, r.candidate).split(",")
            parts.append(f"<circle class={r.status} cx={x} cy={y} r=3 />")
    return f"<svg viewBox='0 0 {w} {h}' width={w} height={h} role=img>{''.join(parts)}</svg>"


def trend_row(
    name: str, metric: str, points: list[tuple[int, Row]], width: int, changes: list[int]
) -> str:
    last = points[-1][1]
    return (
        f"<tr data-row='{esc(name.lower())}'><td>{esc(name)}</td><td>{esc(metric)}</td>"
        f"<td>{sparkline(points, width, changes)}</td><td class=n>{esc(report.number(last.candidate))}</td>"
        f"<td class=n>{esc(report.number(last.baseline))}</td></tr>"
    )


TREND_HEAD = ["row", "metric", "trend", "candidate, latest", "baseline, latest"]


def trends_page(runs: list[Run], scopes: list[str]) -> str:
    sections = []
    for s, scope in enumerate(scopes):
        window = [r for r in runs if r.scope == scope and any(r.rows.values())][-TREND_RUNS:]
        labels = [report.label(run.night.get("baseline")) for run in window]
        changes = [i for i in range(1, len(window)) if labels[i] != labels[i - 1]]
        series: dict[str, dict[tuple[str, str], list[tuple[int, Row]]]] = {}
        for i, run in enumerate(window):
            for check, rows in run.rows.items():
                for key, row in rows.items():
                    series.setdefault(check, {}).setdefault(key, []).append((i, row))
        for check in sorted(series):
            rows = series[check]
            moved = sorted(
                (k for k in rows if any(r.status in CROSSED for _, r in rows[k])),
                key=lambda k: (-sum(r.status in CROSSED for _, r in rows[k]), k),
            )
            body = [f"<h2>{esc(check)} · {esc(scope)}</h2>"]
            if check in RULES:
                body.append(f"<p class=rules>What counts as a regression: {esc(RULES[check])}</p>")
            if moved:
                body.append(
                    f"<p>Moved (crossed a gate in at least one of these {len(window)} runs):</p>"
                    + table(
                        TREND_HEAD, [trend_row(*k, rows[k], len(window), changes) for k in moved]
                    )
                )
            for metric in sorted({k[1] for k in rows if k not in moved}):
                rest = sorted(k for k in rows if k[1] == metric and k not in moved)
                body.append(
                    f"<details class=fold><summary>{esc(metric or 'no metric')}: {len(rest)} rows that "
                    "did not cross</summary>"
                    + table(
                        TREND_HEAD, [trend_row(*k, rows[k], len(window), changes) for k in rest]
                    )
                    + "</details>"
                )
            sections.append(f"<section data-scope=s{s}>{''.join(body)}</section>")
    search = (
        "<p><input id=find type=search placeholder='Find a row' hidden></p><script>"
        "const f=document.getElementById('find');f.hidden=false;f.oninput=()=>{const t=f.value.toLowerCase();"
        "document.querySelectorAll('[data-row]').forEach(r=>r.hidden=!!t&&!r.dataset.row.includes(t));"
        "document.querySelectorAll('details.fold').forEach(d=>d.open=!!t)}</script>"
    )
    return page(
        "Perf trends",
        (
            "<h1>Trends</h1><p class=muted>Both images' own numbers per row, run after run (candidate "
            "solid, baseline dashed; a red or green dot is a run that called the row a regression or "
            f"improvement; a vertical line, a new baseline), so a release that carries a change does not hide it. The newest {TREND_RUNS} "
            f"runs per scope.</p>{scope_filter(scopes)}{search}<main>"
            + ("".join(sections) or "<p>No rows stored yet.</p>")
            + "</main>"
        ),
    )


def build(data: Path, out: Path) -> int:
    """Write the site for every run in <data> into <out>; returns the number of runs."""
    runs = load(data)
    scopes = sorted({run.scope for run in runs}, key=lambda s: (s != "main", s))
    data_href = Path(os.path.relpath(data.resolve(), out.resolve())).as_posix()
    (out / "run").mkdir(parents=True, exist_ok=True)
    for stale in (out / "run").glob("*.html"):  # a run whose folder is gone loses its page
        stale.unlink()
    (out / ".nojekyll").write_text("", encoding="utf-8")
    (out / "index.html").write_text(runs_page(runs, scopes), encoding="utf-8")
    (out / "trends.html").write_text(trends_page(runs, scopes), encoding="utf-8")
    for run in runs:
        (out / "run" / f"{run.folder}.html").write_text(
            run_page(run, data_href, data), encoding="utf-8"
        )
    return len(runs)
