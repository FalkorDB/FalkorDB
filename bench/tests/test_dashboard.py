"""`bench nightly dashboard`: the static site built from the stored runs (falkorbench.nightly.dashboard).

The runs are the example nightlies (tests/fixtures) with a few fake raw/rows.csv files, laid out as
perf-dashboard.yml stores them: data/<date>-<run id>/{nightly.json,run.json,<check>/}.
"""

from __future__ import annotations

import copy
import html
import io
import json
import re
import shutil
import subprocess
import sys
import zipfile
from datetime import UTC
from datetime import datetime
from pathlib import Path

import pytest
import yaml
from click.testing import CliRunner

from falkorbench import contract
from falkorbench import metrics
from falkorbench import verdict
from falkorbench.cli import cli
from falkorbench.fuzz import check as fuzz_check
from falkorbench.fuzz import checks as fuzz_checks
from falkorbench.fuzz import verdict as fuzz_verdict
from falkorbench.ldbc import check as ldbc
from falkorbench.nightly import dashboard
from falkorbench.nightly import failures as nightly_failures
from falkorbench.nightly import store as nightly_store
from falkorbench.nightly import unlisted as nightly_unlisted
from falkorbench.pokec import check as pokec
from falkorbench.soak import check as soak
from fixtures import examples

BENCH = Path(__file__).resolve().parents[1]
ROWS_HEADER = "query,class,metric,status,ratio,candidate_reps,baseline_reps,confirm_candidate_reps,confirm_baseline_reps,note\n"
ROW = "UNWIND id MATCH edge 1 row [writes]"
HOSTILE = "<script>alert('x')</script> & <b>co</b>"


def micro_hw(passed: bool, summary: str = "1 confirmed regression") -> dict:
    result = examples.example("result-micro-hw-regression.json")
    result["summary"] = summary
    result["artifacts"] = ["raw/rows.csv"]
    if passed:
        result.update(verdict="pass", confirmed=False, findings=[])
    return result


def store(data: Path, folder: str, night: dict, run: dict | None, rows: str | None = None) -> None:
    """One run as perf-dashboard.yml stores it; `rows` is micro-hw's raw/rows.csv body."""
    (data / folder).mkdir(parents=True)
    (data / folder / "nightly.json").write_text(json.dumps(night), encoding="utf-8")
    if run is not None:
        (data / folder / "run.json").write_text(json.dumps(run), encoding="utf-8")
    if rows is not None:
        check = data / folder / "micro-hw"
        (check / "raw").mkdir(parents=True)
        (check / "raw" / "rows.csv").write_text(ROWS_HEADER + rows, encoding="utf-8")
        (check / "result.json").write_text(json.dumps(night["checks"][0]), encoding="utf-8")


@pytest.fixture
def site(tmp_path: Path) -> Path:
    """Three main nights (red, then green twice, the baseline moving to 6.2.1 on the last), one
    6.2 release-gate run, an unreadable night and one that claims green but breaks the contract;
    built through the CLI. Returns the site."""
    data = tmp_path / "site" / "data"
    red = examples.nightly("red", [micro_hw(False, HOSTILE)])
    red["candidate"]["label"] = "main@<i>4c10889</i>"
    store(
        data,
        "2026-11-22-101",
        red,
        {"trigger": "nightly", "scope": "main"},
        f"{ROW.split(' [')[0]},writes,cycles,regression,6.3579,8914502;8914510,1402118;1402200,"
        "8914400;8914600,1402100;1402300,confirmed by re-run\n",
    )
    green = examples.nightly("green", [micro_hw(True)])
    store(
        data,
        "2026-11-23-102",
        green,
        {"trigger": "nightly", "scope": "main"},
        f"{ROW.split(' [')[0]},writes,cycles,ok,1.0,1402118;1402120,1402118;1402119,,,\n",
    )
    moved = copy.deepcopy(green)
    moved["baseline"]["label"] = "6.2.1"
    store(
        data,
        "2026-11-24-103",
        moved,
        {"trigger": "<i>manual</i>", "scope": "main"},
        f"{ROW.split(' [')[0]},writes,cycles,ok,1.0,1402000;1402010,1401990;1402001,,,\n",
    )
    store(
        data,
        "2026-11-24-104",
        green,
        {"trigger": "release gate", "scope": "6.2"},
        f"{ROW.split(' [')[0]},writes,cycles,ok,1.0,1500000;1500001,1500000;1500002,,,\n",
    )
    (data / "2026-11-25-105").mkdir()
    (data / "2026-11-25-105" / "nightly.json").write_text("{not json", encoding="utf-8")
    claims_green = examples.nightly("green", [micro_hw(True)])
    claims_green["checks"][0]["candidate"]["image"] = "falkordb/falkordb-server:edge"  # a tag
    store(data, "2026-11-25-106", claims_green, None)
    malformed = examples.nightly("green", [micro_hw(True)])
    bad_row = {"name": ["x"], "metric": 1, "status": "regression"}
    malformed["checks"][0].update(verdict="regression", findings=["not a finding", bad_row])
    malformed["checks"][0]["artifacts"] = ["raw/\0rows.csv"]
    malformed["checks"].append({"check": ["not", "a", "name"], "verdict": "error", "error": "x"})
    store(data, "2026-11-25-107", malformed, None)
    (data / "not-a-run").mkdir()
    (tmp_path / "site" / "run").mkdir(parents=True)
    (tmp_path / "site" / "run" / "2026-11-21-100.html").write_text("a run no longer stored")
    out = tmp_path / "site"
    result = CliRunner().invoke(
        cli, ["nightly", "dashboard", "--data", str(data), "--out", str(out)]
    )
    assert result.exit_code == 0, result.output
    assert result.output.strip() == f"7 run(s) -> {out}"
    return out


def pages(site: Path) -> dict[str, str]:
    return {
        p.relative_to(site).as_posix(): p.read_text(encoding="utf-8") for p in site.rglob("*.html")
    }


def cells(row: str) -> list[str]:
    """A table row's cells as the reader sees them: tags dropped, entities decoded."""
    return [
        html.unescape(re.sub(r"<[^>]+>", "", c)) for c in re.findall(r"<td[^>]*>(.*?)</td>", row)
    ]


def test_the_pages_exist(site):
    assert sorted(pages(site)) == [
        "index.html",
        "run/2026-11-22-101.html",
        "run/2026-11-23-102.html",
        "run/2026-11-24-103.html",
        "run/2026-11-24-104.html",
        "run/2026-11-25-105.html",
        "run/2026-11-25-106.html",
        "run/2026-11-25-107.html",
        "trends.html",
    ]
    # Newest first: each run, its scope, trigger, status, both images and a verdict per check.
    # A night that cannot be read, or that the contract rejects, is an error, never green.
    runs = [cells(r) for r in re.findall(r"<tr data-scope=s\d>.*?</tr>", pages(site)["index.html"])]
    assert runs == [
        ["2026-11-25 · 107", "main", "unknown", "error", "main@4c1088949", "6.2.0", "regression"],
        ["2026-11-25 · 106", "main", "unknown", "error", "main@4c1088949", "6.2.0", "pass"],
        ["2026-11-25 · 105", "main", "unknown", "error", "unknown", "unknown", ""],
        ["2026-11-24 · 104", "6.2", "release gate", "green", "main@4c1088949", "6.2.0", "pass"],
        ["2026-11-24 · 103", "main", "<i>manual</i>", "green", "main@4c1088949", "6.2.1", "pass"],
        ["2026-11-23 · 102", "main", "nightly", "green", "main@4c1088949", "6.2.0", "pass"],
        [
            "2026-11-22 · 101",
            "main",
            "nightly",
            "red",
            "main@<i>4c10889</i>",
            "6.2.0",
            "regression",
        ],
    ]
    unreadable = pages(site)["run/2026-11-25-105.html"]
    assert "could not read nightly.json" in unreadable and "No checks ran." in unreadable
    assert "nightly.json is not valid (image-not-digest)" in pages(site)["run/2026-11-25-106.html"]


def test_the_index_shows_every_soak_check_in_one_column(tmp_path):
    """Soak runs as one check per scenario (soak-s01 ... soak-s12), and older nights as soak-a and
    soak-b: the runs index holds them all in one `soak` column with the worst verdict, and names
    each check's verdict in the cell's title. A run's page keeps one card per check."""

    def result(check: str, verdict: str) -> dict:
        if verdict == "error":
            return {"check": check, "verdict": "error", "error": "no VM came up"}
        doc = micro_hw(False) if verdict == "regression" else micro_hw(True)
        doc["check"] = check
        return doc

    data = tmp_path / "site" / "data"
    # A regression outranks an error, as in the night's status (red, not error).
    old = [result("micro-hw", "pass"), result("soak-a", "error"), result("soak-b", "regression")]
    new = [result("micro-hw", "pass")]
    new += [result(f"soak-s{n:02d}", "error" if n == 2 else "pass") for n in (12, 1, 2)]
    green = [result(f"soak-s{n:02d}", "pass") for n in range(1, 13)]
    for folder, status, checks in (
        ("2026-11-25-201", "red", old),
        ("2026-11-27-202", "error", new),
        ("2026-11-28-203", "green", green),
    ):
        run = {"trigger": "nightly", "scope": "main"}
        store(data, folder, examples.nightly(status, checks), run)
    out = tmp_path / "site"
    built = CliRunner().invoke(
        cli, ["nightly", "dashboard", "--data", str(data), "--out", str(out)]
    )
    assert built.exit_code == 0, built.output
    index = pages(out)["index.html"]
    head = re.search(r"<tr>(<th>.*?)</tr>", index).group(1)
    assert re.findall(r"<th>(.*?)</th>", head)[-2:] == ["micro-hw", "soak"]
    soak_cells = [
        re.findall(r"<td[^>]*>[^<]*</td>", row)[-1]
        for row in re.findall(r"<tr data-scope=s\d>.*?</tr>", index)
    ]
    every = ", ".join(f"soak-s{n:02d} pass" for n in range(1, 13))
    assert soak_cells == [  # newest first
        f"<td class=pass title='{every}'>pass</td>",
        "<td class=error title='soak-s01 pass, soak-s02 error, soak-s12 pass'>error</td>",
        "<td class=regression title='soak-a error, soak-b regression'>regression</td>",
    ]
    run = pages(out)["run/2026-11-27-202.html"]
    assert all(f"<h2>{name} " in run for name in ("soak-s01", "soak-s02", "soak-s12"))


def test_scopes_filter_the_same_pages_without_javascript(site):
    for name in ("index.html", "trends.html"):
        page = pages(site)[name]
        assert (
            "<label for=scope-0>main</label>" in page and "<label for=scope-1>6.2</label>" in page
        )
        assert "#scope-1:checked~main [data-scope]:not([data-scope=s1]){display:none}" in page
        assert "data-scope=s0" in page and "data-scope=s1" in page
    assert "<script" not in pages(site)["index.html"]


def test_a_failing_check_shows_its_row_rerun_rules_and_links(site):
    page = pages(site)["run/2026-11-22-101.html"]
    row = re.search(rf"<tr><td>{re.escape(ROW)}</td>.*?</tr>", page)
    assert row, page
    assert cells(row.group()) == [
        ROW,
        "cycles",
        "8,914,502",
        "1,402,118",
        "6.36x",
        "1.10x",
        "8,914,400 · 8,914,600 vs 1,402,100 · 1,402,300",
    ]
    assert html.escape(dashboard.RULES["micro-hw"]) in page
    run_url = examples.nightly("red", [])["run_url"]
    for href in (
        run_url,
        "../data/2026-11-22-101/nightly.json",
        "../data/2026-11-22-101/micro-hw/result.json",
        "../data/2026-11-22-101/micro-hw/raw/rows.csv",
    ):
        assert f"href='{href}'" in page, href
    assert "Commits since the last green night" in page
    # The linked files are where the links point.
    for href in re.findall(r"href='\.\./(data/[^']+)'", page):
        assert (site / href).is_file(), href


def test_a_trend_has_one_point_per_night_of_its_scope(site):
    trends = pages(site)["trends.html"]
    rows = re.findall(rf"<tr data-row='{re.escape(ROW.lower())}'>.*?</tr>", trends)
    assert len(rows) == 2  # main and 6.2
    main, gate = rows
    for line, nights in ((main, 3), (gate, 1)):
        for side in ("c", "b"):
            points = re.search(rf"<polyline class={side} points='([^']*)'", line).group(1)
            assert len(points.split()) == nights, line
    # The red night's dot, and the line where main's baseline moved to 6.2.1.
    assert main.count("<circle class=regression") == 1 and main.count("<line class=m") == 1
    assert "<details class=fold>" in trends and "Moved (crossed a gate" in trends


def test_text_from_the_result_files_is_escaped(site):
    for name, page in pages(site).items():
        for raw in ("<script>alert", "<b>co</b>", "<i>4c10889</i>", "<i>manual</i>"):
            assert raw not in page, (name, raw)
    red = pages(site)["run/2026-11-22-101.html"]
    assert "&lt;script&gt;alert(&#x27;x&#x27;)&lt;/script&gt; &amp; &lt;b&gt;co&lt;/b&gt;" in red
    assert "main@&lt;i&gt;4c10889&lt;/i&gt;" in red
    assert "&lt;i&gt;manual&lt;/i&gt;" in pages(site)["index.html"]


def test_read_rows_takes_every_checks_layout(tmp_path):
    """Each check's layout, as its writer orders it, keyed so one row has one name every night."""
    soak = "name,metric,candidate,baseline,ratio,threshold,direction,status,crossed_first_pass,kind,note\n"
    cases = {
        "soak": (  # an info row renames itself: "<name> (<note>)", "(p99 shift, not gated)"
            soak + "s1 op (not repeated on re-run),p50_ms,4,2,2,2,higher_is_worse,info,1,perf,"
            'not repeated on re-run\n"s1 op (p99 shift, not gated)",p99_ms,40,9,4.4,,higher_is_worse,'
            "info,0,tail,\n",
            {
                ("s1 op", "p50_ms"): (4.0, 2.0, "info", ""),
                ("s1 op", "p99_ms"): (40.0, 9.0, "info", ""),
            },
        ),
        "pokec": (
            "dataset,workload,name,metric,candidate,baseline,status,re_run\n"
            "medium,read-only,q [read-only],p99_ms,30,10,regression,1\n",
            {("q [read-only]", "p99_ms"): (30.0, 10.0, "regression", "crossed again")},
        ),
        "ldbc": (  # the re-run rows come first; the re-run decides the status
            "query,phase,metric,candidate,baseline,ratio,status\n"
            "IC1,rerun,p50_ms,8.5,3.1,2.7,regression\nIC1,main,p50_ms,9,3,3,crossed: re-run\n",
            {("IC1 p50", "p50_ms"): (9.0, 3.0, "regression", "8.5 vs 3.1")},
        ),
        "fuzz": (  # no metric column, so none on any night
            "phase,side,run,extra,name,value\ncontrol,candidate,1,0,no_freeze,0\n"
            "finding (regression),candidate,,0,control no_freeze,0\n"
            "finding (regression),baseline,,0,control no_freeze,1\n",
            {("control no_freeze", ""): (0.0, 1.0, "regression", "")},
        ),
    }
    for check, (body, want) in cases.items():
        path = tmp_path / f"{check}.csv"
        path.write_text(body, encoding="utf-8")
        rows = dashboard.read_rows(path)
        got = {key: (r.candidate, r.baseline, r.status, r.rerun) for key, r in rows.items()}
        assert got == want, check


# -- what counts as a regression: the text against each check's own numbers -------------------


def _numbers(text: str) -> set[str]:
    found = re.findall(r"(?<![\w.#])#?\d[\d,]*(?:\.\d+)?", text)
    return {n.rstrip(",").replace(",", "") for n in found}


def _micro_hw() -> list[str]:
    names = {
        "instr": "instructions",
        "cycles": "cycles",
        "br_miss": "branch misses",
        "l2_miss": "L2 misses",
        "alloc_bytes": "allocated bytes",
    }
    phrases = [f"runs {verdict.REPS} times a side"]
    for engine, head in (("rust", "Rust: "), ("c", "C (either side): ")):
        gates = []
        for column, (floor, steep, steep_floor) in verdict.HW_GATES[engine].items():
            gate = f"{names[column]} +{round((metrics.THRESHOLDS[column] - 1) * 100)}% and {floor:,.0f}"
            gates.append(gate + (f", or {steep:.1f}x and {steep_floor:,.0f}" if steep else ""))
        phrases.append(head + "; ".join(gates) + ".")
    return phrases


def _pokec() -> list[str]:
    mode, floor, ratio = pokec.MODES["nightly"], pokec.LATENCY_FLOOR_MS, pokec.RATIO_THRESHOLD
    (writes,) = [w.write_ratio for w in pokec.WORKLOADS if w.write_ratio]
    return [
        f"read-only and {writes:.0%} writes, {mode.reps} runs a side",
        f"past {ratio:g}x and past a floor (p50 {floor['p50_ms']:g} ms, p95 {floor['p95_ms']:g} ms, "
        f"p99 {floor['p99_ms']:g} ms)",
        f"re-run of {mode.rerun_reps} runs a side",
        f"drops past {ratio:g}x in both passes",
        f"took {pokec.DEADLINE_FACTOR:g}x the other side's slowest run",
        f"times out in {pokec.REPEAT_RUNS} runs",
        f"PING over {pokec.PING_LIMIT_MS:,.0f} ms in {pokec.REPEAT_RUNS} runs that is past {ratio:g}x",
    ]


def _ldbc() -> list[str]:
    return [
        f"past {ldbc.RATIO:g}x and by more than a floor (p50 {ldbc.FLOOR_MS['p50_ms']:g} ms, "
        f"p95 {ldbc.FLOOR_MS['p95_ms']:g} ms)"
    ]


def _soak() -> list[str]:
    return [
        f"p50 regresses past {soak.P50_RATIO:g}x and by more than {soak.P50_FLOOR_MS:g} ms",
        f"throughput below {soak.TPUT_RATIO:g}x",
        f"setup wall time past {soak.SETUP_RATIO:g}x and by more than {soak.SETUP_FLOOR_S:g} s",
        f"baseline has {soak.MIN_SAMPLES} samples; its latency once the candidate has "
        f"{soak.MIN_CAND_SAMPLES} too",
        f"(past {soak.P99_INFO_RATIO:g}x and {soak.P99_INFO_FLOOR_MS:g} ms, with "
        f"{soak.MIN_P99_SAMPLES:,} baseline samples)",
        *sorted(set(soak.KNOWN_FLAKY.values())),
        f"unless the candidate broke it every time it was checked (at least "
        f"{soak.ALWAYS_BROKEN_MIN_CHECKS} times) in every run",
    ]


def _fuzz() -> list[str]:
    (phase,) = {floor for name, _, floor in fuzz_verdict.BUILD_METRICS if "phase" in name}
    (query,) = {floor for name, _, floor in fuzz_verdict.BUILD_METRICS if "query" in name}
    plain = [
        c.replace("_", " ").replace("dont", "don't")
        for c in (*fuzz_checks.TIMING_CHECKS, *fuzz_checks.DATA_CHECKS)
    ]
    return [
        f"({len(fuzz_check.FIRST_ORDER)} runs: {', '.join(fuzz_check.FIRST_ORDER)})",
        f"no PING later than {fuzz_checks.FREEZE_PING_MS} ms while a plain server answered "
        f"within {fuzz_checks.CONTROL_OK_MS} ms",
        f"at scale {fuzz_check.BUILD_SCALE:g}",
        f"phase regresses past {fuzz_verdict.BUILD_THRESHOLD:g}x and by more than {phase:g} ms",
        f"query past {fuzz_verdict.BUILD_THRESHOLD:g}x and by more than {query:g} ms",
        *plain,
    ]


PHRASES = {
    "micro-hw": _micro_hw,
    "pokec": _pokec,
    "ldbc": _ldbc,
    **dict.fromkeys((f"soak-s{n:02d}" for n in range(1, 13)), _soak),
    "fuzz": _fuzz,
}


def test_every_nightly_check_has_its_rules():
    registry = yaml.safe_load((BENCH / "checks.yaml").read_text(encoding="utf-8")) or []
    assert set(dashboard.RULES) == set(PHRASES) == {"micro-hw", *(e["name"] for e in registry)}


@pytest.mark.parametrize("check", sorted(PHRASES))
def test_the_rules_say_the_checks_own_numbers(check):
    """Each phrase is built from the check's constants; and every number in the text is one of
    them, so a number left behind after the code changed fails here."""
    text, phrases = dashboard.RULES[check], PHRASES[check]()
    for phrase in phrases:
        assert phrase in text, phrase
    assert _numbers(text) <= set().union(*map(_numbers, phrases)), sorted(
        _numbers(text) - set().union(*map(_numbers, phrases))
    )


def test_the_rules_test_catches_drift(monkeypatch):
    # Break it on purpose, both ways. The code's floor moves and the text does not:
    gates = copy.deepcopy(verdict.HW_GATES)
    gates["rust"]["instr"] = (300_000.0, None, 0.0)
    with monkeypatch.context() as patch:
        patch.setattr(verdict, "HW_GATES", gates)
        with pytest.raises(AssertionError, match=r"instructions \+10% and 300,000"):
            test_the_rules_say_the_checks_own_numbers("micro-hw")
    # The text gains a number that no constant says:
    monkeypatch.setitem(dashboard.RULES, "micro-hw", dashboard.RULES["micro-hw"] + " Also 7 runs.")
    with pytest.raises(AssertionError, match="'7'"):
        test_the_rules_say_the_checks_own_numbers("micro-hw")


def test_the_example_nightlies_are_valid():
    # The site's runs are only as honest as their nightlies: these must pass the contract.
    assert contract.validate_nightly(examples.nightly("red", [micro_hw(False, HOSTILE)])) == []
    assert contract.validate_nightly(examples.nightly("green", [micro_hw(True)])) == []


def test_a_rejected_night_with_numbers_for_lists_still_renders(tmp_path):
    data = tmp_path / "data"
    night = examples.nightly("green", [micro_hw(True)])
    night["checks"][0].update(verdict="regression", findings=1, artifacts=1)
    store(data, "2026-11-25-108", night, None)
    out = tmp_path / "site"
    result = CliRunner().invoke(
        cli, ["nightly", "dashboard", "--data", str(data), "--out", str(out)]
    )
    assert result.exit_code == 0, result.output
    assert (out / "run" / "2026-11-25-108.html").is_file()


def test_a_missing_data_folder_fails_and_keeps_the_site(tmp_path):
    out = tmp_path / "site"
    (out / "run").mkdir(parents=True)
    page = out / "run" / "2026-11-21-100.html"
    page.write_text("a good page")
    result = CliRunner().invoke(
        cli, ["nightly", "dashboard", "--data", str(tmp_path / "typo"), "--out", str(out)]
    )
    assert result.exit_code != 0
    assert page.read_text() == "a good page"


# --- the workflow that publishes it --------------------------------------------------

WORKFLOWS = BENCH.parent / ".github" / "workflows"
LEVELS = {"none": 0, "read": 1, "write": 2}


def load_workflow(name):
    doc = yaml.safe_load((WORKFLOWS / name).read_text(encoding="utf-8"))
    if True in doc:  # PyYAML reads the key `on` as True
        doc["on"] = doc.pop(True)
    return doc


def test_perf_nightly_asks_for_no_write_beyond_what_its_callers_grant():
    """perf-nightly.yml is called (the release gate) with the caller's grant as a ceiling,
    and GitHub checks every job's permissions against it before evaluating `if`: one job
    asking for more makes every call invalid. So what writes lives in perf-dashboard.yml."""
    callers = [
        job["permissions"]
        for file in sorted(WORKFLOWS.glob("*.yml"))
        for job in load_workflow(file.name)["jobs"].values()
        if job.get("uses", "").endswith("/perf-nightly.yml")
    ]
    assert callers
    ceiling = {
        scope: min(LEVELS[grant.get(scope, "none")] for grant in callers)
        for grant in callers
        for scope in grant
    }
    for name, job in load_workflow("perf-nightly.yml")["jobs"].items():
        for scope, level in job.get("permissions", {}).items():
            assert LEVELS[level] <= ceiling.get(scope, 0), (name, scope, level)


def test_the_dashboard_is_published_by_its_own_workflow_after_each_perf_run():
    nightly = load_workflow("perf-nightly.yml")
    assert "publish" not in nightly["jobs"]
    comment = load_workflow("perf-comment.yml")
    flow = load_workflow("perf-dashboard.yml")
    assert flow["on"] == {
        "workflow_run": {"workflows": [nightly["name"], comment["name"]], "types": ["completed"]}
    }
    jobs = flow["jobs"]
    assert jobs["decide"]["permissions"] == {"actions": "read", "contents": "read"}
    assert jobs["store"]["permissions"] == {"contents": "write", "actions": "read"}
    assert jobs["deploy"]["permissions"] == {
        "contents": "read",
        "actions": "read",
        "pages": "write",
        "id-token": "write",
    }
    # A /perf run can never displace a waiting night's store: they queue apart.
    assert jobs["store"]["concurrency"] == {
        "group": "perf-dashboard-store",
        "cancel-in-progress": False,
    }
    assert jobs["deploy"]["concurrency"] == {
        "group": "perf-dashboard-deploy",
        "cancel-in-progress": False,
    }
    assert jobs["deploy"]["environment"]["name"] == "github-pages"
    assert jobs["deploy"]["needs"] == ["decide", "store"]
    for job in jobs.values():
        for step in job["steps"]:
            if "uses" in step:  # pinned by SHA, like the repo's other workflows
                assert re.fullmatch(r"[\w./-]+@[0-9a-f]{40}", step["uses"]), step["uses"]
    uses = [step["uses"].split("@")[0] for step in jobs["deploy"]["steps"] if "uses" in step]
    assert "actions/upload-pages-artifact" in uses and "actions/deploy-pages" in uses
    build = publish_step("Build the site")
    # Every deploy rebuilds the unlisted pages: a deploy replaces the whole site.
    assert "bench nightly dashboard" in build and "bench nightly unlisted --out" in build
    assert "--days 30" in build


def test_the_run_is_stored_only_for_runs_that_report_to_chat():
    """A scheduled night, or a dispatch that named no image: the nightly records which. A
    /perf run is never stored: only a comment titled as a request goes on at all."""
    job = load_workflow("perf-dashboard.yml")["jobs"]["decide"]
    assert "'schedule'" in job["if"] and "'workflow_dispatch'" in job["if"]
    assert "startsWith(github.event.workflow_run.display_title, '/perf on #')" in job["if"]
    # By workflow file: a run's `name` is its run-name ("/perf on #N"), not the workflow's.
    assert "workflow_run.name" not in job["if"]
    for path in (".github/workflows/perf-nightly.yml", ".github/workflows/perf-comment.yml"):
        assert f"github.event.workflow_run.path == '{path}'" in job["if"]
        assert (BENCH.parent / path).is_file()
    text = "\n".join(step.get("run", "") for step in job["steps"])
    assert "dispatch.json" in text and "image_overrides" in text
    record = next(
        step["run"]
        for step in load_workflow("perf-nightly.yml")["jobs"]["resolve"]["steps"]
        if step.get("name") == "Record the image overrides"
    )
    assert "dispatch.json" in record and "image_overrides" in record
    run_name = load_workflow("perf-comment.yml")["run-name"]
    assert "format('/perf on #{0}', github.event.issue.number)" in run_name


# --- the publish steps, run for real against a fake gh --------------------------------

FAKE_GH = r"""#!/usr/bin/env bash
# `gh run download` lays out the plan and nightly artifacts from $FAKE_RUN; `gh api` lists
# $FAKE_ARTIFACTS by name (through --jq when given) and serves $FAKE_ZIPS/<id>.zip, else a zip
# whose result.json names the artifact's id.
set -euo pipefail
if [[ $1 == run ]]; then
  dir=
  while [[ $# -gt 0 ]]; do [[ $1 == --dir ]] && dir=$2; shift; done
  cp -R "$FAKE_RUN"/. "$dir"/
  exit 0
fi
url=$2
if [[ $url == */zip ]]; then
  id=$(basename "$(dirname "$url")")
  if [[ -f ${FAKE_ZIPS:-/nonexistent}/$id.zip ]]; then cat "$FAKE_ZIPS/$id.zip"; exit 0; fi
  work=$(mktemp -d); echo "$id" > "$work/result.json"
  (cd "$work" && zip -q -r - result.json)
  exit 0
fi
jqexpr=.
while [[ $# -gt 0 ]]; do [[ $1 == --jq ]] && jqexpr=$2; shift; done
if [[ $url == */compare/* ]]; then
  echo "{\"status\": \"${FAKE_COMPARE:-identical}\"}" | jq -r "$jqexpr"
  exit 0
fi
name=${url#*name=}; name=${name%%&*}
jq --arg n "$name" ".artifacts |= map(select(.name == \$n))" "$FAKE_ARTIFACTS" | jq -r "$jqexpr"
"""

# `uv run --project bench --frozen bench <args>`: bench's own CLI in this interpreter, so the
# workflow step runs the real `bench nightly store`; the renderer has its own tests.
FAKE_UV = f"""#!/bin/sh
shift 4
[ "$1" = bench ] && shift
[ "$1 $2" = "nightly dashboard" ] && exit 0
exec {sys.executable} -c 'import sys; from falkorbench.cli import main; sys.exit(main())' "$@"
"""


def fake_artifacts(tmp_path, entries):
    path = tmp_path / "artifacts.json"
    path.write_text(json.dumps({"artifacts": entries}))
    return path


def publish_step(name):
    jobs = load_workflow("perf-dashboard.yml")["jobs"].values()
    return next(step["run"] for job in jobs for step in job["steps"] if step.get("name") == name)


def run_download(
    tmp_path,
    event,
    dispatch,
    workflow=".github/workflows/perf-nightly.yml",
    artifacts=(),
    branch="main",
    compare="identical",
):
    """Run the decide step; return (stored?, published?)."""
    bin_dir = tmp_path / "bin"
    bin_dir.mkdir()
    (bin_dir / "gh").write_text(FAKE_GH)
    (bin_dir / "gh").chmod(0o755)
    fake_run = tmp_path / "fake_run"
    (fake_run / "perf-plan").mkdir(parents=True)
    (fake_run / "perf-nightly").mkdir()
    (fake_run / "perf-nightly" / "nightly.json").write_text(
        json.dumps({"date": "2026-11-20", "checks": [{"check": "alpha"}, {"check": "beta"}]})
    )
    if dispatch is not None:
        (fake_run / "perf-plan" / "dispatch.json").write_text(dispatch)
    temp, out = tmp_path / "temp", tmp_path / "out"
    temp.mkdir()
    env = {
        "PATH": f"{bin_dir}:/usr/bin:/bin:/usr/local/bin:/opt/homebrew/bin",
        "RUN_ID": "7",
        "RUN_EVENT": event,
        "GITHUB_REPOSITORY": "o/r",
        "GITHUB_OUTPUT": str(out),
        "RUNNER_TEMP": str(temp),
        "FAKE_RUN": str(fake_run),
        "WORKFLOW": workflow,
        "RUN_BRANCH": branch,
        "RUN_SHA": "a" * 40,
        "FAKE_COMPARE": compare,
        "DEFAULT_BRANCH": "main",
        "FAKE_ARTIFACTS": str(fake_artifacts(tmp_path, list(artifacts))),
    }
    subprocess.run(
        ["bash", "-c", publish_step("Download the night's artifacts")],
        env=env,
        check=True,
        capture_output=True,
    )
    said = out.read_text()
    return "store=true" in said, "publish=true" in said


@pytest.mark.parametrize(
    ("event", "dispatch", "stored", "published"),
    [
        ("schedule", None, True, True),
        ("workflow_dispatch", '{"image_overrides": false}', True, True),
        # Named its images: never stored, an unlisted page.
        ("workflow_dispatch", '{"image_overrides": true}', False, True),
        ("workflow_dispatch", None, False, False),
        ("workflow_dispatch", "not json", False, False),
        ("workflow_dispatch", "{}", False, False),
    ],
)
def test_a_dispatch_is_stored_only_when_it_records_no_image_overrides(
    tmp_path, event, dispatch, stored, published
):
    assert run_download(tmp_path, event, dispatch) == (stored, published)


@pytest.mark.parametrize(
    ("artifacts", "published"),
    [([{"id": 5, "name": "perf-nightly", "expired": False}], True), ([], False)],
)
def test_a_perf_run_is_published_unlisted_and_never_stored(tmp_path, artifacts, published):
    """A /perf request that ran left a perf-nightly artifact; a refused one did not."""
    got = run_download(
        tmp_path, "issue_comment", None, ".github/workflows/perf-comment.yml", artifacts
    )
    assert got == (False, published)


def run_store(tmp_path, stored_run, artifacts=()):
    """Run the store step against a local gh-pages remote whose stored run 7 holds
    `stored_run`; the night has checks alpha and beta, and the run's artifacts are
    `artifacts`. Returns (the step, the gh-pages checkout after it)."""
    git = ["git", "-c", "user.name=t", "-c", "user.email=t@t", "-c", "init.defaultBranch=x"]
    seed = tmp_path / "seed"
    (seed / "data" / "2026-11-20-7").mkdir(parents=True)
    for rel in stored_run:
        (seed / "data" / "2026-11-20-7" / rel).parent.mkdir(parents=True, exist_ok=True)
        (seed / "data" / "2026-11-20-7" / rel).write_text("{}")
    subprocess.run([*git, "init", "-q", "-b", "gh-pages", str(seed)], check=True)
    subprocess.run([*git, "-C", str(seed), "add", "-A"], check=True)
    subprocess.run([*git, "-C", str(seed), "commit", "-q", "-m", "seed"], check=True)
    subprocess.run(
        [*git, "-C", str(seed), "config", "receive.denyCurrentBranch", "ignore"], check=True
    )
    bin_dir = tmp_path / "bin"
    bin_dir.mkdir()
    (bin_dir / "base64").write_text("#!/bin/sh\nexec /usr/bin/base64\n")  # no -w0 on macOS
    (bin_dir / "uv").write_text(FAKE_UV)
    (bin_dir / "gh").write_text(FAKE_GH)
    for shim in ("base64", "uv", "gh"):
        (bin_dir / shim).chmod(0o755)
    temp = tmp_path / "temp"
    temp.mkdir()
    # The store fetches the night's nightly.json itself, from its perf-nightly artifact.
    zips = tmp_path / "zips"
    zips.mkdir()
    night = {"date": "2026-11-21", "checks": [{"check": "alpha"}, {"check": "beta"}]}
    (zips / "5.zip").write_bytes(zipped({"nightly.json": json.dumps(night)}))
    artifacts = [{"id": 5, "name": "perf-nightly", "expired": False}, *artifacts]
    env = {
        "PATH": f"{bin_dir}:/usr/bin:/bin:/usr/local/bin:/opt/homebrew/bin",
        "HOME": str(tmp_path),
        "RUN_ID": "7",
        "TRIGGER": "nightly",
        "GH_TOKEN": "x",
        "GITHUB_SERVER_URL": f"file://{tmp_path}",
        "GITHUB_REPOSITORY": "seed",
        "GITHUB_WORKSPACE": str(tmp_path),
        "RUNNER_TEMP": str(temp),
        "FAKE_ARTIFACTS": str(fake_artifacts(tmp_path, list(artifacts))),
        "FAKE_ZIPS": str(zips),
        "PYTHONPATH": str(BENCH / "src"),
    }
    done = subprocess.run(
        ["bash", "-c", publish_step("Store the run on gh-pages and rebuild the dashboard")],
        env=env,
        capture_output=True,
        text=True,
    )
    after = tmp_path / "after"
    subprocess.run([*git, "clone", "-q", "-b", "gh-pages", str(seed), str(after)], check=True)
    return done, after


ALPHA = [
    {"id": 11, "name": "perf-result-alpha", "expired": False},
    {"id": 42, "name": "perf-result-alpha", "expired": False},
    {"id": 99, "name": "perf-result-alpha", "expired": True},
]


def test_the_store_step_stores_the_newest_attempts_artifact_with_bench_nightly_store(tmp_path):
    done, after = run_store(tmp_path, ["nightly.json", "run.json"], ALPHA)
    assert done.returncode == 0, done.stderr
    # A re-run of the night replaces what its first attempt stored, under the night's new date.
    assert not (after / "data" / "2026-11-20-7").exists()
    stored = after / "data" / "2026-11-21-7"
    assert (stored / "alpha" / "result.json").read_text().strip() == "42"
    assert not (stored / "beta").exists()  # beta left no artifact: warned, not stored
    assert json.loads((stored / "run.json").read_text()) == {"trigger": "nightly", "scope": "main"}
    assert (
        "perf run 2026-11-21-7 (nightly)"
        in subprocess.run(
            ["git", "-C", str(after), "log", "-1", "--format=%s"], capture_output=True, text=True
        ).stdout
    )


def test_a_retry_goes_through_when_the_stored_run_never_had_the_missing_check(tmp_path):
    done, after = run_store(tmp_path, ["nightly.json", "run.json"])
    assert done.returncode == 0, done.stderr
    assert (after / "data" / "2026-11-21-7" / "nightly.json").is_file()


def test_a_missing_artifact_never_deletes_a_check_the_stored_run_has(tmp_path):
    done, after = run_store(tmp_path, ["nightly.json", "alpha/result.json"])
    assert done.returncode != 0 and "not replacing" in done.stdout + done.stderr
    # Nothing was pushed: the stored run is as it was.
    assert (after / "data" / "2026-11-20-7" / "alpha" / "result.json").read_text() == "{}"
    assert not (after / "data" / "2026-11-21-7").exists()


# --- bench nightly store: one run's files, laid out as the dashboard reads them ----------


def check_artifact(root: Path, result: dict, extra: dict[str, str]) -> Path:
    """An extracted perf-result-<check>: result.json and other files by relative path."""
    root.mkdir(parents=True)
    (root / "result.json").write_text(json.dumps(result), encoding="utf-8")
    for rel, text in extra.items():
        (root / rel).parent.mkdir(parents=True, exist_ok=True)
        (root / rel).write_text(text, encoding="utf-8")
    return root


def downloads(tmp_path: Path, artifacts: dict[str, Path]):
    """`download(name, dest)` serving these extracted artifacts by name, recording each ask."""
    asked = []

    def download(name, dest):
        asked.append(name)
        if name not in artifacts:
            return False
        shutil.copytree(artifacts[name], dest)
        return True

    return download, asked


def test_store_lays_out_exactly_what_perf_dashboard_stored(tmp_path):
    result = micro_hw(False)
    result["artifacts"] = ["raw/rows.csv", "../escape.txt", "/etc/passwd", "raw/missing.csv"]
    art = check_artifact(
        tmp_path / "art" / "micro-hw",
        result,
        {"perf-vm.json": "{}", "raw/rows.csv": ROWS_HEADER, "raw/unlisted.log": "x"},
    )
    (tmp_path / "art" / "escape.txt").write_text("outside")
    night = examples.nightly("red", [result, {"check": "pokec", "verdict": "error", "error": "x"}])
    night_path = tmp_path / "nightly.json"
    night_path.write_text(json.dumps(night, indent=1), encoding="utf-8")
    download, asked = downloads(tmp_path, {"perf-result-micro-hw": art})
    data = tmp_path / "data"
    folder = nightly_store.store(night_path, "555", data, "pr-comment", "pr-3967", download)
    assert folder == f"{night['date']}-555"
    assert asked == ["perf-result-micro-hw", "perf-result-pokec"]
    dest = data / folder
    files = sorted(p.relative_to(dest).as_posix() for p in dest.rglob("*") if p.is_file())
    # The listed artifacts that exist, the records, nothing unlisted, nothing outside.
    assert files == [
        "micro-hw/perf-vm.json",
        "micro-hw/raw/rows.csv",
        "micro-hw/result.json",
        "nightly.json",
        "run.json",
    ]
    assert (dest / "nightly.json").read_bytes() == night_path.read_bytes()
    # Byte for byte what `jq -n '{trigger: $t, scope: $s}'` wrote before this command.
    assert (dest / "run.json").read_text() == (
        '{\n  "trigger": "pr-comment",\n  "scope": "pr-3967"\n}\n'
    )


def test_a_stored_run_renders_with_each_checks_files(tmp_path):
    result = micro_hw(False)
    art = check_artifact(
        tmp_path / "art", result, {"perf-vm.json": "{}", "raw/rows.csv": ROWS_HEADER}
    )
    night_path = tmp_path / "nightly.json"
    night_path.write_text(json.dumps(examples.nightly("red", [result])), encoding="utf-8")
    site = tmp_path / "site"
    download, _ = downloads(tmp_path, {"perf-result-micro-hw": art})
    folder = nightly_store.store(night_path, "9", site / "data", "pr-comment", "pr-1", download)
    assert dashboard.build(site / "data", site) == 1
    page = (site / "run" / f"{folder}.html").read_text()
    assert "pr-1 · pr-comment" in page
    for name in ("result.json", "perf-vm.json", "raw/rows.csv"):
        href = f"../data/{folder}/micro-hw/{name}"
        assert f"href='{href}'" in page or f'href="{href}"' in page, name
        assert (site / "run" / href).resolve().is_file()


def test_a_store_that_lost_a_stored_check_changes_nothing(tmp_path):
    data = tmp_path / "data"
    old = data / "2026-11-20-7"
    (old / "micro-hw").mkdir(parents=True)
    (old / "micro-hw" / "result.json").write_text("old")
    night_path = tmp_path / "nightly.json"
    night_path.write_text(json.dumps(examples.nightly("green", [micro_hw(True)])))
    download, _ = downloads(tmp_path, {})
    with pytest.raises(
        nightly_store.StoreError, match="not replacing 2026-11-20-7 without perf-result-micro-hw"
    ):
        nightly_store.store(night_path, "7", data, "nightly", "main", download)
    assert (old / "micro-hw" / "result.json").read_text() == "old"
    assert [p.name for p in data.iterdir()] == ["2026-11-20-7"]


@pytest.mark.parametrize(
    ("night", "why"),
    [
        ({"date": "2026-11-2", "checks": []}, "is not YYYY-MM-DD"),
        ({"checks": []}, "is not YYYY-MM-DD"),
        ("not json", "cannot read"),
    ],
)
def test_a_night_with_no_usable_date_is_not_stored(tmp_path, night, why):
    path = tmp_path / "nightly.json"
    path.write_text(night if isinstance(night, str) else json.dumps(night))
    with pytest.raises(nightly_store.StoreError, match=why):
        nightly_store.store(path, "1", tmp_path / "data", "nightly", "main", lambda n, d: False)
    with pytest.raises(nightly_store.StoreError, match="is not a number"):
        nightly_store.store(path, "1; rm", tmp_path / "data", "nightly", "main", lambda n, d: False)


def fake_run(listing, archive: bytes | None, rc_list=0, rc_zip=0):
    calls = []

    def run(argv, **kwargs):
        calls.append(argv[2])
        if argv[2].endswith("/zip"):
            if archive is not None:
                kwargs["stdout"].write(archive)
            return subprocess.CompletedProcess(argv, rc_zip, None, b"")
        return subprocess.CompletedProcess(argv, rc_list, json.dumps(listing), "")

    return run, calls


def zipped(members: dict[str, str]) -> bytes:
    buffer = io.BytesIO()
    with zipfile.ZipFile(buffer, "w") as archive:
        for name, text in members.items():
            archive.writestr(name, text)
    return buffer.getvalue()


def test_fetch_takes_the_newest_live_artifact(tmp_path):
    run, calls = fake_run({"artifacts": ALPHA}, zipped({"result.json": "{}", "raw/a": "z"}))
    dest = tmp_path / "x" / "perf-result-alpha"
    assert nightly_store.fetch("o/r", "7", "perf-result-alpha", dest, run) == "ok"
    assert calls == [
        "repos/o/r/actions/runs/7/artifacts?name=perf-result-alpha&per_page=100",
        "repos/o/r/actions/artifacts/42/zip",
    ]
    assert sorted(p.relative_to(dest).as_posix() for p in dest.rglob("*") if p.is_file()) == [
        "raw/a",
        "result.json",
    ]


def corrupt(members: dict[str, str]) -> bytes:
    """A deflated zip whose directory is intact but whose first member's data is not:
    extracting it raises zlib.error, not BadZipFile."""
    buffer = io.BytesIO()
    with zipfile.ZipFile(buffer, "w", zipfile.ZIP_DEFLATED) as archive:
        for name, text in members.items():
            archive.writestr(name, text * 5000)
    data = bytearray(buffer.getvalue())
    first = zipfile.ZipFile(io.BytesIO(bytes(data))).infolist()[0]
    start = first.header_offset + 30 + len(first.filename)
    for i in range(start, start + 8):
        data[i] ^= 0xFF
    return bytes(data)


@pytest.mark.parametrize(
    ("listing", "archive", "rc_list", "rc_zip", "said"),
    [
        ({"artifacts": []}, None, 0, 0, "missing"),
        ({"artifacts": [{"id": 5, "expired": True}]}, None, 0, 0, "missing"),
        ({}, None, 1, 0, "failed"),
        ({"artifacts": [{"id": 5}]}, b"half a zip", 0, 0, "failed"),
        ({"artifacts": [{"id": 5}]}, zipped({"result.json": "{}"}), 0, 1, "failed"),
        # A member that would land outside fails the whole artifact, as unzip did.
        ({"artifacts": [{"id": 5}]}, zipped({"result.json": "{}", "../evil": "x"}), 0, 0, "failed"),
        ({"artifacts": [{"id": 5}]}, zipped({"result.json": "{}", "/abs": "x"}), 0, 0, "failed"),
        ({"artifacts": [{"id": 5}]}, corrupt({"result.json": "x"}), 0, 0, "failed"),
    ],
)
def test_fetch_that_fails_leaves_no_partial_result(
    tmp_path, no_wait, listing, archive, rc_list, rc_zip, said
):
    run, _ = fake_run(listing, archive, rc_list, rc_zip)
    dest = tmp_path / "x" / "perf-result-alpha"
    assert nightly_store.fetch("o/r", "7", "perf-result-alpha", dest, run) == said
    assert not dest.exists()
    assert not (tmp_path / "x" / "evil").exists()


def test_the_dashboard_and_a_perf_comment_run_share_the_store_command():
    publish = publish_step("Store the run on gh-pages and rebuild the dashboard")
    comment = next(
        step["run"]
        for step in load_workflow("perf-comment.yml")["jobs"]["answer"]["steps"]
        if step.get("name") == "Build this run's dashboard"
    )
    for text in (publish, comment):
        assert "bench nightly store" in text and "bench nightly dashboard" in text
        assert "cp -R --parents" not in text and "artifacts?name=" not in text
    assert "--trigger pr-comment" in comment and '--scope "pr-$PR"' in comment


# --- bench nightly failures: every failure of a stored run, vs the nights before it ------

SOAK_HEADER = (
    "name,metric,candidate,baseline,ratio,threshold,direction,status,crossed_first_pass,kind,note\n"
)
FUZZ_HEADER = "phase,side,run,extra,name,value\n"
LDBC_HEADER = "query,phase,metric,candidate,baseline,ratio,status\n"
KNOWN = "s10 Live (under writes) index == scan during build/after"


def stored_check(folder: Path, check: str, result: dict, rows: str | None) -> None:
    (folder / check / "raw").mkdir(parents=True, exist_ok=True)
    (folder / check / "result.json").write_text(json.dumps(result), encoding="utf-8")
    if rows is not None:
        (folder / check / "raw" / "rows.csv").write_text(rows, encoding="utf-8")


def a_night(data: Path, folder: str, trigger: str, checks: dict) -> Path:
    """A stored run whose checks are {name: (verdict, findings, rows.csv text or None)}."""
    entries = []
    for name, (judged, findings, _) in checks.items():
        result = copy.deepcopy(examples.example("result-micro-pass.json"))
        result.update(check=name, verdict=judged, findings=findings, artifacts=["raw/rows.csv"])
        if judged == "error":
            result["error"] = "no C4 VM came up"
        entries.append(result)
    night = examples.nightly("red", entries)
    night["date"] = folder[:10]
    (data / folder).mkdir(parents=True)
    (data / folder / "nightly.json").write_text(json.dumps(night), encoding="utf-8")
    (data / folder / "run.json").write_text(json.dumps({"trigger": trigger, "scope": "main"}))
    for entry, (_, _, rows) in zip(entries, checks.values(), strict=True):
        stored_check(data / folder, entry["check"], entry, rows)
    return data / folder


def finding(name, metric, status, candidate, baseline, ratio=None):
    return {
        "name": name,
        "metric": metric,
        "candidate": candidate,
        "baseline": baseline,
        "ratio": ratio,
        "threshold": 1.1 if ratio else None,
        "direction": "lower_is_worse"
        if metric in ("invariant", "completed")
        else "higher_is_worse",
        "status": status,
    }


def keys(failures):
    return [(f.check, f.kind, f.name, f.metric, f.both) for f in failures]


SOAK_ROWS = SOAK_HEADER + (
    "s03 range index finds every initial chunk,invariant,0,0,,,lower_is_worse,info,,,\n"
    "s07 counts add up,invariant,0,1,,,lower_is_worse,regression,,,\n"
    f"{KNOWN},invariant,0,0,,,lower_is_worse,known,,known,broke on candidate and baseline\n"
    "s01 holds,invariant,1,1,,,lower_is_worse,ok,,,\n"
    "s02 base only,invariant,1,0,,,lower_is_worse,info,,,\n"
    "s04 ingest_batch,p50_ms,9,3,3.0,2.0,higher_is_worse,regression,True,,\n"
)


def test_failures_list_every_check_the_candidate_broke_from_the_raw_rows(tmp_path):
    """A row absent from the findings was not unmeasured: the known flaky row and the ok-ish
    info row live in raw/rows.csv only, and both are failures of the candidate."""
    data = tmp_path / "data"
    a_night(
        data,
        "2026-11-25-9",
        "manual",
        {
            "soak-b": (
                "regression",
                [
                    finding("s07 counts add up", "invariant", "regression", 0, 1),
                    finding("s04 ingest_batch", "p50_ms", "regression", 9, 3, 3.0),
                ],
                SOAK_ROWS,
            ),
        },
    )
    run = nightly_failures.load_run(data, "2026-11-25-9")
    assert keys(run.failures) == [
        ("soak-b", "regression", "s04 ingest_batch", "p50_ms", False),
        ("soak-b", "broken", "s03 range index finds every initial chunk", "invariant", True),
        ("soak-b", "broken", "s07 counts add up", "invariant", False),
        ("soak-b", "broken", KNOWN, "invariant", True),
    ]
    known = next(f for f in run.failures if f.name == KNOWN)
    assert known.detail == "broken on both sides; known flaky (#3095)"
    assert next(f for f in run.failures if f.name.startswith("s03")).detail == (
        "broken on both sides; judged info"
    )


def test_fuzz_invariants_are_read_per_run_and_side(tmp_path):
    rows = (
        FUZZ_HEADER
        + "".join(
            f"control,{side},{run},0,{name},{value}\n"
            for side, run, name, value in [
                ("candidate", 0, "writes_add_up", 1),
                ("candidate", 1, "writes_add_up", 0),
                ("baseline", 0, "writes_add_up", 1),
                ("baseline", 1, "writes_add_up", 1),
                ("candidate", 0, "no_freeze", 0),
                ("baseline", 0, "no_freeze", 0),
                ("candidate", 0, "lookup p99_ms", 0),
                ("candidate", 0, "server_stays_up", 1),
            ]
        )
        + "finding (regression),candidate,,,build edge phase,900\nfinding (regression),baseline,,,build edge phase,300\n"
    )
    data = tmp_path / "data"
    a_night(
        data,
        "2026-11-25-9",
        "manual",
        {
            "fuzz": (
                "regression",
                [finding("build edge phase", "p50_ms", "regression", 900, 300, 3.0)],
                rows,
            ),
        },
    )
    failures = nightly_failures.load_run(data, "2026-11-25-9").failures
    assert keys(failures) == [
        (
            "fuzz",
            "regression",
            "build edge phase",
            "p50_ms",
            False,
        ),  # the finding, not again from the row
        ("fuzz", "broken", "control no_freeze", "invariant", True),
        ("fuzz", "broken", "control writes_add_up", "invariant", False),
    ]
    assert failures[2].detail == (
        "broken on the candidate only; broke in 1 of 2 candidate runs, 0 of 2 baseline runs"
    )
    assert "a timing check" in failures[1].detail


def test_micro_hw_counts_an_erroring_query_once_and_reads_regressions_past_the_cap(tmp_path):
    row = "{q},{c},{m},{s},{r},1;1,1;1,,,{n}\n"
    rows = ROWS_HEADER + "".join(
        row.format(q=q, c="reads", m=m, s=s, r=r, n=n)
        for q, m, s, r, n in [
            ("UDF call", "instructions", "info", "", "errors on candidate and baseline"),
            ("UDF call", "cycles", "info", "", "errors on candidate and baseline"),
            ("regex", "instructions", "info", "", "errors on baseline"),
            ("new query", "instructions", "info", "", "errors on candidate"),
            ("SKIP", "instructions", "regression", "1.9", "confirmed"),
            ("LIMIT", "cycles", "regression", "1.5", "confirmed"),  # past the cap: no finding
        ]
    )
    data = tmp_path / "data"
    a_night(
        data,
        "2026-11-25-9",
        "manual",
        {
            "micro-hw": (
                "regression",
                [
                    finding("SKIP [reads]", "instructions", "regression", 2, 1, 1.9),
                    finding(
                        "new query [errors on candidate only]", "completed", "regression", 0, 1
                    ),
                    finding(
                        "1 queries error on both sides (e.g. UDF call)", "completed", "info", 0, 0
                    ),
                ],
                rows,
            ),
        },
    )
    assert keys(nightly_failures.load_run(data, "2026-11-25-9").failures) == [
        ("micro-hw", "regression", "LIMIT [reads]", "cycles", False),
        ("micro-hw", "regression", "SKIP [reads]", "instructions", False),
        ("micro-hw", "broken", "UDF call [reads]", "errors", True),
        ("micro-hw", "broken", "new query [reads]", "errors", False),
    ]


def test_ldbc_correctness_lives_in_its_findings_and_a_check_that_could_not_measure_is_an_error(
    tmp_path,
):
    rows = LDBC_HEADER + "IC1,main,p50_ms,9,1,9.0,regression\nIC2,main,p50_ms,1,1,1.0,ok\n"
    data = tmp_path / "data"
    a_night(
        data,
        "2026-11-25-9",
        "manual",
        {
            "ldbc": (
                "regression",
                [
                    finding("IC1 p50", "p50_ms", "regression", 9, 1, 9.0),
                    finding("IC3 completed", "completed", "info", 0, 0),
                    finding("IC9 rows", "rows", "regression", 3, 4),
                ],
                rows,
            ),
            "micro-hw": ("error", [], None),
        },
    )
    assert keys(nightly_failures.load_run(data, "2026-11-25-9").failures) == [
        ("ldbc", "regression", "IC1 p50", "p50_ms", False),
        ("ldbc", "broken", "IC3 completed", "completed", True),
        ("ldbc", "broken", "IC9 rows", "rows", False),
        ("micro-hw", "error", "micro-hw", "", False),
    ]


def soak_night(data, folder, trigger, rows):
    return a_night(data, folder, trigger, {"soak-b": ("pass", [], SOAK_HEADER + rows)})


BROKE = "{n},invariant,0,0,,,lower_is_worse,info,,,\n"


def test_failures_are_new_still_failing_or_fixed_against_earlier_scheduled_nights(tmp_path):
    data = tmp_path / "data"
    soak_night(data, "2026-11-22-1", "nightly", BROKE.format(n="old") + BROKE.format(n="gone"))
    soak_night(data, "2026-11-23-2", "nightly", BROKE.format(n="old") + BROKE.format(n="gone"))
    soak_night(data, "2026-11-24-3", "manual", BROKE.format(n="ignored, not a night"))
    soak_night(data, "2026-11-24-4", "nightly", BROKE.format(n="old") + BROKE.format(n="gone"))
    soak_night(data, "2026-11-27-9", "nightly", BROKE.format(n="later night"))
    soak_night(data, "2026-11-25-5", "pr-comment", BROKE.format(n="old") + BROKE.format(n="new"))
    result = nightly_failures.compare(data, "2026-11-25-5")
    assert [r.folder for r in result.history] == ["2026-11-24-4", "2026-11-23-2", "2026-11-22-1"]
    assert {f.name: result.since[f.key] for f in result.run.failures} == {
        "old": "since 2026-11-22",
        "new": "new",
    }
    assert [f.name for f in result.fixed] == ["gone"]


def test_a_check_this_run_did_not_measure_is_not_fixed(tmp_path):
    data = tmp_path / "data"
    soak_night(data, "2026-11-24-4", "nightly", BROKE.format(n="gone"))
    a_night(data, "2026-11-25-5", "pr-comment", {"micro-hw": ("pass", [], None)})
    result = nightly_failures.compare(data, "2026-11-25-5")
    assert result.fixed == []
    assert "Fixed since 2026-11-24-4 (0):" in nightly_failures.text(result)


def test_with_no_earlier_night_the_report_says_so(tmp_path):
    data = tmp_path / "data"
    soak_night(data, "2026-11-25-5", "nightly", BROKE.format(n="s03 broken"))
    result = nightly_failures.compare(data, "2026-11-25-5")
    assert result.since[result.run.failures[0].key] == "no earlier night"
    assert "No earlier scheduled night is stored" in nightly_failures.text(result)


def test_the_failures_command_prints_text_or_json_and_refuses_a_missing_run(tmp_path, capsys):
    data = tmp_path / "data"
    soak_night(data, "2026-11-24-4", "nightly", BROKE.format(n="x"))
    soak_night(data, "2026-11-25-5", "nightly", BROKE.format(n="x"))
    run = CliRunner().invoke(
        cli, ["nightly", "failures", "--data", str(data), "--run", "2026-11-25-5"]
    )
    assert run.exit_code == 0, run.output
    assert (
        "- soak-b | broken | x (invariant) | candidate 0 vs baseline 0 | broken on both sides"
        in run.output
    )
    assert "since 2026-11-24" in run.output
    run = CliRunner().invoke(
        cli, ["nightly", "failures", "--data", str(data), "--run", "2026-11-25-5", "--json"]
    )
    doc = json.loads(run.output)
    assert doc["failures"][0]["since"] == "since 2026-11-24" and doc["compared_with"] == [
        "2026-11-24-4"
    ]
    assert (
        CliRunner()
        .invoke(cli, ["nightly", "failures", "--data", str(data), "--run", "2026-11-26-6"])
        .exit_code
        == 1
    )


def test_a_fuzz_break_is_one_failure_however_many_places_name_it(tmp_path):
    """The per-run rows, the finding rows and result.json all name a candidate-only break (and
    server_stays_up under the metric `completed`): one failure each."""
    rows = FUZZ_HEADER + "".join(
        f"{phase},{side},{run},0,{name},{value}\n"
        for phase, side, run, name, value in [
            ("control", "candidate", 0, "writes_add_up", 0),
            ("control", "baseline", 0, "writes_add_up", 1),
            ("control", "candidate", 0, "server_stays_up", 0),
            ("control", "baseline", 0, "server_stays_up", 1),
            ("finding (regression)", "candidate", "", "control writes_add_up", 0),
            ("finding (regression)", "baseline", "", "control writes_add_up", 1),
            ("finding (regression)", "candidate", "", "control server_stays_up", 0),
            ("finding (regression)", "baseline", "", "control server_stays_up", 1),
            # past the findings cap: only its finding rows, no per-run rows
            ("finding (info)", "candidate", "", "long index_matches_scan", 0),
            ("finding (info)", "baseline", "", "long index_matches_scan", 0),
        ]
    )
    data = tmp_path / "data"
    a_night(
        data,
        "2026-11-25-9",
        "manual",
        {
            "fuzz": (
                "regression",
                [
                    finding("control writes_add_up", "invariant", "regression", 0, 1),
                    finding("control server_stays_up", "completed", "regression", 0, 1),
                ],
                rows,
            ),
        },
    )
    failures = nightly_failures.load_run(data, "2026-11-25-9").failures
    assert keys(failures) == [
        ("fuzz", "broken", "control server_stays_up", "invariant", False),
        ("fuzz", "broken", "control writes_add_up", "invariant", False),
        ("fuzz", "broken", "long index_matches_scan", "invariant", True),
    ]
    assert failures[2].detail.endswith("from its finding (no per-run rows)")


def test_a_check_the_candidate_broke_in_some_runs_only_is_a_failure(tmp_path):
    soak = SOAK_HEADER + (
        "s05 used_memory stays flat (broke in 1 of 2 candidate runs),invariant,1,1,,,lower_is_worse,info,True,unconfirmed,broke in 1 of 2 candidate runs\n"
    )
    pokec = (
        "dataset,workload,name,metric,candidate,baseline,ratio,direction,status,use,first_pass,re_run,detail\n"
        "medium,read-only,q1 [read-only],completed,1,1,,lower_is_worse,info,,,,"
        '"candidate: 0 errors, 3 timeouts in 1 of 5 runs; baseline: 0 errors, 0 timeouts in 0 of 5 runs"\n'
        "medium,read-only,q2 [read-only],completed,1,1,,lower_is_worse,ok,,,,"
        '"candidate: 0 errors, 0 timeouts in 0 of 5 runs; baseline: 0 errors, 0 timeouts in 0 of 5 runs"\n'
    )
    data = tmp_path / "data"
    a_night(
        data, "2026-11-25-9", "manual", {"soak-a": ("pass", [], soak), "pokec": ("pass", [], pokec)}
    )
    failures = nightly_failures.load_run(data, "2026-11-25-9").failures
    assert keys(failures) == [
        (
            "soak-a",
            "broken",
            "s05 used_memory stays flat (broke in 1 of 2 candidate runs)",
            "invariant",
            False,
        ),
        ("pokec", "broken", "q1 [read-only]", "completed", False),
    ]
    assert "a timing check; broke in 1 of 2 candidate runs" in failures[0].detail
    assert "3 timeouts in 1 of 5 runs" in failures[1].detail


def test_micro_hw_queries_that_error_in_some_passes_are_a_failure(tmp_path):
    some = "2 queries error on the candidate in some passes only (e.g. regex)"
    data = tmp_path / "data"
    a_night(
        data,
        "2026-11-25-9",
        "manual",
        {
            "micro-hw": (
                "pass",
                [
                    finding(some, "completed", "info", None, 1),
                    finding(
                        "3 queries error on both sides (e.g. UDF call)", "completed", "info", 0, 0
                    ),
                ],
                ROWS_HEADER + "regex,reads,instructions,ok,1.0,1;1,1;1,,,\n",
            ),
        },
    )
    assert keys(nightly_failures.load_run(data, "2026-11-25-9").failures) == [
        ("micro-hw", "broken", some, "completed", False),
    ]


def test_ldbc_requests_that_failed_are_read_from_its_summary(tmp_path):
    summary = (
        "query,phase,side,attempted,ok,timeouts,errors,down,rows_total,p50_ms,p95_ms,p99_ms,first_failure\n"
        "IC14,main,candidate,49,48,1,0,0,9,,,,\nIC14,main,baseline,49,49,0,0,0,9,,,,\n"
        "IC1,main,candidate,49,49,0,0,0,9,,,,\nIC1,main,baseline,49,49,0,0,0,9,,,,\n"
    )
    data = tmp_path / "data"
    folder = a_night(data, "2026-11-25-9", "manual", {"ldbc": ("pass", [], LDBC_HEADER)})
    (folder / "ldbc" / "raw" / "summary.csv").write_text(summary)
    [failure] = nightly_failures.load_run(data, "2026-11-25-9").failures
    assert (failure.name, failure.both) == ("IC14 completed", False)
    assert "1 of 49 candidate requests failed, 0 of 49 baseline" in failure.detail


def test_a_check_whose_files_did_not_download_is_an_error_not_a_pass(tmp_path):
    data = tmp_path / "data"
    folder = a_night(data, "2026-11-25-9", "manual", {"soak-b": ("pass", [], SOAK_ROWS)})
    shutil.rmtree(folder / "soak-b")
    [failure] = nightly_failures.load_run(data, "2026-11-25-9").failures
    assert (failure.kind, failure.check) == ("error", "soak-b")
    assert "did not download" in failure.detail


def test_a_skipped_night_is_looked_through(tmp_path):
    data = tmp_path / "data"
    soak_night(data, "2026-11-22-1", "nightly", BROKE.format(n="old") + BROKE.format(n="gone"))
    skipped = soak_night(data, "2026-11-23-2", "nightly", "")
    night = json.loads((skipped / "nightly.json").read_text())
    night.update(status="skipped", checks=[], skip_reason="main has not moved")
    (skipped / "nightly.json").write_text(json.dumps(night))
    soak_night(data, "2026-11-24-3", "nightly", BROKE.format(n="old"))
    result = nightly_failures.compare(data, "2026-11-24-3")
    assert [r.folder for r in result.history] == ["2026-11-22-1"]
    assert result.since[result.run.failures[0].key] == "since 2026-11-22"
    assert [f.name for f in result.fixed] == ["gone"]


def test_regressions_read_where_lower_is_worse_and_past_the_cap(tmp_path):
    data = tmp_path / "data"
    a_night(
        data,
        "2026-11-25-9",
        "manual",
        {
            "micro-hw": (
                "regression",
                [],
                ROWS_HEADER + "SKIP,reads,cycles,regression,1.5,3;5,2;2,,,confirmed\n",
            ),
            "pokec": (
                "regression",
                [
                    {
                        **finding(
                            "throughput [read-only]", "throughput", "regression", 468, 756, 0.62
                        ),
                        "direction": "lower_is_worse",
                        "threshold": 1.3,
                    }
                ],
                None,
            ),
        },
    )
    failures = {f.name: f for f in nightly_failures.load_run(data, "2026-11-25-9").failures}
    assert failures["throughput [read-only]"].detail == "0.62x, below 1/1.30x"
    past = failures["SKIP [reads]"]
    assert (past.candidate, past.baseline) == (4.0, 2.0)  # the medians of the reps
    assert past.detail == "1.50x (raw/rows.csv only: past the findings cap)"


def test_a_run_name_that_is_not_a_stored_folder_is_refused(tmp_path):
    data = tmp_path / "data"
    (data / "latest").mkdir(parents=True)
    (data / "latest" / "nightly.json").write_text("{}")
    assert (
        CliRunner()
        .invoke(cli, ["nightly", "failures", "--data", str(data), "--run", "latest"])
        .exit_code
        == 1
    )


@pytest.fixture
def no_wait(monkeypatch):
    monkeypatch.setattr(nightly_store, "RETRY_DELAYS", (0.0, 0.0))


def stalling_run(listing, archive: bytes, stalls: int):
    """`gh api` whose zip download stalls (times out) `stalls` times before it delivers."""
    calls = []

    def run(argv, **kwargs):
        calls.append(argv[2])
        if argv[2].endswith("/zip"):
            if sum(c.endswith("/zip") for c in calls) <= stalls:
                kwargs["stdout"].write(archive[:10])  # part of it, then nothing
                raise subprocess.TimeoutExpired(argv, kwargs.get("timeout"))
            kwargs["stdout"].write(archive)
            return subprocess.CompletedProcess(argv, 0, None, b"")
        return subprocess.CompletedProcess(argv, 0, json.dumps(listing), "")

    return run, calls


def test_a_stalled_download_is_tried_again(tmp_path, no_wait):
    run, calls = stalling_run({"artifacts": ALPHA}, zipped({"result.json": "{}"}), stalls=2)
    dest = tmp_path / "perf-result-alpha"
    assert nightly_store.download("o/r", "7", "perf-result-alpha", dest, run) == "ok"
    assert (dest / "result.json").read_text() == "{}"  # the stalled attempts' bytes were dropped
    assert sum(c.endswith("/zip") for c in calls) == 3


def test_a_download_that_never_arrives_fails_without_a_partial_result(tmp_path, no_wait):
    run, _ = stalling_run({"artifacts": ALPHA}, zipped({"result.json": "{}"}), stalls=3)
    dest = tmp_path / "perf-result-alpha"
    assert nightly_store.download("o/r", "7", "perf-result-alpha", dest, run) == "failed"
    assert not dest.exists()
    run, _ = stalling_run({"artifacts": []}, b"", stalls=0)
    assert nightly_store.download("o/r", "7", "perf-result-alpha", dest, run) == "missing"


def test_store_fetches_the_runs_nightly_when_none_is_given(tmp_path, no_wait, capsys):
    night = examples.nightly("green", [micro_hw(True)])
    archives = {
        "perf-nightly": zipped({"nightly.json": json.dumps(night)}),
        "perf-result-micro-hw": zipped({"result.json": json.dumps(night["checks"][0])}),
    }

    def run(argv, **kwargs):
        if argv[2].endswith("/zip"):
            name = {"1": "perf-nightly", "2": "perf-result-micro-hw"}[argv[2].split("/")[-2]]
            kwargs["stdout"].write(archives[name])
            return subprocess.CompletedProcess(argv, 0, None, b"")
        name = argv[2].split("name=")[1].split("&")[0]
        ids = {"perf-nightly": [1], "perf-result-micro-hw": [2]}.get(name, [])
        return subprocess.CompletedProcess(
            argv, 0, json.dumps({"artifacts": [{"id": i} for i in ids]}), ""
        )

    data = tmp_path / "data"
    argv = [
        "--run-id",
        "7",
        "--data",
        str(data),
        "--trigger",
        "manual",
        "--scope",
        "main",
        "--repo",
        "o/r",
    ]
    assert nightly_store.main(argv, run=run) == 0
    folder = capsys.readouterr().out.strip()
    assert json.loads((data / folder / "nightly.json").read_text()) == night
    assert (data / folder / "micro-hw" / "result.json").is_file()


@pytest.mark.parametrize(
    ("listing", "said"),
    [
        ({"artifacts": []}, "has no perf-nightly artifact"),
        (None, "did not download"),
    ],
)
def test_store_says_why_a_runs_nightly_is_not_there(tmp_path, no_wait, capsys, listing, said):
    def run(argv, **kwargs):
        if listing is None:
            return subprocess.CompletedProcess(argv, 1, "", "connection reset")
        return subprocess.CompletedProcess(argv, 0, json.dumps(listing), "")

    argv = [
        "--run-id",
        "7",
        "--data",
        str(tmp_path / "d"),
        "--trigger",
        "manual",
        "--scope",
        "main",
    ]
    assert nightly_store.main(argv, run=run) == 1
    assert said in capsys.readouterr().err


def test_a_run_github_does_not_know_is_missing_not_tried_again(tmp_path, no_wait):
    calls = []

    def run(argv, **kwargs):
        calls.append(argv[2])
        return subprocess.CompletedProcess(argv, 1, "", "gh: Not Found (HTTP 404)")

    assert nightly_store.download("o/r", "1", "perf-nightly", tmp_path / "x", run) == "missing"
    assert len(calls) == 1


def test_a_check_no_earlier_night_measured_has_no_earlier_night(tmp_path):
    """A check added since the last stored night (or one that could not measure then) has no
    earlier result: not "new"."""
    data = tmp_path / "data"
    a_night(data, "2026-11-23-1", "nightly", {"micro-hw": ("pass", [], None)})
    a_night(
        data,
        "2026-11-24-2",
        "nightly",
        {
            "micro-hw": ("pass", [], None),
            "soak-b": ("error", [], None),
        },
    )
    a_night(
        data,
        "2026-11-25-3",
        "manual",
        {
            "micro-hw": (
                "regression",
                [finding("SKIP [reads]", "instructions", "regression", 2, 1, 1.9)],
                None,
            ),
            "soak-b": ("pass", [], SOAK_HEADER + BROKE.format(n="s03 broken")),
        },
    )
    result = nightly_failures.compare(data, "2026-11-25-3")
    assert {f.check: result.since[f.key] for f in result.run.failures} == {
        "micro-hw": "new",
        "soak-b": "no earlier night",
    }


# --- what a store keeps, and the unlisted pages ------------------------------------------


def full_artifact(tmp_path: Path) -> Path:
    result = micro_hw(False)
    result["artifacts"] = [
        "raw/rows.csv",
        "raw/big.jsonl",
        "raw/page.html",
        "raw/trace.svg",
        "raw/logs",
    ]
    return check_artifact(
        tmp_path / "art" / "micro-hw",
        result,
        {
            "perf-vm.json": "{}",
            "raw/rows.csv": ROWS_HEADER
            + f"{ROW.split(' [')[0]},writes,cycles,regression,6.4,9;9,1;1,,,x\n",
            "raw/big.jsonl": "{}\n" * 1000,
            "raw/page.html": "<script>alert(1)</script>",
            "raw/trace.svg": "<svg onload=alert(1)>",
            "raw/logs/run.log": "log",
        },
    )


def test_a_store_keeps_only_what_the_run_page_needs_unless_full(tmp_path):
    art = full_artifact(tmp_path)
    night_path = tmp_path / "nightly.json"
    night_path.write_text(json.dumps(examples.nightly("red", [micro_hw(False)])))
    download, _ = downloads(tmp_path, {"perf-result-micro-hw": art})
    trimmed = nightly_store.store(night_path, "9", tmp_path / "trim", "nightly", "main", download)
    download, _ = downloads(tmp_path, {"perf-result-micro-hw": art})
    full = nightly_store.store(
        night_path, "9", tmp_path / "full", "nightly", "main", download, full=True
    )

    def files(root):
        return sorted(p.relative_to(root).as_posix() for p in root.rglob("*") if p.is_file())

    assert files(tmp_path / "trim" / trimmed) == [
        "micro-hw/perf-vm.json",
        "micro-hw/raw/rows.csv",
        "micro-hw/result.json",
        "nightly.json",
        "run.json",
    ]
    assert "micro-hw/raw/big.jsonl" in files(tmp_path / "full" / full)
    assert "micro-hw/raw/logs/run.log" in files(tmp_path / "full" / full)
    # The trimmed store is the run page's subset of the full one, byte for byte.
    for rel in files(tmp_path / "trim" / trimmed):
        assert (tmp_path / "trim" / trimmed / rel).read_bytes() == (
            tmp_path / "full" / full / rel
        ).read_bytes()


def test_the_pages_read_the_same_from_a_trimmed_store(tmp_path):
    """Trends and regression rows come from raw/rows.csv, which a trimmed store keeps: only the
    run page's list of files differs."""
    art = full_artifact(tmp_path)
    night_path = tmp_path / "nightly.json"
    # The night lists the check's files as its result.json does: the run page links from it.
    result = json.loads((art / "result.json").read_text())
    night_path.write_text(json.dumps(examples.nightly("red", [result])))
    sites = {}
    for name, full in (("trim", False), ("full", True)):
        download, _ = downloads(tmp_path, {"perf-result-micro-hw": art})
        site = tmp_path / name
        folder = nightly_store.store(
            night_path, "9", site / "data", "nightly", "main", download, full=full
        )
        dashboard.build(site / "data", site)
        sites[name] = pages(site)
    assert sites["trim"]["trends.html"] == sites["full"]["trends.html"]
    assert sites["trim"]["index.html"] == sites["full"]["index.html"]
    trim_page, full_page = sites["trim"][f"run/{folder}.html"], sites["full"][f"run/{folder}.html"]
    assert re.sub(r"<p>Files:.*?</p>", "", trim_page) == re.sub(r"<p>Files:.*?</p>", "", full_page)
    assert ROW in trim_page  # the regression rows, read from raw/rows.csv
    assert "page.html" in full_page and "big.jsonl" in full_page  # stored in full: linked
    assert "page.html" not in trim_page and "big.jsonl" not in trim_page


class FakeActions:
    """`gh api` over a table of workflow runs and their artifacts (zips by id), found as the
    unlisted pages find them: through the repository's perf-nightly artifacts."""

    def __init__(self):
        self.runs: dict[int, dict] = {}
        self.artifacts: dict[str, list[dict]] = {}
        self.zips: dict[int, bytes] = {}
        self.next_id = 1000
        self.broken: set[int] = set()

    def add_run(self, workflow, run_id, created, title="Perf nightly", event=None):
        event = event or (
            "issue_comment" if workflow == "perf-comment.yml" else "workflow_dispatch"
        )
        self.runs[run_id] = {
            "id": run_id,
            "path": f".github/workflows/{workflow}",
            "event": event,
            "display_title": title,
            "created_at": created,
        }

    def add_artifact(self, run_id, name, members, created=None):
        self.next_id += 1
        self.artifacts.setdefault(str(run_id), []).append(
            {
                "id": self.next_id,
                "name": name,
                "created_at": created or self.runs[run_id]["created_at"],
                "workflow_run": {"id": run_id},
            }
        )
        self.zips[self.next_id] = zipped(members)

    def add_perf_run(self, run_id, night, results):
        self.add_artifact(run_id, "perf-nightly", {"nightly.json": json.dumps(night)})
        for check, files in results.items():
            self.add_artifact(run_id, f"perf-result-{check}", files)

    def __call__(self, argv, **kwargs):
        path = argv[2]
        if path.endswith("/zip"):
            kwargs["stdout"].write(self.zips[int(path.split("/")[-2])])
            return subprocess.CompletedProcess(argv, 0, None, b"")
        if "/actions/artifacts?" in path:
            listed = [
                a for arts in self.artifacts.values() for a in arts if a["name"] == "perf-nightly"
            ]
            page = int(path.split("page=")[-1])
            body = {"artifacts": sorted(listed, key=lambda a: -a["id"]) if page == 1 else []}
            return subprocess.CompletedProcess(argv, 0, json.dumps(body), "")
        run_id = path.split("/runs/")[1].split("/")[0].split("?")[0]
        if "/artifacts" not in path:
            return subprocess.CompletedProcess(argv, 0, json.dumps(self.runs[int(run_id)]), "")
        name = path.split("name=")[1].split("&")[0]
        found = [a for a in self.artifacts.get(run_id, []) if a["name"] == name]
        return subprocess.CompletedProcess(argv, 0, json.dumps({"artifacts": found}), "")


NOW = datetime(2026, 11, 26, 12, 0, tzinfo=UTC)


def perf_night(date, label="pr-3967@727e385"):
    night = examples.nightly("red", [micro_hw(False, HOSTILE)])
    night["date"] = date
    night["candidate"]["label"] = label
    return night


def micro_files():
    result = micro_hw(False, HOSTILE)
    result["artifacts"] = ["raw/rows.csv", "raw/evil.html"]
    return {
        "result.json": json.dumps(result),
        "raw/rows.csv": ROWS_HEADER,
        "raw/evil.html": "<script>alert(1)</script>",
        "perf-vm.json": "{}",
    }


@pytest.fixture
def actions():
    gh = FakeActions()
    # /perf runs: one inside the window, one too old, one comment that was no request
    gh.add_run("perf-comment.yml", 501, "2026-11-20T10:00:00Z", "/perf on #3967")
    gh.add_perf_run(501, perf_night("2026-11-20"), {"micro-hw": micro_files()})
    gh.add_run("perf-comment.yml", 502, "2026-10-26T11:59:59Z", "/perf on #3900")
    gh.add_perf_run(502, perf_night("2026-10-26"), {"micro-hw": micro_files()})
    gh.add_run("perf-comment.yml", 503, "2026-11-21T10:00:00Z", "Comment, not /perf")
    gh.add_perf_run(503, perf_night("2026-11-21"), {"micro-hw": micro_files()})
    # a /perf-titled run that is no comment (#4039's acceptance runs ran on push): no page
    gh.add_run("perf-comment.yml", 504, "2026-11-22T10:00:00Z", "/perf on #4039", "push")
    gh.add_perf_run(504, perf_night("2026-11-22"), {"micro-hw": micro_files()})
    # dispatches: one that named images, one that did not (a night on the public pages), and
    # one that named images but whose artifacts are gone
    gh.add_run("perf-nightly.yml", 601, "2026-11-25T10:00:00Z")
    gh.add_artifact(601, "perf-plan", {"dispatch.json": '{"image_overrides": true}'})
    gh.add_perf_run(601, perf_night("2026-11-25", "main@4d72ac1"), {"micro-hw": micro_files()})
    gh.add_run("perf-nightly.yml", 602, "2026-11-25T11:00:00Z")
    gh.add_artifact(602, "perf-plan", {"dispatch.json": '{"image_overrides": false}'})
    gh.add_perf_run(602, perf_night("2026-11-25", "main@4d72ac1"), {"micro-hw": micro_files()})
    gh.add_run("perf-nightly.yml", 603, "2026-11-24T10:00:00Z")
    gh.add_artifact(603, "perf-plan", {"dispatch.json": '{"image_overrides": true}'})
    # a scheduled night is on the runs page, not unlisted
    gh.add_run("perf-nightly.yml", 604, "2026-11-25T00:00:00Z", event="schedule")
    gh.add_perf_run(604, perf_night("2026-11-25", "main@4d72ac1"), {"micro-hw": micro_files()})
    return gh


def test_unlisted_pages_cover_the_last_30_days_of_manual_and_perf_runs(tmp_path, actions, no_wait):
    site = tmp_path / "site"
    (site / "pr" / "1" / "2026-01-01-1").mkdir(parents=True)  # a page from an earlier deploy
    made = nightly_unlisted.build(site, "o/r", 30, NOW, actions)
    assert sorted(p.relative_to(site).as_posix() for p in made) == [
        "manual/2026-11-25-601/index.html",
        "pr/3967/2026-11-20-501/index.html",
    ]
    # The 31-day-old /perf run, the dispatch that named no images, the one whose artifacts are
    # gone and the comment that was no request get no page; an earlier deploy's page is gone.
    assert not (site / "pr" / "3900").exists() and not (site / "pr" / "1").exists()
    page = (site / "pr/3967/2026-11-20-501/index.html").read_text()
    assert '<meta name="robots" content="noindex">' in page
    assert "← Runs" not in page and "href='../../../index.html'" in page
    assert "pr-3967 · pr-comment" in page
    # Escaped, like every listed page.
    assert "<script>alert" not in page and "&lt;script&gt;" in page
    # Only the page's files, of types shown as text: never the run's own .html.
    stored = site / "pr/3967/2026-11-20-501/data/2026-11-20-501"
    assert sorted(p.relative_to(stored).as_posix() for p in stored.rglob("*") if p.is_file()) == [
        "micro-hw/perf-vm.json",
        "micro-hw/raw/rows.csv",
        "micro-hw/result.json",
        "nightly.json",
        "run.json",
    ]
    for href in re.findall(r"href='([^']+)'", page):
        if not href.startswith(("http", "../")):
            assert (site / "pr/3967/2026-11-20-501" / href).resolve().is_file(), href


def test_the_listed_pages_never_link_an_unlisted_one(tmp_path, actions, no_wait):
    site = tmp_path / "site"
    data = site / "data"
    store(
        data,
        "2026-11-24-101",
        examples.nightly("green", [micro_hw(True)]),
        {"trigger": "nightly", "scope": "main"},
    )
    dashboard.build(data, site)
    nightly_unlisted.build(site, "o/r", 30, NOW, actions)
    listed = {k: v for k, v in pages(site).items() if not k.startswith(("pr/", "manual/"))}
    assert sorted(listed) == ["index.html", "run/2026-11-24-101.html", "trends.html"]
    for name, text in listed.items():
        assert "pr/" not in text and "manual/" not in text, name
        assert "noindex" not in text, name
    assert sorted(k for k in pages(site) if k.startswith(("pr/", "manual/"))) == [
        "manual/2026-11-25-601/index.html",
        "pr/3967/2026-11-20-501/index.html",
    ]


def test_the_window_is_counted_from_each_runs_creation(tmp_path, actions, no_wait):
    """Run 601 was created at 2026-11-25 10:00: a one-day window keeps it until 2026-11-26
    10:00, and the first deploy after that leaves it out."""
    before = nightly_unlisted.build(tmp_path, "o/r", 1, NOW.replace(hour=9), actions)
    assert [p.parent.name for p in before] == ["2026-11-25-601"]
    after = nightly_unlisted.build(tmp_path, "o/r", 1, NOW.replace(hour=10, minute=1), actions)
    assert after == [] and not (tmp_path / "manual" / "2026-11-25-601").exists()


def test_a_runs_page_url_is_known_before_the_page_exists(tmp_path, capsys):
    night = tmp_path / "nightly.json"
    night.write_text(json.dumps({"date": "2026-11-26"}))
    argv = [
        "--repo",
        "FalkorDB/FalkorDB",
        "url",
        "--kind",
        "pr",
        "--nightly",
        str(night),
        "--run-id",
        "77",
        "--pr",
        "3967",
    ]
    assert nightly_unlisted.main(argv, environ={}) == 0
    assert (
        capsys.readouterr().out.strip()
        == "https://falkordb.github.io/FalkorDB/pr/3967/2026-11-26-77/"
    )
    argv[3:5] = ["--kind", "manual"]
    assert nightly_unlisted.main(argv, environ={"PERF_PAGES_URL": "https://perf.example/"}) == 0
    assert capsys.readouterr().out.strip() == "https://perf.example/manual/2026-11-26-77/"
    night.write_text("{}")
    assert nightly_unlisted.main(argv, environ={}) == 1


def test_a_manual_run_that_named_its_images_links_its_page():
    """Only a dispatch that named images gets an unlisted page: the gate and /perf calls, and a
    plain dispatch (on the runs page), print no such link."""
    steps = load_workflow("perf-nightly.yml")["jobs"]["report"]["steps"]
    [step] = [s for s in steps if s.get("name") == "Link the run's page"]
    assert step["if"] == (
        "${{ !cancelled() && inputs.mode == '' && github.event_name == 'workflow_dispatch' "
        "&& (inputs.candidate_image != '' || inputs.baseline_image != '') }}"
    )
    assert "bench nightly unlisted url --kind manual" in step["run"]
    assert step["env"] == {"PERF_PAGES_URL": "${{ vars.PERF_PAGES_URL }}"}


def test_the_chat_line_links_the_dashboard_page_of_the_runs_the_dashboard_stores():
    """PERF_ON_DASHBOARD is perf-dashboard.yml's own rule (the schedule; a dispatch from the
    default branch that named no images); a /perf or gate call (mode set) never has a page."""
    steps = load_workflow("perf-nightly.yml")["jobs"]["report"]["steps"]
    [step] = [s for s in steps if s.get("name") == "Post the night's result"]
    assert step["env"]["PERF_ON_DASHBOARD"] == (
        "${{ inputs.mode == '' && (github.event_name == 'schedule' || "
        "(github.event_name == 'workflow_dispatch' && github.ref_name == "
        "github.event.repository.default_branch && inputs.candidate_image == '' && "
        "inputs.baseline_image == '')) }}"
    )
    assert step["env"]["PERF_PAGES_URL"] == "${{ vars.PERF_PAGES_URL }}"
    decide = load_workflow("perf-dashboard.yml")["jobs"]["decide"]
    text = json.dumps(decide)
    # The same three conditions on the dashboard's side.
    assert "image_overrides" in text and "DEFAULT_BRANCH" in text and "schedule" in text


def test_only_githubs_404_is_a_missing_artifact(tmp_path, no_wait):
    """A 403 (rate limit, permissions) is an answer about the request, not the run: it is tried
    again and, if it lasts, a failed download, never "the run has none"."""
    calls = []

    def run(argv, **kwargs):
        calls.append(argv[2])
        return subprocess.CompletedProcess(argv, 1, "", "gh: API rate limit exceeded (HTTP 403)")

    assert nightly_store.download("o/r", "7", "perf-nightly", tmp_path / "x", run) == "failed"
    assert len(calls) == 3


def test_a_download_that_failed_stops_the_store_with_nothing_changed(tmp_path):
    """A check whose artifact exists but did not download is not stored as a run without it:
    the store stops, so the workflow fails and a re-run tries again."""
    data = tmp_path / "data"
    night_path = tmp_path / "nightly.json"
    night_path.write_text(json.dumps(examples.nightly("green", [micro_hw(True)])))
    with pytest.raises(nightly_store.StoreError, match="perf-result-micro-hw did not download"):
        nightly_store.store(night_path, "7", data, "nightly", "main", lambda name, dest: "failed")
    assert not data.exists() or not any(data.iterdir())
    # A check that left no artifact at all is tolerated, as before.
    folder = nightly_store.store(
        night_path, "7", data, "nightly", "main", lambda name, dest: "missing"
    )
    assert (data / folder / "nightly.json").is_file() and not (data / folder / "micro-hw").exists()


def test_a_run_whose_page_cannot_be_built_is_left_out_and_the_others_still_are(
    tmp_path, actions, no_wait, monkeypatch
):
    real = dashboard.build_unlisted

    def flaky(data, folder, out, root):
        if folder.endswith("-601"):
            raise OSError("File name too long")
        return real(data, folder, out, root)

    monkeypatch.setattr(dashboard, "build_unlisted", flaky)
    made = nightly_unlisted.build(tmp_path, "o/r", 30, NOW, actions)
    assert [p.parent.name for p in made] == ["2026-11-20-501"]
    # Nothing of the run whose page failed is left in the site.
    assert not (tmp_path / "manual").exists() or not any((tmp_path / "manual").iterdir())


def test_trim_copies_each_run_with_only_what_its_pages_need(tmp_path, monkeypatch):
    src, dest = tmp_path / "src", tmp_path / "dest"
    run = src / "2026-11-25-9"
    files = {
        "nightly.json": "{}",
        "run.json": "{}",
        "ldbc/result.json": "{}",
        "ldbc/raw/rows.csv": "a",
        "ldbc/raw/summary.csv": "query,phase,side,attempted,ok\n",
        "ldbc/raw/samples.csv": "kept out",
        "soak-a/raw/s01_social/candidate-1/stdout.log": "kept out",
        "soak-a/raw/rows.csv": "x" * 64,
        "soak-a/result.json": "{}",
    }
    for rel, text in files.items():
        (run / rel).parent.mkdir(parents=True, exist_ok=True)
        (run / rel).write_text(text)
    (src / "not-a-run").mkdir()
    monkeypatch.setattr(nightly_store, "MAX_PAGE_FILE", 32)  # soak-a's rows.csv is too big
    assert nightly_store.trim(src, dest) == 1
    assert sorted(p.relative_to(dest).as_posix() for p in dest.rglob("*") if p.is_file()) == [
        "2026-11-25-9/ldbc/raw/rows.csv",
        "2026-11-25-9/ldbc/raw/summary.csv",
        "2026-11-25-9/ldbc/result.json",
        "2026-11-25-9/nightly.json",
        "2026-11-25-9/run.json",
        "2026-11-25-9/soak-a/result.json",
    ]


def run_build(tmp_path, remote):
    """Run the deploy job's build step against a gh-pages remote that is `present` (holding one
    night stored in full), `absent` (a repository without the branch) or `unreachable`."""
    git = ["git", "-c", "user.name=t", "-c", "user.email=t@t", "-c", "init.defaultBranch=x"]
    seed = tmp_path / "seed"
    night = seed / "data" / "2026-11-24-7"
    (night / "soak-a" / "raw" / "s01").mkdir(parents=True)
    (night / "nightly.json").write_text(json.dumps(examples.nightly("green", [])))
    (night / "soak-a" / "result.json").write_text("{}")
    (night / "soak-a" / "raw" / "s01" / "stdout.log").write_text("big")
    subprocess.run(
        [*git, "init", "-q", "-b", "gh-pages" if remote == "present" else "main", str(seed)],
        check=True,
    )
    subprocess.run([*git, "-C", str(seed), "add", "-A"], check=True)
    subprocess.run([*git, "-C", str(seed), "commit", "-q", "-m", "seed"], check=True)
    bin_dir = tmp_path / "bin"
    bin_dir.mkdir()
    (bin_dir / "base64").write_text("#!/bin/sh\nexec /usr/bin/base64\n")
    (bin_dir / "uv").write_text(FAKE_UV)
    (bin_dir / "gh").write_text(FAKE_GH)
    for shim in ("base64", "uv", "gh"):
        (bin_dir / shim).chmod(0o755)
    temp = tmp_path / "temp"
    temp.mkdir()
    env = {
        "PATH": f"{bin_dir}:/usr/bin:/bin:/usr/local/bin:/opt/homebrew/bin",
        "HOME": str(tmp_path),
        "GH_TOKEN": "x",
        "GITHUB_SERVER_URL": f"file://{tmp_path}" + ("/nowhere" if remote == "unreachable" else ""),
        "GITHUB_REPOSITORY": "seed",
        "RUNNER_TEMP": str(temp),
        "FAKE_ARTIFACTS": str(fake_artifacts(tmp_path, [])),
        "PYTHONPATH": str(BENCH / "src"),
    }
    done = subprocess.run(
        ["bash", "-c", publish_step("Build the site")],
        env=env,
        capture_output=True,
        text=True,
        cwd=BENCH.parent,
    )
    return done, temp / "pages"


def test_the_deploy_publishes_the_stored_nights_trimmed(tmp_path):
    done, pages = run_build(tmp_path, "present")
    assert done.returncode == 0, done.stderr
    assert (pages / "data" / "2026-11-24-7" / "soak-a" / "result.json").is_file()
    assert not (pages / "data" / "2026-11-24-7" / "soak-a" / "raw").exists()


def test_the_deploy_starts_empty_only_when_there_is_no_gh_pages_branch(tmp_path):
    done, _ = run_build(tmp_path / "a", "absent")
    assert done.returncode == 0, done.stderr
    assert "no gh-pages branch yet" in done.stdout
    done, _ = run_build(tmp_path / "b", "unreachable")
    assert (
        done.returncode != 0 and "could not read the gh-pages branch" in done.stdout + done.stderr
    )


def test_the_answer_goes_out_without_a_page_address_it_could_not_make():
    steps = load_workflow("perf-comment.yml")["jobs"]["answer"]["steps"]
    [step] = [s for s in steps if s.get("name") == "Answer on the pull request"]
    assert '--pr "$PR") || page=""' in step["run"]


@pytest.mark.parametrize(
    ("event", "dispatch"), [("workflow_dispatch", '{"image_overrides": false}')]
)
def test_a_feature_branch_dispatch_is_no_night(tmp_path, event, dispatch):
    """It runs that branch's checks: stored, it would read as one of main's nights."""
    assert run_download(tmp_path, event, dispatch, branch="ci/try-something") == (False, False)
    # One that named its images still gets its unlisted page, from whatever branch.
    (tmp_path / "b").mkdir()
    got = run_download(
        tmp_path / "b", event, '{"image_overrides": true}', branch="ci/try-something"
    )
    assert got == (False, True)


def test_a_rerun_that_lists_fewer_checks_never_drops_a_stored_one(tmp_path):
    data = tmp_path / "data"
    old = data / "2026-11-20-7"
    for check in ("micro-hw", "pokec"):
        (old / check).mkdir(parents=True)
        (old / check / "result.json").write_text("old")
    stub = tmp_path / "nightly.json"
    stub.write_text(json.dumps(examples.nightly("error", [micro_hw(True)])))  # no pokec now
    with pytest.raises(nightly_store.StoreError, match="does not list pokec"):
        nightly_store.store(stub, "7", data, "nightly", "main", lambda name, dest: True)
    assert (old / "pokec" / "result.json").read_text() == "old"


def test_an_artifact_without_a_result_never_replaces_a_stored_one(tmp_path):
    """The VM died before the check wrote result.json: its upload holds only the runner's record."""
    data = tmp_path / "data"
    old = data / "2026-11-20-7" / "micro-hw"
    old.mkdir(parents=True)
    (old / "result.json").write_text("old")
    night = tmp_path / "nightly.json"
    night.write_text(json.dumps(examples.nightly("error", [micro_hw(True)])))

    def record_only(name, dest):
        dest.mkdir(parents=True)
        (dest / "perf-vm.json").write_text('{"command_exit": null}')
        return "ok"

    with pytest.raises(nightly_store.StoreError, match=r"has no result\.json"):
        nightly_store.store(night, "7", data, "nightly", "main", record_only)
    assert (old / "result.json").read_text() == "old"
    # A first store of that run keeps the record (the page shows why the check has no result).
    folder = nightly_store.store(night, "8", tmp_path / "fresh", "nightly", "main", record_only)
    assert (tmp_path / "fresh" / folder / "micro-hw" / "perf-vm.json").is_file()


@pytest.mark.parametrize("body", ["not json", "[]", '{"total_count": 0}'])
def test_a_listing_that_is_not_one_is_a_failure_not_a_missing_artifact(tmp_path, no_wait, body):
    def run(argv, **kwargs):
        return subprocess.CompletedProcess(argv, 0, body, "")

    assert nightly_store.download("o/r", "7", "perf-result-alpha", tmp_path / "x", run) == "failed"


def test_a_failed_listing_fails_the_deploy_and_removes_nothing(tmp_path, actions, no_wait, capsys):
    (tmp_path / "pr" / "1" / "2026-11-01-1").mkdir(parents=True)

    def broken(argv, **kwargs):
        if "/actions/artifacts?" in argv[2]:
            return subprocess.CompletedProcess(argv, 1, "", "HTTP 502")
        return actions(argv, **kwargs)

    with pytest.raises(nightly_unlisted.ListingError):
        nightly_unlisted.build(tmp_path, "o/r", 30, NOW, broken)
    assert (tmp_path / "pr" / "1" / "2026-11-01-1").exists()
    assert (
        nightly_unlisted.main(["--repo", "o/r", "--out", str(tmp_path)], environ={}, run=broken)
        == 1
    )
    assert "not deploying a site without its unlisted pages" in capsys.readouterr().err


def test_a_measurement_that_is_no_number_is_none():
    assert [
        nightly_failures._number(v) for v in ("nan", "inf", "-inf", "1e999", "3", 2.5, True)
    ] == [
        None,
        None,
        None,
        None,
        3.0,
        2.5,
        None,
    ]


def test_a_check_whose_files_are_not_read_neither_fixes_nor_continues_a_failure(tmp_path):
    data = tmp_path / "data"
    soak_night(data, "2026-11-23-1", "nightly", BROKE.format(n="old"))
    soak_night(data, "2026-11-24-2", "nightly", BROKE.format(n="old"))
    now = soak_night(data, "2026-11-25-3", "nightly", BROKE.format(n="old"))
    # The middle night's soak-b files did not download: not a night without the failure.
    shutil.rmtree(data / "2026-11-24-2" / "soak-b")
    result = nightly_failures.compare(data, "2026-11-25-3")
    [failure] = [f for f in result.run.failures if f.name == "old"]
    assert result.since[failure.key] == "since 2026-11-23"
    # And when this run's files are the ones missing, the earlier failure is not "fixed".
    shutil.rmtree(now / "soak-b")
    result = nightly_failures.compare(data, "2026-11-25-3")
    assert result.fixed == []


def test_an_unreadable_result_or_night_is_an_error(tmp_path):
    data = tmp_path / "data"
    run = soak_night(data, "2026-11-25-3", "nightly", BROKE.format(n="x"))
    (run / "soak-b" / "result.json").write_text("{not json")
    [failure] = [
        f for f in nightly_failures.load_run(data, "2026-11-25-3").failures if f.kind == "error"
    ]
    assert "cannot be read" in failure.detail
    (run / "nightly.json").write_text("{not json")
    failures = nightly_failures.load_run(data, "2026-11-25-3").failures
    assert [(f.check, f.kind) for f in failures] == [("nightly.json", "error")]
    argv = ["nightly", "failures", "--data", str(data), "--run", "2026-11-25-3"]
    assert CliRunner().invoke(cli, argv).exit_code == 1


def test_an_error_compares_only_with_nights_that_planned_the_check(tmp_path):
    """A night that did not plan soak-b says nothing about its error: looked through, not a
    break in the streak."""
    data = tmp_path / "data"
    a_night(data, "2026-11-22-1", "nightly", {"soak-b": ("error", [], None)})
    a_night(data, "2026-11-23-2", "nightly", {"micro-hw": ("pass", [], None)})  # no soak-b
    a_night(data, "2026-11-24-3", "nightly", {"soak-b": ("error", [], None)})
    a_night(data, "2026-11-25-4", "nightly", {"soak-b": ("error", [], None)})
    result = nightly_failures.compare(data, "2026-11-25-4")
    [failure] = result.run.failures
    assert result.since[failure.key] == "since 2026-11-22"


@pytest.mark.parametrize(
    ("compare", "stored"),
    [("identical", True), ("behind", True), ("ahead", False), ("diverged", False)],
)
def test_a_night_is_a_commit_on_the_default_branch_not_only_its_name(tmp_path, compare, stored):
    """A dispatch on a tag named like the default branch carries the branch's name but not its
    history: the compare API says where its commit is."""
    got = run_download(tmp_path, "workflow_dispatch", '{"image_overrides": false}', compare=compare)
    assert got == (stored, stored)


def test_the_unlisted_build_stops_starting_runs_when_out_of_time(
    tmp_path, actions, no_wait, capsys
):
    made = nightly_unlisted.build(tmp_path, "o/r", 30, NOW, actions, budget=-1)
    assert made == []
    assert "out of time" in capsys.readouterr().err


def test_a_store_that_fails_half_way_leaves_the_previous_run(tmp_path, monkeypatch):
    data = tmp_path / "data"
    old = data / "2026-11-20-7" / "micro-hw"
    old.mkdir(parents=True)
    (old / "result.json").write_text("old")
    night = tmp_path / "nightly.json"
    night.write_text(json.dumps(examples.nightly("green", [micro_hw(True)])))
    art = check_artifact(tmp_path / "art", micro_hw(True), {})
    download, _ = downloads(tmp_path, {"perf-result-micro-hw": art})

    def broken(src, dest):
        raise OSError("disk full")

    monkeypatch.setattr(nightly_store, "copy_page_files", broken)
    with pytest.raises(OSError, match="disk full"):
        nightly_store.store(night, "7", data, "nightly", "main", download)
    assert (old / "result.json").read_text() == "old"
    assert sorted(p.name for p in data.iterdir()) == ["2026-11-20-7"]  # no staging left behind


def previous_site(tmp_path):
    """The last deployed site: a page for /perf run 501 (in the window), for 502 (31 days), and
    for 505, another run of the same pull request that is no candidate now."""
    prev = tmp_path / "previous"
    for rel in ("pr/3967/2026-11-20-501", "pr/3900/2026-10-26-502", "pr/3967/2026-11-21-505"):
        (prev / rel / "data").mkdir(parents=True)
        (prev / rel / "index.html").write_text(f"old page {rel}")
    return prev


def test_a_run_not_rebuilt_keeps_the_page_the_last_deploy_had(tmp_path, actions, no_wait):
    """Its download failed this time: the page the last deploy published stays up (a link in a
    /perf answer keeps working for its 30 days), and nothing older than the window comes back."""
    prev = previous_site(tmp_path)

    def failing(argv, **kwargs):
        if "/runs/501/artifacts?name=perf-result" in argv[2]:
            return subprocess.CompletedProcess(argv, 1, "", "HTTP 502")
        return actions(argv, **kwargs)

    site = tmp_path / "site"
    made = nightly_unlisted.build(site, "o/r", 30, NOW, failing, previous=prev)
    assert sorted(p.relative_to(site).as_posix() for p in made) == [
        "manual/2026-11-25-601/index.html",
        "pr/3967/2026-11-20-501/index.html",
    ]
    assert (
        site / "pr/3967/2026-11-20-501/index.html"
    ).read_text() == "old page pr/3967/2026-11-20-501"
    assert not (site / "pr" / "3900").exists()


def test_a_manual_run_whose_plan_did_not_download_keeps_its_last_page(tmp_path, actions, no_wait):
    """Its perf-plan (which says whether it named images) failed to download: that is no answer,
    so the run is not taken for a public night and dropped; it keeps the page it had."""
    prev = previous_site(tmp_path)
    (prev / "manual/2026-11-25-601").mkdir(parents=True)
    (prev / "manual/2026-11-25-601/index.html").write_text("old page manual/2026-11-25-601")

    def failing(argv, **kwargs):
        if "/runs/601/artifacts?name=perf-plan" in argv[2]:
            return subprocess.CompletedProcess(argv, 1, "", "HTTP 502")
        return actions(argv, **kwargs)

    site = tmp_path / "site"
    made = nightly_unlisted.build(site, "o/r", 30, NOW, failing, previous=prev)
    assert sorted(p.relative_to(site).as_posix() for p in made) == [
        "manual/2026-11-25-601/index.html",
        "pr/3967/2026-11-20-501/index.html",
    ]
    assert (
        site / "manual/2026-11-25-601/index.html"
    ).read_text() == "old page manual/2026-11-25-601"


def test_out_of_time_every_run_keeps_its_last_page(tmp_path, actions, no_wait):
    prev = previous_site(tmp_path)
    site = tmp_path / "site"
    made = nightly_unlisted.build(site, "o/r", 30, NOW, actions, budget=-1, previous=prev)
    # 501 is carried (its own page, not another of the pull request's); 601 had no page
    # before, so it waits for the next deploy.
    assert [p.relative_to(site).as_posix() for p in made] == ["pr/3967/2026-11-20-501/index.html"]
    assert (
        site / "pr/3967/2026-11-20-501/index.html"
    ).read_text() == "old page pr/3967/2026-11-20-501"
    assert not (site / "pr" / "3900").exists() and not (site / "pr/3967/2026-11-21-505").exists()


def test_the_deploy_hands_the_last_site_to_the_unlisted_build():
    build = publish_step("Build the site")
    assert "-n github-pages" in build and "artifact.tar" in build
    assert 'bench nightly unlisted --out "$pages" --days 30 --previous "$previous"' in build
    steps = load_workflow("perf-dashboard.yml")["jobs"]["deploy"]["steps"]
    [upload] = [s for s in steps if s.get("uses", "").startswith("actions/upload-pages-artifact@")]
    assert upload["with"]["retention-days"] == 35  # past the pages' 30 days


def test_a_run_page_names_each_pass_fail_check_broken_on_the_candidate(tmp_path):
    """#4065: the run page names every pass/fail check broken on the candidate, whether the
    baseline breaks it too, and its issue where one exists."""
    doc = examples.example("result-soak-both-broken.json")
    doc["broken"][0]["issue"] = 3512
    doc["findings"].insert(
        1,
        {**doc["findings"][0], "name": "s10 <b>x</b>", "baseline": 1},
    )
    doc["broken"].append(
        {"name": "s10 <b>x</b>", "baseline_too": False, "issue": None, "introduced": True}
    )
    night = examples.nightly("red", [doc])
    assert contract.validate_nightly(night) == []
    data = tmp_path / "data"
    store(data, "2026-11-25-200", night, {"trigger": "nightly", "scope": "main"})
    page = dashboard.run_page(dashboard.load(data)[0], "data", data)
    rows = re.findall(r"<tr><td>[^<]*(?:&lt;[^<]*)*</td><td>(?:yes|no)</td>.*?</tr>", page)
    assert [cells(r) for r in rows] == [
        ["s7 MERGE re-run creates no duplicate :Device", "yes", "#3512"],
        ["s10 <b>x</b>", "no", "none known"],
    ]
    assert "https://github.com/FalkorDB/FalkorDB/issues/3512" in page
    assert "<b>x</b>" not in page  # escaped
