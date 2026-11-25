"""`bench nightly dashboard`: the static site built from the stored runs (falkorbench.nightly.dashboard).

The runs are the example nightlies (tests/fixtures) with a few fake raw/rows.csv files, laid out as
perf-nightly.yml's publish job stores them: data/<date>-<run id>/{nightly.json,run.json,<check>/}.
"""

from __future__ import annotations

import copy
import html
import json
import re
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
    """One run as the publish job stores it; `rows` is micro-hw's raw/rows.csv body."""
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
    "soak-a": _soak,
    "soak-b": _soak,
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
