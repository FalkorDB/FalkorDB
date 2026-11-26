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


def test_the_dashboard_is_published_by_its_own_workflow_after_the_night():
    nightly = load_workflow("perf-nightly.yml")
    assert "publish" not in nightly["jobs"]
    flow = load_workflow("perf-dashboard.yml")
    assert flow["on"] == {"workflow_run": {"workflows": [nightly["name"]], "types": ["completed"]}}
    assert flow["concurrency"] == {"group": "perf-dashboard", "cancel-in-progress": False}
    job = flow["jobs"]["publish"]
    assert job["permissions"] == {
        "contents": "write",
        "actions": "read",
        "pages": "write",
        "id-token": "write",
    }
    assert job["environment"]["name"] == "github-pages"
    for step in job["steps"]:
        if "uses" in step:  # pinned by SHA, like the repo's other workflows
            assert re.fullmatch(r"[\w./-]+@[0-9a-f]{40}", step["uses"]), step["uses"]
    uses = [step["uses"].split("@")[0] for step in job["steps"] if "uses" in step]
    assert "actions/upload-pages-artifact" in uses
    assert "actions/deploy-pages" in uses


def test_the_run_is_stored_only_for_runs_that_report_to_chat():
    """A scheduled night, or a dispatch that named no image: the nightly records which."""
    job = load_workflow("perf-dashboard.yml")["jobs"]["publish"]
    assert "'schedule'" in job["if"] and "'workflow_dispatch'" in job["if"]
    text = "\n".join(step.get("run", "") for step in job["steps"])
    assert "dispatch.json" in text and "image_overrides" in text
    record = next(
        step["run"]
        for step in load_workflow("perf-nightly.yml")["jobs"]["resolve"]["steps"]
        if step.get("name") == "Record the image overrides"
    )
    assert "dispatch.json" in record and "image_overrides" in record


# --- the publish steps, run for real against a fake gh --------------------------------

FAKE_GH = r"""#!/usr/bin/env bash
# `gh run download` lays out the plan and nightly artifacts from $FAKE_RUN; `gh api` lists
# $FAKE_ARTIFACTS by name and serves a zip whose result.json names the artifact's id.
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
  work=$(mktemp -d); echo "$id" > "$work/result.json"
  (cd "$work" && zip -q -r - result.json)
  exit 0
fi
name=${url#*name=}; name=${name%%&*}
jq --arg n "$name" ".artifacts |= map(select(.name == \$n))" "$FAKE_ARTIFACTS"
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
    steps = load_workflow("perf-dashboard.yml")["jobs"]["publish"]["steps"]
    return next(step["run"] for step in steps if step.get("name") == name)


def run_download(tmp_path, event, dispatch):
    """Run the download step; return whether it stores the run."""
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
    }
    subprocess.run(
        ["bash", "-c", publish_step("Download the night's artifacts")],
        env=env,
        check=True,
        capture_output=True,
    )
    return "store=true" in out.read_text()


@pytest.mark.parametrize(
    ("event", "dispatch", "stored"),
    [
        ("schedule", None, True),
        ("workflow_dispatch", '{"image_overrides": false}', True),
        ("workflow_dispatch", '{"image_overrides": true}', False),
        ("workflow_dispatch", None, False),
        ("workflow_dispatch", "not json", False),
        ("workflow_dispatch", "{}", False),
    ],
)
def test_a_dispatch_is_stored_only_when_it_records_no_image_overrides(
    tmp_path, event, dispatch, stored
):
    assert run_download(tmp_path, event, dispatch) is stored


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
    (temp / "run" / "perf-nightly").mkdir(parents=True)
    (temp / "run" / "perf-nightly" / "nightly.json").write_text(
        json.dumps({"date": "2026-11-21", "checks": [{"check": "alpha"}, {"check": "beta"}]})
    )
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
    assert nightly_store.fetch("o/r", "7", "perf-result-alpha", dest, run)
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
    ("listing", "archive", "rc_list", "rc_zip"),
    [
        ({"artifacts": []}, None, 0, 0),
        ({"artifacts": [{"id": 5, "expired": True}]}, None, 0, 0),
        ({}, None, 1, 0),
        ({"artifacts": [{"id": 5}]}, b"half a zip", 0, 0),
        ({"artifacts": [{"id": 5}]}, zipped({"result.json": "{}"}), 0, 1),
        # A member that would land outside fails the whole artifact, as unzip did.
        ({"artifacts": [{"id": 5}]}, zipped({"result.json": "{}", "../evil": "x"}), 0, 0),
        ({"artifacts": [{"id": 5}]}, zipped({"result.json": "{}", "/abs": "x"}), 0, 0),
        ({"artifacts": [{"id": 5}]}, corrupt({"result.json": "x"}), 0, 0),
    ],
)
def test_fetch_that_fails_leaves_no_partial_result(
    tmp_path, no_wait, listing, archive, rc_list, rc_zip
):
    run, _ = fake_run(listing, archive, rc_list, rc_zip)
    dest = tmp_path / "x" / "perf-result-alpha"
    assert not nightly_store.fetch("o/r", "7", "perf-result-alpha", dest, run)
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
