"""micro-hw's verdict and `bench hw`, over CSV fixtures laid out the way `bench hw`
writes them. Each test names the wrong answer it prevents."""

from __future__ import annotations

import importlib.util
import json
from pathlib import Path
from typing import ClassVar

import pytest

from falkorbench import verdict as v
from falkorbench.counters import L2_MISS_EVENT
from falkorbench.metrics import read_rows
from falkorbench.metrics import write_rows
from falkorbench.model import Query


def _load_validator():
    """tests/perf/validate.py, the contract's reference, by path: bench does not
    import tests/perf, but every result written here must pass it."""
    path = Path(__file__).resolve().parents[2] / "tests" / "perf" / "validate.py"
    spec = importlib.util.spec_from_file_location("perf_validate", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


validate = _load_validator()

CAND = "ghcr.io/falkordb/falkordb-server@sha256:" + "a" * 64
BASE = "falkordb/falkordb-server:6.2.0@sha256:" + "b" * 64

QUERIES = [
    Query("RETURN 1", False, "RETURN 1"),
    Query("read a", False, "MATCH (n) RETURN n"),
    Query("write a", True, "CREATE ()"),
    Query("write b", True, "CREATE ()"),
]

COLS = ("instr", "cycles", "br_miss", "l2_miss", "alloc_bytes")


def row(instr, cycles, br, l2, alloc):
    return dict(zip(COLS, (instr, cycles, br, l2, alloc), strict=True))


# One engine's readings per execution, comfortably clear of every floor.
ROWS = {
    "RETURN 1": row(150_000.0, 300_000.0, 500.0, 1_000.0, 1_000.0),
    "read a": row(2_000_000.0, 4_000_000.0, 5_000.0, 20_000.0, 200_000.0),
    "write a": row(5_000_000.0, 10_000_000.0, 8_000.0, 30_000.0, 300_000.0),
    "write b": row(3_000_000.0, 6_000_000.0, 6_000.0, 25_000.0, 50_000.0),
}


def write_side(root: Path, side: str, rows, *, confirm: bool = False) -> None:
    """Write both passes of one side. A list gives each pass its own rows; a
    query left out is one whose warmup errored, a None cell an empty column."""
    root.mkdir(parents=True, exist_ok=True)
    path_of = v.confirm_path if confirm else v.screen_path
    for rep in range(1, v.REPS + 1):
        table = rows[rep - 1] if isinstance(rows, list) else rows
        write_rows(str(path_of(root, side, rep)), list(table.items()), ("query", *COLS))


def inputs(root: Path, queries=QUERIES, engine: str = "rust") -> v.Inputs:
    return v.Inputs(
        queries=queries,
        screen={side: v.load(root, side) for side in v.SIDES},
        confirm={side: v.load_confirm(root, side) for side in v.SIDES},
        metrics=v.hw_metrics(engine),
    )


def run(root: Path, queries=QUERIES) -> dict:
    res, _ = v.build_result(
        inputs(root, queries),
        candidate=v.ImageRef(CAND),
        baseline=v.ImageRef(BASE, "6.2.0"),
        mode="nightly",
        started="2026-11-24T00:00:00Z",
        machine={"runner": "gce", "type": "c4-standard-8", "cpu": "x", "cores": 8},
        finished="2026-11-24T00:30:00Z",
    )
    assert validate.validate_result(res) == []
    return res


def by_status(res: dict, status: str) -> list[dict]:
    return [f for f in res["findings"] if f["status"] == status]


def scaled(rows: dict, query: str, col: str, factor: float) -> dict:
    out = {q: dict(r) for q, r in rows.items()}
    out[query][col] *= factor
    return out


def without(rows: dict, query: str) -> dict:
    return {q: r for q, r in rows.items() if q != query}


def blank(rows: dict, query: str, col: str) -> dict:
    out = {q: dict(r) for q, r in rows.items()}
    out[query][col] = None
    return out


class TestCrossing:
    def test_overlapping_reps_do_not_cross(self):
        """A bimodal row (`multi-type traverse`: 39.25M or 47.19M on one binary)
        must not flag on a median that happens to straddle the modes. The medians
        here are 1.20x apart, so only the overlap holds it back."""
        cand, base = [47.19, 48.0], [32.0, 47.19]
        assert v.ratio_of(cand, base) > 1.1
        assert v.crossing(cand, base, threshold=1.1, floor=0) is None

    def test_clear_regression_crosses(self):
        assert v.crossing([120, 121], [100, 101], threshold=1.1, floor=0) == "regression"

    def test_below_threshold_does_not_cross(self):
        assert v.crossing([109, 109.5], [100, 100.2], threshold=1.1, floor=0) is None

    def test_floor_holds_back_a_small_absolute_change(self):
        """2 KB -> 3 KB is 1.5x of an allocation that is not the query's."""
        assert v.crossing([3000, 3000], [2000, 2000], threshold=1.1, floor=4096) is None
        assert v.crossing([9000, 9000], [2000, 2000], threshold=1.1, floor=4096) == "regression"

    def test_improvement_mirrors_regression(self):
        assert v.crossing([80, 81], [100, 101], threshold=1.1, floor=0) == "improvement"

    @pytest.mark.parametrize(
        ("on", "past", "base", "gate", "status"),
        [
            (110, 111, 100, {"floor": 0}, "regression"),
            (100, 99, 110, {"floor": 0}, "improvement"),
            (150, 151, 100, {"floor": 50}, "regression"),
            (300, 301, 100, {"floor": 1e9, "steep": 3.0, "steep_floor": 10}, "regression"),
            (400, 401, 100, {"floor": 1e9, "steep": 3.0, "steep_floor": 300}, "regression"),
            (50, 49, 100, {"floor": 50}, "improvement"),
            (25, 24, 100, {"floor": 1e9, "steep": 4.0, "steep_floor": 10}, "improvement"),
            (20, 19, 100, {"floor": 1e9, "steep": 4.0, "steep_floor": 80}, "improvement"),
            (50, 51, 0, {"floor": 50}, "regression"),
            (10, 11, 0, {"floor": 1e9, "steep": 3.0, "steep_floor": 10}, "regression"),
        ],
    )
    def test_every_bound_is_strict(self, on, past, base, gate, status):
        """A row exactly on a ratio or a floor does not cross; one step past it
        does (contract v1.12: ratio > threshold and delta > floor)."""
        assert v.crossing([on, on], [base, base], threshold=1.1, **gate) is None
        assert v.crossing([past, past], [base, base], threshold=1.1, **gate) == status

    def test_zero_baseline_crosses_on_the_floor_alone(self):
        """0 -> 100 KB cannot be a ratio, and must not vanish because of that."""
        assert v.crossing([1e5, 1e5], [0, 0], threshold=1.1, floor=4096) == "regression"
        assert v.crossing([10, 10], [0, 0], threshold=1.1, floor=0) is None

    ALLOC: ClassVar = {m.column: m for m in v.hw_metrics("rust")}["alloc_bytes"]

    def test_zero_baseline_is_steep(self):
        """0 -> 50 KB is under the 64 KiB floor but steeper than any ratio."""
        assert self.ALLOC.crossing([5e4, 5e4], [0.0, 0.0]) == "regression"
        assert self.ALLOC.crossing([3e3, 3e3], [0.0, 0.0]) is None

    def test_a_steep_allocation_change_crosses_under_the_floor(self):
        """6.2.0 vs main on the C4 (run 45403484820): `correlated hash join`
        allocates 10.3x more (+18 KB), under the 64 KiB floor. The 8x tier
        catches it."""
        cand, base = [19998.392, 19917.968], [1939.696, 1933.752]
        assert self.ALLOC.crossing(cand, base) == "regression"
        assert self.ALLOC.crossing(base, cand) == "improvement"

    def test_the_steep_rule_needs_its_own_delta(self):
        """A 10x move of a few hundred bytes is an allocation that is not the query's."""
        assert self.ALLOC.crossing([3000.0, 3000.0], [300.0, 300.0]) is None

    def test_hosted_a_a_noise_stays_under_the_allocation_gate(self):
        """The same gate held on hosted runners: `pending degree` read 1.33x
        between two runs of 6.2.0 (45332622044), and `multi-label write` could
        split 5.02x. Neither crosses."""
        assert self.ALLOC.crossing([22675.168, 23249.152], [14030.8, 20566.688]) is None
        assert self.ALLOC.crossing([13438.0, 30682.0], [4361.0, 4421.0]) is None


class TestVerdict:
    def test_identical_images_pass_with_no_regression(self, tmp_path):
        write_side(tmp_path, "candidate", ROWS)
        write_side(tmp_path, "baseline", ROWS)
        res = run(tmp_path)
        assert res["check"] == "micro-hw"
        assert res["verdict"] == "pass"
        assert res["confirmed"] is False
        assert res["error"] is None
        assert by_status(res, "regression") == []
        assert "\n" not in res["summary"]

    def test_confirmed_regression(self, tmp_path):
        slow = scaled(ROWS, "read a", "instr", 3.0)
        for confirm in (False, True):
            write_side(tmp_path, "candidate", slow, confirm=confirm)
            write_side(tmp_path, "baseline", ROWS, confirm=confirm)
        res = run(tmp_path)
        assert res["verdict"] == "regression"
        assert res["confirmed"] is True
        (f,) = by_status(res, "regression")
        assert (f["name"], f["metric"]) == ("read a [reads]", "instructions")
        assert f["ratio"] == pytest.approx(3.0)
        assert res["findings"][0] is f, "worst first"

    def test_crossing_once_is_not_a_regression(self, tmp_path):
        """The contract: a perf regression needs a second run that repeats it."""
        write_side(tmp_path, "candidate", scaled(ROWS, "read a", "instr", 3.0))
        write_side(tmp_path, "baseline", ROWS)
        write_side(tmp_path, "candidate", ROWS, confirm=True)
        write_side(tmp_path, "baseline", ROWS, confirm=True)
        res = run(tmp_path)
        assert res["verdict"] == "pass"
        assert by_status(res, "regression") == []
        assert any(f["name"] == "read a [reads]" for f in by_status(res, "info"))

    def test_crossed_but_never_re_measured_is_an_error(self, tmp_path):
        """No second-run reading must not read as 'not confirmed, so pass'."""
        write_side(tmp_path, "candidate", scaled(ROWS, "read a", "instr", 3.0))
        write_side(tmp_path, "baseline", ROWS)
        res = run(tmp_path)
        assert res["verdict"] == "error"
        assert "re-run" in res["error"]

    def test_a_missing_pass_is_an_error_not_a_smaller_run(self, tmp_path):
        write_side(tmp_path, "candidate", ROWS)
        write_side(tmp_path, "baseline", ROWS)
        (tmp_path / "measure-baseline-r2.csv").unlink()
        res = run(tmp_path)
        assert res["verdict"] == "error"
        assert "measure-baseline-r2.csv" in res["error"]
        assert res["confirmed"] is False

    def test_a_regression_outranks_a_missing_pass(self, tmp_path):
        """Contract v1.4: `error` may not carry a regression finding. The night
        is red either way, and the gap is still said in the summary."""
        slow = scaled(ROWS, "read a", "instr", 3.0)
        for confirm in (False, True):
            write_side(tmp_path, "candidate", slow, confirm=confirm)
            write_side(tmp_path, "baseline", ROWS, confirm=confirm)
        (tmp_path / "measure-baseline-r2.csv").unlink()
        res = run(tmp_path)
        assert res["verdict"] == "regression"
        assert res["confirmed"] is True
        assert res["error"] is None
        assert "could not measure" in res["summary"]
        assert "measure-baseline-r2.csv" in res["summary"]

    def test_query_erroring_on_candidate_only_is_a_regression_at_once(self, tmp_path):
        """A correctness failure needs no second run (contract §1) and must not
        vanish from the comparison as a row nobody measured."""
        write_side(tmp_path, "candidate", without(ROWS, "write b"))
        write_side(tmp_path, "baseline", ROWS)
        res = run(tmp_path)
        assert res["verdict"] == "regression"
        assert res["confirmed"] is True
        (f,) = by_status(res, "regression")
        assert (f["metric"], f["candidate"], f["baseline"]) == ("completed", 0, 1)
        assert f["direction"] == "lower_is_worse"

    def test_query_erroring_on_both_sides_is_information(self, tmp_path):
        write_side(tmp_path, "candidate", without(ROWS, "write b"))
        write_side(tmp_path, "baseline", without(ROWS, "write b"))
        res = run(tmp_path)
        assert res["verdict"] == "pass"
        (f,) = [f for f in by_status(res, "info") if "both sides" in f["name"]]
        assert (f["candidate"], f["baseline"]) == (0, 0), "v1.4: 0 and 0 is info"
        assert "1 queries error on both images" in res["summary"]

    def test_one_row_with_no_reading_is_an_error(self, tmp_path):
        """A counter missing on one side is a gap, not a pass on the rest."""
        write_side(tmp_path, "candidate", blank(ROWS, "read a", "l2_miss"))
        write_side(tmp_path, "baseline", ROWS)
        res = run(tmp_path)
        assert res["verdict"] == "error"
        assert "read a" in res["error"]

    def test_a_row_missing_from_one_rep_is_still_compared(self, tmp_path):
        """One rep is enough to compare; only a side with none is a gap."""
        write_side(tmp_path, "candidate", [ROWS, blank(ROWS, "read a", "l2_miss")])
        write_side(tmp_path, "baseline", ROWS)
        assert run(tmp_path)["verdict"] == "pass"

    def test_an_error_on_some_candidate_passes_is_reported(self, tmp_path):
        """Not a regression (an A/A pair can produce one), but never silent."""
        write_side(tmp_path, "candidate", [ROWS, without(ROWS, "write b")])
        write_side(tmp_path, "baseline", ROWS)
        res = run(tmp_path)
        assert res["verdict"] == "pass"
        (f,) = [f for f in by_status(res, "info") if "some passes" in f["name"]]
        assert (f["metric"], f["candidate"], f["baseline"]) == ("completed", None, 1)

    def test_a_side_with_no_reading_is_an_error(self, tmp_path):
        nothing = {q: {**r, "cycles": None} for q, r in ROWS.items()}
        write_side(tmp_path, "candidate", nothing)
        write_side(tmp_path, "baseline", ROWS)
        res = run(tmp_path)
        assert res["verdict"] == "error"
        assert "no cycles reading for the candidate" in res["error"]

    def test_class_rows_show_a_fixed_cost_no_row_crosses(self, tmp_path):
        """+3% on every read crosses no 10% threshold; the class row shows it."""
        cand = {q: {**r, "cycles": r["cycles"] * 1.03} for q, r in ROWS.items()}
        write_side(tmp_path, "candidate", cand)
        write_side(tmp_path, "baseline", ROWS)
        res = run(tmp_path)
        assert res["verdict"] == "pass"
        reads = next(
            f for f in res["findings"] if f["name"].startswith("reads:") and f["metric"] == "cycles"
        )
        assert reads["ratio"] == pytest.approx(1.03)
        assert "2 up, 0 down" in reads["name"]

    def test_notices_are_never_crowded_out_by_class_rows(self, tmp_path):
        """The contract allows 10 info findings. Rows that say a query errors or
        crossed once and not on the second run come first; the per-class rows
        (6 here) fill what is left, so they are the ones cut."""
        qs = [Query(f"q{i}", i % 2 == 1, "RETURN 1") for i in range(8)]
        base = {q.name: row(1e6, 2e6, 5e3, 2e4, 1e5) for q in qs}
        once = {q: dict(r) for q, r in base.items()}
        for i in range(5):
            once[f"q{i}"]["instr"] *= 3.0  # crosses at screen, not on the second run
        broken = {q: r for q, r in base.items() if q != "q7"}  # errors on both sides
        write_side(tmp_path, "candidate", without(once, "q7"))
        write_side(tmp_path, "baseline", broken)
        write_side(tmp_path, "candidate", broken, confirm=True)
        write_side(tmp_path, "baseline", broken, confirm=True)
        res = run(tmp_path, queries=qs)
        names = [f["name"] for f in by_status(res, "info")]
        for i in range(5):
            assert any(n.startswith(f"q{i} [") for n in names), f"q{i} crossed once"
        assert any("both sides" in n for n in names)
        assert sum(f["status"] in ("ok", "info") for f in res["findings"]) == 10

    def test_improvement_does_not_fail_the_run(self, tmp_path):
        write_side(tmp_path, "candidate", scaled(ROWS, "write a", "instr", 0.5))
        write_side(tmp_path, "baseline", ROWS)
        res = run(tmp_path)
        assert res["verdict"] == "pass"
        assert [f["name"] for f in by_status(res, "improvement")] == ["write a [writes]"]

    def test_contract_limits(self, tmp_path):
        """80 regressions: run() checks the 50 cap; at most 10 ok/info, worst first."""
        qs = [Query(f"q{i}", False, "RETURN 1") for i in range(80)]
        base = {q.name: row(1e6, 2e6, 5e3, 2e4, 1e5) for q in qs}
        cand = {q.name: {**base[q.name], "instr": 1e6 * (2 + i / 100)} for i, q in enumerate(qs)}
        for confirm in (False, True):
            write_side(tmp_path, "candidate", cand, confirm=confirm)
            write_side(tmp_path, "baseline", base, confirm=confirm)
        res = run(tmp_path, queries=qs)
        assert sum(f["status"] in ("ok", "info") for f in res["findings"]) <= 10
        regs = by_status(res, "regression")
        assert regs[0]["name"] == "q79 [reads]", "worst ratio first"
        ratios = [f["ratio"] for f in regs]
        assert ratios == sorted(ratios, reverse=True)

    def test_bad_inputs_are_an_error(self, tmp_path):
        write_side(tmp_path, "candidate", ROWS)
        write_side(tmp_path, "baseline", ROWS)
        res, _ = v.build_result(
            inputs(tmp_path),
            candidate=v.ImageRef("falkordb/falkordb-server:edge"),
            baseline=v.ImageRef(BASE),
            mode="weekly",
            started="2026-11-24T00:00:00Z",
            machine={"runner": "gce", "type": "c4-standard-8", "cpu": "x", "cores": 8},
        )
        assert res["verdict"] == "error"
        assert "digest" in res["error"] and "mode" in res["error"]

    def test_finished_is_never_before_started(self, tmp_path):
        """`started` and `finished` can come from two clocks; a result whose
        finish precedes its start would be invalid."""
        res, _ = v.build_result(
            None,
            candidate=v.ImageRef(CAND),
            baseline=v.ImageRef(BASE),
            mode="nightly",
            started="2999-01-01T00:00:00Z",
            machine={"runner": "gce", "type": "c4-standard-8", "cpu": "x", "cores": 8},
        )
        assert res["finished"] == res["started"]

    def test_no_nan_or_infinity(self, tmp_path):
        """A zero baseline gives ratio null, never inf; run() validates the rest."""
        zero = {q: {**r, "alloc_bytes": 0.0} for q, r in ROWS.items()}
        for confirm in (False, True):
            write_side(tmp_path, "candidate", ROWS, confirm=confirm)
            write_side(tmp_path, "baseline", zero, confirm=confirm)
        res = run(tmp_path)
        assert res["verdict"] == "regression"
        assert {f["ratio"] for f in by_status(res, "regression")} == {None}


class TestFindings:
    def test_a_regression_reports_a_ratio_that_crossed(self):
        """Both runs crossed (2.01x, then 1.50x), but their readings pooled into
        one median read 1.004x, under the 1.10x threshold. The finding reports
        the weaker run that crossed, and candidate / baseline equals its ratio."""
        r = v.RowResult(
            "read a",
            "reads",
            "instructions",
            cand=[101e6, 102e6],
            base=[1e6, 100e6],
            confirm_cand=[1002e6, 2000e6],
            confirm_base=[1000e6, 1001e6],
            status="regression",
            threshold=1.1,
        )
        f = v.row_finding(r)
        assert f["ratio"] == pytest.approx(1501e6 / 1000.5e6, rel=1e-6)
        assert f["ratio"] > f["threshold"]
        assert f["candidate"] / f["baseline"] == pytest.approx(f["ratio"])

    def test_other_findings_report_the_screen(self):
        r = v.RowResult(
            "read a", "reads", "instructions", cand=[3e6, 3e6], base=[1e6, 1e6], status="info"
        )
        r.confirm_cand, r.confirm_base = [1e6, 1e6], [1e6, 1e6]
        assert v.row_finding(r)["ratio"] == pytest.approx(3.0)


class TestLabels:
    def test_default_label_is_repo_and_twelve_hex(self):
        assert v.default_label(BASE) == "falkordb/falkordb-server@" + "b" * 12
        assert v.ImageRef(CAND).as_json() == {
            "image": CAND,
            "commit": None,
            "label": "ghcr.io/falkordb/falkordb-server@" + "a" * 12,
        }

    def test_malformed_commit_and_blank_label_keep_the_result_valid(self):
        """The bad value is reported in `error`, never written where the
        contract wants 40 hex or a non-empty label."""
        ref = v.ImageRef(CAND, "  ", "4c10889")
        assert ref.as_json()["commit"] is None
        assert ref.as_json()["label"] == v.default_label(CAND)
        assert any("commit" in e for e in v.input_errors(ref, v.ImageRef(BASE), "nightly"))

    def test_image_and_commit_rules_are_the_validators(self):
        assert v.IMAGE_RE.pattern == validate.IMAGE_RE.pattern
        assert v.COMMIT_RE.pattern == validate.COMMIT_RE.pattern

    @pytest.mark.parametrize(
        ("image", "commit", "says"),
        [
            (CAND.replace("falkordb/", "FalkorDB/"), None, "digest"),
            (CAND + "\n", None, "digest"),
            (CAND, "4c1088949a5852e3949c05bfb3d92a603c4cd4ca\n", "commit"),
        ],
    )
    def test_what_the_validator_rejects_is_an_input_error(self, image, commit, says):
        """An uppercase repo or a trailing newline passed the old `^...$` rule,
        so the check went on and wrote a result the validator rejects. Now it is
        an input error, and a bad commit is written as null."""
        ref = v.ImageRef(image, commit=commit)
        assert any(says in e for e in v.input_errors(ref, v.ImageRef(BASE), "nightly"))
        assert ref.as_json()["commit"] is None

    def test_given_label_and_commit_win(self):
        ref = v.ImageRef(CAND, "main@4c10889", "4c1088949a5852e3949c05bfb3d92a603c4cd4ca")
        assert ref.as_json()["label"] == "main@4c10889"
        assert ref.as_json()["commit"] == "4c1088949a5852e3949c05bfb3d92a603c4cd4ca"


class FakeDocker:
    """Stands in for docker on a C4 VM: copies out a Rust or C module, answers the
    perf probe, and writes the CSV a `bench measure` pass would."""

    def __init__(
        self, raw: Path, readings, *, c_images=(), counters=True, pull_fails=(), crash=(), blank=()
    ):
        self.raw = raw
        self.readings = readings  # (side, confirm) -> {query: {col: value}}
        self.c_images = set(c_images)
        self.counters = counters
        self.pull_fails = set(pull_fails)
        self.crash = set(crash)  # CSV names whose pass dies (exit 139, no CSV)
        self.blank = set(blank)  # CSV names written with no counter columns
        self.calls: list[list[str]] = []
        self.created: dict[str, str] = {}

    def __call__(self, argv):
        import subprocess

        argv = list(argv)
        self.calls.append(argv)
        ok = subprocess.CompletedProcess(argv, 0, "", "")
        if argv[:2] == ["docker", "pull"]:
            return subprocess.CompletedProcess(
                argv, 1 if argv[-1] in self.pull_fails else 0, "", ""
            )
        if argv[:2] == ["docker", "create"]:
            cid = f"cid{len(self.created)}"
            self.created[cid] = argv[2]
            return subprocess.CompletedProcess(argv, 0, cid + "\n", "")
        if argv[:2] == ["docker", "cp"]:
            image = self.created[argv[2].split(":", 1)[0]]
            body = b"\x7fELF ... " + (b"" if image in self.c_images else b"/rustc/abc/library")
            Path(argv[3]).write_bytes(body)
            return ok
        if "perf" in argv and "stat" in argv:
            stat = argv.index("stat")  # perf's -e, not docker's
            event = argv[argv.index("-e", stat) + 1]
            if event == L2_MISS_EVENT:
                # The L2 event in a call of its own; "no-l2" is a PMU that
                # does not count it, "zero" one whose filter makes it read 0.
                counts = self.counters in (True, "no-branch", "zero-branch")
                if self.counters == "zero":
                    return subprocess.CompletedProcess(argv, 0, "", f"0,,{event},1,100.00,,\n")
                line = (
                    f"20,,{event},1,100.00,,\n"
                    if counts
                    else (f"<not supported>,,{event},0,100.00,,\n")
                )
                return subprocess.CompletedProcess(argv, 0, "", line)
            full = (
                "1000,,instructions,1,100.00,,\n2000,,cycles,1,100.00,,\n"
                "30,,branch-misses,1,100.00,,\n"
            )
            err = {
                True: full,
                "no-l2": full,
                "zero": full,
                "no-branch": full.replace(
                    "30,,branch-misses,1", "<not supported>,,branch-misses,0"
                ),
                "zero-branch": full.replace("30,,branch-misses,1", "0,,branch-misses,1"),
                False: "<not supported>,,instructions,0,100.00,,\n"
                "<not supported>,,cycles,0,100.00,,\n",
            }[self.counters]
            return subprocess.CompletedProcess(argv, 0, "", err)
        if "measure" in argv:
            side = Path(argv[argv.index("--module") + 1]).stem
            name = Path(argv[argv.index("--out") + 1]).name
            if name in self.crash:
                return subprocess.CompletedProcess(argv, 139, "", "Segmentation fault")
            rows = self.readings[(side, name.startswith("confirm-"))]
            cols = ("query", "instr", "cycles", "br_miss", "l2_miss", "alloc_bytes")
            out = self.raw / name
            if name in self.blank:
                rows = {q: {"alloc_bytes": r["alloc_bytes"]} for q, r in rows.items()}
            # Like bench measure: merged into a CSV that already exists.
            merged = read_rows(str(out)) if out.exists() else {}
            merged.update({q: dict(r) for q, r in rows.items()})
            write_rows(str(out), list(merged.items()), cols)
            return ok
        return ok


HW_ROWS = {
    q.name: {"instr": 1e6, "cycles": 2e6, "br_miss": 5e3, "l2_miss": 4e3, "alloc_bytes": 1e5}
    for q in QUERIES
}


def hw_run(tmp_path, docker, *, cand=CAND, base=BASE, server_cpus="0-1,4-5"):
    from falkorbench import hw

    return hw.HwRun(
        bench_image="ghcr.io/falkordb/falkordb-bench@sha256:" + "c" * 64,
        candidate=v.ImageRef(cand),
        baseline=v.ImageRef(base),
        mode="nightly",
        out=tmp_path / "out",
        repo_root=tmp_path,
        server_cpus=server_cpus,
        queries=QUERIES,
        runner=docker,
        echo=lambda _: None,
    )


def measure_calls(docker):
    out = []
    for argv in docker.calls:
        if "measure" in argv:
            out.append((Path(argv[argv.index("--out") + 1]).name, "--c-compat" in argv, argv))
    return out


class TestHw:
    @pytest.fixture(autouse=True)
    def _every_result_passes_the_validator(self, monkeypatch):
        """`bench hw`'s own path (its machine, artifacts and errors) is checked
        too; an image the validator rejects cannot be written validly."""
        from falkorbench import hw

        real = hw.run_check

        def checked(run, **kwargs):
            res = real(run, **kwargs)
            if all(v.IMAGE_RE.fullmatch(r.image) for r in (run.candidate, run.baseline)):
                assert validate.validate_result(res) == []
            return res

        monkeypatch.setattr(hw, "run_check", checked)

    def readings(self, cand=None, cand_confirm=None, base=None):
        return {
            ("candidate", False): cand or base or HW_ROWS,
            ("baseline", False): base or HW_ROWS,
            ("candidate", True): cand_confirm or cand or base or HW_ROWS,
            ("baseline", True): base or HW_ROWS,
        }

    def test_identical_images_pass_in_the_interleaved_order(self, tmp_path):
        from falkorbench import hw

        docker = FakeDocker(tmp_path / "out" / "raw", self.readings())
        res = hw.run_check(hw_run(tmp_path, docker), started="2026-11-24T00:00:00Z")
        assert res["check"] == "micro-hw"
        assert res["verdict"] == "pass", res
        calls = measure_calls(docker)
        assert [c[0] for c in calls] == [
            "measure-candidate-r1.csv",
            "measure-baseline-r1.csv",
            "measure-baseline-r2.csv",
            "measure-candidate-r2.csv",
        ], "candidate, baseline, baseline, candidate; no re-run when nothing crossed"
        for _, c_compat, argv in calls:
            assert "--privileged" in argv, "seccomp blocks perf_event_open otherwise"
            # Pinned, so the VM's size does not change what a row allocates.
            assert "OMP_NUM_THREADS=4" in argv
            assert argv[argv.index("THREAD_COUNT") + 1 :][:2] == ["--module-args", "4"]
            assert argv[argv.index("--cpuset-cpus") + 1] == "0-1,4-5"
            assert not c_compat
        # Class rows for the cost metrics; every metric is judged (raw/rows.csv).
        assert {f["metric"] for f in res["findings"]} == {"instructions", "cycles", "alloc_bytes"}
        judged = (tmp_path / "out" / "raw" / "rows.csv").read_text()
        for metric in ("branch_misses", "l2_misses", "alloc_bytes"):
            assert f",{metric}," in judged
        assert (tmp_path / "out" / "result.json").is_file()
        assert not (tmp_path / "out" / "raw" / "candidate.so").exists(), "modules are not results"

    def test_a_cycles_regression_is_re_run_baseline_first(self, tmp_path):
        from falkorbench import hw

        slow = {q: dict(r) for q, r in HW_ROWS.items()}
        slow["read a"]["cycles"] *= 2.0  # past the 1.8x steep tier
        docker = FakeDocker(tmp_path / "out" / "raw", self.readings(slow))
        res = hw.run_check(hw_run(tmp_path, docker))
        assert res["verdict"] == "regression"
        (f,) = by_status(res, "regression")
        assert (f["name"], f["metric"], f["threshold"]) == ("read a [reads]", "cycles", 1.1)
        assert [c[0] for c in measure_calls(docker)][4:] == [
            "confirm-measure-baseline-r1.csv",
            "confirm-measure-candidate-r1.csv",
            "confirm-measure-candidate-r2.csv",
            "confirm-measure-baseline-r2.csv",
        ]

    @pytest.mark.parametrize(
        ("col", "base", "factor", "verdict"),
        [
            # Branch misses: +25% past the 12,000 floor, or 2.6x past 2,000.
            ("br_miss", 100_000, 1.2, "pass"),  # +20,000 past the floor, 1.2x: under 25%
            ("br_miss", 50_000, 1.3, "regression"),  # +15,000, 1.3x: floor tier
            ("br_miss", 5_000, 2.7, "pass"),  # +8,500: under the floor, under 2.8x
            ("br_miss", 5_000, 3.0, "regression"),  # +10,000, 3x: steep tier
            # L2 misses: +25% past 225,000, or 3x past 15,000.
            ("l2_miss", 2_000_000, 1.2, "pass"),  # +400,000 past the floor, 1.2x
            ("l2_miss", 1_000_000, 1.3, "regression"),  # +300,000, 1.3x: floor tier
            ("l2_miss", 10_000, 2.8, "pass"),  # +18,000, under 3x
            ("l2_miss", 10_000, 4.0, "regression"),  # +30,000, 4x: steep tier
            # Cycles: +10% past 20,000,000, or 1.8x past 250,000.
            ("cycles", 100_000_000, 1.12, "pass"),  # +12M, under the floor and 1.8x
            ("cycles", 100_000_000, 1.25, "regression"),  # +25M, 1.25x: floor tier
            ("cycles", 1_000_000, 1.7, "pass"),  # +700,000, under 1.8x
            ("cycles", 1_000_000, 1.9, "regression"),  # +900,000, 1.9x: steep tier
            # Allocated bytes: +10% past 64 KiB, or 8x past 4 KiB.
            ("alloc_bytes", 500_000, 1.12, "pass"),  # +60,000: under the floor
            ("alloc_bytes", 500_000, 1.14, "regression"),  # +70,000, 1.14x: floor tier
            ("alloc_bytes", 1_000_000, 1.08, "pass"),  # +80,000 past the floor, under 10%
            ("alloc_bytes", 5_000, 7.5, "pass"),  # +32,500, under 8x
            ("alloc_bytes", 5_000, 9.0, "regression"),  # +40,000, 9x: steep tier
            ("alloc_bytes", 400, 9.0, "pass"),  # +3,200, 9x but under 4 KiB
            ("alloc_bytes", 600, 9.0, "regression"),  # +4,800, 9x: steep tier
            # Instructions: +10% past 250,000.
            ("instr", 5_000_000, 1.08, "pass"),  # +400,000 past the floor, under 10%
            ("instr", 3_000_000, 1.15, "regression"),  # +450,000, 1.15x
        ],
    )
    def test_each_counter_gate_from_both_sides(self, tmp_path, col, base, factor, verdict):
        """Each threshold, floor and steep tier, just under and just over."""
        from falkorbench import hw

        rows = {q: {**r, col: float(base)} for q, r in HW_ROWS.items()}
        moved = {q: dict(r) for q, r in rows.items()}
        moved["read a"][col] = base * factor
        docker = FakeDocker(tmp_path / "out" / "raw", self.readings(moved, base=rows))
        assert hw.run_check(hw_run(tmp_path, docker))["verdict"] == verdict

    def test_a_c_run_is_judged_on_the_c_gates(self, tmp_path):
        """+600,000 instructions at 3x is C busy-wait noise (its worst A/A split
        was +629,577 at 3.66x) and a Rust regression. The same readings must
        pass on two C images and fail on two Rust ones."""
        from falkorbench import hw

        rows = {q: {**r, "instr": 300_000.0} for q, r in HW_ROWS.items()}
        moved = {q: dict(r) for q, r in rows.items()}
        moved["read a"]["instr"] = 900_000.0
        for c_images, verdict in (((CAND, BASE), "pass"), ((), "regression")):
            out = tmp_path / ("c" if c_images else "rust")
            docker = FakeDocker(
                out / "out" / "raw", self.readings(moved, base=rows), c_images=c_images
            )
            assert hw.run_check(hw_run(out, docker))["verdict"] == verdict, c_images

    @pytest.mark.parametrize(
        ("mode", "says"),
        [
            ("zero", "does not count l2_rqsts"),
            ("no-branch", "does not count branch-misses"),
            ("zero-branch", "does not count branch-misses"),
        ],
    )
    def test_a_counter_reading_zero_or_missing_stops_the_probe(self, tmp_path, mode, says):
        """A PMU filter makes a blocked event read 0, not fail. That must not
        leave a metric gated on zeros that can never cross."""
        from falkorbench import hw

        docker = FakeDocker(tmp_path / "out" / "raw", self.readings(), counters=mode)
        res = hw.run_check(hw_run(tmp_path, docker))
        assert res["verdict"] == "error"
        assert says in res["error"]
        assert measure_calls(docker) == []

    def test_a_pass_whose_counter_is_all_zero_is_an_error(self, tmp_path):
        from falkorbench import hw

        zero = {q: {**r, "l2_miss": 0.0} for q, r in HW_ROWS.items()}
        docker = FakeDocker(tmp_path / "out" / "raw", self.readings(zero))
        res = hw.run_check(hw_run(tmp_path, docker))
        assert res["verdict"] == "error"
        assert "counted no l2_miss" in res["error"]

    def test_a_c_module_puts_both_sides_on_c_compat(self, tmp_path):
        """Otherwise the two sides build different graphs."""
        from falkorbench import hw

        docker = FakeDocker(tmp_path / "out" / "raw", self.readings(), c_images={BASE})
        hw.run_check(hw_run(tmp_path, docker))
        calls = measure_calls(docker)
        assert calls and all(c_compat for _, c_compat, _ in calls)

    def test_no_counters_is_an_error_before_any_pass(self, tmp_path):
        """A VM without a PMU must never read as nothing moved."""
        from falkorbench import hw

        docker = FakeDocker(tmp_path / "out" / "raw", self.readings(), counters=False)
        res = hw.run_check(hw_run(tmp_path, docker))
        assert res["verdict"] == "error"
        assert "no hardware counters" in res["error"]
        assert measure_calls(docker) == []

    def test_no_l2_event_is_an_error_before_any_pass(self, tmp_path):
        """No L2-miss candidate counting must stop the run at the probe, naming
        what it tried, not one pass later."""
        from falkorbench import hw

        docker = FakeDocker(tmp_path / "out" / "raw", self.readings(), counters="no-l2")
        res = hw.run_check(hw_run(tmp_path, docker))
        assert res["verdict"] == "error"
        assert "does not count l2_rqsts.all_demand_miss" in res["error"]
        assert "never substitutes" in res["error"]
        assert measure_calls(docker) == []

    def test_the_l2_event_that_counts_goes_to_every_pass(self, tmp_path):
        from falkorbench import hw

        docker = FakeDocker(tmp_path / "out" / "raw", self.readings())
        hw.run_check(hw_run(tmp_path, docker))
        chosen = f"BENCH_L2_EVENT={L2_MISS_EVENT}"
        calls = measure_calls(docker)
        assert calls and all(chosen in argv for _, _, argv in calls)
        notes = json.loads((tmp_path / "out" / "raw" / "hw.json").read_text())
        assert notes["l2_event"] == L2_MISS_EVENT == "l2_rqsts.all_demand_miss"

    def test_a_relative_out_is_resolved_where_the_check_starts(self, tmp_path, monkeypatch):
        """_perf-vm.yml passes `--out out` (contract v1.8). Resolved at start, it
        cannot be read against another directory later: not by the docker mount,
        and not if something changes directory before result.json is written."""
        from falkorbench import hw

        start, elsewhere = tmp_path / "checkout", tmp_path / "elsewhere"
        start.mkdir()
        elsewhere.mkdir()
        monkeypatch.chdir(start)
        docker = FakeDocker(start / "out" / "raw", self.readings())
        run = hw.HwRun(
            bench_image="ghcr.io/falkordb/falkordb-bench@sha256:" + "c" * 64,
            candidate=v.ImageRef(CAND),
            baseline=v.ImageRef(BASE),
            mode="nightly",
            out=Path("out"),
            repo_root=Path("."),
            queries=QUERIES,
            runner=docker,
            echo=lambda _: None,
        )
        monkeypatch.chdir(elsewhere)
        assert hw.run_check(run)["verdict"] == "pass"
        assert (start / "out" / "result.json").is_file()
        assert not (elsewhere / "out").exists()
        mounts = [a[a.index("-v") + 1 :] for a in docker.calls if "-v" in a]
        assert mounts and all(m[0].startswith("/") and m[2].startswith("/") for m in mounts)

    def test_a_failed_pull_still_writes_an_error_result(self, tmp_path, monkeypatch):
        from falkorbench import hw

        monkeypatch.setattr(hw.time, "sleep", lambda _: None)
        docker = FakeDocker(tmp_path / "out" / "raw", self.readings(), pull_fails={CAND})
        res = hw.run_check(hw_run(tmp_path, docker))
        assert res["verdict"] == "error"
        assert "could not pull" in res["error"]
        assert sum(1 for a in docker.calls if a[:2] == ["docker", "pull"] and a[-1] == CAND) == 3

    def test_an_unpinned_image_is_never_pulled(self, tmp_path):
        from falkorbench import hw

        docker = FakeDocker(tmp_path / "out" / "raw", self.readings())
        res = hw.run_check(hw_run(tmp_path, docker, cand="falkordb/falkordb-server:edge"))
        assert res["verdict"] == "error"
        assert "digest" in res["error"]
        assert docker.calls == []

    def test_a_failed_re_run_keeps_what_the_screen_found(self, tmp_path):
        """A correctness failure needs no re-run, and a regression outranks an
        error (contract v1.4): a crash in the re-run must not erase it."""
        from falkorbench import hw

        broken = {q: dict(r) for q, r in HW_ROWS.items() if q != "write b"}
        broken["read a"]["cycles"] *= 2.0  # crosses the 1.8x steep tier, so the re-run happens
        docker = FakeDocker(
            tmp_path / "out" / "raw",
            self.readings(broken),
            crash={"confirm-measure-candidate-r2.csv"},
        )
        res = hw.run_check(hw_run(tmp_path, docker))
        assert res["verdict"] == "regression"
        assert any(f["metric"] == "completed" for f in by_status(res, "regression"))
        assert "confirm-measure-candidate-r2.csv" in res["summary"]

    def test_a_reused_out_never_reads_an_earlier_runs_rows(self, tmp_path):
        """bench measure merges into an existing CSV. Left in place, a row from
        an earlier run would stand in for a query that errors now."""
        from falkorbench import hw

        raw = tmp_path / "out" / "raw"
        write_side(raw, "candidate", ROWS)  # the earlier run's files
        broken = {q: dict(r) for q, r in HW_ROWS.items() if q != "write b"}
        docker = FakeDocker(raw, self.readings(broken))
        res = hw.run_check(hw_run(tmp_path, docker))
        assert res["verdict"] == "regression"
        (f,) = by_status(res, "regression")
        assert (f["name"], f["metric"]) == ("write b [errors on candidate only]", "completed")

    def test_a_late_failed_pass_keeps_the_second_run_it_finished(self, tmp_path):
        """The last second-run pass dies; the three before it still repeat the
        regression, which outranks the error."""
        from falkorbench import hw

        slow = {q: dict(r) for q, r in HW_ROWS.items()}
        slow["read a"]["cycles"] *= 2.0
        docker = FakeDocker(
            tmp_path / "out" / "raw",
            self.readings(slow),
            crash={"confirm-measure-baseline-r2.csv"},
        )
        res = hw.run_check(hw_run(tmp_path, docker))
        assert res["verdict"] == "regression"
        (f,) = by_status(res, "regression")
        assert (f["name"], f["metric"]) == ("read a [reads]", "cycles")
        assert "confirm-measure-baseline-r2.csv" in res["summary"]

    def test_one_pass_that_counted_nothing_is_an_error(self, tmp_path):
        """bench measure falls back to no counters by itself; one such pass
        among four must not leave a pass standing on the other three."""
        from falkorbench import hw

        docker = FakeDocker(
            tmp_path / "out" / "raw", self.readings(), blank={"measure-baseline-r2.csv"}
        )
        res = hw.run_check(hw_run(tmp_path, docker))
        assert res["verdict"] == "error"
        assert "measure-baseline-r2.csv counted no instr, cycles, br_miss, l2_miss" in res["error"]


class TestMachine:
    def test_a_vm_reports_its_own_machine(self, tmp_path, monkeypatch):
        """On a _perf-vm.yml VM the result names the VM from $PERF_MACHINE_JSON."""
        from falkorbench import hw

        vm = {"runner": "gce", "type": "c4-standard-8", "cpu": "Intel Xeon 8581C", "cores": 8}
        (tmp_path / "machine.json").write_text(json.dumps(vm))
        monkeypatch.setenv("PERF_MACHINE_JSON", str(tmp_path / "machine.json"))
        assert v.vm_machine() == vm
        assert hw.machine() == vm


def worst_split(readings):
    """The worst draw the non-overlap rule can see in four same-image readings:
    the top two as candidate, the bottom two as baseline."""
    s = sorted(readings)
    return s[2:], s[:2]


class TestHwGates:
    """micro-hw's gates, pinned to the readings of the A/A acceptance pairs (runs
    45376924910 / 45376924661 Rust, 45376924684 C) and of 6.2.0 vs main
    (45376924693). Each A/A case is the worst split its four readings allow."""

    RUST: ClassVar[dict] = {m.column: m for m in v.hw_metrics("rust")}
    C: ClassVar[dict] = {m.column: m for m in v.hw_metrics("c")}

    @pytest.mark.parametrize(
        ("col", "readings", "why"),
        [
            (
                "cycles",
                [97941449, 89749005, 95001550, 83630664],
                "labelPropagation all types 1.113x +9.8M",
            ),
            ("cycles", [465994, 396697, 383122, 540193], "write 1 1.29x +113k"),
            ("instr", [314789, 209379, 312750, 210921], "edge index lt range 1.49x, by pass"),
            ("br_miss", [21563, 25640, 26708, 19588], "maxFlow KNOWS 1.27x"),
            ("l2_miss", [271527, 183486, 224723, 89758], "labelPropagation all types 1.82x"),
            ("l2_miss", [7975, 25903, 17166, 9235], "cartesian 3 branches 2.50x +12.9k"),
            (
                "alloc_bytes",
                [37971.6, 20238.224, 27668.4, 10091.824],
                "create delete indexed 2.16x",
            ),
            ("alloc_bytes", [5445.536, 13941.232, 3693.808, 3706.576], "delete path 2.62x +6.0k"),
        ],
    )
    def test_rust_a_a_worst_splits_do_not_cross(self, col, readings, why):
        assert self.RUST[col].crossing(*worst_split(readings)) is None, why

    @pytest.mark.parametrize(
        ("col", "readings", "why"),
        [
            ("instr", [866391, 237470, 866779, 236544], "cross product filter 3.66x, busy-wait"),
            ("cycles", [312896, 197933, 352246, 197523], "cross product filter 1.68x"),
            ("br_miss", [15766, 3201, 14759, 5328], "algo.pageRank 3.58x"),
            ("l2_miss", [1101985, 807596, 593983, 651057], "algo.labelPropagation 1.53x"),
            ("alloc_bytes", [15030224.0, 16835062.8, 17879849.2, 16610609.2], "betweenness 1.097x"),
        ],
    )
    def test_c_a_a_worst_splits_do_not_cross(self, col, readings, why):
        assert self.C[col].crossing(*worst_split(readings)) is None, why

    @pytest.mark.parametrize(
        ("row", "cand", "base"),
        [
            ("UNWIND id MATCH edge 1 row", [7030370, 7034448], [731005, 715138]),
            ("UNWIND id MATCH edge 100 rows", [7963267, 7991593], [1619038, 1638279]),
            ("UNWIND index MATCH edge 1 row", [3128526, 3146044], [717949, 726185]),
        ],
    )
    def test_6_2_0_vs_main_crosses_on_cycles(self, row, cand, base):
        """The #3963 rows, 4.3x-9.9x and +2.4M-6.3M cycles: under the 20M floor,
        so only the steep tier can flag them."""
        assert self.RUST["cycles"].crossing(cand, base) == "regression", row

    def test_the_branch_miss_steep_tier_from_both_sides(self):
        """2.8x over 2,000. Under it: main against itself (WCC typed, 1.716x,
        +2,385, reps apart; run 45403484617) and 6.2.0's UNWIND index MATCH
        edge 1 row at 2.72x (45403484820), which still flags on cycles,
        instructions and L2. Over it: the same row at 2.9x."""
        br = self.RUST["br_miss"]
        assert br.crossing([3684.0, 7749.0], [3135.0, 3528.0]) is None
        assert br.crossing([7264.0, 7349.0], [2644.0, 2728.0]) is None
        assert br.crossing([7800.0, 7900.0], [2644.0, 2728.0]) == "regression"

    def test_the_c_allocation_floor_from_both_sides(self):
        """C's `betweenness default` moved 1.097x (+1.5 MB) between two runs of
        one image, with no margin under 1.10x; the 3.1 MB floor gives it 2x. A
        slightly noisier night (1.15x, +2.4 MB) does not cross; 1.2x over +3.2 MB
        does."""
        alloc = self.C["alloc_bytes"]
        base = [16_000_000.0, 16_100_000.0]
        assert alloc.crossing([18_400_000.0, 18_500_000.0], base) is None
        assert alloc.crossing([19_300_000.0, 19_400_000.0], base) == "regression"
        # The steep tier, the same as Rust's: 8x over 4 KiB.
        assert alloc.crossing([37_500.0, 37_500.0], [5_000.0, 5_000.0]) is None  # 7.5x
        assert alloc.crossing([45_000.0, 45_000.0], [5_000.0, 5_000.0]) == "regression"
        assert alloc.crossing([3_600.0, 3_600.0], [400.0, 400.0]) is None  # 9x, +3,200

    def test_the_100_row_index_form_stays_at_parity(self):
        cand, base = [4699308, 4796160], [4773313, 4793680]
        assert self.RUST["cycles"].crossing(cand, base) is None

    # The worst A/A ratio of a split at least each steep tier's floor (README).
    STEEP_BASIS: ClassVar[dict] = {
        ("rust", "cycles"): 1.113,
        ("rust", "br_miss"): 1.716,  # WCC typed, final acceptance run's main A/A
        ("rust", "l2_miss"): 1.816,
        ("c", "l2_miss"): 1.534,
        ("rust", "alloc_bytes"): 2.620,  # delete path, +5,993 (45376924661 second run)
        ("c", "alloc_bytes"): 1.097,  # betweenness default, +1,537,039 (45376924684)
    }

    def test_every_steep_tier_sits_well_above_its_basis(self):
        """At least 1.6x, the allocation rule's footing (8x over 5.02x). The
        cycles tier was 1.25x over 1.113x (1.12x) until this was checked."""
        tiers = {
            (engine, col): steep
            for engine, gates in v.HW_GATES.items()
            for col, (_, steep, _) in gates.items()
            if steep is not None
        }
        assert set(tiers) == set(self.STEEP_BASIS), "a steep tier with no recorded basis"
        for key, steep in tiers.items():
            assert steep / self.STEEP_BASIS[key] >= 1.6, key

    def test_c_gates_are_chosen_when_a_side_is_c(self, tmp_path):
        run = hw_run(tmp_path, FakeDocker(tmp_path, {}))
        assert run.metrics == v.hw_metrics("rust")
        run.c_compat = True
        assert run.metrics == v.hw_metrics("c")

    def test_an_unreadable_machine_file_falls_back(self, tmp_path, monkeypatch):
        """A $PERF_MACHINE_JSON that is not JSON describes no machine; the
        result still gets one instead of failing over it."""
        (tmp_path / "machine.json").write_text("{not json")
        monkeypatch.setenv("PERF_MACHINE_JSON", str(tmp_path / "machine.json"))
        assert v.vm_machine() is None

    def test_no_lscpu_still_names_the_host(self, monkeypatch):
        from falkorbench import hw

        def no_lscpu(*_a, **_k):
            raise OSError("lscpu: not found")

        monkeypatch.delenv("PERF_MACHINE_JSON", raising=False)
        monkeypatch.setattr(hw.subprocess, "run", no_lscpu)
        m = hw.machine()
        assert m["runner"] == "local"
        assert m["cpu"] and m["cores"] >= 1
