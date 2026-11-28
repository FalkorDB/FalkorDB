"""From runs to findings: the 0/1 checks, the one rule for confirming a break on a repeat, and
the graph-build timing (README, "How a check becomes a finding"). Pure: no server."""

from __future__ import annotations

import statistics

from falkorbench import kit

from .checks import DATA_CHECKS
from .checks import TIMING_CHECKS

CONTROL_CHECKS = (*TIMING_CHECKS, *DATA_CHECKS)
SERVER_CHECK = "server_stays_up"
BUILD_THRESHOLD = 1.5
#: (finding name, key in a build run, minimum difference in ms for a regression or improvement)
BUILD_METRICS = (
    ("build edge phase", "edges_ms", 500.0),
    ("build node phase", "nodes_ms", 500.0),
    ("build edge query 1 row", "probe_1_ms", 2.0),
    ("build edge query 100 rows", "probe_100_ms", 2.0),
)


def judge_repeat(first_c, first_b, rep_c, rep_b) -> tuple[int | None, int | None, str]:
    """The one rule for a timing break (control epoch, long-mode seed, repro): it counts only if it
    repeats. Values are 1 held, 0 broke, None not measured; rep_* are the repeat's (None when no
    repeat ran). Returns (candidate, baseline, status).

    A candidate break (first 0) needs the repeat, whatever the baseline did (#4065):
      the repeat did not run or measure     -> (None, b)    not judged: info
      the candidate breaks again            -> (0, b)       regression
      the candidate holds on the repeat     -> (1, b)       info (it broke once)
    b is 0 when the baseline broke in the first runs or on the repeat, so the release gate's older
    rule (kit.introduced on these values) counts the break only when the baseline did not break
    it either time, and an unmeasured baseline blocks it, as the older "not judged" error did.
    Without a candidate break the first runs decide (kit.correctness_status)."""
    if first_c != 0:
        return first_c, first_b, kit.correctness_status(first_c, first_b)
    base = 0 if 0 in (first_b, rep_b) else first_b
    if rep_c is None:
        return None, base, "info"
    if rep_c == 0:
        return 0, base, "regression"
    return 1, base, "info"



def side_value(values: list) -> int | None:
    """0 if any run broke, 1 if every run held, None if a run did not measure it."""
    if any(v is False for v in values):
        return 0
    if values and all(v is True for v in values):
        return 1
    return None


NA = "n/a"  # a check that does not apply to an epoch (None: it applied but was not measured)


def epoch_values(checks_: dict, names, na=None) -> dict:
    """{check: True / False / None, or `na` when it does not apply} from an epoch's checks."""
    out = {}
    for c in names:
        v = checks_.get(c)
        if v is not None and v.get("applies") is False:
            out[c] = na
        else:
            out[c] = v.get("pass") if v is not None and v.get("pass") in (True, False) else None
    return out


def run_checks(run: dict) -> dict:
    """{check: True/False/None} for one control run (None: not measured or does not apply)."""
    if run.get("started") is False:
        return {
            **{c: None for c in CONTROL_CHECKS},
            SERVER_CHECK: False if run.get("server_died") else None,
        }
    cks = (run.get("epoch") or {}).get("checks", {})
    out = epoch_values(cks, CONTROL_CHECKS)
    if run.get("server_died"):
        out[SERVER_CHECK] = False
    elif "server_stays_up" in cks:
        out[SERVER_CHECK] = bool(cks["server_stays_up"].get("pass"))
    else:
        out[SERVER_CHECK] = None
    return out


def by_side(runs: list[dict], check: str, first_only: bool = False) -> dict[str, list]:
    out = {side: [] for side in kit.SIDES}
    for run in runs:
        if first_only and run.get("extra"):
            continue
        out[run["side"]].append(run_checks(run)[check])
    return out


def needs_confirmation(runs: list[dict]) -> list[str]:
    """Timing checks the candidate broke, whatever the baseline did (#4065)."""
    return [c for c in TIMING_CHECKS if side_value(by_side(runs, c)["candidate"]) == 0]


def control_findings(runs: list[dict], confirmed_pair: bool) -> tuple[list[dict], list[str]]:
    """Findings for the control epoch, and notes for the summary. `confirmed_pair`: an extra
    (baseline, candidate) pair ran after the first runs (runs with "extra": True).

    Timing checks are judged on the first runs, the extra pair only deciding (judge_repeat) whether
    a break repeats. Data checks and the server count every run."""
    findings, notes = [], []
    first = [r for r in runs if not r.get("extra")]
    extra = [r for r in runs if r.get("extra")]
    for c in CONTROL_CHECKS:
        name = f"control {c}"
        allv, firstv = by_side(runs, c), by_side(first, c)
        if c not in TIMING_CHECKS:
            cand, base = side_value(allv["candidate"]), side_value(allv["baseline"])
            findings.append(
                _invariant(name, "invariant", cand, base, kit.correctness_status(cand, base))
            )
            continue
        first_c, first_b = side_value(firstv["candidate"]), side_value(firstv["baseline"])
        ext = by_side(extra, c) if confirmed_pair else {"candidate": [], "baseline": []}
        rep_c, rep_b = side_value(ext["candidate"]), side_value(ext["baseline"])
        cand, base, status = judge_repeat(first_c, first_b, rep_c, rep_b)
        if first_c == 0:
            broke = sum(1 for v in allv["candidate"] if v is False)
            notes.append(
                f"{name}: broke in {broke} of {len(allv['candidate'])} candidate runs, "
                + {"regression": "and again on the repeat", "info": "not confirmed by the repeat"}[
                    status
                ]
            )
        if status == "ok" and any(v is False for v in allv["candidate"]):
            status = "info"
        findings.append(_invariant(name, "invariant", cand, base, status))
    svr = by_side(runs, SERVER_CHECK)
    cand, base = side_value(svr["candidate"]), side_value(svr["baseline"])
    findings.append(
        _invariant(
            f"control {SERVER_CHECK}", "completed", cand, base, kit.correctness_status(cand, base)
        )
    )
    return findings, notes


def _invariant(name, metric, cand, base, status) -> dict:
    return kit.finding(name, metric, cand, base, status, direction="lower_is_worse")



# --- graph-build timing ----------------------------------------------------------------------------


def _median(xs):
    xs = [x for x in xs if x is not None]
    return statistics.median(xs) if xs else None


def build_side(runs: list[dict], key: str, side: str, extra: bool | None = None) -> float | None:
    return _median(
        [
            r.get(key)
            for r in runs
            if r["side"] == side
            and r.get("ok")
            and (extra is None or bool(r.get("extra")) == extra)
        ]
    )


def crosses(cand: float | None, base: float | None, floor: float) -> int:
    """+1 regression, -1 improvement, 0 neither: ratio > 1.5 and difference > floor, either way."""
    return kit.crossed(cand, base, BUILD_THRESHOLD, floor)


def build_needs_confirmation(runs: list[dict]) -> list[str]:
    return [
        name
        for name, key, floor in BUILD_METRICS
        if crosses(
            build_side(runs, key, "candidate", False),
            build_side(runs, key, "baseline", False),
            floor,
        )
        > 0
    ]


def build_findings(runs: list[dict]) -> tuple[list[dict], list[str]]:
    """A row per build metric, showing the first runs' medians: the values that decide its status.
    A repeat pair only confirms a crossing; its medians are in the notes."""
    findings, notes = [], []
    has_extra = any(r.get("extra") for r in runs)
    for name, key, floor in BUILD_METRICS:
        c1, b1 = build_side(runs, key, "candidate", False), build_side(runs, key, "baseline", False)
        first = crosses(c1, b1, floor)
        if first > 0:
            ce, be = (
                build_side(runs, key, "candidate", True),
                build_side(runs, key, "baseline", True),
            )
            again = crosses(ce, be, floor) > 0 if has_extra else False
            status = "regression" if again else "info"
            notes.append(
                f"{name}: {c1:.0f} vs {b1:.0f} ms" + (f" ({c1 / b1:.2f}x)" if b1 else "")
                + (
                    f", repeat {ce:.0f} vs {be:.0f} ms"
                    if has_extra and ce is not None and be is not None
                    else ", no repeat ran"
                )
                + ("" if again else ", not repeated")
            )
        elif first < 0:
            status = "improvement"
        else:
            status = "ok" if c1 is not None and b1 is not None else "info"
        findings.append(
            kit.finding(name, "p50_ms", _r(c1), _r(b1), status, threshold=BUILD_THRESHOLD)
        )
    return findings, notes


def _r(x):
    return None if x is None else round(x, 3)


# --- the whole result --------------------------------------------------------------------------------

ORDER = {"regression": 0, "improvement": 1, "info": 2, "ok": 3}


def worst_first(findings: list[dict]) -> list[dict]:
    """Findings in the order kit.select keeps them: by status, then the worst first. A timing by its
    ratio, largest first; a 0/1 check (no ratio) broken against held (0 vs 1) first, then held on
    both, then the rest."""

    def key(f):
        c, b, r = f["candidate"], f["baseline"], f["ratio"]
        if f["metric"] in ("invariant", "completed"):
            return ORDER[f["status"]], c / b if c is not None and b else 2
        return ORDER[f["status"]], -(r if r is not None else 0)

    return sorted(findings, key=key)
