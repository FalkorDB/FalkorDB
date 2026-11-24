"""From runs to contract findings (result.json schema 1, contract v1.5). Pure: no server.

A run is one server-hour on one side. Every check of the control epoch comes out of a run as
True (held), False (broke) or None (not measured, or does not apply).

How a check becomes a finding (metric `invariant`, 1 held, 0 broke; `completed` for the server):

* Timing checks (`checks.TIMING_CHECKS`): a side broke if it broke in any of its runs. When the
  candidate broke and the baseline held in every run, one more pair runs (`needs_confirmation`):
  the candidate's value is 0 only if it broke again there. If it held, its value is 1 and the
  finding is `info`, with "broke in i of n candidate runs" in the summary.
* Data checks (`checks.DATA_CHECKS`) and the server staying up: a break counts at once.
* Graph-build timing: the median over a side's runs, per phase; a regression needs the ratio at or
  above the threshold and the difference above a floor, in the first runs and again in the extra
  pair (`build_needs_confirmation`).

Status follows contract v1.4: 0 on the candidate only is `regression`, 0 on both is `info`, 0 on
the baseline only is `improvement`.
"""

from __future__ import annotations

import statistics

from .checks import DATA_CHECKS, TIMING_CHECKS

CONTROL_CHECKS = (*TIMING_CHECKS, *DATA_CHECKS)
SERVER_CHECK = "server_stays_up"
SIDES = ("candidate", "baseline")
BUILD_THRESHOLD = 1.5
#: (finding name, key in a build run, minimum difference in ms for a regression or improvement)
BUILD_METRICS = (("build edge phase", "edges_ms", 500.0), ("build node phase", "nodes_ms", 500.0),
                 ("build edge query 1 row", "probe_1_ms", 2.0), ("build edge query 100 rows", "probe_100_ms", 2.0))


def status_of(candidate: int | None, baseline: int | None) -> str:
    """Contract v1.4 for a 0/1 metric: 0 on the candidate is a regression unless the baseline is 0
    too (the validator requires it whenever the baseline is not 0, null included)."""
    if candidate == 0:
        return "info" if baseline == 0 else "regression"
    if baseline == 0 and candidate == 1:
        return "improvement"
    if candidate is None or baseline is None:
        return "info"
    return "ok"


def side_value(values: list) -> int | None:
    """0 if any run broke, 1 if every run held, None if a run did not measure it."""
    if any(v is False for v in values):
        return 0
    if values and all(v is True for v in values):
        return 1
    return None


def run_checks(run: dict) -> dict:
    """{check: True/False/None} for one run (None: not measured or does not apply)."""
    out = {}
    if run.get("started") is False:
        return {**{c: None for c in CONTROL_CHECKS}, SERVER_CHECK: False if run.get("server_died") else None}
    ep = run.get("epoch") or {}
    cks = ep.get("checks", {})
    for c in CONTROL_CHECKS:
        v = cks.get(c, {})
        p = v.get("pass")
        if v.get("applies") is False:
            p = None
        out[c] = p if p in (True, False) else None
    if run.get("server_died"):
        out[SERVER_CHECK] = False
    elif "server_stays_up" in cks:
        out[SERVER_CHECK] = bool(cks["server_stays_up"].get("pass"))
    else:
        out[SERVER_CHECK] = None
    return out


def by_side(runs: list[dict], check: str, first_only: bool = False) -> dict[str, list]:
    out = {side: [] for side in SIDES}
    for run in runs:
        if first_only and run.get("extra"):
            continue
        out[run["side"]].append(run_checks(run)[check])
    return out


def needs_confirmation(runs: list[dict]) -> list[str]:
    """Timing checks the candidate broke while the baseline held in every one of its runs."""
    out = []
    for c in TIMING_CHECKS:
        v = by_side(runs, c)
        if side_value(v["candidate"]) == 0 and side_value(v["baseline"]) == 1:
            out.append(c)
    return out


def control_findings(runs: list[dict], confirmed_pair: bool) -> tuple[list[dict], list[str]]:
    """Findings for the control epoch, and notes for the summary. `confirmed_pair`: an extra
    (baseline, candidate) pair ran after the first runs (runs with "extra": True).

    Timing checks are judged on the first runs; the extra pair only decides whether a candidate
    break that the baseline did not show repeats. Data checks and the server count every run."""
    findings, notes = [], []
    first = [r for r in runs if not r.get("extra")]
    extra = [r for r in runs if r.get("extra")]
    for c in CONTROL_CHECKS:
        name = f"control {c}"
        allv, firstv = by_side(runs, c), by_side(first, c)
        if c not in TIMING_CHECKS:
            cand, base = side_value(allv["candidate"]), side_value(allv["baseline"])
            findings.append(_invariant(name, "invariant", cand, base, status_of(cand, base)))
            continue
        cand, base = side_value(firstv["candidate"]), side_value(firstv["baseline"])
        status = None
        if cand == 0 and base == 1:  # needs the repeat
            broke = sum(1 for v in allv["candidate"] if v is False)
            n = len(allv["candidate"])
            ext = by_side(extra, c) if confirmed_pair and extra else {"candidate": [], "baseline": []}
            ext_c, ext_b = side_value(ext["candidate"]), side_value(ext["baseline"])
            if not ext["candidate"] or not ext["baseline"]:
                cand = None  # the repeat could not run: neither a regression nor a pass
                notes.append(f"{name}: broke in {broke} of {n} candidate runs, no repeat could run")
            elif ext_b == 0:
                base = 0  # the baseline broke on the repeat: broken on both sides
                notes.append(f"{name}: broke on the candidate and, on the repeat, on the baseline")
            elif ext_b is None or ext_c is None:
                cand = None  # a repeat that did not measure the check confirms nothing
                notes.append(f"{name}: broke in {broke} of {n} candidate runs, the repeat did not measure it")
            elif ext_c == 0:
                notes.append(f"{name}: broke in {broke} of {n} candidate runs, held in "
                             f"{sum(1 for v in allv['baseline'] if v is True)} of {len(allv['baseline'])} baseline runs")
            else:
                cand, status = 1, "info"
                notes.append(f"{name}: broke in {broke} of {n} candidate runs, held on the repeat")
        elif cand == 0 and base is None:
            cand = None  # a timing break against an unmeasured baseline cannot be judged
        status = status or status_of(cand, base)
        if status == "ok" and any(v is False for v in allv["candidate"]):
            status = "info"
        findings.append(_invariant(name, "invariant", cand, base, status))
    svr = by_side(runs, SERVER_CHECK)
    cand, base = side_value(svr["candidate"]), side_value(svr["baseline"])
    findings.append(_invariant(f"control {SERVER_CHECK}", "completed", cand, base, status_of(cand, base)))
    return findings, notes


def _invariant(name, metric, cand, base, status) -> dict:
    return {"name": name, "metric": metric, "candidate": cand, "baseline": base,
            "ratio": round(cand / base, 4) if cand is not None and base else None,
            "threshold": None, "direction": "lower_is_worse", "status": status}


# --- graph-build timing ----------------------------------------------------------------------------


def _median(xs):
    xs = [x for x in xs if x is not None]
    return statistics.median(xs) if xs else None


def build_side(runs: list[dict], key: str, side: str, extra: bool | None = None) -> float | None:
    return _median([r.get(key) for r in runs if r["side"] == side and r.get("ok")
                    and (extra is None or bool(r.get("extra")) == extra)])


def crosses(cand: float | None, base: float | None, floor: float) -> int:
    """+1 regression, -1 improvement, 0 neither (ratio 1.5x either way and a difference > floor)."""
    if cand is None or base is None or base <= 0 or cand <= 0:
        return 0
    if cand / base >= BUILD_THRESHOLD and cand - base >= floor:
        return 1
    if base / cand >= BUILD_THRESHOLD and base - cand >= floor:
        return -1
    return 0


def build_needs_confirmation(runs: list[dict]) -> list[str]:
    return [name for name, key, floor in BUILD_METRICS
            if crosses(build_side(runs, key, "candidate", False), build_side(runs, key, "baseline", False), floor) > 0]


def build_findings(runs: list[dict]) -> tuple[list[dict], list[str]]:
    findings, notes = [], []
    has_extra = any(r.get("extra") for r in runs)
    for name, key, floor in BUILD_METRICS:
        c1, b1 = build_side(runs, key, "candidate", False), build_side(runs, key, "baseline", False)
        cand, base = build_side(runs, key, "candidate"), build_side(runs, key, "baseline")
        first = crosses(c1, b1, floor)
        if first > 0:
            ce, be = build_side(runs, key, "candidate", True), build_side(runs, key, "baseline", True)
            again = crosses(ce, be, floor) > 0 if has_extra else False
            status = "regression" if again else "info"
            notes.append(f"{name}: {c1:.0f} vs {b1:.0f} ms ({c1 / b1:.2f}x)"
                         + (f", repeat {ce:.0f} vs {be:.0f} ms" if has_extra and ce and be else ", no repeat ran")
                         + ("" if again else ", not repeated"))
        elif first < 0:
            status = "improvement"
        else:
            status = "ok" if cand is not None and base is not None else "info"
        findings.append({"name": name, "metric": "p50_ms", "candidate": _r(cand), "baseline": _r(base),
                         "ratio": round(cand / base, 4) if cand and base else None,
                         "threshold": BUILD_THRESHOLD, "direction": "higher_is_worse", "status": status})
    return findings, notes


def _r(x):
    return None if x is None else round(x, 3)


# --- the whole result --------------------------------------------------------------------------------

ORDER = {"regression": 0, "improvement": 1, "info": 2, "ok": 3}


def select(findings: list[dict], cap_informative: int = 10, cap_total: int = 50) -> list[dict]:
    """Regressions and improvements first (worst ratio first), then up to 10 info/ok rows."""
    def key(f):
        r = f.get("ratio")
        worse = -(r if r is not None else 0) if f["direction"] == "higher_is_worse" else (r if r is not None else 2)
        return ORDER[f["status"]], worse
    out, informative = [], 0
    for f in sorted(findings, key=key):
        if f["status"] in ("ok", "info"):
            if informative >= cap_informative:
                continue
            informative += 1
        out.append(f)
    return out[:cap_total]


def verdict_of(findings: list[dict], unmeasured: list[str]) -> str:
    if any(f["status"] == "regression" for f in findings):
        return "regression"
    if unmeasured:
        return "error"
    return "pass"
