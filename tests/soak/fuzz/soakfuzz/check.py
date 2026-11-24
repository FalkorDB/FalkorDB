"""The fuzz check as one contract command (`soak-fuzz --candidate … --baseline … --out … --mode
queue|nightly|long`; README). Writes <out>/result.json whatever happens, then exits 0."""

from __future__ import annotations

import argparse
import csv
import json
import os
import platform
import re
import signal
import subprocess
import sys
import time
import traceback
from datetime import datetime, timezone
from pathlib import Path

from . import checks, model, verdict

CHECK = "fuzz"
# Docker's image reference with a digest, as the contract's validator (tests/perf/validate.py) reads it
_DOMAIN = r"(?:[A-Za-z0-9]|[A-Za-z0-9][A-Za-z0-9-]*[A-Za-z0-9])"
_PATH = r"[a-z0-9]+(?:(?:[._]|__|-+)[a-z0-9]+)*"
DIGEST = re.compile(rf"(?:(?:localhost|{_DOMAIN}(?:\.{_DOMAIN})+)(?::[0-9]+)?/|{_DOMAIN}:[0-9]+/)?"
                    rf"{_PATH}(?:/{_PATH})*(?::[A-Za-z0-9_][A-Za-z0-9_.-]{{0,127}})?@sha256:[0-9a-f]{{64}}")
FIRST_ORDER = ("candidate", "baseline", "baseline", "candidate")
EXTRA_ORDER = ("baseline", "candidate")
#: long_s: seeded epochs and repros end this long after the start; deadline_s: no run starts that
#: would end after it (the nightly's timeout_minutes is 90)
MODES = {
    "queue": {"build": False, "long_s": 0, "deadline_s": 80 * 60},
    "nightly": {"build": True, "long_s": 0, "deadline_s": 85 * 60},
    "long": {"build": True, "long_s": 3.5 * 3600, "deadline_s": 4.5 * 3600},
}
CONTROL_RUN_S = 180  # a control run besides its epoch and drain: start, build at scale 1, checks, stop
BUILD_RUN_S = 180


class Stopped(BaseException):
    """SIGTERM (the workflow's time limit): stop, clean up, and still write result.json. A
    BaseException, so that no `except Exception` on the way up can swallow it."""


def time_for(opts: dict, need_s: float, what: str, run) -> bool:
    if time.time() + need_s <= opts["deadline"]:
        return True
    run.unmeasured.append(f"{what}: not run, it would end after the time limit")
    return False
BUILD_SCALE = 0.25
LONG_EPOCHS_PER_HOUR = 3
LONG_REPLAYS = 2  # seeds replayed on both sides to confirm a long-mode break


def utc_now() -> str:
    return datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


def default_label(image: str) -> str:
    name, _, digest = image.partition("@sha256:")
    if name.rfind(":") > name.rfind("/"):
        name = name[: name.rfind(":")]
    return f"{name}@{digest[:12]}"


def machine() -> dict:
    """The VM workflow's machine object (PERF_MACHINE_JSON), else this host as `local`."""
    path = os.environ.get("PERF_MACHINE_JSON")
    if path:
        try:
            return json.loads(Path(path).read_text())
        except (OSError, ValueError) as e:
            print(f"PERF_MACHINE_JSON unreadable ({e}); recording this host as local", file=sys.stderr)
    return {"runner": "local", "type": f"{platform.system()}-{platform.machine()}",
            "cpu": platform.processor() or platform.machine() or "unknown", "cores": os.cpu_count() or 1}


def parse_args(argv):
    p = argparse.ArgumentParser(prog="soak-fuzz", description=__doc__.splitlines()[0])
    p.add_argument("--candidate", required=True)
    p.add_argument("--baseline", required=True)
    p.add_argument("--out", required=True)
    p.add_argument("--mode", required=True, choices=sorted(MODES))
    p.add_argument("--candidate-label")
    p.add_argument("--baseline-label")
    p.add_argument("--candidate-commit")
    p.add_argument("--baseline-commit")
    dev = p.add_argument_group("development (not part of the contract; defaults are the documented modes)")
    dev.add_argument("--epoch-s", type=float, default=300, help="control and long epochs, seconds")
    dev.add_argument("--drain-s", type=float, default=60)
    dev.add_argument("--control-scales", default=",".join(str(x) for x in model.CONTROL_SCALES))
    dev.add_argument("--build-scale", type=float, default=BUILD_SCALE)
    dev.add_argument("--long-hours", type=float, default=None, help="long: override the 3.5 h budget")
    dev.add_argument("--seed", default=None, help="long: epoch seed (default: from the candidate digest)")
    dev.add_argument("--net", choices=["host", "publish"], default="host" if sys.platform == "linux" else "publish")
    dev.add_argument("--max-conns", type=int, default=20000)
    dev.add_argument("--skip-control", action="store_true", help="development: build timing only")
    dev.add_argument("--long-scale-range", default=None, help="long: per-graph scale lo,hi and their total cap")
    dev.add_argument("--repros", default=None, help="long: comma-separated repro names (default: all)")
    return p.parse_args(argv)


class Run:
    """Collects rows for raw/rows.csv and every run's summary for raw/runs.json."""

    def __init__(self, out: Path):
        self.out, self.raw = out, out / "raw"
        self.raw.mkdir(parents=True, exist_ok=True)
        self.rows: list[dict] = []
        self.runs: dict[str, list] = {"control": [], "build": [], "long": [], "repros": []}
        self.unmeasured: list[str] = []
        self.notes: list[str] = []

    def row(self, phase, side, index, extra, name, value):
        self.rows.append({"phase": phase, "side": side, "run": index, "extra": int(bool(extra)), "name": name,
                          "value": "" if value is None else value})

    def save(self):
        with open(self.raw / "rows.csv", "w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=["phase", "side", "run", "extra", "name", "value"])
            w.writeheader()
            w.writerows(self.rows)
        with open(self.raw / "runs.json", "w") as f:
            json.dump(self.runs, f, indent=1, default=str)


# --- control epoch -------------------------------------------------------------------------------


def control_run(opts, info, side, index, extra, run: Run, hp, ep) -> dict:
    from . import harness
    tag = f"control-{index}-{side}"
    hour = harness.run_server_hour(opts, info[side], tag, hp, [ep], str(run.raw / tag))
    epoch = hour["epochs"][0] if hour["epochs"] else None
    rec = {"side": side, "index": index, "extra": extra, "tag": tag,
           "started": epoch is not None, "server_died": hour.get("server_alive") is False,
           "error": hour.get("error"), "epoch": epoch, "build": hour.get("build"),
           "module_version": hour.get("module_version")}
    for c, v in verdict.run_checks(rec).items():
        run.row("control", side, index, extra, c, None if v is None else int(v))
    for b in hour.get("build", []):
        for k in ("nodes_s", "edges_s"):
            run.row("control-build", side, index, extra, f"g{b['graph']} {k}", b[k])
    if epoch:
        for op, v in epoch.get("info", {}).get("reads_by_op", {}).items():
            run.row("control", side, index, extra, f"{op} p99_ms", v["p99_ms"])
        nt = epoch.get("checks", {}).get("no_timeouts_while_waiting", {})
        run.row("control", side, index, extra, "cheap read timeouts", nt.get("spurious_timeouts"))
        run.row("control", side, index, extra, "all read timeouts", nt.get("all_read_timeouts"))
        run.row("control", side, index, extra, "worst ping ms", epoch.get("checks", {}).get("no_freeze", {}).get("worst_ping_ms"))
    if rec["error"] and not rec["server_died"]:
        run.unmeasured.append(f"{tag}: {rec['error'][:160]}")
    slim = {k: v for k, v in rec.items() if k != "epoch"}
    slim["failed"] = epoch.get("failed") if epoch else None
    run.runs["control"].append(slim)
    return rec


def control_phase(opts, info, run: Run, scales) -> list[dict]:
    hp, ep = model.control_params(scales)
    need_s = opts["epoch_s"] + opts["drain_s"] + CONTROL_RUN_S
    runs = [control_run(opts, info, side, i, False, run, hp, ep) for i, side in enumerate(FIRST_ORDER)
            if time_for(opts, need_s, f"control run {i} ({side})", run)]
    need = verdict.needs_confirmation(runs)
    if need and time_for(opts, 2 * need_s, "control repeat pair", run):
        run.notes.append(f"repeat pair for {', '.join(need)}")
        for j, side in enumerate(EXTRA_ORDER):
            runs.append(control_run(opts, info, side, len(FIRST_ORDER) + j, True, run, hp, ep))
    return runs


# --- graph-build timing --------------------------------------------------------------------------


def build_phase(opts, info, run: Run, scale) -> list[dict]:
    from . import build

    def one(side, index, extra):
        tag = f"build-{index}-{side}"
        res = build.build_run(opts, info[side], tag, scale, str(run.raw / tag))
        res.update(side=side, index=index, extra=extra)
        for _name, key, _floor in verdict.BUILD_METRICS:
            run.row("build", side, index, extra, key, res.get(key))
        if not res["ok"]:
            if res.get("server_alive") is False:
                res["server_died"] = True
            else:
                run.unmeasured.append(f"{tag}: {res.get('error', '')[:160]}")
        run.runs["build"].append({k: v for k, v in res.items() if k != "plan"})
        return res

    runs = [one(side, i, False) for i, side in enumerate(FIRST_ORDER)
            if time_for(opts, BUILD_RUN_S, f"build run {i} ({side})", run)]
    if verdict.build_needs_confirmation(runs) and time_for(opts, 2 * BUILD_RUN_S, "build repeat pair", run):
        runs += [one(side, len(FIRST_ORDER) + j, True) for j, side in enumerate(EXTRA_ORDER)]
    return runs


def build_crash_finding(runs) -> dict | None:
    """A server that died during a build run: completed 0 (a crash is a regression at once). A side
    is 1 only if one of its builds finished, and unknown (null) if none died and none finished."""
    def completed(side):
        rs = [r for r in runs if r["side"] == side]
        if any(r.get("server_died") for r in rs):
            return 0
        return 1 if any(r.get("ok") for r in rs) else None

    c, b = completed("candidate"), completed("baseline")
    if 0 not in (c, b):
        return None
    return verdict._invariant("build server_stays_up", "completed", c, b, verdict.status_of(c, b))


# --- long mode: seeded random epochs and the repros ---------------------------------------------------

#: memory_steady is recorded in epoch.json but not judged: it exists only on an hour's last epoch,
#: so a one-epoch replay could never confirm it
#: memory_steady is recorded in epoch.json but not judged: it exists only on an hour's last epoch
LONG_CHECKS = (*checks.LONG_TIMING_CHECKS, *checks.LONG_DATA_CHECKS, "server_stays_up")
LONG_IMMEDIATE = (*checks.LONG_DATA_CHECKS, "server_stays_up")  # a break counts at once
NA = verdict.NA


REPRO_S = 90  # about what one repro takes on one side (build, a ~30 s long read, server start/stop)


def long_tag(h, side, eps, replay) -> str:
    """A server-hour's tag and directory; a replay names its epochs, so two replays never share one."""
    tag = f"long-h{h:02d}-{side}"
    return tag + "-replay-" + "-".join(f"e{e['epoch']}" for e in eps) if replay else tag


def long_hour(opts, info, run: Run, side, h, seed, deadline, epochs=None, replay=False) -> dict:
    from . import harness
    hp = model.hour_params(seed, h, opts["long_scale_range"])
    hp["seed"] = seed
    eps = epochs or [model.epoch_params(seed, h, e, hp["graphs"]) for e in range(1, LONG_EPOCHS_PER_HOUR + 1)]
    tag = long_tag(h, side, eps, replay)
    hour = harness.run_server_hour(opts, info[side], tag, hp, eps, str(run.raw / tag), deadline=deadline)
    res = {"side": side, "hour": h, "replay": replay, "error": hour.get("error"),
           "server_died": hour.get("server_alive") is False, "epochs": {}}
    for e in hour["epochs"]:
        res["epochs"][e["params"]["seed"]] = verdict.epoch_values(e.get("checks", {}), LONG_CHECKS, NA)
    if res["server_died"] and not hour["epochs"]:  # died before its first epoch: a crash on every seed
        for e in eps:
            res["epochs"][e["seed"]] = {**{c: None for c in LONG_CHECKS}, "server_stays_up": False}
    elif res["error"] and not res["server_died"]:
        run.unmeasured.append(f"{tag}: {res['error'][:160]}")
    run.runs["long"].append(res)
    return res


def _pair_estimate(opts, seed, h) -> float:
    """Seconds a server-hour pair should take (model.hour_seconds on each side)."""
    hp = model.hour_params(seed, h, opts["long_scale_range"])
    eps = [model.epoch_params(seed, h, e, hp["graphs"]) for e in range(1, LONG_EPOCHS_PER_HOUR + 1)]
    return 2 * model.hour_seconds(opts["epoch_s"], opts["drain_s"], hp, eps)


def long_phase(opts, info, run: Run, seed, t_end) -> list[dict]:
    """Server-hours h1, h2, … on both sides (order swapped every hour) until t_end (unix time) less
    room for the repros; then the repros (one more pair for a timing break); then replays of seeds
    where the candidate broke a timing check the baseline held."""
    from . import repros
    names = opts["repros"] or list(repros.REPROS)
    reserve = len(names) * 2 * REPRO_S + 10 * 60
    hours, h, slow = [], 1, 1.0
    # A pair (the same seeded hour on both sides) starts only if it fits, and then runs whole: a time
    # limit checked per epoch would let the first side run an epoch the second side has no time for.
    while True:
        est = _pair_estimate(opts, seed, h) * slow
        if h > 1 and time.time() + est > t_end - reserve:
            break
        if not time_for(opts, est, f"long hour {h} on both sides", run):  # the hard limit, first pair included
            break
        order = ("candidate", "baseline") if h % 2 else ("baseline", "candidate")
        t0 = time.time()
        for side in order:
            hours.append(long_hour(opts, info, run, side, h, seed, None))
        slow = max(slow, (time.time() - t0) / _pair_estimate(opts, seed, h))
        h += 1
    for i, name in enumerate(names):
        if not time_for(opts, 2 * REPRO_S, f"repro {name}", run):
            continue
        order = ("candidate", "baseline") if i % 2 == 0 else ("baseline", "candidate")
        for side in order:
            res = repros.run_repro(opts, info[side], f"repro-{name}-{side}", name)
            run.runs["repros"].append({"name": name, "side": side, "replay": False, **res})
            run.row("repro", side, 0, False, name, None if res.get("pass") is None else int(res["pass"]))
    first = {(r["name"], r["side"]): r for r in run.runs["repros"]}
    for name in names:
        c, b = first.get((name, "candidate"), {}), first.get((name, "baseline"), {})
        if (c.get("pass") is False and not c.get("crashed") and b.get("pass") is True
                and time_for(opts, 2 * REPRO_S, f"repro {name} repeat", run)):
            for side in ("baseline", "candidate"):
                res = repros.run_repro(opts, info[side], f"repro-{name}-{side}-replay", name)
                run.runs["repros"].append({"name": name, "side": side, "replay": True, **res})
                run.row("repro", side, 1, True, name, None if res.get("pass") is None else int(res["pass"]))
    suspects = []
    for c in LONG_CHECKS:
        if c not in LONG_IMMEDIATE:
            suspects += [(s_, c) for s_ in _seeds_where(hours, c)]
    replayed = set()
    for s_ in dict.fromkeys(s_ for s_, _c in suspects):  # each seed once, in the order found
        if len(replayed) >= LONG_REPLAYS:
            break
        h_, e_ = (int(x[1:]) for x in s_.split("-")[-2:])
        hp = model.hour_params(seed, h_, opts["long_scale_range"])
        ep = model.epoch_params(seed, h_, e_, hp["graphs"])
        one = model.hour_seconds(opts["epoch_s"], opts["drain_s"], hp, [ep])
        if not time_for(opts, 2 * one * slow, f"replay of {s_}", run):  # as slow as the hours ran
            continue
        replayed.add(s_)
        for side in ("baseline", "candidate"):
            hours.append(long_hour(opts, info, run, side, h_, seed, None, epochs=[ep], replay=True))
    return hours


def _by_seed(hours, side, replay):
    out = {}
    for hr in hours:
        if hr["side"] == side and hr["replay"] == replay:
            out.update(hr["epochs"])
    return out


def _seeds_where(hours, check):
    c, b = _by_seed(hours, "candidate", False), _by_seed(hours, "baseline", False)
    return sorted(s_ for s_ in c if s_ in b and c[s_].get(check) is False and b[s_].get(check) is True)


def _v(x):
    """True / False / None / NA -> 1 / 0 / None (as judge_repeat takes them)."""
    return None if x in (None, NA) else int(x)


def _long_timing(chk, vals, sus, c_rep, b_rep):
    """A timing check across seeds: a candidate-only break counts only if its replay repeats it
    (verdict.judge_repeat); otherwise seeds broken on both sides give info, baseline-only breaks an
    improvement."""
    judged = [verdict.judge_repeat(0, 1, _v(c_rep.get(s_, {}).get(chk)), _v(b_rep.get(s_, {}).get(chk)))
              for s_ in sus]
    if any(st == "regression" for _, _, st in judged):
        return 0, 1
    if any(c is None for c, _, _ in judged):
        return None, 0 if any(b is False for _, b in vals) else 1
    if any(c == 0 for c, _, _ in judged) or any(c is False and b is False for c, b in vals):
        return 0, 0
    if any(b is False and c is True for c, b in vals):
        return 1, 0
    return 1, 1


def long_findings(hours, repro_runs) -> tuple[list[dict], list[str], list[str]]:
    """Long-mode findings, paired by seed (README, "How a check becomes a finding"); also returns
    the summary notes and what was not measured."""
    findings, notes, unmeasured = [], [], []
    c_first, b_first = _by_seed(hours, "candidate", False), _by_seed(hours, "baseline", False)
    c_rep, b_rep = _by_seed(hours, "candidate", True), _by_seed(hours, "baseline", True)
    if not c_first or not b_first:
        unmeasured.append("long: no seeded epoch ran on both sides")
    lone = sorted(set(c_first) ^ set(b_first))
    if lone:
        unmeasured.append(f"long: {len(lone)} seeds ran on one side only")
    paired = [s_ for s_ in c_first if s_ in b_first]
    for chk in LONG_CHECKS:
        def pairs(cs, bs, seeds):
            out = [(cs[s_].get(chk), bs.get(s_, {}).get(chk)) for s_ in seeds]
            return [(c, b) for c, b in out if c != NA and b != NA]
        vals = pairs(c_first, b_first, paired)
        # immediate checks also count seeds (and replays) the baseline never ran
        lone_c = [c for c in ([c_first[s_].get(chk) for s_ in c_first if s_ not in b_first]
                              + [c_rep[s_].get(chk) for s_ in c_rep if s_ not in b_rep]) if c != NA]
        if chk not in LONG_IMMEDIATE:
            lone_c = []
        if not vals and not lone_c:
            continue
        if any(None in v for v in vals):
            unmeasured.append(f"long {chk}: not measured on every seed")
        if chk in LONG_IMMEDIATE and not vals:  # only seeds the baseline never ran: its value is unknown
            cand = 0 if any(c is False for c in lone_c) else (1 if all(c is True for c in lone_c) else None)
            base = None
        elif chk in LONG_IMMEDIATE:
            seen = vals + pairs(c_rep, b_rep, [s_ for s_ in c_rep if s_ in b_rep])
            if any(c is False and b is True for c, b in seen):
                cand, base = 0, 1
            elif any(c is False and b is None for c, b in seen) or any(c is False for c in lone_c):
                cand, base = 0, None  # broke where the baseline measured nothing: still a break at once
            elif any(c is False and b is False for c, b in seen):
                cand, base = 0, 0
            elif any(b is False and c is True for c, b in seen):
                cand, base = 1, 0
            else:
                cand, base = 1, 1
        else:
            sus = _seeds_where(hours, chk)
            cand, base = _long_timing(chk, vals, sus, c_rep, b_rep)
            if cand is None:
                unmeasured.append(f"long {chk}: a candidate-only break was not replayed")
            if sus:
                notes.append(f"long {chk}: candidate-only on {len(sus)} of {len(vals)} seeds")
        status = verdict.status_of(cand, base)
        if status == "ok" and any(c is False for c, _ in vals):
            status = "info"
        metric = "completed" if chk == "server_stays_up" else "invariant"
        findings.append(verdict._invariant(f"long {chk}", metric, cand, base, status))
    by = {}
    for r in repro_runs:
        by.setdefault(r["name"], {}).setdefault(r["side"], []).append(r)
    for name, sides in by.items():
        def pick(side, replay):
            return next((r for r in sides.get(side, []) if bool(r.get("replay")) == replay), {})
        c, b = pick("candidate", False), pick("baseline", False)
        cv, bv = _v(c.get("pass")), _v(b.get("pass"))
        if cv == 0 and c.get("crashed"):  # a crash counts at once
            status = verdict.status_of(cv, bv)
        else:  # a timing break: it must repeat (a crash on the repeat counts as breaking again)
            ca, ba = pick("candidate", True), pick("baseline", True)
            cv, bv, status = verdict.judge_repeat(cv, bv, 0 if ca.get("crashed") else _v(ca.get("pass")),
                                                  _v(ba.get("pass")))
        if cv is None or bv is None:
            unmeasured.append(f"long repro {name}: not measured on both sides")
        findings.append(verdict._invariant(f"long repro {name}", "invariant", cv, bv, status))
    return findings, notes, unmeasured


# --- main -------------------------------------------------------------------------------------------


def cleanup(prefix: str):
    """Remove this run's containers (named `<prefix>-…`), and only those."""
    p = subprocess.run(["docker", "ps", "-a", "--format", "{{.Names}}"], capture_output=True, text=True, timeout=60)
    names = [x for x in p.stdout.split() if x.startswith(prefix + "-")]
    if names:
        subprocess.run(["docker", "rm", "-f", *names], capture_output=True, timeout=300)


def summarise(verdict_s, findings, notes, unmeasured, error) -> str:
    regs = [f["name"] for f in findings if f["status"] == "regression"]
    both = [f["name"] for f in findings if f["status"] == "info" and f["candidate"] == 0 and f["baseline"] == 0]
    parts = []
    if verdict_s == "regression":
        parts.append("regression in " + ", ".join(regs))
    elif verdict_s == "error":
        parts.append("could not measure: " + (error or "; ".join(unmeasured[:2])))
    else:
        parts.append("no check broke on the candidate where it held on the baseline")
    if both:
        parts.append(f"{len(both)} broken on both sides (not new): " + ", ".join(b.replace("control ", "") for b in both))
    parts += notes
    text = "; ".join(parts)
    text = " ".join(text.split())
    return text if len(text) <= 600 else text[:597] + "..."


def build_result(args, started, finished, findings, notes, unmeasured, error, machine_obj) -> dict:
    """The result.json document (contract schema 1) from the findings of every phase."""
    if not findings and not error:
        unmeasured = [*unmeasured, "nothing was measured"]
    v = verdict.verdict_of(findings, unmeasured + ([error] if error else []))
    if v == "error" and not error:
        error = "; ".join(unmeasured[:3])[:500] or "not every check was measured"
    return {
        "schema": 1,
        "check": CHECK,
        "verdict": v,
        "confirmed": v == "regression",
        "summary": summarise(v, findings, notes, unmeasured, error),
        "candidate": {"image": args.candidate, "commit": args.candidate_commit or None,
                      "label": args.candidate_label or default_label(args.candidate)},
        "baseline": {"image": args.baseline, "commit": args.baseline_commit or None,
                     "label": args.baseline_label or default_label(args.baseline)},
        "machine": machine_obj,
        "started": started,
        "finished": finished,
        "findings": verdict.select(findings),
        "error": error if v == "error" else None,
        "artifacts": ["raw/rows.csv", "raw/runs.json"],
    }


def main(argv=None) -> int:
    args = parse_args(argv)
    out = Path(args.out).resolve()  # contract v1.8: the workflow passes `--out out`, relative to where it starts
    out.mkdir(parents=True, exist_ok=True)
    run = Run(out)
    started, t_start = utc_now(), time.time()
    images = {"candidate": args.candidate, "baseline": args.baseline}
    prefix = f"perf-fuzz-{os.getpid()}"
    opts = {"epoch_s": args.epoch_s, "drain_s": args.drain_s, "max_conns": args.max_conns, "net": args.net,
            "name_prefix": prefix, "server_cpus": os.environ.get("PERF_SERVER_CPUS") or None,
            "client_cpus": os.environ.get("PERF_CLIENT_CPUS") or None,
            "long_scale_range": (tuple(float(x) for x in args.long_scale_range.split(","))
                                 if args.long_scale_range else (0.25, 2.0, 4.0)),
            "repros": [x for x in (args.repros or "").split(",") if x]}
    mode = MODES[args.mode]
    opts["deadline"] = t_start + mode["deadline_s"]
    findings, error = [], None

    def on_term(signum, _frame):
        if harness_mod is not None:
            harness_mod.STOPPING = True
        raise Stopped(f"stopped by signal {signum} (the time limit)")

    harness_mod = None
    previous = signal.signal(signal.SIGTERM, on_term)
    try:
        bad = [f"--{side} {image}" for side, image in images.items() if not DIGEST.fullmatch(image)]
        if bad:
            raise ValueError("images must be pinned by digest (name@sha256:<64 hex>): " + ", ".join(bad))
        from . import harness
        harness_mod = harness
        harness.set_start_method()
        info, identity = {}, {}
        for side, image in images.items():
            try:
                info[side] = harness.probe_image(image)
            except RuntimeError as e:
                raise RuntimeError(f"{side} image: {e}") from None
            identity[side] = info[side]
        with open(run.raw / "images.json", "w") as f:
            json.dump(identity, f, indent=1)
        if not args.skip_control:
            scales = tuple(float(x) for x in args.control_scales.split(","))
            runs = control_phase(opts, info, run, scales)
            cf, notes = verdict.control_findings(runs, any(r["extra"] for r in runs))
            findings += cf
            run.notes += notes
            for f in cf:
                if f["candidate"] is None or f["baseline"] is None:
                    run.unmeasured.append(f"{f['name']}: not measured on every run")
        if mode["build"]:
            bruns = build_phase(opts, info, run, args.build_scale)
            bf, notes = verdict.build_findings(bruns)
            findings += bf
            run.notes += notes
            for f in bf:
                if f["candidate"] is None or f["baseline"] is None:
                    run.unmeasured.append(f"{f['name']}: not measured on both sides")
            crash = build_crash_finding(bruns)
            if crash:
                findings.append(crash)
        if mode["long_s"]:
            budget = (args.long_hours * 3600 if args.long_hours else mode["long_s"])
            seed = args.seed or f"long-{args.candidate.rsplit(':', 1)[-1][:12]}"
            run.notes.append(f"long seed {seed}")
            hours = long_phase(opts, info, run, seed, t_start + budget)
            lf, notes, unmeasured = long_findings(hours, run.runs["repros"])
            findings += lf
            run.notes += notes
            run.unmeasured += unmeasured
    except (Exception, Stopped) as e:
        signal.signal(signal.SIGTERM, signal.SIG_IGN)  # a second SIGTERM must not stop the write below
        error = f"{type(e).__name__}: {e}"[:500]
        (run.raw / "error.txt").write_text(traceback.format_exc())
    signal.signal(signal.SIGTERM, signal.SIG_IGN)  # what follows must finish
    for f in findings:  # every finding, ok rows included (result.json lists only the others)
        for side in verdict.SIDES:
            run.row(f"finding ({f['status']})", side, "", False, f["name"], f[side])
    try:
        run.save()
        result = build_result(args, started, utc_now(), findings, run.notes, run.unmeasured, error, machine())
        (out / "result.json").write_text(json.dumps(result, indent=2, allow_nan=False) + "\n")
        print(f"verdict: {result['verdict']}: {result['summary']}", flush=True)
    finally:
        try:
            cleanup(prefix)
        except Exception as e:  # result.json is already written; say what may be left running
            print(f"warning: removing the {prefix}-* containers failed: {e}", file=sys.stderr, flush=True)
        signal.signal(signal.SIGTERM, previous)
        if harness_mod is not None:
            harness_mod.STOPPING = False
    return 0


if __name__ == "__main__":
    sys.exit(main())
