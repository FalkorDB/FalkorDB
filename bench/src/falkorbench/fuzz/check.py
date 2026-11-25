"""The fuzz check as one contract command (`bench fuzz --candidate … --baseline … --out …
--mode queue|nightly|long`; README.md). kit.run_check writes <out>/result.json whatever happens;
the command then exits 0."""

from __future__ import annotations

import argparse
import csv
import json
import os
import sys
import time
from pathlib import Path

from falkorbench import kit

from . import checks
from . import model
from . import verdict

CHECK = "fuzz"
# every container of a run carries LABEL=<run id>: the clean-up removes them
LABEL = "falkordb-fuzz"
ARTIFACTS = ("raw/rows.csv", "raw/runs.json")
FIRST_ORDER = ("candidate", "baseline", "baseline", "candidate")
EXTRA_ORDER = ("baseline", "candidate")
#: long_s: seeded epochs and repros end this long after the start; deadline_s: no run starts that
#: would end after it (the nightly's timeout_minutes is 90)
MODES = {
    "queue": {"build": False, "long_s": 0, "deadline_s": 80 * 60},
    "nightly": {"build": True, "long_s": 0, "deadline_s": 85 * 60},
    "long": {"build": True, "long_s": 3.5 * 3600, "deadline_s": 4.5 * 3600},
}
# a control run besides its epoch and drain: start, build at scale 1, checks, stop
CONTROL_RUN_S = 180
BUILD_RUN_S = 180
BUILD_SCALE = 0.25
LONG_EPOCHS_PER_HOUR = 3
LONG_REPLAYS = 2  # seeds replayed on both sides to confirm a long-mode break


def time_for(opts: dict, need_s: float, what: str, run) -> bool:
    if time.time() + need_s <= opts["deadline"]:
        return True
    run.unmeasured.append(f"{what}: not run, it would end after the time limit")
    return False


def parse_args(argv):
    p = argparse.ArgumentParser(prog="bench fuzz", description=__doc__.splitlines()[0])
    kit.add_args(p)
    dev = p.add_argument_group(
        "development (not part of the contract; defaults are the documented modes)"
    )
    dev.add_argument("--epoch-s", type=float, default=300, help="control and long epochs, seconds")
    dev.add_argument("--drain-s", type=float, default=60)
    dev.add_argument("--control-scales", default=",".join(str(x) for x in model.CONTROL_SCALES))
    dev.add_argument("--build-scale", type=float, default=BUILD_SCALE)
    dev.add_argument(
        "--long-hours", type=float, default=None, help="long: override the 3.5 h budget"
    )
    dev.add_argument(
        "--seed", default=None, help="long: epoch seed (default: from the candidate digest)"
    )
    dev.add_argument(
        "--net",
        choices=["host", "publish"],
        default="host" if sys.platform == "linux" else "publish",
    )
    dev.add_argument("--max-conns", type=int, default=20000)
    dev.add_argument("--skip-control", action="store_true", help="development: build timing only")
    dev.add_argument(
        "--long-scale-range", default=None, help="long: per-graph scale lo,hi and their total cap"
    )
    dev.add_argument(
        "--repros", default=None, help="long: comma-separated repro names (default: all)"
    )
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
        self.rows.append(
            {
                "phase": phase,
                "side": side,
                "run": index,
                "extra": int(bool(extra)),
                "name": name,
                "value": "" if value is None else value,
            }
        )

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
    rec = {
        "side": side,
        "index": index,
        "extra": extra,
        "tag": tag,
        "started": epoch is not None,
        "server_died": hour.get("server_alive") is False,
        "error": hour.get("error"),
        "epoch": epoch,
        "build": hour.get("build"),
        "module_version": hour.get("module_version"),
    }
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
        run.row(
            "control",
            side,
            index,
            extra,
            "worst ping ms",
            epoch.get("checks", {}).get("no_freeze", {}).get("worst_ping_ms"),
        )
    if rec["error"] and not rec["server_died"]:
        run.unmeasured.append(f"{tag}: {rec['error'][:160]}")
    slim = {k: v for k, v in rec.items() if k != "epoch"}
    slim["failed"] = epoch.get("failed") if epoch else None
    run.runs["control"].append(slim)
    return rec


def control_phase(opts, info, run: Run, scales) -> list[dict]:
    hp, ep = model.control_params(scales)
    need_s = opts["epoch_s"] + opts["drain_s"] + CONTROL_RUN_S
    runs = [
        control_run(opts, info, side, i, False, run, hp, ep)
        for i, side in enumerate(FIRST_ORDER)
        if time_for(opts, need_s, f"control run {i} ({side})", run)
    ]
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

    runs = [
        one(side, i, False)
        for i, side in enumerate(FIRST_ORDER)
        if time_for(opts, BUILD_RUN_S, f"build run {i} ({side})", run)
    ]
    if verdict.build_needs_confirmation(runs) and time_for(
        opts, 2 * BUILD_RUN_S, "build repeat pair", run
    ):
        runs += [one(side, len(FIRST_ORDER) + j, True) for j, side in enumerate(EXTRA_ORDER)]
    return runs


def unconfirmed_builds(notes: list[str]) -> list[str]:
    """A build timing that crossed but whose confirming repeat did not run is unmeasured."""
    return [
        f"{n.split(':')[0]}: the confirming repeat was not measured"
        for n in notes
        if "no repeat ran" in n
    ]


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
    return verdict._invariant(
        "build server_stays_up", "completed", c, b, kit.correctness_status(c, b)
    )


# --- long mode: seeded random epochs and the repros ---------------------------------------------------

#: memory_steady is recorded in epoch.json but not judged: it exists only on an hour's last epoch,
#: so a one-epoch replay could never confirm it
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
    eps = epochs or [
        model.epoch_params(seed, h, e, hp["graphs"]) for e in range(1, LONG_EPOCHS_PER_HOUR + 1)
    ]
    tag = long_tag(h, side, eps, replay)
    hour = harness.run_server_hour(
        opts, info[side], tag, hp, eps, str(run.raw / tag), deadline=deadline
    )
    res = {
        "side": side,
        "hour": h,
        "replay": replay,
        "error": hour.get("error"),
        "server_died": hour.get("server_alive") is False,
        "epochs": {},
    }
    for e in hour["epochs"]:
        res["epochs"][e["params"]["seed"]] = verdict.epoch_values(
            e.get("checks", {}), LONG_CHECKS, NA
        )
    # the server died during the hour: the first epoch it did not finish is a crash on that seed
    # (the ones after it never ran)
    if res["server_died"]:
        lost = next((e for e in eps if e["seed"] not in res["epochs"]), None)
        if lost is not None:
            res["epochs"][lost["seed"]] = {
                **{c: None for c in LONG_CHECKS},
                "server_stays_up": False,
            }
    elif res["error"]:
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
        # the hard limit, first pair included
        if not time_for(opts, est, f"long hour {h} on both sides", run):
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
            run.row(
                "repro", side, 0, False, name, None if res.get("pass") is None else int(res["pass"])
            )
    first = {(r["name"], r["side"]): r for r in run.runs["repros"]}
    for name in names:
        c, b = first.get((name, "candidate"), {}), first.get((name, "baseline"), {})
        if (
            c.get("pass") is False
            and not c.get("crashed")
            and b.get("pass") is True
            and time_for(opts, 2 * REPRO_S, f"repro {name} repeat", run)
        ):
            for side in ("baseline", "candidate"):
                res = repros.run_repro(opts, info[side], f"repro-{name}-{side}-replay", name)
                run.runs["repros"].append({"name": name, "side": side, "replay": True, **res})
                run.row(
                    "repro",
                    side,
                    1,
                    True,
                    name,
                    None if res.get("pass") is None else int(res["pass"]),
                )
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
    return sorted(
        s_ for s_ in c if s_ in b and c[s_].get(check) is False and b[s_].get(check) is True
    )


def _v(x):
    """True / False / None / NA -> 1 / 0 / None (as judge_repeat takes them)."""
    return None if x in (None, NA) else int(x)


def _long_timing(chk, vals, sus, c_rep, b_rep):
    """A timing check across seeds: a candidate-only break counts only if its replay repeats it
    (verdict.judge_repeat); otherwise seeds broken on both sides give info, baseline-only breaks an
    improvement."""
    judged = [
        verdict.judge_repeat(0, 1, _v(c_rep.get(s_, {}).get(chk)), _v(b_rep.get(s_, {}).get(chk)))
        for s_ in sus
    ]
    if any(st == "regression" for _, _, st in judged):
        return 0, 1
    if any(c is None for c, _, _ in judged):
        return None, 0 if any(b is False for _, b in vals) else 1
    if any(c == 0 for c, _, _ in judged) or any(c is False and b is False for c, b in vals):
        return 0, 0
    if any(b is False and c is True for c, b in vals):
        return 1, 0
    # held wherever measured: a side held only if it was measured on every seed
    return (
        None if any(c is None for c, _ in vals) else 1,
        None if any(b is None for _, b in vals) else 1,
    )


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

        def pairs(cs, bs, seeds, chk=chk):
            out = [(cs[s_].get(chk), bs.get(s_, {}).get(chk)) for s_ in seeds]
            return [(c, b) for c, b in out if c != NA and b != NA]

        vals = pairs(c_first, b_first, paired)
        # immediate checks also count seeds (and replays) the baseline never ran
        lone_c = [
            c
            for c in (
                [c_first[s_].get(chk) for s_ in c_first if s_ not in b_first]
                + [c_rep[s_].get(chk) for s_ in c_rep if s_ not in b_rep]
            )
            if c != NA
        ]
        if chk not in LONG_IMMEDIATE:
            lone_c = []
        if not vals and not lone_c:
            continue
        if any(None in v for v in vals):
            unmeasured.append(f"long {chk}: not measured on every seed")
        # only seeds the baseline never ran: its value is unknown
        if chk in LONG_IMMEDIATE and not vals:
            cand = (
                0
                if any(c is False for c in lone_c)
                else (1 if all(c is True for c in lone_c) else None)
            )
            base = None
        elif chk in LONG_IMMEDIATE:
            seen = vals + pairs(c_rep, b_rep, [s_ for s_ in c_rep if s_ in b_rep])
            if any(c is False and b is True for c, b in seen):
                cand, base = 0, 1
            elif any(c is False and b is None for c, b in seen) or any(c is False for c in lone_c):
                # broke where the baseline measured nothing: still a break at once
                cand, base = 0, None
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
        status = kit.correctness_status(cand, base)
        if status == "ok" and any(c is False for c, _ in vals):
            status = "info"
        metric = "completed" if chk == "server_stays_up" else "invariant"
        findings.append(verdict._invariant(f"long {chk}", metric, cand, base, status))
    by = {}
    for r in repro_runs:
        by.setdefault(r["name"], {}).setdefault(r["side"], []).append(r)
    for name, sides in by.items():

        def pick(side, replay, sides=sides):
            return next((r for r in sides.get(side, []) if bool(r.get("replay")) == replay), {})

        c, b = pick("candidate", False), pick("baseline", False)
        ca, ba = pick("candidate", True), pick("baseline", True)
        cv, bv = _v(c.get("pass")), _v(b.get("pass"))
        if c.get("crashed") or ca.get("crashed"):
            # a crash counts at once, in the first run or the repeat, against the baseline's crashes
            ran = b.get("pass") is not None or ba.get("pass") is not None
            cv, bv = 0, 0 if b.get("crashed") or ba.get("crashed") else (1 if ran else None)
            status = kit.correctness_status(cv, bv)
        else:  # a timing break: it must repeat
            cv, bv, status = verdict.judge_repeat(cv, bv, _v(ca.get("pass")), _v(ba.get("pass")))
        if cv is None or bv is None:
            unmeasured.append(f"long repro {name}: not measured on both sides")
        findings.append(verdict._invariant(f"long repro {name}", "invariant", cv, bv, status))
    return findings, notes, unmeasured


# --- main -------------------------------------------------------------------------------------------


def summarise(verdict_s, findings, notes, errors) -> str:
    regs = [f["name"] for f in findings if f["status"] == "regression"]
    both = [
        f["name"]
        for f in findings
        if f["status"] == "info" and f["candidate"] == 0 and f["baseline"] == 0
    ]
    parts = []
    if verdict_s == "regression":
        parts.append("regression in " + ", ".join(regs))
    elif verdict_s == "error":
        parts.append("could not measure: " + "; ".join(errors[:2]))
    else:
        parts.append("no check broke on the candidate where it held on the baseline")
    if both:
        parts.append(
            f"{len(both)} broken on both sides (not new): "
            + ", ".join(b.replace("control ", "") for b in both)
        )
    parts += notes
    return "; ".join(parts)  # kit.result makes it one line and cuts it


def build_result(args, started, findings, notes, unmeasured, error, machine_info=None) -> dict:
    """The result.json document (contract schema 1) from the findings of every phase: an error
    first, then what was not measured."""
    errors = [*([error] if error else []), *unmeasured]
    if not findings and not errors:
        errors = ["nothing was measured"]
    ordered = verdict.worst_first(findings)
    outcome = kit.verdict(kit.select(ordered), [*kit.input_errors(args), *errors])
    return kit.result(
        CHECK,
        args,
        started=started,
        findings=ordered,
        summary=summarise(outcome, findings, notes, errors),
        errors=errors,
        artifacts=ARTIFACTS,
        machine_info=machine_info,
    )


class Check:
    """One run of the check; kit.run_check calls work(), then build(error), then cleanup()."""

    def __init__(self, args):
        self.args = args
        self.started = kit.utc_now()
        self.t_start = time.time()
        self.run: Run | None = None
        self.findings: list[dict] = []
        self.run_id = f"perf-fuzz-{os.getpid()}"
        self.launched = False

    def work(self) -> None:
        args = self.args
        self.run = run = Run(Path(args.out))  # absolute by now
        if kit.input_errors(args):
            return  # build() reports them
        opts = {
            "epoch_s": args.epoch_s,
            "drain_s": args.drain_s,
            "max_conns": args.max_conns,
            "net": args.net,
            "name_prefix": self.run_id,
            "labels": {LABEL: self.run_id},
            "server_cpus": os.environ.get("PERF_SERVER_CPUS") or None,
            "client_cpus": os.environ.get("PERF_CLIENT_CPUS") or None,
            "long_scale_range": (
                tuple(float(x) for x in args.long_scale_range.split(","))
                if args.long_scale_range
                else (0.25, 2.0, 4.0)
            ),
            "repros": [x for x in (args.repros or "").split(",") if x],
        }
        mode = MODES[args.mode]
        opts["deadline"] = self.t_start + mode["deadline_s"]
        from . import harness

        harness.set_start_method()
        info = {}
        self.launched = True
        for side in kit.SIDES:
            try:
                info[side] = harness.probe_image(getattr(args, side))
            except RuntimeError as e:
                raise RuntimeError(f"{side} image: {e}") from None
        with open(run.raw / "images.json", "w") as f:
            json.dump(info, f, indent=1)
        if not args.skip_control:
            scales = tuple(float(x) for x in args.control_scales.split(","))
            runs = control_phase(opts, info, run, scales)
            cf, notes = verdict.control_findings(runs, any(r["extra"] for r in runs))
            self.findings += cf
            run.notes += notes
            for f in cf:
                if f["candidate"] is None or f["baseline"] is None:
                    run.unmeasured.append(f"{f['name']}: not measured on every run")
        if mode["build"]:
            bruns = build_phase(opts, info, run, args.build_scale)
            bf, notes = verdict.build_findings(bruns)
            self.findings += bf
            run.notes += notes
            for f in bf:
                if f["candidate"] is None or f["baseline"] is None:
                    run.unmeasured.append(f"{f['name']}: not measured on both sides")
            run.unmeasured += unconfirmed_builds(notes)
            crash = build_crash_finding(bruns)
            if crash:
                self.findings.append(crash)
        if mode["long_s"]:
            budget = args.long_hours * 3600 if args.long_hours else mode["long_s"]
            seed = args.seed or f"long-{args.candidate.rsplit(':', 1)[-1][:12]}"
            run.notes.append(f"long seed {seed}")
            hours = long_phase(opts, info, run, seed, self.t_start + budget)
            lf, notes, unmeasured = long_findings(hours, run.runs["repros"])
            self.findings += lf
            run.notes += notes
            run.unmeasured += unmeasured

    def build(self, error: str | None) -> dict:
        """raw/rows.csv (every finding, ok rows included) and raw/runs.json, then the result."""
        args = self.args
        refused = kit.input_errors(args)
        if refused:
            return kit.error_result(CHECK, args, self.started, "; ".join(refused))
        run = self.run or Run(Path(args.out))
        for f in self.findings:
            for side in kit.SIDES:
                run.row(f"finding ({f['status']})", side, "", False, f["name"], f[side])
        run.save()
        return build_result(args, self.started, self.findings, run.notes, run.unmeasured, error)

    def cleanup(self) -> None:
        """Remove this run's containers (labelled LABEL=<run id>), and only those."""
        if self.launched:
            kit.remove_labelled(f"{LABEL}={self.run_id}")


def main(argv=None) -> int:
    """`bench fuzz`: exits 0 whenever result.json was written, whatever its verdict."""
    args = parse_args(argv)
    check = Check(args)
    result = kit.run_check(CHECK, args, check.work, check.build, check.cleanup)
    print(f"verdict: {result['verdict']}: {result['summary']}", flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
