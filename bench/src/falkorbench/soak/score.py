"""Score soak results. usage: python score.py <results-root> [--layout full|smoke]

full layout:  <root>/<scenario>-rust/summary.json and <root>/<scenario>-c/summary.json, each once the
              runner marked it finished (.done); a finished archive <scenario>-rust@<label> stands in
              while the Rust run repeats, and a C run that did not exit 0 is no baseline (see main)
smoke layout: <root>/work/<scenario>/<run>/summary.json — the agent's largest Rust run and its C run
              are picked by meta.engine / module path heuristics (see pick_smoke).
Writes <root>/scores.json.

Rubric per scenario (100):
  correctness 40  any critical check failed -> 0; each major -8; each minor -2
  stability   25  crash markers or server dead while not intentionally killed -> 0;
                  unexpected error rate (ops not starting with "_" or "expect"/"chaos"/"bad") :
                  25 * (1 - min(1, err_rate / 0.01))
  resources   15  7.5 for RSS slope during soak (<= 1% of rss_max per hour full, >= 10%/h zero)
                  7.5 for peak RSS vs C (<= 1.0 full, >= 2.0 zero; no C -> 7.5 * 0.5)
  performance 20  10 for p99 vs C, geometric mean over shared ops (<= 1.0 full, >= 3.0 zero)
                  5 for throughput vs C on shared ops (>= 0.95 full, <= 0.5 zero)
                  5 for latency drift (worst late/early p99 ratio <= 1.5 full, >= 4 zero)
Release score = mean of scenario scores; release is BLOCKED if any critical check fails on Rust,
or a Rust run did not exit 0 (failed_runs; scored if it left a summary).
"""
import json
import math
import sys
from pathlib import Path

EXPECTED_ERR = ("_", "e1_", "e2_", "e3_", "e4_", "e5_", "e6_", "e7_", "e8_", "e9_", "expect", "chaos", "bad", "malformed", "err_", "timeout", "kill", "slow",
                "fail", "ro_write", "violat", "invalid")


def clamp(x, lo=0.0, hi=1.0):
    return max(lo, min(hi, x))


def lin(x, good, bad):
    """1 at good, 0 at bad, linear in between (works for either direction)."""
    if x is None or (isinstance(x, float) and math.isnan(x)):
        return None
    return clamp((x - bad) / (good - bad))


def unexpected(op):
    return not op.lower().startswith(EXPECTED_ERR) and not any(k in op.lower() for k in EXPECTED_ERR[1:])


def pre_soak_rss(d):
    """Peak RSS before the first load phase = memory for the loaded dataset (equal data on both engines)."""
    m = d / "monitor.csv"
    if not m.exists():
        return None
    best = None
    lines = m.read_text().splitlines()[1:]
    if not any(len(l.split(",")) >= 8 and l.split(",")[7].startswith(("soak", "load")) for l in lines):
        return None  # run never reached its load phase (e.g. crashed during ingest): no equal-data point
    for l in lines:
        p = l.split(",")
        if len(p) < 8:
            continue
        if p[7].startswith(("soak", "load")):
            break
        try:
            v = float(p[2])
        except ValueError:
            continue
        if v == v:
            best = v if best is None else max(best, v)
    return best


def total_ok_ops(summ):
    return sum(d["throughput_ops_s"] for o, d in summ.get("ops", {}).items() if unexpected(o))


def score(rust, c, rdir=None, cdir=None):
    out = {"notes": []}
    checks = rust.get("checks", [])
    ov = [o for o in OVERRIDES if o["scenario"] == rust.get("meta", {}).get("scenario") and o["engine"] == "rust"]
    if ov:
        kept = []
        for x in checks:
            hit = next((o for o in ov if not x["ok"] and o.get("check_prefix") and x["name"].startswith(o["check_prefix"])), None)
            if hit:
                out["notes"].append(f"override: '{x['name']}' counted as pass. {hit['reason']}")
                x = {**x, "ok": True}
            kept.append(x)
        checks = kept
    crit = [x for x in checks if not x["ok"] and x.get("severity") == "critical"]
    major = [x for x in checks if not x["ok"] and x.get("severity") == "major"]
    minor = [x for x in checks if not x["ok"] and x.get("severity") == "minor"]
    corr = 0 if crit else max(0, 40 - 8 * len(major) - 2 * len(minor))

    ops = rust.get("ops", {})
    n = sum(d["count"] for o, d in ops.items() if unexpected(o))
    e = sum(d["errors"] for o, d in ops.items() if unexpected(o))
    er = e / n if n else 0
    crashed = bool(rust.get("server_log_crash_markers"))
    stab = 0 if crashed else 25 * (1 - min(1, er / 0.01))

    mon, cmon = rust.get("monitor", {}), (c or {}).get("monitor", {})
    rmax = mon.get("rss_max_mb") or 0
    slope = mon.get("rss_slope_mb_per_h")
    # Memory growth: workloads that add data grow on both engines, so compare used_memory growth
    # with C's on the same workload when available; else fall back to RSS slope as % of peak per hour.
    um_slope, c_um_slope = mon.get("used_mem_slope_mb_per_h"), cmon.get("used_mem_slope_mb_per_h")
    r_ops, c_ops = total_ok_ops(rust), total_ok_ops(c or {})
    if um_slope is not None and c_um_slope is not None and c_um_slope > 1 and r_ops and c_ops:
        # growth per unit of work done: engines run at different throughputs and durations
        growth_ratio = (um_slope / r_ops) / (c_um_slope / c_ops)
        s_slope = lin(growth_ratio, 1.2, 3.0)
        out["notes"].append(f"used_memory growth vs C: {growth_ratio:.2f}x")
    else:
        slope_pct = (slope / rmax * 100) if (slope is not None and rmax) else None
        s_slope = lin(slope_pct, 1, 10)
    r_pre, c_pre = (pre_soak_rss(rdir) if rdir else None), (pre_soak_rss(cdir) if cdir else None)
    if r_pre and c_pre:
        mem_ratio = r_pre / c_pre  # same dataset loaded on both
        out["notes"].append(f"RSS after ingest: Rust {r_pre:.0f} MB, C {c_pre:.0f} MB")
    else:
        mem_ratio = (rmax / cmon["rss_max_mb"]) if cmon.get("rss_max_mb") else None
    s_mem = lin(mem_ratio, 1.0, 2.0)
    res = 7.5 * (s_slope if s_slope is not None else 0.5) + 7.5 * (s_mem if s_mem is not None else 0.5)

    cops = (c or {}).get("ops", {})
    shared = [o for o in ops if o in cops and unexpected(o) and ops[o]["count"] > 50 and cops[o]["count"] > 50
              and ops[o]["p99_ms"] == ops[o]["p99_ms"] and cops[o]["p99_ms"] == cops[o]["p99_ms"]]
    p99r = {o: ops[o]["p99_ms"] / cops[o]["p99_ms"] for o in shared if cops[o]["p99_ms"] > 0}
    gm = math.exp(sum(math.log(v) for v in p99r.values()) / len(p99r)) if p99r else None
    thr = {o: ops[o]["throughput_ops_s"] / cops[o]["throughput_ops_s"] for o in shared
           if cops[o]["throughput_ops_s"] > 0}
    thr_gm = math.exp(sum(math.log(max(v, 1e-6)) for v in thr.values()) / len(thr)) if thr else None
    drifts = [d["drift"]["ratio"] for o, d in ops.items()
              if unexpected(o) and d.get("drift") and d["drift"].get("ratio")]
    worst_drift = max(drifts) if drifts else None
    s_p99, s_thr, s_drift = lin(gm, 1.0, 3.0), lin(thr_gm, 0.95, 0.5), lin(worst_drift, 1.5, 4.0)
    gw = next((o for o in OVERRIDES if o.get("growth_workload") and o["scenario"] == rust.get("meta", {}).get("scenario")), None)
    if gw:  # data grows by design: slope and drift measure the data, not the engine
        s_drift = 1.0
        res = 7.5 * 1.0 + 7.5 * (s_mem if s_mem is not None else 0.5)
        out["notes"].append("growth workload: " + gw["reason"])
    perf = (10 * (s_p99 if s_p99 is not None else 0.5) + 5 * (s_thr if s_thr is not None else 0.5)
            + 5 * (s_drift if s_drift is not None else 0.5))
    out.update({
        "correctness": corr, "stability": round(stab, 1), "resources": round(res, 1),
        "performance": round(perf, 1), "total": round(corr + stab + res + perf, 1),
        "critical_failures": crit, "major_failures": major, "minor_failures": minor,
        "unexpected_err_rate": er, "crashed": crashed, "rss_max_mb": rmax, "rss_slope_mb_h": slope,
        "rss_vs_c": mem_ratio, "p99_vs_c_gmean": gm, "p99_vs_c": p99r, "thr_vs_c_gmean": thr_gm,
        "worst_drift": worst_drift, "cpu_avg": mon.get("cpu_avg_pct"), "c_cpu_avg": cmon.get("cpu_avg_pct"),
        "c_failed_checks": [x["name"] for x in (c or {}).get("checks_failed", [])]})
    return out


OVERRIDES = []


def finished(d):
    """The runner marked the run finished (run_*.sh write .rc, then .done, whatever the exit)."""
    return (d / ".done").exists()


def exit_status(d):
    """The finished run's exit status (.rc), or None when it has none to read."""
    try:
        return int((d / ".rc").read_text().strip())
    except (OSError, ValueError):
        return None


def succeeded(d):
    return finished(d) and exit_status(d) == 0


def read_summary(p):
    """summary.json's content, or None when there is none or it does not parse (a run killed
    while it wrote the file)."""
    try:
        return json.loads(p.read_text())
    except (OSError, ValueError):
        return None


def rust_runs(root):
    """The Rust run to score per scenario, and the scenarios with none finished yet.

    The current <scenario>-rust counts once it finished (a run in progress has written
    summary.json only part way). While it runs again, or after run_rust_main.sh discarded
    all its contended attempts, the newest finished archive <scenario>-rust@<label> (any
    label, by .done time) stands in for it."""
    runs, running = {}, set()
    for d in sorted(root.glob("*-rust")):
        name = d.name[:-len("-rust")]
        if finished(d):
            runs[name] = d
        else:
            running.add(name)
    archives = sorted((d for d in root.glob("*-rust@*") if finished(d)),
                      key=lambda d: (d / ".done").stat().st_mtime, reverse=True)
    for d in archives:
        runs.setdefault(d.name.split("-rust@")[0], d)
    return runs, sorted(running - set(runs))


def main():
    global OVERRIDES
    root = Path(sys.argv[1])
    if (root / "overrides.json").exists():
        OVERRIDES = json.loads((root / "overrides.json").read_text())
    res = {}
    failed = {}  # scenario -> why its Rust run failed: a blocker, scored or not
    runs, pending = rust_runs(root)
    for name, rdir in sorted(runs.items()):
        rc = exit_status(rdir)
        if rc != 0:
            failed[name] = f"{rdir.name} exited {rc if rc is not None else 'with no status'}"
        rust = read_summary(rdir / "summary.json")
        if rust is None:  # nothing to score: a blocker whatever its exit status said
            failed.setdefault(name, f"{rdir.name} left no readable summary.json")
            continue
        base = [root / f"{name}-c", root / f"{name}-c@jul-master"]  # C re-run in progress: the old one meanwhile
        same = [root / f"{name}-c-samewindow", root / f"{name}-c-samewindow@jul-master"]
        if not any(finished(d) for d in base + same):
            pending.append(name)  # score only once a C baseline has finished
            continue
        # a C run that timed out or crashed is no baseline: the first that exited 0, the
        # same-size C rerun before the short baseline
        cdir = next((d for d in same if succeeded(d)), None)
        notes = []
        if cdir is None:
            cdir = next((d for d in base if succeeded(d)), None)
            if cdir is None:
                notes.append("no C baseline exited 0: scored without C ratios")
            elif any(o.get("no_c_compare") and o["scenario"] == name
                     and (o.get("build") != "jul-master" or not (cdir / ".commit").exists()) for o in OVERRIDES):
                cdir = None  # C baseline not comparable: score without C ratios
        c = read_summary(cdir / "summary.json") if cdir else None
        if cdir and c is None:  # no C ratios at all: not even RSS from its monitor.csv
            notes.append(f"{cdir.name} left no readable summary.json: scored without C ratios")
            cdir = None
        res[name] = score(rust, c, rdir, cdir)
        if name in failed:
            notes.append(f"run failed: {failed[name]}")
        res[name]["notes"] += notes
        res[name]["run"], res[name]["c_run"] = rdir.name, cdir.name if cdir else None  # what was scored
    tot = [v["total"] for v in res.values()]
    blocked = sorted({k for k, v in res.items() if v["critical_failures"] or v["crashed"]} | set(failed))
    summary = {"scenarios": res, "release_score": round(sum(tot) / len(tot), 1) if tot else None,
               "blocked_by": blocked, "failed_runs": failed, "pending": sorted(pending)}
    (root / "scores.json").write_text(json.dumps(summary, indent=1, default=str))
    for k, v in res.items():
        print(f"{k:28} total {v['total']:5.1f}  C{v['correctness']:4} S{v['stability']:5} "
              f"R{v['resources']:5} P{v['performance']:5}  p99/C={v['p99_vs_c_gmean']}  rss/C={v['rss_vs_c']}")
    print("release score", summary["release_score"], "blocked by", blocked)


if __name__ == "__main__":
    main()
