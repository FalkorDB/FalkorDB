#!/usr/bin/env python3
"""Per-flow-test-file benchmark: redis-server instructions/cycles/peak memory
plus wall time and pass/fail counts. Compare C vs Rust module per flow file.

Runs each flow test file through ./flow.sh (RLTest, --parallelism 1) against a
given module. RLTest spawns transient redis-server processes (and macOS wait4
rusage does NOT fold grandchildren), so a background thread polls every
redis-server pid that appears during the run via proc_pid_rusage and keeps its
last-seen counters; the row is the SUM over all servers the file spawned.
Teardown work after the final poll (~25 ms) is lost — fine for comparisons.

Usage:
  python3 bench/flow_bench.py --out bench/results/flow_rust.csv
  python3 bench/flow_bench.py \
      --module ~/repos/FalkorDB/bin/macos-arm64v8-release/falkordb.so \
      --out bench/results/flow_c.csv
  python3 bench/flow_bench.py --compare bench/results/flow_c.csv \
      [--current bench/results/flow_rust.csv]
  python3 bench/flow_bench.py test_index test_constraint ...   # subset, merged
"""
import argparse, csv, ctypes, os, re, subprocess, sys, tempfile, threading, time

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
SUMMARY_RE = re.compile(r"Total Tests Run:\s*(\d+), Total Tests Failed:\s*(\d+)")

FIELDS = ["file", "wall_s", "instr", "cycles", "peak_mem_mb", "servers",
          "tests_run", "tests_failed"]

# This tool measures per-process instructions via macOS `proc_pid_rusage`, which
# has no Linux equivalent. Fail with a clear message instead of an opaque ctypes
# load error. (`run_bench.py` does have a Linux backend; this one does not yet.)
if sys.platform != "darwin":
    sys.exit(
        "flow_bench.py needs macOS: per-process instruction counters come from "
        "proc_pid_rusage, which Linux does not provide. Use bench/run_bench.py, "
        "which has a perf-based Linux backend."
    )

libproc = ctypes.CDLL("/usr/lib/libproc.dylib")
RUSAGE_INFO_V4 = 4


def rusage(pid):
    buf = ctypes.create_string_buffer(1024)
    if libproc.proc_pid_rusage(ctypes.c_int(pid), ctypes.c_int(RUSAGE_INFO_V4), buf) != 0:
        return None
    u64 = (ctypes.c_uint64 * 40).from_buffer_copy(buf.raw[16:336])
    # ri_instructions, ri_cycles, ri_lifetime_max_phys_footprint
    return u64[29], u64[30], u64[28]


def redis_pids():
    # redis rewrites its proc title to "redis-server *:PORT", so -x won't match
    out = subprocess.run(["pgrep", "-f", "redis-server"],
                         capture_output=True, text=True).stdout
    return {int(p) for p in out.split()}


def flow_files():
    with open(os.path.join(ROOT, "flow_tests_done.txt")) as f:
        return [line.strip().removesuffix(".py") for line in f if line.strip()]


def run_one(test, env):
    pre = redis_pids()
    seen = {}  # pid -> last (instr, cycles, peak_footprint)
    stop = threading.Event()

    def poll():
        while not stop.is_set():
            for pid in redis_pids() - pre:
                r = rusage(pid)
                if r:
                    seen[pid] = r
            stop.wait(0.025)

    t = threading.Thread(target=poll, daemon=True)
    t0 = time.time()
    t.start()
    out = subprocess.run(["./flow.sh"], cwd=ROOT, capture_output=True,
                         text=True, env={**env, "TEST": test})
    stop.set()
    t.join()
    wall = time.time() - t0
    text = re.sub(r"\x1b\[[0-9;]*m", "", out.stdout + out.stderr)
    m = SUMMARY_RE.search(text)
    run, failed = (int(m.group(1)), int(m.group(2))) if m else (0, -1)
    return {
        "file": os.path.basename(test),
        "wall_s": round(wall, 2),
        "instr": sum(r[0] for r in seen.values()),
        "cycles": sum(r[1] for r in seen.values()),
        "peak_mem_mb": round(sum(r[2] for r in seen.values()) / 1e6, 1),
        "servers": len(seen),
        "tests_run": run,
        "tests_failed": failed,
    }


def compare(current, baseline):
    with open(current, newline="") as cur_f, open(baseline, newline="") as base_f:
        cur = {r["file"]: r for r in csv.DictReader(cur_f)}
        base = {r["file"]: r for r in csv.DictReader(base_f)}

    def ratio(b, c, key):
        bv, cv = float(b.get(key, 0) or 0), float(c.get(key, 0) or 0)
        return f"{cv / bv:>6.2f}" if bv else f"{'-':>6}"

    print(f"{'file':<32} {'base instr':>13} {'cur instr':>13} {'ratio':>6}  "
          f"{'cyc':>6} {'mem':>6} {'wall':>6}  {'run':>4} {'fail b/c':>8}")
    for name, b in base.items():
        c = cur.get(name)
        if not c:
            print(f"{name:<32} MISSING from current")
            continue
        print(f"{name:<32} {float(b['instr']):>13,.0f} {float(c['instr']):>13,.0f} "
              f"{ratio(b, c, 'instr')}  {ratio(b, c, 'cycles')} "
              f"{ratio(b, c, 'peak_mem_mb')} {ratio(b, c, 'wall_s')}  "
              f"{c['tests_run']:>4} {b['tests_failed']:>3}/{c['tests_failed']}")
    for name in cur:
        if name not in base:
            print(f"{name:<32} NEW (not in baseline)")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--module",
                    default=os.path.join(ROOT, "target/release/libfalkordb.dylib"))
    ap.add_argument("--out", default=os.path.join(HERE, "results/flow_current.csv"))
    ap.add_argument("--compare", metavar="BASELINE",
                    help="compare --current CSV against BASELINE CSV and exit")
    ap.add_argument("--current",
                    default=os.path.join(HERE, "results/flow_current.csv"))
    ap.add_argument("names", nargs="*",
                    help="flow file names (e.g. test_index); default: all in "
                         "flow_tests_done.txt")
    args = ap.parse_args()

    if args.compare:
        compare(args.current, args.compare)
        return

    tests = flow_files()
    if args.names:
        only = {n.removesuffix(".py") for n in args.names}
        tests = [t for t in tests if os.path.basename(t) in only]
        missing = only - {os.path.basename(t) for t in tests}
        if missing:
            sys.exit(f"unknown flow files: {missing}")

    env = {**os.environ, "VERBOSE": "0", "PARALLELISM": "--parallelism 1"}
    target = "libfalkordb.dylib" if sys.platform == "darwin" else "libfalkordb.so"
    module = os.path.abspath(os.path.expanduser(args.module))
    tmp = None
    if os.path.basename(module) == target:
        env["TARGET_DIR"] = os.path.dirname(module)
    else:
        # flow.sh hardcodes the module filename; symlink foreign modules
        # (e.g. the C falkordb.so) under the expected name
        tmp = tempfile.TemporaryDirectory()
        os.symlink(module, os.path.join(tmp.name, target))
        env["TARGET_DIR"] = tmp.name
    print(f"module: {module}\n{len(tests)} flow files", flush=True)

    rows = []
    for t in tests:
        row = run_one(t, env)
        rows.append(row)
        print(f"{row['file']:<32} {row['instr']/1e9:>7.2f}G instr "
              f"{row['cycles']/1e9:>7.2f}G cyc {row['peak_mem_mb']:>8.1f}MB "
              f"{row['wall_s']:>6.1f}s  {row['servers']} srv, "
              f"{row['tests_run']} run, {row['tests_failed']} failed", flush=True)

    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    merged = {}
    if os.path.exists(args.out):
        with open(args.out, newline="") as f:
            merged = {r["file"]: r for r in csv.DictReader(f)}
    for r in rows:
        merged[r["file"]] = {k: str(v) for k, v in r.items()}
    with open(args.out, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=FIELDS)
        w.writeheader()
        w.writerows(merged.values())
    print(f"wrote {args.out}")


if __name__ == "__main__":
    main()
