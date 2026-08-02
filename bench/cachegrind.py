#!/usr/bin/env python3
"""Deterministic per-query instruction counts via callgrind.

Why this exists: no hosted CI runner exposes a PMU. Measured, not assumed —
GCE rejects `--performance-monitoring-unit` on the v1 API and on the beta API
alike, `perf` there reports `<not supported>`, macOS runners return 0 for
`proc_pid_rusage`'s ri_instructions, and kperf inside a macOS runner fails with
`kpep_db_create failed: 7`. Hardware counters are simply unavailable.

Callgrind counts instructions in *software*, so it needs no PMU and no
privileges, and its counts are near-deterministic: far steadier than sampled
counters, though not bit-exact (see "Measured precision").

The cost is a large slowdown, so this measures a curated subset on a smaller
graph than `bench/run_bench.py`. See "Not comparable to run_bench.py" below.

## How a query is isolated: differencing, not windowing

Callgrind reports one total when the process exits. The obvious approach is to
window with `callgrind_control --instr=on/off --dump` around each query, and
that is what the first version of this script did. **It does not work in a
container.** `callgrind_control` reaches the process through vgdb FIFOs in
/tmp, and reproduced locally in `debian:trixie-slim` (valgrind 3.24.0, the CI
version):

    ==236== open fifo /tmp/vgdb-pipe-from-vgdb-to-236-by-???-on-???
    ==236== valgrind: fatal error: vgdb FIFO cannot be opened.

The server dies on the first control command, so every dump silently never
arrives and every query reports nothing. Setting USER/LOGNAME does not help;
`--vgdb-prefix` makes `callgrind_control` hang instead. In CI this produced
two empty CSVs after 16 minutes.

So instead each query is measured by **differencing two complete runs** of the
same query at different repeat counts:

    T(n2) = startup + setup + compile + n2 * exec
    T(n1) = startup + setup + compile + n1 * exec
    exec  = (T(n2) - T(n1)) / (n2 - n1)

Startup, graph setup and one-time plan compilation appear identically in both
runs and cancel *exactly* — not approximately — because the counts are
deterministic. Nothing but the query's steady-state execution survives the
subtraction. No vgdb, no dumps, no timing races, and each run reads a single
number from a single file at exit.

The price is two valgrind runs per query, each paying setup, which is why
`CG_SETUP` below is deliberately small.

## Measured precision

Measured on arm64 against a bare redis-server (see GRAPH_CMD for why the module
is not used there), with `--hz 1` and one connection per run:

  - three identical runs: totals within 3,450 instr, 0.015% of the total
  - two independent (n1, n2) pairs on the same command agreed to 0.45%

The residual comes from work whose amount depends on how long the process
lives, so it appears as a roughly fixed absolute drift per run rather than a
per-execution error: drift spread over a span of `n2 - n1` executions.

**The 3.5k figure above is the bare-redis case and badly understates the real
thing.** With the module loaded, CI measured per-run drift of ~300-600k
instructions on a ~236M baseline. At a fixed span of 100 that is 3-6k
instr/exec of error — nothing for a 7M-instruction query, but **6.7% for
`RETURN 1`**, which is how the control row came to read 1.0673x on two builds
whose Rust was byte-identical:

    RETURN 1              90,512 ->    96,607   1.0673x   (+6,095)
    create node          191,445 ->   192,544   1.0057x   (+1,099)
    arithmetic         1,465,420 -> 1,469,746   1.0030x   (+4,326)
    reduce             4,788,178 -> 4,787,865   0.9999x     (-313)
    list comprehension 7,414,221 -> 7,412,876   0.9998x   (-1,345)

Note the absolute deltas are all a few thousand regardless of query cost —
the error is absolute, so "treat sub-1% as noise" is wrong: it is far too
strict for expensive queries and far too lax for cheap ones.

So the span is now chosen per query as `DRIFT / (TARGET_REL * cost)`, which
holds the *differenced work* constant instead of the span. Cheap queries get a
wide span (a cheap thing to do — `RETURN 1` at span ~3300 is still only ~300M
instructions), expensive ones keep the default. Each row prints the span it
used and the resulting error bar.

An earlier version of this file claimed the counts were exactly reproducible
with "no run-to-run noise to threshold against". That was wrong, and the
validation above is what caught it: at the default hz=10 and with a fresh
connection per execution, two (n1, n2) pairs disagreed by 44%.

End-to-end confirmation from CI, comparing a branch that changes no Rust at
all against main — so every ratio should read 1.0000x:

    two-hop      3,492,183 -> 3,491,397   0.9998x
    reduce       4,787,089 -> 4,787,235   1.0000x
    CASE         2,161,954 -> 2,163,617   1.0008x
    arithmetic   1,463,580 -> 1,467,917   1.0030x

## Not comparable to run_bench.py

`CG_SETUP` builds a 1,000-node graph, not the 10,000-node one in queries.py,
and skips the vector/fulltext indexes, constraints, UDFs and DEBUG RELOAD that
the full harness sets up. Absolute numbers here are therefore *not* comparable
to run_bench.py rows. They are only meaningful as a main-vs-PR ratio, where
both sides run this identical setup.

Usage:
    bench/cachegrind.py --module path/to/libfalkordb.so [--out out.csv]
                        [--n1 3] [--n2 13] [query name ...]
"""

import argparse
import csv
import glob
import math
import os
import shutil
import subprocess
import sys
import time

# Per-run drift in the whole-process total, measured in CI with the module
# loaded: the two runs of a pair differ by ~300-600k instructions even for
# identical code. Divided by the span this becomes the per-execution error, so
# it is an *absolute* budget, not a relative one.
DRIFT_INSTR = 600_000
# Per-execution precision to aim for. span = DRIFT / (TARGET_REL * cost), so
# the differenced work per query is DRIFT/TARGET_REL = 300M instructions
# regardless of how cheap the query is — a few seconds under valgrind.
TARGET_REL = 0.002
MAX_SPAN = 4000

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
sys.path.insert(0, HERE)
from queries import QUERIES  # noqa: E402

SHLIB_EXT = "dylib" if sys.platform == "darwin" else "so"

# A small graph that supports the subset below. Deliberately not queries.py's
# SETUP: that builds 10k nodes, 10k edges, vector and fulltext indexes,
# constraints, UDFs and a DEBUG RELOAD, and every one of those instructions
# would be paid twice per measured query under instrumentation.
#
# The Person index is created before the ring so the ring build is index-driven
# rather than a 1000x1000 nested scan.
CG_SETUP = [
    (
        "UNWIND range(0, 999) AS i "
        "CREATE (:Person {id: i, name: 'p' + toString(i), age: i % 80, score: i * 1.5})"
    ),
    "CREATE INDEX FOR (p:Person) ON (p.id)",
    (
        "UNWIND range(0, 999) AS i "
        "MATCH (a:Person {id: i}) MATCH (b:Person {id: (i + 1) % 1000}) "
        "CREATE (a)-[:KNOWS]->(b)"
    ),
    # `delete node` deletes one :Tmp per execution, so there must be more of
    # them than the highest repeat count — which is now `MAX_SPAN` plus n1,
    # since a cheap delete query would get its span widened. Running dry would
    # not fail loudly: the remaining executions would measure a no-op delete
    # and quietly halve the reported cost.
    "UNWIND range(0, 4999) AS i CREATE (:Tmp {x: i})",
]

# The command each statement/query is sent as. A seam: valgrind on arm64 cannot
# run this module at all (`unhandled instruction 0xB8BFC108` — an ARMv8.1 LSE
# atomic in RediSearch's slots_tracker, valgrind's limitation rather than a
# module bug), so the differencing mechanism is validated on arm64 against a
# plain redis-server by patching this to () and CG_SETUP to [].
GRAPH_CMD = ("GRAPH.QUERY", "bench")

# Extra `--loadmodule` arguments, set from --module-args.
#
# This exists because thread count destroys reproducibility under valgrind. The
# C engine defaults to a pool sized to the host ("Thread pool created, using 11
# threads") while this module runs queries inline; valgrind serialises threads
# but schedules them nondeterministically, and the same C module measured on two
# consecutive CI runs disagreed by up to 10x on the same query (`reduce`
# 3,027,301 then 32,263,982). Pinning every engine to one thread is both fairer
# and the only way these numbers mean anything.
MODULE_ARGS = []

# One representative per family that has shown movement in this work.
#
# Each query costs two instrumented server lifecycles and each lifecycle pays
# CG_SETUP again, which sounded expensive enough to keep this list to five —
# but CI measured ~10s per query (a run is ~236M instructions, a few seconds
# under valgrind), so the whole set costs well under two minutes per build.
# Widen it freely. Extra queries can also be passed as positional arguments
# without editing this list.
DEFAULT_SUBSET = [
    "RETURN 1",              # fixed per-query floor; the control
    "arithmetic",            # scalar expression evaluation
    "CASE",                  # branchy expression + property access
    "list comprehension",    # scoped iteration
    "reduce",                # accumulator loop
    "create node",           # small write path
    "delete node",           # small delete path
    "two-hop",               # matrix traversal
]


# Names must exist in queries.py. Checked at import rather than after a
# 2-minute build and a container spin-up, which is how `traverse 2 hops` — a
# name I assumed rather than looked up — reached CI.
_UNKNOWN = sorted(set(DEFAULT_SUBSET) - {q[0] for q in QUERIES})
assert not _UNKNOWN, f"DEFAULT_SUBSET names not in queries.py: {_UNKNOWN}"


def parse_total(path):
    """Instruction count from a callgrind output file.

    Callgrind writes `totals:` (and `summary:`) with the first field being
    instruction reads. Returns None when neither is present, which happens for
    a file still being written.
    """
    try:
        with open(path, errors="replace") as f:
            for line in f:
                if line.startswith(("totals:", "summary:")):
                    parts = line.split(":", 1)[1].split()
                    if parts:
                        return int(parts[0])
    except OSError:
        return None
    return None


def run_total(module, port, outdir, query, reps, also_run=()):
    """Run one instrumented server lifecycle; return its total instruction count.

    The server runs `query` `reps` times after building CG_SETUP, then exits so
    callgrind writes its total. `also_run` executes each of those queries once
    first — used only by the warm-up, to force every kernel the subset needs to
    be compiled and cached before anything is measured.
    """
    shutil.rmtree(outdir, ignore_errors=True)
    os.makedirs(outdir, exist_ok=True)

    cmd = [
        "valgrind",
        "--tool=callgrind",
        f"--callgrind-out-file={outdir}/callgrind.out.%p",
        "redis-server",
        "--port", str(port),
        "--save", "",
        # serverCron does work proportional to how long the process lives, and
        # under instrumentation the two runs being differenced live for
        # different durations — so cron lands in the subtraction as drift.
        # Measured at the default hz=10 it is ~240k instr per second of life,
        # which swamped a PING (~20k) and made two (n1,n2) pairs disagree by
        # 44%. hz=1 is the lowest redis accepts and cuts it 10x.
        "--hz", "1",
    ]
    # module=None runs a bare redis-server. Used only by the arm64 validation
    # described at GRAPH_CMD, where the module cannot run under valgrind.
    if module:
        cmd += ["--loadmodule", module, *MODULE_ARGS]

    server = subprocess.Popen(cmd, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)

    try:
        for _ in range(1200):  # instrumented startup is far slower than native
            if server.poll() is not None:
                raise RuntimeError("server exited during startup under callgrind")
            if cli(port, "ping", check=False).strip() == "PONG":
                break
            time.sleep(0.5)
        else:
            raise RuntimeError("server did not answer PING under callgrind")

        for stmt in CG_SETUP:
            cli(port, *GRAPH_CMD, stmt)

        for extra in also_run:
            cli(port, *GRAPH_CMD, extra, check=False)

        # One redis-cli with -r, not `reps` of them: a fresh connection per
        # execution would put accept/handshake/teardown into the measurement,
        # and the extra wall time feeds the cron drift described above.
        if reps:
            cli(port, "-r", reps, *GRAPH_CMD, query)

        cli(port, "shutdown", "nosave", check=False)
        server.wait(timeout=600)
    finally:
        if server.poll() is None:
            server.terminate()
            try:
                server.wait(timeout=120)
            except subprocess.TimeoutExpired:
                server.kill()

    # Redis forks, and valgrind profiles the child too; the child's total is
    # tiny, so the server's own run is the maximum.
    totals = [
        t
        for t in (parse_total(p) for p in glob.glob(os.path.join(outdir, "callgrind.out.*")))
        if t is not None
    ]
    if not totals:
        raise RuntimeError(f"no parseable callgrind output in {outdir}")
    return max(totals)


def cli(port, *args, check=True, timeout=1800):
    # Under callgrind a query can take minutes, but never forever: without a
    # timeout a wedged redis-cli would hang the whole job instead of failing
    # the one query.
    try:
        out = subprocess.run(
            ["redis-cli", "-p", str(port)] + [str(a) for a in args],
            capture_output=True,
            text=True,
            timeout=timeout,
        )
    except subprocess.TimeoutExpired as e:
        raise RuntimeError(
            f"redis-cli {' '.join(str(a) for a in args)} timed out after {timeout}s"
        ) from e
    if check and out.returncode != 0:
        raise RuntimeError(
            f"redis-cli {' '.join(str(a) for a in args)} failed "
            f"(exit {out.returncode}): {out.stderr.strip()[:200]}"
        )
    return out.stdout


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--module", default=os.path.join(ROOT, f"target/release/libfalkordb.{SHLIB_EXT}"))
    ap.add_argument("--port", default="6402")
    ap.add_argument("--out", default=os.path.join(HERE, "results/cachegrind.csv"))
    ap.add_argument("--n1", type=int, default=20, help="low repeat count")
    ap.add_argument(
        "--n2",
        type=int,
        default=120,
        help="high repeat count. exec cost = (T(n2) - T(n1)) / (n2 - n1). The "
        "span divides the per-run drift, so a wider span is more precise: at "
        "span 100 the ~3.5k drift is ~35 instr/exec.",
    )
    ap.add_argument(
        "--module-args",
        nargs="*",
        default=[],
        help="extra --loadmodule arguments, e.g. --module-args THREAD_COUNT 1. "
        "Pin the thread count: see MODULE_ARGS.",
    )
    ap.add_argument("names", nargs="*", help="queries to measure (default: a curated subset)")
    args = ap.parse_args()
    MODULE_ARGS[:] = args.module_args

    if args.n2 <= args.n1:
        sys.exit(f"--n2 ({args.n2}) must exceed --n1 ({args.n1})")

    for tool in ("valgrind", "redis-server", "redis-cli"):
        if not shutil.which(tool):
            sys.exit(
                f"{tool} not found. This tool needs valgrind and redis on PATH. "
                "Note valgrind does not support macOS on Apple silicon, so this "
                "is Linux-only in practice."
            )
    if not os.path.exists(args.module):
        sys.exit(f"module not found: {args.module}")

    wanted = args.names or DEFAULT_SUBSET
    queries = [q for q in QUERIES if q[0] in set(wanted)]
    missing = set(wanted) - {q[0] for q in queries}
    if missing:
        sys.exit(f"unknown queries: {sorted(missing)}")

    outdir = os.path.join(HERE, "results/callgrind")
    span = args.n2 - args.n1
    rows = []

    # Warm-up, discarded. GraphBLAS compiles kernels on first use and caches
    # them on disk, so the first server lifecycle in a job pays for that and no
    # later one does. Measured: the first run came in ~30M instructions above
    # its pair, which made T(n2) < T(n1) and cost the `RETURN 1` control row —
    # the skip guard caught it rather than reporting a negative cost.
    #
    # The warm-up runs *every* query in the subset once, not just one. GraphBLAS
    # compiles kernels on first use, and a query whose kernels no earlier query
    # needed pays that compile inside its own first measured run. Measured on
    # the C engine with a one-query warm-up: `two-hop` reported
    # T(n1)=3,384,143,011 against T(n2)=2,546,675,797 — the n1 run cost 838M
    # *more* while doing 3,000 fewer queries, so the guard skipped the row. Same
    # for `RETURN 1`, the first row measured. Warming the whole subset removes
    # the asymmetry.
    print("warm-up run (compiles every kernel the subset needs)...", flush=True)
    try:
        run_total(
            args.module,
            args.port,
            outdir,
            "RETURN 1",
            1,
            also_run=[q for _, _, q, *_ in queries],
        )
    except (RuntimeError, subprocess.TimeoutExpired) as e:
        sys.exit(f"warm-up run failed, so nothing else will work: {e}")

    for name, _is_write, q, *_rest in queries:
        t0 = time.time()
        try:
            lo = run_total(args.module, args.port, outdir, q, args.n1)
            hi = run_total(args.module, args.port, outdir, q, args.n2)
        except (RuntimeError, subprocess.TimeoutExpired) as e:
            print(f"{name:<24} FAILED: {e}", flush=True)
            continue

        if hi <= lo:
            # Deterministic counts cannot go down when work is added; if this
            # trips, the two runs did not do what the model above assumes.
            print(
                f"{name:<24} SKIPPED: T(n2)={hi:,} <= T(n1)={lo:,} — "
                "the runs are not differing only by repeat count",
                flush=True,
            )
            continue

        per = (hi - lo) / span
        used_span, note = span, ""

        # Widen the span for cheap queries. The per-run drift is roughly a
        # fixed number of instructions, so dividing it by the span makes the
        # *absolute* error per execution fixed — which is negligible for a
        # 7M-instruction query and enormous for a 90k one. Choosing
        # span = DRIFT / (TARGET_REL * cost) makes the differenced work
        # constant instead, so precision is uniform and the extra cost is
        # bounded at DRIFT/TARGET_REL instructions per query.
        if per > 0:
            want = math.ceil(DRIFT_INSTR / (TARGET_REL * per))
            want = min(want, MAX_SPAN)
            if want > span:
                try:
                    hi2 = run_total(args.module, args.port, outdir, q, args.n1 + want)
                except (RuntimeError, subprocess.TimeoutExpired) as e:
                    print(f"{name:<24} refine failed, keeping span {span}: {e}", flush=True)
                    hi2 = None
                if hi2 is not None and hi2 > lo:
                    per = (hi2 - lo) / want
                    used_span, note = want, f" [span widened {span}->{want}]"

        rows.append({"query": name, "instr": f"{per:.0f}"})
        print(
            f"{name:<24}{per:>15,.0f} instr/exec   "
            f"(span {used_span}, +-{DRIFT_INSTR / used_span / per * 100:.2f}%, "
            f"{time.time() - t0:.0f}s){note}",
            flush=True,
        )

    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    with open(args.out, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=["query", "instr"])
        w.writeheader()
        w.writerows(rows)
    print(f"wrote {args.out} ({len(rows)} rows)", flush=True)
    if not rows:
        sys.exit("no query produced a count")


if __name__ == "__main__":
    main()
