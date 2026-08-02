#!/usr/bin/env python3
"""Benchmark harness: per-query instructions/cycles (+ optional PMU counters).

Starts a redis-server with the module, builds the benchmark graph, then for
each query measures redis-server-process instructions & cycles (macOS
proc_pid_rusage, no root needed) around `redis-benchmark -c 1 -n N`.
If ./pmc_tool exists and works (setuid root or run under sudo), also records
system-wide branches / branch-misses / L1D-misses, idle-adjusted.
If the server is jemalloc-built (stock redis is), also records per-query
allocated/deallocated bytes from `MEMORY MALLOC-STATS` merged-arena deltas.

Usage:
  python3 bench/run_bench.py [options] [query name ...]
    --module PATH   module to load (default target/release/libfalkordb.<dylib|so>)
    --port P        (default 6399)
    --out FILE      CSV output (default bench/results/current.csv); named
                    queries are merged into an existing CSV, so subset re-runs
                    patch only those rows
    --n N           requests per query (default 1000)
    --once          run each query exactly once via redis-cli, no measurement
                    (used by coverage.sh; server env gets LLVM_PROFILE_FILE
                    passthrough automatically)
    --keep-server   leave the server running after the run (for profiling)
    --reuse         assume a server is already on --port with the graph built
    --c-compat      benchmarking the C FalkorDB module: skip setup commands
                    known to crash it (composite unique constraint) and drop
                    queries whose warmup reply is an error instead of writing
                    artifact rows
"""
import argparse, csv, ctypes, os, shutil, signal, subprocess, sys, time

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
sys.path.insert(0, HERE)
from queries import QUERIES, SETUP, SETUP_COMMANDS, ERROR_QUERIES, IMPORT_DIR, CSV_FILES

# Path to the server log, set when this harness starts the server. Stays None
# with --reuse, where the server is someone else's and its log is not ours.
SERVER_LOG = None

# Per-process instruction/cycle counters, by platform.
#
#   macOS  "rusage" — `proc_pid_rusage` gives a running total for any pid with
#          no privileges, so a window is read-before/read-after.
#   Linux  "perf"   — there is no rusage equivalent; the PMU is reached through
#          `perf stat -p <pid> -- <cmd>`, which measures the redis process for
#          exactly as long as `cmd` runs. That is a window measurement, not a
#          running total, so the two backends cannot share one code path —
#          hence `run_and_count` below.
#   none            — neither available. instr/cycles are then reported EMPTY,
#          never substituted with wall-clock: a time-based stand-in would turn
#          the regression gate into a noise detector while still looking like a
#          measurement.
SHLIB_EXT = "dylib" if sys.platform == "darwin" else "so"

if sys.platform == "darwin":
    libproc = ctypes.CDLL("/usr/lib/libproc.dylib")
    RUSAGE_INFO_V4 = 4
    COUNTER_BACKEND = "rusage"

    def rusage(pid):
        buf = ctypes.create_string_buffer(1024)
        if libproc.proc_pid_rusage(ctypes.c_int(pid), ctypes.c_int(RUSAGE_INFO_V4), buf) != 0:
            raise OSError("proc_pid_rusage failed")
        u64 = (ctypes.c_uint64 * 40).from_buffer_copy(buf.raw[16:336])
        return u64[29], u64[30]  # ri_instructions, ri_cycles

else:
    _PERF = shutil.which("perf")

    def _perf_counters_work():
        """True only if perf actually returns counter values.

        The binary being on PATH is not enough: without PMU access (GCE without
        vPMU, or a strict `kernel.perf_event_paranoid`) perf runs fine and
        reports `<not supported>` for every event. Selecting the backend on
        `which perf` alone made `counters_available()` lie, so the
        graceful-degradation path never engaged and the run died on the first
        measurement instead of reporting instr/cycles as absent.
        """
        if not _PERF:
            return False
        try:
            out = subprocess.run(
                [_PERF, "stat", "-x,", "-e", "instructions,cycles", "--", "true"],
                capture_output=True, text=True, timeout=30,
            )
        except (OSError, subprocess.SubprocessError):
            return False
        for line in out.stderr.splitlines():
            parts = line.split(",")
            if len(parts) >= 3:
                try:
                    float(parts[0])
                    return True
                except ValueError:
                    continue  # "<not supported>" / "<not counted>"
        return False

    COUNTER_BACKEND = "perf" if _perf_counters_work() else "none"

    def perf_window(pid, cmd):
        """instructions, cycles for `pid` while `cmd` runs, plus elapsed seconds.

        `perf stat -p PID -- CMD` attaches to PID, runs CMD, and stops counting
        when CMD exits, so the counters cover exactly the benchmark window.
        `-x,` gives machine-readable `value,unit,event,...` lines on stderr.
        """
        t0 = time.time()
        out = subprocess.run(
            [_PERF, "stat", "-x,", "-e", "instructions,cycles", "-p", str(pid), "--"] + cmd,
            capture_output=True,
            text=True,
        )
        dt = time.time() - t0
        vals = {}
        for line in out.stderr.splitlines():
            parts = line.split(",")
            if len(parts) >= 3:
                raw, event = parts[0].strip(), parts[2].strip()
                # "<not counted>"/"<not supported>" land here too — skip them.
                try:
                    vals[event] = int(float(raw))
                except ValueError:
                    continue
        if "instructions" not in vals or "cycles" not in vals:
            raise OSError(
                "perf stat returned no instructions/cycles. Needs PMU access: "
                "kernel.perf_event_paranoid <= 0 (or CAP_PERFMON), and a host "
                "that exposes the PMU (bare metal or a VM with vPMU enabled). "
                f"stderr: {out.stderr.strip()[:300]}"
            )
        return vals["instructions"], vals["cycles"], dt


def counters_available():
    return COUNTER_BACKEND != "none"


def run_and_count(pid, cmd, pmc, pmc_ok):
    """Run `cmd`, returning (instr, cycles, elapsed, pmc_events) for `pid`.

    instr/cycles are None when no counter backend is available. `pmc_events` is
    the extra branch/L1D dict, which only the macOS `pmc_tool` path fills.
    """
    if COUNTER_BACKEND == "rusage":
        i0, c0 = rusage(pid)
        if pmc_ok:
            ev, dt = pmc_run(pmc, cmd)
        else:
            t0 = time.time()
            subprocess.run(cmd, capture_output=True)
            dt = time.time() - t0
            ev = {}
        i1, c1 = rusage(pid)
        return i1 - i0, c1 - c0, dt, ev
    if COUNTER_BACKEND == "perf":
        instr, cycles, dt = perf_window(pid, cmd)
        return instr, cycles, dt, {}
    t0 = time.time()
    subprocess.run(cmd, capture_output=True)
    return None, None, time.time() - t0, {}


def pmc_run(pmc, cmd):
    out = subprocess.run([pmc, "runcmd"] + cmd, capture_output=True, text=True)
    if "EVENT" not in out.stdout:
        return None, None
    ev, elapsed = {}, 0.0
    for line in out.stdout.splitlines():
        p = line.split()
        if not p:
            continue
        if p[0] == "ELAPSED":
            elapsed = float(p[1])
        elif p[0] == "EVENT":
            ev[p[1]] = int(p[2])
    return ev, elapsed


def server_death_details():
    """Panic/crash lines from the server log, for when a command fails because
    the server is gone. Returns "" when there is nothing to add."""
    if not SERVER_LOG or not os.path.exists(SERVER_LOG):
        return ""
    try:
        with open(SERVER_LOG, errors="replace") as f:
            lines = f.readlines()
    except OSError:
        return ""
    marked = [
        line.rstrip()
        for line in lines
        if any(m in line for m in ("panicked at", "FalkorDB panic", "Redis crashed",
                                   "signal:", "=== REDIS BUG REPORT"))
    ]
    tail = marked[:6] or [line.rstrip() for line in lines[-6:]]
    return "\n  server log: " + "\n              ".join(tail)


def cli(port, *args, check=True):
    """Run redis-cli and return stdout.

    Failures were silent — the caller got an empty string and blew up later in a
    parse, far from the cause (redis-cli missing, connection refused, bad
    command). `check=False` is for probes where a failure is a legitimate
    answer.
    """
    out = subprocess.run(["redis-cli", "-p", str(port)] + [str(a) for a in args],
                         capture_output=True, text=True)
    if check and out.returncode != 0:
        raise RuntimeError(
            f"redis-cli {' '.join(str(a) for a in args)} failed "
            f"(exit {out.returncode}): {out.stderr.strip()[:200]}"
            f"{server_death_details()}"
        )
    return out.stdout


def jemalloc_totals(port):
    """Cumulative allocated/deallocated bytes from jemalloc's merged-arenas
    stats: sum size*nmalloc / size*ndalloc over the bins: and large: size-class
    tables. Returns (None, None) if the server isn't jemalloc-built."""
    out = cli(port, "memory", "malloc-stats", check=False)
    if "Merged arenas stats:" not in out:
        return None, None
    alloc = dealloc = 0
    in_merged = in_table = False
    # Column positions are read from each table's own header rather than
    # hardcoded. They were hardcoded, and the dealloc index was wrong: jemalloc
    # 5.x emits `size ind allocated nmalloc ndalloc nrequests ...`, so index 5
    # is *nrequests*, not ndalloc. That inflated deallocated bytes and made the
    # per-query deltas meaningless (absurd ratios, negative values). Reading the
    # header also survives column changes between jemalloc versions.
    i_nmalloc = i_ndalloc = None
    for line in out.splitlines():
        if line.startswith("Merged arenas stats:"):
            in_merged = True
        elif line.startswith("arenas["):
            break
        elif in_merged:
            s = line.split()
            if not s:
                continue
            if s[0] in ("bins:", "large:") and "size" in s:
                in_table = True
                # Header tokens include the leading "bins:"/"large:" label,
                # which data rows do not, hence the -1.
                try:
                    i_nmalloc = s.index("nmalloc") - 1
                    i_ndalloc = s.index("ndalloc") - 1
                except ValueError:
                    i_nmalloc = i_ndalloc = None
            elif s[0] == "extents:":
                in_table = False
            elif in_table and s[0].isdigit() and i_nmalloc is not None:
                if len(s) <= max(i_nmalloc, i_ndalloc):
                    continue
                size = int(s[0])
                alloc += size * int(s[i_nmalloc])
                dealloc += size * int(s[i_ndalloc])
    return alloc, dealloc


def main():
    exit_code = 0
    ap = argparse.ArgumentParser()
    ap.add_argument(
        "--module",
        default=os.path.join(ROOT, f"target/release/libfalkordb.{SHLIB_EXT}"),
    )
    ap.add_argument("--port", default="6399")
    ap.add_argument("--out", default=os.path.join(HERE, "results/current.csv"))
    ap.add_argument("--n", type=int, default=1000)
    ap.add_argument("--once", action="store_true")
    ap.add_argument("--keep-server", action="store_true")
    ap.add_argument("--reuse", action="store_true")
    ap.add_argument(
        "--setup",
        action="store_true",
        help="build the benchmark graph even with --reuse. Needed when the server "
        "is already running but empty (e.g. a container started from a published "
        "image in CI); without it the harness would measure an empty graph.",
    )
    ap.add_argument(
        "--pid",
        type=int,
        default=None,
        help="host pid of the redis-server to attach counters to. Needed with "
        "--reuse when the server runs in a container: INFO server reports a "
        "namespaced pid that perf on the host cannot see "
        "(docker inspect -f '{{.State.Pid}}' <container>).",
    )
    ap.add_argument("--c-compat", action="store_true")
    ap.add_argument("names", nargs="*")
    args = ap.parse_args()

    queries = QUERIES
    if args.names:
        only = set(args.names)
        queries = [q for q in QUERIES if q[0] in only]
        missing = only - {q[0] for q in queries}
        if missing:
            sys.exit(f"unknown queries: {missing}")

    server = None
    if not args.reuse:
        # check=False: "connection refused" is the answer we want here — it
        # means the port is free.
        if cli(args.port, "ping", check=False).strip() == "PONG":
            sys.exit(f"port {args.port} already in use; use --reuse or another --port")
        os.makedirs(IMPORT_DIR, exist_ok=True)
        for fname, content in CSV_FILES.items():
            with open(os.path.join(IMPORT_DIR, fname), "w") as f:
                f.write(content)
        # Isolated, clean data dir: DEBUG RELOAD in setup writes a dump.rdb
        # into the server dir, and a later server would reload it (making
        # setup fail with "already indexed").
        work_dir = os.path.join(HERE, "results", "server_dir")
        shutil.rmtree(work_dir, ignore_errors=True)
        os.makedirs(work_dir)
        server_args = ["redis-server", "--port", str(args.port), "--save", "",
                       "--enable-debug-command", "local", "--dir", work_dir,
                       # trailing slash: the C module concatenates IMPORT_FOLDER
                       # + filename directly, without inserting a separator
                       "--loadmodule", args.module, "IMPORT_FOLDER", IMPORT_DIR + "/"]
        if args.once:
            # AOF makes the module emit replication effects on every write
            # (Pending::build_effects_buffer), covering those paths.
            server_args += ["--appendonly", "yes", "--appendfsync", "no"]
        # Keep the server's log instead of discarding it. When the module
        # panics, redis prints the panic and backtrace here and then dies, and
        # every client command afterwards fails with the useless "Server closed
        # the connection". With DEVNULL that was all CI ever reported, and
        # diagnosing one such failure took an instrumented local rebuild to
        # recover a message the server had already printed.
        global SERVER_LOG
        SERVER_LOG = os.path.join(work_dir, "server.log")
        # Popen dups the fd into the child, so close the parent's copy rather
        # than leaking it for the life of the run.
        with open(SERVER_LOG, "w") as log:
            server = subprocess.Popen(server_args, stdout=log, stderr=subprocess.STDOUT)
        for _ in range(100):
            # check=False: connection refused is the expected answer while
            # the server is still coming up.
            if cli(args.port, "ping", check=False).strip() == "PONG":
                break
            time.sleep(0.1)
        else:
            sys.exit("server did not start")
    # Graph setup is separate from server lifecycle. `--reuse` means "do not
    # start a server"; it must NOT silently skip building the graph, or the
    # harness benchmarks an empty database and reports numbers that look real.
    # CI reuses a server from a published image and so needs --setup.
    if not args.reuse or args.setup:
        print(f"server up on :{args.port}, building graph...", flush=True)
        for stmt in SETUP:
            out = cli(args.port, "GRAPH.QUERY", "bench", stmt)
            # Successful GRAPH.QUERY replies always end with the execution-time
            # stat; anything else (error message, empty reply) is a failure.
            if "execution time" not in out:
                sys.exit(f"setup failed: {out[:200]}")
        for cmd in SETUP_COMMANDS:
            # The C server's async validation of a composite (2-property)
            # unique constraint crashes it; everything else in setup works.
            if args.c_compat and cmd[:2] == ["GRAPH.CONSTRAINT", "CREATE"] and "2" in cmd:
                continue
            # The C server's RDB round-trip drops numeric 0 from the Person
            # range index: after DEBUG RELOAD, MATCH (:Person {id: 0}) finds
            # nothing, turning every id-0 row into a no-op (e.g. algo.BFS).
            if args.c_compat and cmd == ["DEBUG", "RELOAD"]:
                continue
            out = cli(args.port, *[a.replace("{graph}", "bench") for a in cmd])
            if "error" in out.lower() or out.startswith("ERR"):
                sys.exit(f"setup command failed: {cmd[0]} {cmd[1]}: {out[:200]}")

    try:
        if args.once:
            fails = 0
            for name, is_write, q, *_ in queries:
                cmd = "GRAPH.QUERY" if is_write else "GRAPH.RO_QUERY"
                out = cli(args.port, cmd, "bench", q)
                if "execution time" not in out:
                    print(f"FAIL {name}: {out.strip().splitlines()[0][:120] if out.strip() else '(empty reply)'}")
                    fails += 1
            # Expected-error queries: cover parse/bind/eval error paths and
            # constraint rollback. Pass iff the reply is a non-empty error.
            if not args.names:
                for name, cmd, q in ERROR_QUERIES:
                    out = cli(args.port, cmd, "bench", q)
                    if not out.strip() or "execution time" in out:
                        print(f"FAIL (expected error) {name}: {out.strip().splitlines()[0][:120] if out.strip() else '(empty reply)'}")
                        fails += 1
            print(f"once-mode done, {fails} failures")
            if server and not args.keep_server:
                cli(args.port, "shutdown", "nosave")  # graceful: flushes .profraw
                server.wait()
                server = None
            if fails:
                sys.exit(1)
            return

        pid = args.pid or int(
            cli(args.port, "info", "server").split("process_id:")[1].split()[0]
        )
        pmc = os.path.join(HERE, "pmc_tool")
        pmc_ok = os.path.exists(pmc) and pmc_run(pmc, ["true"])[0] is not None
        if not pmc_ok:
            print("pmc_tool unavailable — branches/L1D columns will be empty "
                  "(see bench/README.md to build+setuid)", flush=True)

        # idle baseline
        have_counters = counters_available()
        # Calibrate the process's idle counter rate, then subtract it per query
        # so background work (cron/serverCron, bgsave) is not attributed to the
        # query being measured.
        idle_i, idle_c, idle_dt, idle_ev = run_and_count(pid, ["sleep", "3"], pmc, pmc_ok)
        # A backend can exist and still return nothing: `proc_pid_rusage` reports
        # ri_instructions = 0 inside a virtualised macOS runner, and selecting
        # the backend without checking made the harness emit a column of zeros
        # that looked like measurements. A live redis process cannot execute
        # zero instructions in three seconds, so zero here means "no counters".
        if have_counters and idle_i <= 0:
            have_counters = False
            print(
                f"pid {pid}, {COUNTER_BACKEND} backend returned 0 instructions "
                f"over {idle_dt:.0f}s — no usable hardware counters (typical "
                f"inside a VM). instr/cycles will be empty rather than zero.",
                flush=True,
            )
        idle_ips = (idle_i / idle_dt) if have_counters else 0.0
        idle_cps = (idle_c / idle_dt) if have_counters else 0.0
        idle_rate = {k: v / idle_dt for k, v in idle_ev.items()}
        if have_counters:
            print(
                f"pid {pid}, idle process rate {idle_ips/1e6:.1f}M instr/s "
                f"({COUNTER_BACKEND} backend)",
                flush=True,
            )
        else:
            print(
                f"pid {pid}, per-process instruction counters unavailable on "
                f"{sys.platform}. Usually a virtualised host with no PMU exposed: "
                f"on Linux perf reports '<not supported>', on macOS "
                f"proc_pid_rusage reports 0. instr/cycles columns will be empty "
                f"rather than zero; alloc_bytes/dealloc_bytes/ms are unaffected",
                flush=True,
            )

        rows = []
        failed = []
        for name, is_write, q, *rest in queries:
            n = rest[0] if rest else args.n
            cmd = ["GRAPH.QUERY" if is_write else "GRAPH.RO_QUERY", "bench", q]
            warm = cli(args.port, *cmd)  # warmup / plan cache
            if "execution time" not in warm:
                first = warm.strip().splitlines()[0][:100] if warm.strip() else "(empty reply)"
                if args.c_compat:
                    # Expected: the C engine does not implement everything here
                    # (UDFs are Rust-only), so skip the row and carry on.
                    print(f"{name:<20} SKIPPED (C error: {first})", flush=True)
                else:
                    # Not expected. Without this the harness would benchmark the
                    # *error* path and report a plausible, fast row for a query
                    # that never ran — worse than failing, because the number
                    # looks real. Collected and reported at the end so one bad
                    # query does not cost the other 316.
                    print(f"{name:<20} FAILED ({first})", flush=True)
                    failed.append((name, first))
                continue
            bench = ["redis-benchmark", "-p", str(args.port), "-c", "1",
                     "-n", str(n)] + cmd
            # memory snapshots outside the i0..i1 window so the MALLOC-STATS
            # call's own work doesn't pollute the instruction counts
            m0a, m0d = jemalloc_totals(args.port)
            q_i, q_c, dt, ev = run_and_count(pid, bench, pmc, pmc_ok)
            m1a, m1d = jemalloc_totals(args.port)
            row = {
                "query": name,
                # Empty, not 0: compare.py treats a non-numeric cell as
                # "absent" and skips the metric, whereas 0 would read as a
                # real measurement and flag every row as an infinite change.
                # Clamped at 0: subtracting the idle rate can overshoot on a
                # short/cheap query, and a negative value made compare.py skip
                # the metric (it needs a positive baseline), silently disabling
                # the gate for that row — see the negative branches/br_miss/
                # l1d_miss columns in earlier baselines.
                "instr": max(0.0, (q_i - idle_ips * dt) / n) if have_counters else "",
                "cycles": max(0.0, (q_c - idle_cps * dt) / n) if have_counters else "",
                "branches": "", "br_miss": "", "l1d_miss": "",
                "alloc_bytes": (m1a - m0a) / n if m0a is not None else "",
                "dealloc_bytes": (m1d - m0d) / n if m0d is not None else "",
                "ms": dt / n * 1000,
            }
            if pmc_ok:
                # Same clamp as instr/cycles: a negative adjusted counter
                # makes compare.py skip the metric and stop gating it.
                adj = {k: max(0.0, (v - idle_rate[k] * dt) / n) for k, v in ev.items()}
                row["branches"] = adj["INST_BRANCH"]
                row["br_miss"] = adj["BRANCH_MISPRED_NONSPEC"]
                row["l1d_miss"] = adj["L1D_CACHE_MISS_LD"] + adj["L1D_CACHE_MISS_ST"]
            rows.append(row)
            # instr/cycles are "" when no counter backend is available; a
            # ",.0f" on an empty string raises ValueError and would abort the
            # whole run partway through (exactly what happens on Linux CI).
            def _n(v, w):
                return f"{v:>{w},.0f}" if v != "" else f"{'-':>{w}}"

            print(f"{name:<20} {_n(row['instr'], 13)} instr {_n(row['cycles'], 12)} cyc "
                  + f"{row['ms']:>8.3f} ms"
                  + (f" {row['alloc_bytes']:>12,.0f} B alloc" if row['alloc_bytes'] != "" else ""),
                  flush=True)

        os.makedirs(os.path.dirname(args.out), exist_ok=True)
        merged = {}
        if os.path.exists(args.out):
            with open(args.out, newline="") as f:
                merged = {r["query"]: r for r in csv.DictReader(f)}
        for r in rows:
            merged[r["query"]] = {k: str(v) for k, v in r.items()}
        fields = ["query", "instr", "cycles", "branches", "br_miss", "l1d_miss",
                  "alloc_bytes", "dealloc_bytes", "ms"]
        with open(args.out, "w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=fields)
            w.writeheader()
            w.writerows({k: r.get(k, "") for k in fields} for r in merged.values())
        print(f"wrote {args.out}")
        if failed:
            # Non-zero exit: a query in QUERIES that does not answer is a real
            # problem, and the CSV is now missing that row rather than carrying
            # a wrong one. Reported after the CSV is written so the rest of the
            # run is still usable.
            print(f"\n{len(failed)} query(ies) failed and were not measured:", flush=True)
            for name, why in failed:
                print(f"  {name}: {why}", flush=True)
            exit_code = 1
    finally:
        if server and not args.keep_server:
            server.send_signal(signal.SIGTERM)
            server.wait()
        elif server:
            print(f"server left running on :{args.port} (pid {server.pid})")

    if exit_code:
        sys.exit(exit_code)


if __name__ == "__main__":
    main()
