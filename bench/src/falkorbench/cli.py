"""The `bench` command.

One entry point with subcommands, rather than four scripts that each re-declared
--module/--port/--out and one shell script that needed a different one of them to
have been run first. The shared options are defined once, in `common_options`.
"""

from __future__ import annotations

import argparse
import os
import subprocess
import sys
from pathlib import Path

import click

from falkorbench import callgrind as cg
from falkorbench import client as client_mod
from falkorbench import compare as compare_mod
from falkorbench import contract
from falkorbench import cov_gate
from falkorbench import coverage as coverage_mod
from falkorbench import flow as flow_mod
from falkorbench import hw as hw_mod
from falkorbench import measure as measure_mod
from falkorbench import metrics
from falkorbench import profile as profile_mod
from falkorbench import queries as query_set
from falkorbench import report as report_mod
from falkorbench.counters import select_backend
from falkorbench.fuzz import check as fuzz_check
from falkorbench.ldbc import check as ldbc_check
from falkorbench.nightly import dashboard as nightly_dashboard
from falkorbench.nightly import merge as nightly_merge
from falkorbench.nightly import missed_run
from falkorbench.nightly import report as nightly_report
from falkorbench.soak import check as soak_check

# bench/ is the project root; the repo is its parent.
BENCH_DIR = Path(__file__).resolve().parents[2]
REPO_ROOT = BENCH_DIR.parent
RESULTS = BENCH_DIR / "results"


def _select(names: tuple[str, ...], *, cg_only: bool = False):
    """Resolve query names to Query objects, erroring on an unknown name."""
    pool = [q for q in query_set.QUERIES if q.cg] if cg_only else list(query_set.QUERIES)
    if not names:
        return pool
    wanted = set(names)
    chosen = [q for q in pool if q.name in wanted]
    missing = wanted - {q.name for q in chosen}
    if missing:
        raise click.ClickException(f"unknown queries: {sorted(missing)}")
    # Pull in whatever the chosen rows depend on, transitively. A row that
    # measures against a graph an earlier row builds is meaningless without it,
    # and silently so -- it reports a plausible number for the wrong graph.
    by_name = {q.name: q for q in pool}
    while True:
        need = {n for q in chosen for n in q.needs} - {q.name for q in chosen}
        if not need:
            break
        unknown = need - by_name.keys()
        if unknown:
            raise click.ClickException(f"unknown prerequisite queries: {sorted(unknown)}")
        chosen += [by_name[n] for n in need]
    # Back into suite order: a prerequisite has to run before its dependent.
    order = {q.name: i for i, q in enumerate(pool)}
    return sorted(chosen, key=lambda q: order[q.name])


def common_options(fn):
    """--module/--port, shared by every subcommand that starts a server."""
    fn = click.option(
        "--module",
        default=None,
        help="module to load (default: this repo's target/release build)",
    )(fn)
    fn = click.option("--port", default=6399, show_default=True, type=int)(fn)
    return fn


@click.group(context_settings={"help_option_names": ["-h", "--help"]})
def cli() -> None:
    """Per-query performance harness for FalkorDB."""


# --- measure -----------------------------------------------------------------


@cli.command()
@common_options
@click.option("--out", default=None, type=click.Path(), help="CSV output path")
@click.option("--n", "reps", default=1000, show_default=True, help="requests per query")
@click.option("--once", is_flag=True, help="run each query once, unmeasured (coverage)")
@click.option("--keep-server", is_flag=True, help="leave the server running afterwards")
@click.option("--reuse", is_flag=True, help="attach to a server already on --port")
@click.option("--setup/--no-setup", default=None, help="build the graph (implied unless --reuse)")
@click.option("--c-compat", is_flag=True, help="measuring the C engine: skip what it cannot do")
@click.option("--module-args", multiple=True, help="extra --loadmodule args, e.g. THREAD_COUNT 4")
@click.option(
    "--failed-out",
    default=None,
    type=click.Path(),
    help="write {query: why} for the queries that did not answer, and exit 0 for them",
)
@click.argument("names", nargs=-1)
def measure(
    module,
    port,
    out,
    reps,
    once,
    keep_server,
    reuse,
    setup,
    c_compat,
    module_args,
    failed_out,
    names,
):
    """Measure queries and write a CSV.

    Named queries are merged into an existing CSV, so a subset re-run patches
    only those rows.
    """
    queries = _select(names)
    out_path = Path(out) if out else RESULTS / "current.csv"
    module_path = client_mod.find_module(module, REPO_ROOT)
    # --reuse means "do not start a server", and by default also "do not build
    # the graph": the server is assumed to hold it already (from --keep-server).
    # Against a fresh server that measures an empty database and reports numbers
    # that look real, so pass --setup there (#3658).
    do_setup = (not reuse) if setup is None else setup

    server = client_mod.Server(port=port)
    if reuse:
        if not client_mod.is_server_up(port):
            raise click.ClickException(f"--reuse given but nothing answers on :{port}")
    else:
        if client_mod.is_server_up(port):
            raise click.ClickException(f"port {port} already in use; use --reuse or another --port")
        if not module_path.exists():
            raise click.ClickException(f"module not found: {module_path}")
        client_mod.write_csv_fixtures(Path(query_set.IMPORT_DIR), query_set.CSV_FILES)
        server = client_mod.start_server(
            module_path,
            port,
            RESULTS / "server_dir",
            Path(query_set.IMPORT_DIR),
            appendonly=once,
            module_args=list(module_args),
        )

    exit_code = 0
    try:
        bench = client_mod.connect(server)
        if do_setup:
            click.echo(f"server up on :{port}, building graph...")
            client_mod.build_graph(
                bench,
                query_set.SETUP,
                query_set.SETUP_COMMANDS,
                c_compat=c_compat,
            )

        if once:
            fails = measure_mod.run_once(
                bench,
                queries,
                query_set.ERROR_QUERIES,
                include_errors=not names,
                echo=click.echo,
            )
            if server.proc is not None and not keep_server:
                bench.shutdown()  # graceful: flushes .profraw
            raise SystemExit(1 if fails else 0)

        backend = counter_backend()
        rows, failures = measure_mod.measure_queries(
            bench,
            backend,
            queries,
            default_reps=reps,
            c_compat=c_compat,
            echo=click.echo,
        )
        measure_mod.merge_into_csv(out_path, rows)
        click.echo(f"wrote {out_path}")
        if failed_out:
            # The release gate judges them (bench cov-gate): a query neither side
            # answers is not this pull request's doing.
            measure_mod.write_failures(Path(failed_out), failures)
            click.echo(f"wrote {failed_out} ({len(failures)} query(ies) did not answer)")
            failures = []
        if failures:
            # A query in the set that does not answer is a real problem, and the
            # CSV is now missing that row rather than carrying a wrong one.
            click.echo(f"\n{len(failures)} query(ies) failed and were not measured:")
            for name, why in failures:
                click.echo(f"  {name}: {why}")
            exit_code = 1
    except client_mod.SetupFailed as e:
        raise click.ClickException(str(e)) from e
    finally:
        if server.proc is not None and not keep_server:
            server.stop()
        elif server.proc is not None:
            click.echo(f"server left running on :{port} (pid {server.proc.pid})")

    if exit_code:
        raise SystemExit(exit_code)


def counter_backend():
    """The counter backend, with pmc_tool picked up if it has been built.

    Without pmc_tool the branch/L1D columns stay empty, which is fine — the
    regression-gating columns are instructions and allocated bytes.
    """
    pmc = BENCH_DIR / "pmc_tool"
    backend = select_backend(str(pmc) if pmc.exists() else None)
    if pmc.exists() and getattr(backend, "pmc", None) is None:
        click.echo("pmc_tool present but not usable — branches/L1D columns stay empty")
    return backend


# --- callgrind ---------------------------------------------------------------


@cli.command()
@common_options
@click.option("--out", default=None, type=click.Path())
@click.option("--n1", default=20, show_default=True, help="low repeat count")
@click.option("--n2", default=120, show_default=True, help="high repeat count")
@click.option("--shard", default=None, help="measure only shard I/N (1-based, round-robin)")
@click.option("--job-total", default=None, type=int, help="cross-check N against the CI matrix")
@click.option("--module-args", multiple=True, help="extra --loadmodule args, e.g. THREAD_COUNT 1")
@click.option(
    "--bare",
    is_flag=True,
    help="no module, no graph: validates the differencing maths where valgrind "
    "cannot run this module (arm64)",
)
@click.argument("names", nargs=-1)
def callgrind(module, port, out, n1, n2, shard, job_total, module_args, bare, names):
    """Deterministic instruction counts, by differencing two runs."""
    if n2 <= n1:
        raise click.ClickException(f"--n2 ({n2}) must exceed --n1 ({n1})")
    cg.require_tools()

    module_path = cg.resolve_module(module, bare)
    if bare:
        # Refuse rather than ignore. --bare measures one fixed payload, so a
        # shard or a name list cannot be honoured — and silently dropping a flag
        # that changes what gets measured is the failure mode this harness exists
        # to avoid.
        conflicting = [n for n, v in (("--shard", shard), ("NAMES", names)) if v]
        if conflicting:
            raise click.ClickException(
                f"--bare measures a single fixed payload, so {', '.join(conflicting)} "
                f"cannot apply. Drop it, or drop --bare."
            )
        queries = [cg.bare_payload()]
    else:
        queries = _select(names, cg_only=True)
        if shard:
            try:
                queries = cg.shard(queries, shard, job_total)
            except ValueError as e:
                raise click.ClickException(str(e)) from e
            click.echo(f"shard {shard}: {len(queries)} queries")
            if not queries:
                return

    runner = cg.Runner(
        module=module_path,
        port=port,
        outdir=cg.default_outdir(BENCH_DIR),
        module_args=list(module_args),
        bare=bare,
    )

    if not bare:
        # GraphBLAS compiles kernels on first use and caches them on disk, so the
        # first server lifecycle in a job pays for that and no later one does.
        # Measured: the first run came in ~30M instructions above its pair, which
        # made T(n2) < T(n1) and cost the control row. The warm-up runs *every*
        # query in the subset, not just one: a query whose kernels no earlier
        # query needed would otherwise pay that compile inside its own first
        # measured run.
        click.echo("warm-up run (compiles every kernel the subset needs)...")
        try:
            runner.total("RETURN 1", 1, also_run=[q.cypher for q in queries])
        except cg.Skipped as e:
            raise click.ClickException(f"warm-up failed, so nothing else will work: {e}") from e

    rows: dict[str, metrics.Row] = {}
    for query in queries:
        try:
            m = cg.measure_one(runner, query, n1, n2)
        except cg.Skipped as e:
            click.echo(f"{query.name:<24} SKIPPED: {e}")
            continue
        # Rounded to whole instructions, matching the artifact format the CI
        # report already consumes (the pre-refactor writer used "%.0f").
        rows[m.query] = metrics.Row(instr=round(m.instr))
        note = f" [span widened {m.widened_from}->{m.span}]" if m.widened_from else ""
        click.echo(
            f"{m.query:<24}{m.instr:>15,.0f} instr/exec   (span {m.span}, "
            f"±{m.rel_err * 100:.2f}%, drift {m.drift:,.0f}, {m.seconds:.0f}s){note}"
        )

    out_path = Path(out) if out else RESULTS / "callgrind.csv"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    metrics.write_rows(str(out_path), rows.items(), ("query", "instr"))
    click.echo(f"wrote {out_path} ({len(rows)} rows)")
    if not rows:
        raise click.ClickException("no query produced a count")


# --- compare -----------------------------------------------------------------


@cli.command()
@click.argument("current", type=click.Path(exists=True), required=False)
@click.argument("baseline", type=click.Path(exists=True), required=False)
@click.option("--threshold", type=float, default=None, help="override every metric's threshold")
@click.option("--metrics", "metric_list", default=None, help="comma-separated subset to gate")
def compare(current, baseline, threshold, metric_list):
    """Gate a measurement CSV against a baseline CSV. Exits 1 on a regression."""
    cur_path = current or str(RESULTS / "current.csv")
    base_path = baseline or str(BENCH_DIR / "baseline/rust.csv")
    for path in (cur_path, base_path):
        if not Path(path).exists():
            raise click.ClickException(f"{path} not found")

    wanted = None
    if metric_list:
        wanted = [m.strip() for m in metric_list.split(",") if m.strip()]
        unknown = [m for m in wanted if m not in metrics.THRESHOLDS]
        if unknown:
            raise click.ClickException(f"unknown metric(s): {', '.join(unknown)}")

    cur = metrics.read_rows(cur_path)
    base = metrics.read_rows(base_path)
    result = compare_mod.compare(cur, base, metrics=wanted, threshold=threshold)
    for line in compare_mod.render(result):
        click.echo(line)
    if result.regressions:
        raise SystemExit(1)


# --- report ------------------------------------------------------------------


@cli.command()
@click.option("--measure", "measure_specs", multiple=True, metavar="NAME=CSV")
@click.option("--callgrind", "cg_specs", multiple=True, metavar="NAME=GLOB")
@click.option("--provenance", default=None, type=click.Path())
@click.option("--coverage", default=None, type=click.Path())
@click.option(
    "--strict/--no-strict",
    default=True,
    help="exit 1 when a side required for the PR-vs-base reading is missing",
)
def report(measure_specs, cg_specs, provenance, coverage, strict):
    """Build the PR comment from the CSVs the CI jobs produced."""

    def parse(specs: tuple[str, ...]) -> dict[str, dict[str, metrics.Row]]:
        out: dict[str, dict[str, metrics.Row]] = {}
        for spec in specs:
            if "=" not in spec:
                raise click.ClickException(f"expected name=path, got {spec!r}")
            name, path = spec.split("=", 1)
            out[name] = metrics.read_rows(path)
        return out

    rep = report_mod.build(
        parse(measure_specs),
        parse(cg_specs),
        provenance=provenance,
        coverage=coverage,
    )
    click.echo(rep.text())
    if strict and rep.fatal:
        # The run-level verdict lives here, not in the workflow: GitHub collapses
        # a matrix into one `result`, so a YAML gate cannot tell "the C side
        # failed" (a lost column) from "the base side failed" (a lost reading).
        for reason in rep.fatal:
            click.echo(f"::error::{reason}", err=True)
        raise SystemExit(1)


# --- micro-hw (the nightly micro check) ---------------------------------------


@cli.command("micro-hw")
@click.option("--bench-image", required=True, help="the measurement image, by digest")
@click.option("--candidate", required=True, help="image, name[:tag]@sha256:<64 hex>")
@click.option("--baseline", required=True, help="image, name[:tag]@sha256:<64 hex>")
@click.option("--out", required=True, help="directory for result.json and raw/")
@click.option("--mode", required=True, help="queue, nightly or long")
@click.option("--candidate-label", default=None)
@click.option("--baseline-label", default=None)
@click.option("--candidate-commit", default=None)
@click.option("--baseline-commit", default=None)
def micro_hw(bench_image, **flags):
    """The micro-hw check: hardware counters on both images, on this host.

    Needs docker and a PMU (a GCE C4 VM through _perf-vm.yml with pmu: true).
    Servers run on $PERF_SERVER_CPUS when it is set. Always writes
    <out>/result.json; exits 0 whenever it did.
    """
    run = hw_mod.HwRun(
        bench_image=bench_image,
        args=argparse.Namespace(**flags),
        repo_root=REPO_ROOT,
        server_cpus=os.environ.get("PERF_SERVER_CPUS") or None,
        queries=query_set.QUERIES,
        echo=click.echo,
    )
    res = hw_mod.run_check(run)
    click.echo(f"{res['verdict']}: {res['summary']}")
    for f in res["findings"]:
        ratio = f"{f['ratio']:.3f}x" if f["ratio"] is not None else "-"
        click.echo(f"  [{f['status']}] {f['name']} {f['metric']} {ratio}")
    if res["error"]:
        click.echo(f"error: {res['error']}")


# --- coverage ----------------------------------------------------------------


@cli.command()
@click.option("--port", default=6401, show_default=True, type=int)
def coverage(port):
    """Instrumented build, one pass over the query set, graph-crate line coverage.

    Reports a percentage and enforces no floor — it is a validator of the query
    set, not a coverage gate. It does fail if any query stopped working.
    """
    try:
        coverage_mod.run(REPO_ROOT, BENCH_DIR, port=port, echo=click.echo)
    except (RuntimeError, subprocess.CalledProcessError) as e:
        raise click.ClickException(str(e)) from e


# --- profile -----------------------------------------------------------------


@cli.command()
@common_options
@click.option("--out", default=None, type=click.Path())
@click.option("--seconds", default=5, show_default=True)
@click.option("--reuse", is_flag=True, help="attach to a server already on --port")
@click.option("--open-ui", is_flag=True, help="open the profiler UI instead of just saving")
@click.argument("name")
def profile(module, port, out, seconds, reuse, open_ui, name):
    """Profile the server while one query runs in a loop."""
    # `_select` returns the row's prerequisites too, in suite order, so the row asked for is not
    # necessarily first. They run once, below, before the row is profiled.
    queries = _select((name,))
    query = next(q for q in queries if q.name == name)
    if query.drains:
        raise click.ClickException(
            f"{name!r} deletes from a finite pool: profiling repeats it until the pool is empty, "
            "then samples a query that deletes nothing. Use `measure` for it."
        )
    prerequisites = [q for q in queries if q.name != name]
    out_path = Path(out) if out else RESULTS / f"profile_{query.name.replace(' ', '_')}.json.gz"

    server = client_mod.Server(port=port)
    started_here = False
    if reuse:
        if not client_mod.is_server_up(port):
            raise click.ClickException(f"--reuse given but nothing answers on :{port}")
    else:
        if client_mod.is_server_up(port):
            raise click.ClickException(f"port {port} in use; use --reuse or another --port")
        module_path = client_mod.find_module(module, REPO_ROOT)
        client_mod.write_csv_fixtures(Path(query_set.IMPORT_DIR), query_set.CSV_FILES)
        server = client_mod.start_server(
            module_path, port, RESULTS / "server_dir", Path(query_set.IMPORT_DIR)
        )
        started_here = True

    try:
        bench = client_mod.connect(server)
        if started_here:
            click.echo("building graph...")
            client_mod.build_graph(bench, query_set.SETUP, query_set.SETUP_COMMANDS)
        for pre in prerequisites:
            bench.run(pre.cypher, write=pre.write)
        profile_mod.profile_query(
            bench, query, out_path, seconds=seconds, save_only=not open_ui, echo=click.echo
        )
    finally:
        if started_here:
            server.stop()


# --- flow --------------------------------------------------------------------


@cli.command()
@click.option("--module", default=None)
@click.option("--out", default=None, type=click.Path())
@click.option("--compare", "baseline", default=None, type=click.Path(exists=True))
@click.option("--current", default=None, type=click.Path())
@click.argument("names", nargs=-1)
def flow(module, out, baseline, current, names):
    """Per-flow-test-file server instructions/cycles/peak memory (macOS only)."""
    out_path = Path(out) if out else RESULTS / "flow_current.csv"

    if baseline:
        cur = Path(current) if current else out_path
        if not cur.exists():
            raise click.ClickException(f"{cur} not found")
        for line in flow_mod.compare_csvs(cur, Path(baseline)):
            click.echo(line)
        return

    flow_mod.require_macos()
    tests = flow_mod.flow_files(REPO_ROOT)
    if names:
        wanted = {n.removesuffix(".py") for n in names}
        tests = [t for t in tests if Path(t).name in wanted]
        missing = wanted - {Path(t).name for t in tests}
        if missing:
            raise click.ClickException(f"unknown flow files: {sorted(missing)}")

    module_path = client_mod.find_module(module, REPO_ROOT)
    env, tmp = flow_mod.build_env(REPO_ROOT, module_path)
    try:
        click.echo(f"module: {module_path}\n{len(tests)} flow files")
        rows = []
        for test in tests:
            row = flow_mod.run_one(REPO_ROOT, test, env)
            rows.append(row)
            click.echo(
                f"{row.file:<32} {row.instr / 1e9:>7.2f}G instr {row.cycles / 1e9:>7.2f}G cyc "
                f"{row.peak_mem_mb:>8.1f}MB {row.wall_s:>6.1f}s  {row.servers} srv, "
                f"{row.tests_run} run, {row.tests_failed} failed"
            )
        flow_mod.merge_csv(out_path, rows)
        click.echo(f"wrote {out_path}")
    finally:
        if tmp is not None:
            tmp.cleanup()


# --- pokec -------------------------------------------------------------------


@cli.command(
    context_settings={
        "ignore_unknown_options": True,
        "allow_extra_args": True,
        "help_option_names": [],
    },
    add_help_option=False,
)
@click.argument("argv", nargs=-1, type=click.UNPROCESSED)
def pokec(argv):
    """The pokec check: FalkorDB/benchmark on Pokec, candidate against baseline.

    Takes the contract's flags (`bench pokec --help` lists them) and always writes
    <out>/result.json; exits 0 whenever it did. See src/falkorbench/pokec/README.md.
    """
    from falkorbench.pokec import check as pokec_check  # its flags are argparse's (kit.add_args)

    raise SystemExit(pokec_check.main(list(argv)))


# --- validate ----------------------------------------------------------------


@cli.command()
@click.option(
    "--kind",
    type=click.Choice(["result", "nightly"]),
    default=None,
    help="skip shape detection",
)
@click.argument("files", nargs=-1, required=True)
def validate(kind, files):
    """Check perf result.json / nightly.json files against the result contract.

    Prints each problem as <file>: <code>: <path>: <message>, or <file>: ok
    (<kind>); exits 1 if any file has a problem.
    """
    if contract.check_files(files, kind, echo=click.echo):
        raise SystemExit(1)


@cli.group()
def nightly() -> None:
    """The nightly perf pipeline: merge the checks, report the night, alarm on a missed one."""


@nightly.command("report")
@click.option(
    "--nightly",
    "path",
    required=True,
    type=click.Path(path_type=Path),
    help="the night's nightly.json",
)
def nightly_report_command(path):
    """One line to Chat ($PERF_CHAT_WEBHOOK), the details to $GITHUB_STEP_SUMMARY.

    Exits 0 when the night was reported, whatever its status; 1 when it could not be.
    """
    raise SystemExit(nightly_report.run(path, os.environ))


@nightly.command("missed-run")
@click.option(
    "--date",
    "night",
    default=None,
    type=click.DateTime(["%Y-%m-%d"]),
    help="the night to check (UTC); default: the night a scheduled run was created for, else today",
)
@click.option("--workflow", default="perf-nightly.yml", show_default=True)
@click.option("--branch", default="main", show_default=True)
@click.option("--event", default="schedule", show_default=True, help="the runs' trigger")
@click.option("--artifact", default="perf-nightly", show_default=True)
def missed_run_command(night, workflow, branch, event, artifact):
    """Alarm (and exit 1) when the night's scheduled run left no perf-nightly artifact."""
    raise SystemExit(
        missed_run.run(
            os.environ,
            night.date() if night else None,
            workflow=workflow,
            branch=branch,
            event=event,
            artifact=artifact,
        )
    )


@nightly.command(
    "merge",
    context_settings={
        "ignore_unknown_options": True,
        "allow_extra_args": True,
        "help_option_names": [],
    },
)
@click.argument("args", nargs=-1, type=click.UNPROCESSED)
def nightly_merge_command(args):
    """The night's bookkeeping: plan | context | results | conclude (each takes --help)."""
    raise SystemExit(nightly_merge.main(list(args)))


@cli.command(
    "cov-gate",
    context_settings={
        "ignore_unknown_options": True,
        "allow_extra_args": True,
        "help_option_names": [],
    },
)
@click.argument("args", nargs=-1, type=click.UNPROCESSED)
def cov_gate_command(args):
    """benchmark-cov's PR-vs-base CSVs -> the release gate's result.json (--help: the flags)."""
    raise SystemExit(cov_gate.main(list(args)))


@nightly.command("dashboard")
@click.option(
    "--data",
    required=True,
    type=click.Path(exists=True, file_okay=False, path_type=Path),
    help="the stored runs: one <date>-<run id>/ folder each",
)
@click.option(
    "--out", required=True, type=click.Path(file_okay=False, path_type=Path), help="the site"
)
def nightly_dashboard_command(data, out):
    """Build the perf dashboard, a static site, from every stored run."""
    click.echo(f"{nightly_dashboard.build(data, out)} run(s) -> {out}")


# --- ldbc --------------------------------------------------------------------


@cli.group()
def ldbc() -> None:
    """LDBC SNB Interactive v1 complex reads.

    Internal instrument, not an auditable LDBC result: that needs the official
    Java driver, LDBC membership and a commissioned audit.
    """


@ldbc.command(
    "check",
    context_settings={
        "ignore_unknown_options": True,
        "allow_extra_args": True,
        "help_option_names": [],
    },
)
@click.argument("args", nargs=-1, type=click.UNPROCESSED)
def ldbc_check_command(args):
    """Compare a candidate image with a baseline image on the complex reads (--help: the flags).

    The nightly perf check: each image runs as its own server in docker, both
    load the dataset, and OUT/result.json plus raw/ are written. Exits 0 whenever
    result.json was written, whatever its verdict.
    """
    raise SystemExit(ldbc_check.main(list(args)))


@cli.command(
    "soak",
    context_settings={
        "ignore_unknown_options": True,
        "allow_extra_args": True,
        "help_option_names": [],
    },
)
@click.argument("args", nargs=-1, type=click.UNPROCESSED)
def soak_command(args):
    """Compare a candidate image with a baseline image on the release soak (--help: the flags).

    The nightly perf check (needs `--group soak`): each scenario runs once per image, each
    image as its own server in docker, and OUT/result.json plus raw/ are written. Exits 0
    whenever result.json was written, whatever its verdict.
    """
    raise SystemExit(soak_check.main(list(args)))


@cli.command(
    "fuzz",
    context_settings={
        "ignore_unknown_options": True,
        "allow_extra_args": True,
        "help_option_names": [],
    },
)
@click.argument("args", nargs=-1, type=click.UNPROCESSED)
def fuzz_command(args):
    """Compare a candidate image with a baseline image on the seeded soak (--help: the flags).

    The nightly perf check (needs `--group fuzz`): long reads, writes and cheap reads, open loop,
    against each image's server in docker, beside a plain server from the same image. Exits 0
    whenever OUT/result.json was written, whatever its verdict.
    """
    raise SystemExit(fuzz_check.main(list(args)))


def main() -> int:
    try:
        cli.main(standalone_mode=False)
    except click.ClickException as e:
        e.show()
        return e.exit_code
    except SystemExit as e:
        return int(e.code or 0)
    return 0


if __name__ == "__main__":
    sys.exit(main())
