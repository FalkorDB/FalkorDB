"""Sampling profile of one query, via samply.

This was `profile.sh`: 25 lines of shell that re-derived the server pid with the
same `redis-cli info | grep process_id` the harness already did, started
`redis-benchmark` to keep the server busy, sampled with macOS `sample`, and
required you to pass the query twice — once by name to the harness so it would
leave a server up, then again as raw Cypher to the script.

Two things changed in the fold-in. The query is named once, because this runs
inside the harness that already has the server, the pid and the query text. And
the profiler is **samply** rather than macOS `sample`: samply is what the profile
skill and the team performance-toolbox path both name, and unlike `sample` it
exists on Linux too — the shell version was silently macOS-only.

The old text "hot leaves" summary is gone with `sample`; samply's output is a
profile for the Firefox Profiler UI, which is the workflow the profile skill
documents.
"""

from __future__ import annotations

import shutil
import subprocess
from pathlib import Path

from falkorbench.client import BenchClient
from falkorbench.model import Query


def profile_query(
    bench: BenchClient,
    query: Query,
    out: Path,
    *,
    seconds: int = 5,
    save_only: bool = True,
    echo=print,
) -> Path:
    """Profile the server while it executes `query` in a loop.

    The load generator is `redis-benchmark` with a repeat count high enough to
    outlast the sampling window; it is killed once sampling stops. Same reasoning
    as the measurement path: the thing driving the server is a C binary, so the
    profile shows the server's work and not a Python client's.
    """
    if not shutil.which("samply"):
        raise RuntimeError(
            "samply not found. Install with `cargo install --locked samply` "
            "(see the profile skill)."
        )
    if not shutil.which("redis-benchmark"):
        raise RuntimeError("redis-benchmark not found on PATH; it drives the load")

    pid = bench.pid
    out.parent.mkdir(parents=True, exist_ok=True)

    load = subprocess.Popen(
        [
            "redis-benchmark",
            "-p",
            str(bench.server.port),
            "-c",
            "1",
            "-n",
            "100000000",
            query.command,
            bench.graph_name,
            query.cypher,
        ],
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
    )
    try:
        argv = ["samply", "record", "-p", str(pid), "-d", str(seconds), "-o", str(out)]
        if save_only:
            argv.append("--save-only")
        echo(f"profiling pid {pid} for {seconds}s while '{query.name}' runs...")
        subprocess.run(argv, check=True)
    finally:
        load.terminate()
        try:
            load.wait(timeout=10)
        except subprocess.TimeoutExpired:
            load.kill()

    echo(f"wrote {out}")
    if save_only:
        echo(f"open it with: samply load {out}")
    return out
