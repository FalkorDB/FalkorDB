"""Graph-build timing: build the soak's graph on a fresh server and time its two phases.

The node phase is `UNWIND range CREATE` batches. The edge phase is the id-wiring shape
(`UNWIND range … WITH j WHERE j % 8 = i MATCH (a) WHERE id(a) = … MATCH (b) WHERE id(b) = … CREATE`),
which an engine that plans the second id() lookup as a scan of every node (a hash join over all
nodes) slows by a cost per query that grows with the node count. After the build, the same shape
runs on its own with 1 and 100 UNWIND rows (5 times each, median), and its plan is recorded.
"""

from __future__ import annotations

import json
import os
import statistics
import time

from . import harness, model

GRAPH = "soak_build"
PROBE_REPS = 5
PROBE_ROWS = (1, 100)
BUILD_THREADS = 16


def probe_query(n: dict, base: dict, rows: int) -> str:
    return model.edge_query("PROBE", 0, rows - 1, f"{base[model.HUB]} + j", f"{base[model.FAN]} + j % {n[model.FAN]}")


def build_run(opts: dict, info: dict, tag: str, scale: float, rdir: str) -> dict:
    """One fresh server, one build. Returns {"ok", "nodes_ms", "edges_ms", "probe_1_ms",
    "probe_100_ms", ...} or {"ok": False, "error", "server_alive"}."""
    os.makedirs(rdir, exist_ok=True)
    server = harness.Server(info, f"{opts['name_prefix']}-{tag}", opts.get("server_cpus"), opts["net"], BUILD_THREADS)
    out = {"tag": tag, "scale": scale, "ok": False}
    try:
        r = server.start()
        out["module_version"] = harness.module_version(r)
        n, timing, base = harness.build_graph(r, GRAPH, scale)
        out.update(timing)
        out["nodes_ms"] = round(timing["nodes_s"] * 1000, 1)
        out["edges_ms"] = round(timing["edges_s"] * 1000, 1)
        plan = r.execute_command("GRAPH.EXPLAIN", GRAPH, model.edge_query(
            "PROBE", 0, model.BUILD_BATCH - 1, f"{base[model.HUB]} + j", f"{base[model.FAN]} + j % {n[model.FAN]}",
            f" WHERE j % {len(model.LINK_TYPES)} = 0"))
        out["plan"] = [harness.s(x) for x in plan]
        for rows in PROBE_ROWS:
            q = probe_query(n, base, rows)
            ts = []
            for _ in range(PROBE_REPS):
                t0 = time.perf_counter()
                r.execute_command("GRAPH.QUERY", GRAPH, q)
                ts.append((time.perf_counter() - t0) * 1000)
            out[f"probe_{rows}_ms"] = round(statistics.median(ts), 3)
            out[f"probe_{rows}_all_ms"] = [round(x, 3) for x in ts]
        out["ok"] = True
    except Exception as e:
        out["error"] = f"{type(e).__name__}: {e}"[:500]
        out.update(harness.after_failure(server, e))
    finally:
        with open(os.path.join(rdir, "server-tail.log"), "w") as f:
            f.write(server.stop())
    with open(os.path.join(rdir, "build.json"), "w") as f:
        json.dump(out, f, indent=1)
    harness.log(f"{tag}: build {json.dumps({k: out.get(k) for k in ('ok', 'nodes_ms', 'edges_ms', 'probe_1_ms', 'probe_100_ms', 'error')})}")
    return out
