"""Gap 1: the auxiliary structure's SPACE, and the ABSENCE of a transition.

Two properties the existing read-differencing bound says nothing about,
measured on both engines: (A) container-per-cell (the C module) and (C)
inline-first with lazy overflow (the Rust module).

(a) SPACE
    The always-materialised design (B) holds every edge id in the auxiliary
    structure, including for single-edge pairs; (C) holds only multi-edge ids
    there and (A) allocates a container only for multi-edge pairs. So the
    quantity (B)'s space penalty is made of is *bytes per id in the auxiliary
    structure*.

    Method: hold PAIR COUNT (500,000) and NODE COUNT (2,500) fixed and vary ids
    per pair k in {1, 2, 4, 8}; |E| = 500,000 * k. Metric is
    `relation_matrices_sz_mb` from GRAPH.MEMORY, which is the engines' own
    accounting of exactly the relationship-storage matrices — it excludes the
    edge blocks and attribute stores that scale with |E| for reasons unrelated
    to edge identity, which is why it is preferred here to total resident size.
    Resident deltas (jemalloc live bytes, RSS after MEMORY PURGE) are recorded
    alongside as a cross-check.

    The slope over k >= 2, where every id is in the auxiliary structure, is its
    marginal bytes per id. k = 1 is the all-inline point.

    That slope AMORTISES the auxiliary structure's per-pair (per-row/per-
    container) overhead over k ids, so as an estimate of what (B) would pay per
    id at one id per pair it is a LOWER BOUND, not (B)'s number.

    `reference` = 1,000,000 pairs x 1 edge: the same |E| as k=2 with no
    multi-edge pair anywhere. Different pair count, so it is context, not a
    controlled comparison.

    Edges are deliberately property-free (`CREATE (a)-[:R]->(b)`) so nothing in
    the attribute store moves with k.

(b) NO TRANSITION
    (B) never promotes or demotes because it has no state change. (C) promotes a
    pair when it gains a second edge; (A) allocates a container. A pair built
    multi-edge in a single batch does not pay the same transition as one built
    incrementally — in (C), `set_all_from_slices` resolves within-batch
    duplicates through its own map and promotes the pending inline slot without
    probing the committed matrix. The gap between the two build orders isolates
    the transition cost that (B) would not pay at all.

    Method: build the SAME final graph (500,000 pairs x 2 edges) two ways and
    count server-side instructions for the whole build:
      incremental = one statement per source row creating edge 1, then a second
                    pass creating edge 2 on the same pairs;
      batch       = one statement per source row creating both edges at once.
    `single` builds 1,000,000 single-edge pairs for the same |E| with no
    promotion anywhere. 3 repetitions, each on a fresh server.

Instructions come from `proc_pid_rusage` on the SERVER pid, so this driver's
work is not in the numbers. Ports 6600-6699.
"""

from __future__ import annotations

import json
import shutil
import sys
import time
from pathlib import Path

sys.path.insert(0, "/Users/aviavni/repos/FalkorDB/bench/src")

from falkorbench import client as client_mod
from falkorbench.counters import select_backend, rss

SRCS = 500  # :S nodes; the multi configs use the first MULTI_SRCS
DSTS = 2_000  # :D nodes
MULTI_SRCS = 250
PAIRS = MULTI_SRCS * DSTS  # 500,000
PORT = 6613
REPS = 3

WORK = Path(
    "/private/tmp/claude-501/-Users-aviavni-repos-FalkorDB/"
    "315a73ee-413e-4bd9-9431-31a67ffd04de/scratchpad/gap1work"
)


def fresh(module: Path):
    shutil.rmtree(WORK, ignore_errors=True)
    WORK.mkdir(parents=True, exist_ok=True)
    server = client_mod.start_server(module, PORT, WORK / "srv", WORK / "imp")
    bench = client_mod.connect(server)
    bench.run(f"UNWIND range(0, {SRCS - 1}) AS i CREATE (:S {{id: i}})")
    bench.run(f"UNWIND range(0, {DSTS - 1}) AS i CREATE (:D {{id: i}})")
    bench.run("CREATE INDEX FOR (n:S) ON (n.id)")
    return server, bench


def build_rows(bench, src_lo, src_hi, k):
    """One statement per source row: k property-free edges to every :D node."""
    for s in range(src_lo, src_hi):
        if k == 1:
            bench.run(f"MATCH (a:S {{id: {s}}}) MATCH (b:D) CREATE (a)-[:R]->(b)")
        else:
            bench.run(
                f"MATCH (a:S {{id: {s}}}) MATCH (b:D) "
                f"UNWIND range(1, {k}) AS j CREATE (a)-[:R]->(b)"
            )


def edge_count(bench):
    return bench.graph.query("MATCH ()-[r:R]->() RETURN count(r)").result_set[0][0]


def memory(bench):
    """GRAPH.MEMORY USAGE as a dict (values are integer MB)."""
    try:
        raw = bench.command("GRAPH.MEMORY", "USAGE", bench.graph_name)
        return dict(zip(raw[0::2], raw[1::2]))
    except Exception as e:  # noqa: BLE001
        return {"error": str(e)}


# --- (a) space ----------------------------------------------------------------


def space(engine: str, module: Path, out: dict) -> None:
    # (label, src_lo, src_hi, k)
    configs = [
        ("k=1", 0, MULTI_SRCS, 1),
        ("k=2", 0, MULTI_SRCS, 2),
        ("k=4", 0, MULTI_SRCS, 4),
        ("k=8", 0, MULTI_SRCS, 8),
        ("reference", 0, SRCS, 1),
    ]
    for rep in range(REPS):
        for label, lo, hi, k in configs:
            server, bench = fresh(module)
            try:
                a0, d0 = bench.jemalloc_totals()
                base_live = (a0 - d0) if a0 is not None else None
                try:
                    bench.command("MEMORY", "PURGE")
                except Exception:  # noqa: BLE001
                    pass
                base_rss = rss(server.proc.pid)
                build_rows(bench, lo, hi, k)
                edges = edge_count(bench)
                pairs = (hi - lo) * DSTS
                assert edges == pairs * k, f"{label}: {edges} != {pairs} * {k}"
                a1, d1 = bench.jemalloc_totals()
                live = ((a1 - d1) - base_live) if a1 is not None else None
                mem = memory(bench)
                try:
                    bench.command("MEMORY", "PURGE")
                except Exception:  # noqa: BLE001
                    pass
                row = {
                    "k": k,
                    "pairs": pairs,
                    "edges": edges,
                    "rel_matrices_mb": mem.get("relation_matrices_sz_mb"),
                    "total_graph_mb": mem.get("total_graph_sz_mb"),
                    "edge_block_mb": mem.get("amortized_edge_block_sz_mb"),
                    "live_bytes": live,
                    "rss_delta": (rss(server.proc.pid) - base_rss) if base_rss else None,
                }
                out.setdefault(engine, {}).setdefault("space", {}).setdefault(
                    label, []
                ).append(row)
                print(
                    f"{engine:>5} space r{rep} {label:>10}  pairs={pairs:>7} "
                    f"edges={edges:>8} rel_mb={row['rel_matrices_mb']} "
                    f"total_mb={row['total_graph_mb']} live={live} rss={row['rss_delta']}",
                    flush=True,
                )
            finally:
                server.stop()


# --- (b) transition -----------------------------------------------------------


def transition(engine: str, module: Path, out: dict) -> None:
    backend = select_backend(None)
    for rep in range(REPS):
        res: dict = {}

        # incremental: edge 1 for every pair, then edge 2 for every pair.
        server, bench = fresh(module)
        try:
            pid = server.proc.pid
            with backend.count_window(pid) as w:
                t0 = time.time()
                build_rows(bench, 0, MULTI_SRCS, 1)
                pass1_s = time.time() - t0
            pass1 = w.reading.instr
            with backend.count_window(pid) as w:
                t0 = time.time()
                build_rows(bench, 0, MULTI_SRCS, 1)  # a 2nd edge on the same pairs
                pass2_s = time.time() - t0
            pass2 = w.reading.instr
            res["incremental"] = {
                "pass1_instr": pass1,
                "pass2_instr": pass2,
                "total_instr": pass1 + pass2,
                "seconds": pass1_s + pass2_s,
                "edges": edge_count(bench),
                "pairs": PAIRS,
            }
        finally:
            server.stop()

        # batch: both edges of every pair in one statement per source row.
        server, bench = fresh(module)
        try:
            pid = server.proc.pid
            with backend.count_window(pid) as w:
                t0 = time.time()
                build_rows(bench, 0, MULTI_SRCS, 2)
                secs = time.time() - t0
            res["batch"] = {
                "total_instr": w.reading.instr,
                "seconds": secs,
                "edges": edge_count(bench),
                "pairs": PAIRS,
            }
        finally:
            server.stop()

        # single: the same |E| as 1,000,000 single-edge pairs, no promotion.
        server, bench = fresh(module)
        try:
            pid = server.proc.pid
            with backend.count_window(pid) as w:
                t0 = time.time()
                build_rows(bench, 0, SRCS, 1)
                secs = time.time() - t0
            res["single"] = {
                "total_instr": w.reading.instr,
                "seconds": secs,
                "edges": edge_count(bench),
                "pairs": SRCS * DSTS,
            }
        finally:
            server.stop()

        out.setdefault(engine, {}).setdefault("transition", []).append(res)
        for k, v in res.items():
            print(
                f"{engine:>5} trans r{rep} {k:>12}  edges={v['edges']:>8} "
                f"pairs={v['pairs']:>7} instr={v['total_instr']:,.0f} ({v['seconds']:.2f}s)",
                flush=True,
            )


def transition_controlled(engine: str, module: Path, out: dict) -> None:
    """Promotion cost per pair with statement count and edge count held fixed.

    `incremental` vs `batch` above answers "what does it cost to build this graph
    two ways", but the incremental build runs the MATCH pipeline twice, so its
    excess is promotion *plus* a second pass of parse/plan/scan. This isolates
    the promotion instead:

      pass1     250 statements, 1 edge to each of 500,000 pairs (setup, not
                measured as the comparison).
      promote   250 statements adding a 2nd edge to those SAME 500,000 pairs.
      control   250 statements adding a 1st edge to 500,000 FRESH pairs
                (source band 250..499), so the same number of statements, the
                same number of created edges and the same plan shape — but no
                promotion.

    Promotion cost per pair = (promote - control) / 500,000. Not held fixed: the
    control grows the pair count to 1,000,000 while the promote pass leaves it at
    500,000, so the control does its work against a slightly larger adjacency.
    """
    backend = select_backend(None)
    for rep in range(REPS):
        row = {}
        for label, lo, hi in (("promote", 0, MULTI_SRCS), ("control", MULTI_SRCS, SRCS)):
            server, bench = fresh(module)
            try:
                pid = server.proc.pid
                build_rows(bench, 0, MULTI_SRCS, 1)  # setup: 500,000 single-edge pairs
                assert edge_count(bench) == PAIRS
                with backend.count_window(pid) as w:
                    t0 = time.time()
                    build_rows(bench, lo, hi, 1)
                    secs = time.time() - t0
                row[label] = {
                    "instr": w.reading.instr,
                    "seconds": secs,
                    "edges_after": edge_count(bench),
                }
            finally:
                server.stop()
        diff = row["promote"]["instr"] - row["control"]["instr"]
        row["promote_instr_per_pair"] = diff / PAIRS
        out.setdefault(engine, {}).setdefault("transition_controlled", []).append(row)
        print(
            f"{engine:>5} ctrl  r{rep}  promote={row['promote']['instr']:,.0f} "
            f"control={row['control']['instr']:,.0f} "
            f"promote_per_pair={diff / PAIRS:,.0f} instr",
            flush=True,
        )


def transition_controlled2(engine: str, module: Path, out: dict) -> None:
    """Promotion cost per pair, controlled against an insert that does NOT promote.

    `transition_controlled` used fresh pairs as its control and came out negative
    on the C engine: creating 500,000 new pairs grows the adjacency from 500,000
    to 1,000,000 entries, and that growth costs more than a promotion, so the
    control was measuring matrix growth rather than a plain insert.

    The control here adds an edge to a pair that is ALREADY multi-edge, so the
    auxiliary container/row already exists and no state changes:

      promote   setup 500,000 pairs x 1 edge; measure 250 statements adding a
                2nd edge to every pair. Every pair transitions.
      control   setup 500,000 pairs x 2 edges; measure 250 statements adding a
                3rd edge to every pair. No pair transitions.

    Held fixed: pair count (500,000), statement count (250), edges created by the
    measured pass (500,000), plan shape. Not held fixed, and it biases against the
    control: the control's auxiliary structure already holds 1,000,000 ids when
    the measured pass runs, while the promote pass starts with an empty one. So
    (promote - control) is a LOWER BOUND on the transition cost.
    """
    backend = select_backend(None)
    for rep in range(REPS):
        row = {}
        for label, setup_k in (("promote", 1), ("control", 2)):
            server, bench = fresh(module)
            try:
                pid = server.proc.pid
                build_rows(bench, 0, MULTI_SRCS, setup_k)
                assert edge_count(bench) == PAIRS * setup_k
                with backend.count_window(pid) as w:
                    t0 = time.time()
                    build_rows(bench, 0, MULTI_SRCS, 1)  # +1 edge on every pair
                    secs = time.time() - t0
                after = edge_count(bench)
                assert after == PAIRS * (setup_k + 1), after
                row[label] = {"instr": w.reading.instr, "seconds": secs, "edges_after": after}
            finally:
                server.stop()
        diff = row["promote"]["instr"] - row["control"]["instr"]
        row["promote_instr_per_pair"] = diff / PAIRS
        out.setdefault(engine, {}).setdefault("transition_controlled2", []).append(row)
        print(
            f"{engine:>5} ctrl2 r{rep}  promote={row['promote']['instr']:,.0f} "
            f"control={row['control']['instr']:,.0f} "
            f"promote_per_pair={diff / PAIRS:,.0f} instr",
            flush=True,
        )


if __name__ == "__main__":
    root = Path("/Users/aviavni/repos/FalkorDB")
    engines = {
        "C": root / "bin/macos-arm64v8-release/falkordb.so",
        "Rust": root / ".claude/worktrees/agent-aaba21650d1efbb58/target/release/libfalkordb.dylib",
    }
    results: dict = {}
    for name, mod in engines.items():
        if not mod.exists():
            print(f"skip {name}: {mod} missing", flush=True)
            continue
        stages = {
            "space": space,
            "transition": transition,
            "controlled": transition_controlled,
            "controlled2": transition_controlled2,
        }
        want = sys.argv[1:] or list(stages)
        for fn in [stages[s] for s in want]:
            try:
                fn(name, mod, results)
            except Exception as e:  # noqa: BLE001
                print(f"{name} {fn.__name__} FAILED: {type(e).__name__}: {e}", flush=True)
    outp = WORK.parent / ("gap1_" + "_".join(sys.argv[1:] or ["all"]) + ".json")
    outp.write_text(json.dumps(results, indent=2, default=str))
    print(f"\nwrote {outp}")
