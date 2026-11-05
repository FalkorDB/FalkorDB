"""#2430: which stage steps?

`ExpandInto` reads a bound pair with `tensor.get(src, dst)`, which is exactly what
`issue_2430_bench.rs` measures at the boundary — and there it only *drifts* (~108
instructions across the range) where the whole query *steps* (~1,150). So the
step is around the read, not in it. This narrows where.

Three measurements per graph size, all on the same fixture:

  full     MATCH (a)-[r:R]->(b) RETURN count(r)   — scan, unwind, project, ExpandInto, aggregate
  base     the same query with the edge match removed — everything but ExpandInto
  control  `full`, but the filler edges are type :FILL rather than :R

`full - base` isolates ExpandInto's contribution. And `control` is the decisive
one: its `:R` tensor is byte-for-byte the same at every size (always exactly the
1,000 two-edge pairs that get read), while the graph around it grows. If the step
survives in `control`, it cannot be a property of the tensor being read.

Usage:
    python3 stage_isolation.py <path-to-module> <port>
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "src"))

from falkorbench import client as bc
from falkorbench.counters import read_rusage

MOD = Path(sys.argv[1])
PORT = int(sys.argv[2])
WORK = Path("/tmp/i2430_stage")
PROBE = 1_000
NODES = 1_000
SIZES = [1_000, 11_000, 41_000, 88_000, 121_000]

PREFIX = (
    f"MATCH (n:N) WITH collect(n) AS ns UNWIND range(0,{PROBE - 1}) AS i "
    f"WITH ns, i, ns[i%{NODES}] AS a, ns[(i+1)%{NODES}] AS b "
)
Q_FULL = PREFIX + "MATCH (a)-[r:R]->(b) RETURN count(r)"
Q_BASE = PREFIX + "RETURN count(a)"


def build(cl, pairs: int, filler_type: str) -> None:
    cl.run(f"UNWIND range(0, {NODES - 1}) AS i CREATE (:N {{id:i}})")
    # the pairs that are actually read: always the same 1,000, always two-edge
    cl.run(
        f"MATCH (n:N) WITH collect(n) AS ns UNWIND range(0,{PROBE - 1}) AS i "
        f"WITH ns, i, ns[i%{NODES}] AS a, ns[(i+1)%{NODES}] AS b "
        f"UNWIND [1,2] AS r CREATE (a)-[:R {{k:i}}]->(b)"
    )
    # filler: grows the graph without changing which pairs are read
    for lo in range(PROBE, pairs, 20_000):
        hi = min(lo + 20_000, pairs)
        cl.run(
            f"MATCH (n:N) WITH collect(n) AS ns UNWIND range({lo},{hi - 1}) AS i "
            f"WITH ns, i, ns[i%{NODES}] AS a, ns[(i*7+3)%{NODES}] AS b "
            f"CREATE (a)-[:{filler_type} {{k:i}}]->(b)"
        )


def cost(cl, q: str, reps: int = 20) -> float:
    for _ in range(3):
        cl.run(q, write=False)
    r0 = read_rusage(cl.pid)
    for _ in range(reps):
        cl.run(q, write=False)
    r1 = read_rusage(cl.pid)
    return (r1.instructions - r0.instructions) / (reps * PROBE)


def delta_state(cl) -> str:
    """`GRAPH.MEMORY` does not expose delta sizes, so use the plan's own view:
    a trivial write forces the fold policy to run, which is the cheapest probe
    of whether an unflushed delta is what the read is paying for."""
    cl.run("CREATE (:Flush)")
    cl.run("MATCH (n:Flush) DELETE n")
    return "flushed"


def run(filler_type: str) -> list[dict]:
    rows = []
    for pairs in SIZES:
        srv = bc.start_server(MOD, PORT, WORK / "srv", WORK / "imp")
        try:
            cl = bc.connect(srv)
            build(cl, pairs, filler_type)
            row = {
                "filler": filler_type,
                "pairs": pairs,
                "full": round(cost(cl, Q_FULL), 1),
                "base": round(cost(cl, Q_BASE), 1),
            }
            row["expand_into"] = round(row["full"] - row["base"], 1)
            # does forcing the fold move it? if the high state is an unflushed
            # delta, a write should collapse it back to the low state.
            delta_state(cl)
            row["full_after_write"] = round(cost(cl, Q_FULL), 1)
            row["expand_after_write"] = round(row["full_after_write"] - row["base"], 1)
            rows.append(row)
            print(json.dumps(row), flush=True)
            cl.shutdown()
        finally:
            srv.stop()
    return rows


def main() -> None:
    WORK.mkdir(parents=True, exist_ok=True)
    print("--- filler edges are :R (the tensor being read grows) ---", flush=True)
    same = run("R")
    print("--- filler edges are :FILL (the :R tensor is identical at every size) ---", flush=True)
    other = run("FILL")

    def span(rows, key):
        vals = [r[key] for r in rows]
        return f"{min(vals):.0f} -> {max(vals):.0f}  (delta {max(vals) - min(vals):.0f})"

    print("\n=== summary, instructions per probed pair ===")
    for label, rows in ((":R filler", same), (":FILL filler", other)):
        print(f"{label:>14}  full {span(rows, 'full'):<28} "
              f"ExpandInto {span(rows, 'expand_into'):<28} base {span(rows, 'base')}")
        print(f"{'':>14}  after a write: ExpandInto {span(rows, 'expand_after_write')}")


if __name__ == "__main__":
    main()
