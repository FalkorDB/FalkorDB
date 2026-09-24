"""Reproduce #2430's two cost states at engine level.

The issue reports a bound multi-edge point read landing in one of two cost
states, selected by graph size. This is the engine-level half of the diagnosis;
the structure-level half is `issue_2430_bench.rs`, and the point of running both
is that they disagree about the *shape* of the effect.

The fixture detail that matters: node count is held at 1,000 while the pair count
grows, so a bigger graph means **longer adjacency rows**, not more of them. That
is the variable a point read can be sensitive to, since
`GrB_Matrix_extractElement` searches within a row.

Usage:
    python3 engine_two_states.py <path-to-module> <port>
"""
import sys, json
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "src"))
from falkorbench import client as bc
from falkorbench.counters import read_rusage

MOD = Path(sys.argv[1]); PORT = int(sys.argv[2])
work = Path("/tmp/i2430"); work.mkdir(parents=True, exist_ok=True)
PROBE = 1000

def build(cl, pairs):
    cl.run("UNWIND range(0, 999) AS i CREATE (:N {id:i})")
    # `PROBE` two-edge pairs, then single-edge pairs out to `pairs`
    cl.run(f"MATCH (n:N) WITH collect(n) AS ns UNWIND range(0,{PROBE-1}) AS i "
           f"WITH ns,i,ns[i%1000] AS a, ns[(i+1)%1000] AS b "
           f"UNWIND [1,2] AS r CREATE (a)-[:R {{k:i}}]->(b)")
    if pairs > PROBE:
        for lo in range(PROBE, pairs, 20000):
            hi = min(lo+20000, pairs)
            cl.run(f"MATCH (n:N) WITH collect(n) AS ns UNWIND range({lo},{hi-1}) AS i "
                   f"WITH ns,i,ns[i%1000] AS a, ns[(i*7+3)%1000] AS b "
                   f"CREATE (a)-[:R {{k:i}}]->(b)")

Q = (f"MATCH (n:N) WITH collect(n) AS ns UNWIND range(0,{PROBE-1}) AS i "
     f"WITH ns,i,ns[i%1000] AS a, ns[(i+1)%1000] AS b "
     f"MATCH (a)-[r:R]->(b) RETURN count(r)")

out=[]
for pairs in [1000, 11000, 41000, 88000, 121000]:
    srv = bc.start_server(MOD, PORT, work/"srv", work/"imp")
    try:
        cl = bc.connect(srv)
        build(cl, pairs)
        for _ in range(3): cl.run(Q, write=False)
        r0 = read_rusage(cl.pid)
        for _ in range(20): cl.run(Q, write=False)
        r1 = read_rusage(cl.pid)
        per = (r1.instructions - r0.instructions)/(20*PROBE)
        out.append({"pairs":pairs,"instr_per_pair":round(per,1)})
        print(json.dumps(out[-1]), flush=True)
        cl.shutdown()
    finally:
        srv.stop()
