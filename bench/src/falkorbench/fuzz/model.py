"""The soak's synthetic graph (14 labels from a halving series, edges wired by node id), its
queries, and the seeded parameters of each epoch (README, "What one run is"). Pure: no server."""

from __future__ import annotations

import math
import random

# --- the graph ---------------------------------------------------------------------------------

BASE_TS = 1_700_000_000_000  # a round epoch in ms; every timestamp in the graph is relative to it
LABELS = tuple(f"L{i}" for i in range(14))
#: nodes per label at scale 1: a halving series from 1.2M, rounded to hundreds
COUNTS = {lbl: max(100, int(round(1_200_000 / 2**i, -2))) for i, lbl in enumerate(LABELS)}
LINK_TYPES = tuple(f"R{i}" for i in range(8))
HUB, FAN, HOLDER = "L0", "L4", "L5"  # L0 -[R*]-> L4 and L5 -[S]-> L0
CHAIN = tuple(LABELS[1:9])  # L1 .. L8, each -[R*]-> the label before it
#: the pairs the upserts extend: (source label, type, target label); initially source[j % n_src] -> target[j]
PAIRS = (("L11", "A", "L10"), ("L9", "B", "L12"))
SINGLE = "L13"  # single-node upserts and the small lookups
REL_TYPES = (*LINK_TYPES, "S", "A", "B")
LOOKUP_LABELS = ("L9", "L10", SINGLE)
FILTERED = "L6"  # filtered sort and distinct
WIDE = "L4"  # has a second key, k2
BULK_LABELS = ("L9", "L10", "L11", "L12")
UPSERT_LABELS = ("L11", "L10", "L9", "L12", SINGLE)
GROUPS = 16
BUCKETS = 1000
MODES = 5
KEY_GROWTH = 1.2  # upsert keys drawn from 1.2x the label size: about 1 in 6 creates a node
BUILD_BATCH = 100_000

# --- the traffic -------------------------------------------------------------------------------

#: cheap reads: name -> (requests/s per graph, TIMEOUT in s)
#: scan-lookup's budget keeps its cost (a scan of the largest label) well above 5% of it at scale 1
#: and well below it at scale 0.1, so whether it is judged never hangs on a few milliseconds
READ_OPS = {
    "lookup": (30.0, 1.0),
    "filter-sort": (10.0, 1.0),
    "distinct": (4.0, 1.0),
    "scan-lookup": (1.0, 0.6),
}
WRITE_RATE = {"upsert": 10.0, "bulk-set": 2.0}  # requests/s per busy graph
LONG_TIMEOUT = {"full-scan": 6.0, "full-count": 6.0, "type-count": 6.0}  # TIMEOUT in s
SCAN_SHARE = 0.6  # of long reads, the share that are full scans (the rest: a count batch)
SCAN_WRITE_P = 0.5  # share of long reads with an upsert sent 1 ms after them
LONG_PAUSE_S = 5.0  # mean pause between two long reads of one stream
LONG_START_S = 2.0  # a long-read stream starts within this many seconds of the epoch
CHEAP_SHARE = 0.05  # a read whose cost alone is under 5% of its TIMEOUT must never time out
UPSERT_MIX = (0.4, 0.4)  # pair A, pair B; the rest single-node
MAX_READ_RATE = 400.0  # one Python driver lags beyond about 400 requests/s in all

SINCE = BASE_TS - 100_000_000  # every timestamp in the graph is after this
SCAN_W = 3  # the full scan returns edges with w < 3
COUNT_W = 96  # the full count counts edges with w >= 96

FULL_SCAN = (
    "MATCH (a)-[r]->(b) WHERE r.w < $w RETURN type(r), id(a), labels(a), a.k, a.h, "
    "id(b), labels(b), b.k, b.h, r.ts"
)
FULL_COUNT = (
    "MATCH (a)-[r]->(b) WHERE r.w >= $w RETURN labels(a)[0] AS src, type(r) AS t, "
    "labels(b)[0] AS dst, count(r) AS n"
)
TYPE_COUNT = "MATCH (a)-[r:{t}]->(b) WHERE r.w >= $w RETURN labels(a)[0] AS src, labels(b)[0] AS dst, count(r) AS n"
LOOKUP = "MATCH (n:{label} {{k: $k}}) WHERE n.ts >= $since RETURN n.h"
FILTER_SORT = (
    f"MATCH (s:{FILTERED}) WHERE s.g = {{g}} AND s.h = {{h}} AND s.k = {{k}} WITH DISTINCT s "
    "WHERE s.ts >= $since RETURN s ORDER BY s.h, s.k, s.ts LIMIT 100"
)
DISTINCT = (
    f"MATCH (n:{FILTERED}) WHERE n.m = {{m}} AND n.ts >= $since RETURN DISTINCT n.k LIMIT 20000"
)
# a key past the end: every node is read
SCAN_LOOKUP = f"MATCH (n:{HUB}) WHERE n.k = $k RETURN n.h LIMIT 1"
UPSERT_PAIR = (
    "MERGE (a:{src} {{k: $ka, g: $ga}}) ON CREATE SET a.ts = $now, a.h = $ha SET a.c = $c, a.u = $now "
    "WITH count(*) AS _ MERGE (b:{dst} {{k: $kb, g: $gb}}) ON CREATE SET b.ts = $now, b.h = $hb "
    "SET b.c = $c, b.u = $now WITH count(*) AS _ "
    "MATCH (a:{src} {{k: $ka, g: $ga}}), (b:{dst} {{k: $kb, g: $gb}}) "
    "MERGE (a)-[r:{t}]->(b) ON CREATE SET r.ts = $now SET r.c = $c, r.u = $now RETURN 1"
)
UPSERT_SINGLE = (
    f"MERGE (n:{SINGLE} {{k: $k, g: $g}}) ON CREATE SET n.ts = $now, n.h = $h RETURN id(n)"
)
BULK_SET = "MATCH (n:{label}) WHERE n.g = $g SET n.c = $c, n.u = $now"

#: indexes of index epochs (read-path labels)
EPOCH_INDEXES = ((FILTERED, ("g", "h", "k")), (WIDE, ("k", "k2")), (HUB, ("k", "h")))
#: unique constraints of constraint epochs, on the upsert keys
CONSTRAINTS = (
    ("L11", ("k", "g")),
    ("L10", ("k", "g")),
    ("L9", ("k", "g")),
    ("L12", ("k", "g")),
    (SINGLE, ("k",)),
)
#: (label, property) pairs whose index lookups must match a scan
INDEX_PROBES = ((FILTERED, "k"), (WIDE, "k"), (HUB, "k"))


def label_sizes(scale: float) -> dict:
    return {k: max(5, int(v * scale)) for k, v in COUNTS.items()}


def wiring(n: dict) -> list[tuple[str, tuple[str, ...], str, str, int]]:
    """Edge sets of a graph with sizes n: (source, types, target, driver, count).

    For edge j of a set (0 <= j < count): source[j % n_source] -> target[j % n_target], so the driver
    (the label with `count` nodes) gets one edge per node. With several types, edge j has type
    types[j % len(types)]."""
    out = [(HUB, LINK_TYPES, FAN, HUB, n[HUB]), (HOLDER, ("S",), HUB, HUB, n[HUB])]
    prev = HUB
    for lbl in CHAIN:
        out.append((lbl, LINK_TYPES, prev, lbl, n[lbl]))
        prev = lbl
    for src, t, dst in PAIRS:
        out.append((src, (t,), dst, dst, n[dst]))
    return out


def node_query(label: str, lo: int, hi: int) -> str:
    extra = ""
    if label == WIDE:
        extra = ", k2: i + 10000000"
    elif label == FILTERED:
        extra = f", m: i % {MODES}"
    return (
        f"UNWIND range({lo}, {hi}) AS i CREATE (:{label} {{k: i, g: i % {GROUPS}, h: i % {BUCKETS}, "
        f"ts: {BASE_TS} - i{extra}}})"
    )


def edge_query(t: str, lo: int, hi: int, src_expr: str, dst_expr: str, where: str = "") -> str:
    """The id-wiring query: the UNWIND + two id() seeks + CREATE shape the build timing measures."""
    return (
        f"UNWIND range({lo}, {hi}) AS j WITH j{where} MATCH (a) WHERE id(a) = {src_expr} "
        f"MATCH (b) WHERE id(b) = {dst_expr} CREATE (a)-[:{t} {{w: j % 100, ts: {BASE_TS} - j}}]->(b)"
    )


def build_plan(n: dict, base: dict | None = None, batch: int = BUILD_BATCH):
    """The build's queries, in order: ("nodes", label, query) for every node batch, then
    ("edges", type, query) for every edge batch. Edge queries need `base` (label -> id of its
    first node; labels get consecutive ids), so they are generated only when it is given."""
    for label, cnt in n.items():
        for lo in range(0, cnt, batch):
            yield "nodes", label, node_query(label, lo, min(cnt, lo + batch) - 1)
    if base is None:
        return
    for src, types, dst, _driver, cnt in wiring(n):
        s_expr = f"{base[src]} + j % {n[src]}"
        d_expr = f"{base[dst]} + j % {n[dst]}"
        for lo in range(0, cnt, batch):
            hi = min(cnt, lo + batch) - 1
            if len(types) == 1:
                yield "edges", types[0], edge_query(types[0], lo, hi, s_expr, d_expr)
            else:
                for i, t in enumerate(types):
                    yield (
                        "edges",
                        t,
                        edge_query(t, lo, hi, s_expr, d_expr, f" WHERE j % {len(types)} = {i}"),
                    )


def initial_pair(n: dict, src: str, dst: str, a: int, b: int) -> bool:
    """Is (a, b) one of the pair edges the build created (source[j % n_src] -> target[j])?"""
    return b < n[dst] and a == b % n[src]


# --- query generators: rng -> cypher, params -------------------------------------------------------


def make_read(op: str, n: dict, rng: random.Random) -> tuple[str, dict]:
    if op == "lookup":
        label = rng.choice(LOOKUP_LABELS)
        k = rng.randrange(int(n[label] * 1.1))  # some keys miss
        return LOOKUP.format(label=label), {"k": k, "since": SINCE}
    if op == "filter-sort":
        j = rng.randrange(n[FILTERED])
        return FILTER_SORT.format(g=j % GROUPS, h=j % BUCKETS, k=j), {"since": SINCE}
    if op == "distinct":
        return DISTINCT.format(m=rng.randrange(MODES)), {"since": SINCE}
    if op == "scan-lookup":
        return SCAN_LOOKUP, {"k": n[HUB] + rng.randrange(n[HUB])}
    raise ValueError(op)


def make_long(kind: str) -> list[tuple[str, str, dict]]:
    """One long read: [(op, cypher, params)]; a count is followed by one count per type."""
    if kind == "full-scan":
        return [("full-scan", FULL_SCAN, {"w": SCAN_W})]
    return [("full-count", FULL_COUNT, {"w": COUNT_W})] + [
        ("type-count", TYPE_COUNT.format(t=t), {"w": COUNT_W}) for t in REL_TYPES
    ]


def make_upsert(n: dict, rng: random.Random) -> tuple[str, dict, list]:
    """Returns cypher, params, keys; keys = [(label, k), ...] plus (type, a, b) for the edge."""
    x = rng.random()
    now = BASE_TS + rng.randrange(1_000_000)
    c = rng.randrange(1000)
    for (src, t, dst), cut in zip(
        PAIRS, (UPSERT_MIX[0], UPSERT_MIX[0] + UPSERT_MIX[1]), strict=True
    ):
        if x < cut:
            ka = rng.randrange(int(n[src] * KEY_GROWTH))
            kb = rng.randrange(int(n[dst] * KEY_GROWTH))
            params = {
                "ka": ka,
                "ga": ka % GROUPS,
                "ha": ka % BUCKETS,
                "kb": kb,
                "gb": kb % GROUPS,
                "hb": kb % BUCKETS,
                "c": c,
                "now": now,
            }
            return (
                UPSERT_PAIR.format(src=src, dst=dst, t=t),
                params,
                [(src, ka), (dst, kb), (t, ka, kb)],
            )
    k = rng.randrange(int(n[SINGLE] * KEY_GROWTH))
    return UPSERT_SINGLE, {"k": k, "g": k % GROUPS, "h": k % BUCKETS, "now": now}, [(SINGLE, k)]


def make_bulk_set(rng: random.Random) -> tuple[str, dict]:
    return BULK_SET.format(label=rng.choice(BULK_LABELS)), {
        "g": rng.randrange(GROUPS),
        "c": rng.randrange(1000),
        "now": BASE_TS + rng.randrange(1_000_000),
    }


def pair_of(edge_type: str) -> tuple[str, str, str]:
    return next(p for p in PAIRS if p[1] == edge_type)


# --- parameters --------------------------------------------------------------------------------------


def hour_params(
    seed: str, h: int, scale_range: tuple[float, float, float] = (0.25, 2.0, 4.0)
) -> dict:
    """A server-hour: a fresh server with THREAD_COUNT drawn, and 2-4 graphs at their own scales
    (their sum capped)."""
    rng = random.Random(f"{seed}-h{h}")
    ng = rng.choice([2, 3, 4])
    lo, hi, cap = scale_range
    while True:
        scales = [round(rng.uniform(lo, hi), 2) for _ in range(ng)]
        if sum(scales) <= cap:
            break
    return {"hour": h, "graphs": ng, "scales": scales, "thread_count": rng.choice([4, 8, 16, 32])}


def epoch_params(seed: str, h: int, e: int, ng: int) -> dict:
    """An epoch: which graphs are busy, long-read streams, rates, write pattern, indexes, constraints,
    write timeouts and BGSAVE, all drawn from the seed."""
    rng = random.Random(f"{seed}-h{h}-e{e}")
    busy = sorted(rng.sample(range(ng), rng.randint(1, ng)))
    long_k = {g: rng.choices([0, 1, 2, 3], [0.1, 0.5, 0.25, 0.15])[0] for g in busy}
    read_mult = math.exp(rng.uniform(math.log(0.5), math.log(10)))
    base_read = sum(rt for rt, _ in READ_OPS.values())
    read_mult = min(read_mult, MAX_READ_RATE / (base_read * ng))
    constraints = rng.random() < 1 / 3
    return {
        "hour": h,
        "epoch": e,
        "seed": f"{seed}-h{h}-e{e}",
        "busy": busy,
        "long_streams": {str(g): k for g, k in long_k.items()},
        "write_mult": round(math.exp(rng.uniform(math.log(0.25), math.log(4))), 3),
        "read_mult": round(read_mult, 3),
        "write_pattern": "holdoff" if rng.random() < 0.5 else "poisson",
        "indexes": rng.random() < 0.5,
        "constraints": constraints,
        "write_timeout_ms": rng.choice([500, 1500, 3000]) if rng.random() < 1 / 3 else 0,
        "bgsave": (not constraints) and rng.random() < 0.5,
    }


def epoch_seconds(epoch_s: float, drain_s: float, ep: dict) -> float:
    """What one epoch takes at most: the load, its drain, the end-of-epoch checks (120 s), and up
    to 300 s after the load for an index step and for a constraint step still building."""
    return epoch_s + drain_s + 120 + 300 * (bool(ep.get("indexes")) + bool(ep.get("constraints")))


def hour_seconds(epoch_s: float, drain_s: float, hp: dict, eps: list) -> float:
    """A server-hour: start and calibration (120 s), the build (about 120 s per unit of scale), its epochs."""
    return 120 + 120 * sum(hp["scales"]) + sum(epoch_seconds(epoch_s, drain_s, ep) for ep in eps)


#: the control epoch's graphs: a busy one at full scale and a quiet one beside it
CONTROL_SCALES = (1.0, 0.1)


def control_params(scales: tuple[float, ...] = CONTROL_SCALES) -> tuple[dict, dict]:
    """The fixed-seed control epoch: graph 0 busy (one long-read stream, Poisson writes arriving
    during the long reads), graph 1 quiet (reads only), THREAD_COUNT 32."""
    hp = {"hour": 0, "graphs": len(scales), "scales": list(scales), "thread_count": 32}
    ep = {
        "hour": 0,
        "epoch": 0,
        "seed": "control",
        "busy": [0],
        "long_streams": {"0": 1},
        "write_mult": 1.0,
        "read_mult": 1.0,
        "write_pattern": "poisson",
        "indexes": False,
        "constraints": False,
        "write_timeout_ms": 0,
        "bgsave": False,
    }
    return hp, ep
