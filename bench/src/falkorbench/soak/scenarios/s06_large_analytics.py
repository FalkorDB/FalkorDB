"""s06_large_analytics — the largest graph in the soak: citation network + road lattice,
heavy analytics (var-length, shortest paths, algo.*, full-graph aggregation, big ORDER BY)
running concurrently with OLTP point lookups and a small write stream (noisy-neighbour test).

CLI contract (same as s00_pilot):
  python -I s06_large_analytics.py --module <path> --port <port> --out <dir> --duration <sec> --scale <f>

Graph (scale 1.0):
  :Paper    NP=2.0M  {id, year, cat}      year = 1990 + id*35//NP, cat = id % 50
  :Author   NA=1.0M  {id}
  :Venue    NV=2000  {id, name}
  :Junction G*G=~0.4M {id, x, y}          lattice, id = y*G + x
  :Island   ~5K nodes {id}                K small directed cycles (known WCC components)
  (:Paper)-[:CITES {w}]->(:Paper)         12 per paper, to earlier papers (skewed => hubs)
  (:Author)-[:WROTE]->(:Paper)            3 per paper, skewed authors
  (:Paper)-[:PUBLISHED_IN]->(:Venue)
  (:Junction)-[:ROAD {w}]->(:Junction)    right (w=1) and down (w=2) only
  (:Island)-[:LINK]->(:Island)            cycles
OLTP writes only touch :Draft nodes / :REFS edges and Paper.views, so every analytic answer
over Paper/CITES/Junction/ROAD/Island stays exactly known during the soak.
"""
import argparse
import math
import os
import struct
import sys
import time
from collections import Counter
from pathlib import Path

import numpy as np
import redis

from falkorbench.soak import soakkit as sk

G_NAME = "citeroad"
CITES_K = 12
AUTH_PER_PAPER = 3
NCAT = 50
SEED = 606

# filled by setup_sizes()
S: dict = {}
EXP: dict = {}


def setup_sizes(scale: float):
    np_ = max(5000, int(2_000_000 * scale) // NCAT * NCAT)
    na = max(2000, int(1_000_000 * scale))
    nv = max(50, int(2000 * scale))
    side = max(40, int(math.sqrt(400_000 * scale)))
    k_isl = max(20, int(1000 * scale))
    S.update(NP=np_, NA=na, NV=nv, G=side, K=k_isl)
    S["A0"] = np_
    S["V0"] = np_ + na
    S["J0"] = S["V0"] + nv
    S["I0"] = S["J0"] + side * side
    S["isl_sizes"] = [3 + (k % 6) for k in range(k_isl)]
    S["NI"] = sum(S["isl_sizes"])
    S["NN"] = S["I0"] + S["NI"]


# ---------------------------------------------------------------- bulk encoding
def hdr(names: str, props: list[str]) -> bytes:
    b = names.encode() + b"\0" + struct.pack("=I", len(props))
    for p in props:
        b += p.encode() + b"\0"
    return b


NODE3 = np.dtype([("b0", "u1"), ("v0", "<i8"), ("b1", "u1"), ("v1", "<i8"), ("b2", "u1"), ("v2", "<i8")])
NODE1 = np.dtype([("b0", "u1"), ("v0", "<i8")])
NODE3b = NODE3
EDGE0 = np.dtype([("s", "<u8"), ("d", "<u8")])
EDGE1 = np.dtype([("s", "<u8"), ("d", "<u8"), ("b", "u1"), ("v", "<i8")])
BI_LONG, BI_STRING = 4, 3


def rec3(a, b, c) -> bytes:
    n = len(a)
    r = np.empty(n, NODE3)
    r["b0"] = r["b1"] = r["b2"] = BI_LONG
    r["v0"], r["v1"], r["v2"] = a, b, c
    return r.tobytes()


def rec1(a) -> bytes:
    r = np.empty(len(a), NODE1)
    r["b0"] = BI_LONG
    r["v0"] = a
    return r.tobytes()


def erec1(s, d, w) -> bytes:
    r = np.empty(len(s), EDGE1)
    r["s"], r["d"], r["b"], r["v"] = s, d, BI_LONG, w
    return r.tobytes()


def erec0(s, d) -> bytes:
    r = np.empty(len(s), EDGE0)
    r["s"], r["d"] = s, d
    return r.tobytes()


class Bulk:
    def __init__(self, r, graph):
        self.r, self.graph, self.first = r, graph, True
        self.sent_nodes = 0
        self.sent_edges = 0

    def send(self, nodes: int, edges: int, ntok: list[bytes], rtok: list[bytes]):
        args = ["GRAPH.BULK", self.graph]
        if self.first:
            args.append("BEGIN")
        args += [nodes, edges, len(ntok), len(rtok), *ntok, *rtok]
        res = self.r.execute_command(*args)
        self.first = False
        self.sent_nodes += nodes
        self.sent_edges += edges
        return res


def load_graph(r, ex: dict):
    """Generate with numpy + GRAPH.BULK. Fills ex with python-side ground truth."""
    NP, NA, NV, G, K = S["NP"], S["NA"], S["NV"], S["G"], S["K"]
    bulk = Bulk(r, G_NAME)
    CH = 1_000_000

    # ---- nodes
    for lo in range(0, NP, CH):
        ids = np.arange(lo, min(NP, lo + CH), dtype=np.int64)
        tok = hdr("Paper", ["id", "year", "cat"]) + rec3(ids, 1990 + ids * 35 // NP, ids % NCAT)
        bulk.send(len(ids), 0, [tok], [])
    for lo in range(0, NA, CH):
        ids = np.arange(lo, min(NA, lo + CH), dtype=np.int64)
        bulk.send(len(ids), 0, [hdr("Author", ["id"]) + rec1(ids)], [])
    vb = hdr("Venue", ["id", "name"])
    for i in range(NV):
        vb += bytes([BI_LONG]) + struct.pack("=q", i) + bytes([BI_STRING]) + f"venue{i}".encode() + b"\0"
    bulk.send(NV, 0, [vb], [])
    gx = np.arange(G * G, dtype=np.int64)
    bulk.send(G * G, 0, [hdr("Junction", ["id", "x", "y"]) + rec3(gx, gx % G, gx // G)], [])
    ib = hdr("Island", ["id"]) + rec1(np.arange(S["NI"], dtype=np.int64))
    bulk.send(S["NI"], 0, [ib], [])

    # ---- edges
    cites_in = np.zeros(NP, dtype=np.int64)
    sum_w = 0
    auth_deg = np.zeros(NA, dtype=np.int64)
    venue_deg = np.zeros(NV, dtype=np.int64)
    n_cites = n_wrote = n_pub = 0
    ECH = 1_000_000 // CITES_K  # papers per call
    for c, lo in enumerate(range(0, NP, ECH)):
        hi = min(NP, lo + ECH)
        rng = np.random.default_rng([SEED, c])
        ids = np.arange(max(lo, 1), hi, dtype=np.int64)  # paper 0 cites nothing
        n = len(ids)
        src = np.repeat(ids, CITES_K)
        u = rng.random(n * CITES_K)
        dst = (np.repeat(ids, CITES_K) * u * u).astype(np.int64)
        w = rng.integers(1, 101, size=n * CITES_K)
        cites_in += np.bincount(dst, minlength=NP)
        sum_w += int(w.sum())
        n_cites += len(src)
        t_c = hdr("CITES", ["w"]) + erec1(src, dst, w)
        # authors
        pids = np.arange(lo, hi, dtype=np.int64)
        asrc = (NA * rng.random(len(pids) * AUTH_PER_PAPER) ** 1.5).astype(np.int64)
        adst = np.repeat(pids, AUTH_PER_PAPER)
        auth_deg += np.bincount(asrc, minlength=NA)
        n_wrote += len(asrc)
        t_w = hdr("WROTE", []) + erec0(asrc + S["A0"], adst)
        # venues
        v = rng.integers(0, NV, size=len(pids))
        venue_deg += np.bincount(v, minlength=NV)
        n_pub += len(pids)
        t_p = hdr("PUBLISHED_IN", []) + erec0(pids, v + S["V0"])
        bulk.send(0, len(src) + len(asrc) + len(pids), [], [t_c, t_w, t_p])
    # grid
    xs = np.arange(G * G, dtype=np.int64)
    x, y = xs % G, xs // G
    m = x < G - 1
    right_s, right_d = xs[m], xs[m] + 1
    m = y < G - 1
    down_s, down_d = xs[m], xs[m] + G
    gs = np.concatenate([right_s, down_s]) + S["J0"]
    gd = np.concatenate([right_d, down_d]) + S["J0"]
    gw = np.concatenate([np.ones(len(right_s), np.int64), np.full(len(down_s), 2, np.int64)])
    n_road = len(gs)
    for lo in range(0, n_road, 1_500_000):
        sl = slice(lo, lo + 1_500_000)
        bulk.send(0, len(gs[sl]), [], [hdr("ROAD", ["w"]) + erec1(gs[sl], gd[sl], gw[sl])])
    # islands: directed cycles
    ls_, ld_ = [], []
    base = S["I0"]
    for sz in S["isl_sizes"]:
        for j in range(sz):
            ls_.append(base + j)
            ld_.append(base + (j + 1) % sz)
        base += sz
    n_link = len(ls_)
    bulk.send(0, n_link, [], [hdr("LINK", []) + erec0(np.array(ls_), np.array(ld_))])

    # ---- ground truth
    ci = cites_in[cites_in > 0]
    vals, cnts = np.unique(ci, return_counts=True)
    ex["indeg_hist"] = {int(a): int(b) for a, b in zip(vals, cnts)}
    ex["cites_in"] = cites_in
    ex["auth_deg"] = auth_deg
    ex["sum_w"] = sum_w
    ex["n_cites"], ex["n_wrote"], ex["n_pub"], ex["n_road"], ex["n_link"] = n_cites, n_wrote, n_pub, n_road, n_link
    ex["iso_auth"] = int((auth_deg == 0).sum())
    ex["iso_venue"] = int((venue_deg == 0).sum())
    ex["edges"] = n_cites + n_wrote + n_pub + n_road + n_link
    ex["nodes"] = S["NN"]
    ex["wcc"] = 2 + K + ex["iso_auth"] + ex["iso_venue"]  # giant + lattice + islands + isolated
    ex["wrote_hist_max"] = int(auth_deg.max())


# ---------------------------------------------------------------- helpers
class Wrong(Exception):
    """A result that disagrees with ground truth (counted as an error under `op`)."""

    def __init__(self, op, msg):
        super().__init__("WRONG " + msg)
        self.op = op


def q(ctx_or_r, text, params=None, ro=True, timeout_ms=None):
    # a client is used as-is; a worker context is unwrapped via .r
    is_client = hasattr(ctx_or_r, "execute_command")
    r = ctx_or_r if is_client else ctx_or_r.r
    return sk.query(r, G_NAME, text, params, ro=ro, timeout_ms=timeout_ms)


def jid(x, y):
    return y * S["G"] + x


def comb(n, k):
    return math.comb(n, k)


HEAVY_TO = 300_000
ANALYST_RATE = 3.0  # per analyst stream; open loop (a slow op just starts the next one late)


# ---------------------------------------------------------------- workload ops
def op_paper_lookup(ctx):
    i = ctx.rng.randrange(S["NP"])
    res = q(ctx, "MATCH (p:Paper {id: $id}) RETURN p.year, p.cat", {"id": i})
    rs = sk.rows(res)
    if len(rs) != 1 or sk.cval(rs[0][0]) != 1990 + i * 35 // S["NP"] or sk.cval(rs[0][1]) != i % NCAT:
        raise Wrong("lookup_paper", f"paper {i}: {rs}")
    return "lookup_paper"


def op_paper_cites(ctx):
    i = ctx.rng.randrange(S["NP"])
    res = q(ctx, "MATCH (p:Paper {id: $id})-[e:CITES]->(c) RETURN count(e)", {"id": i})
    n = sk.scalar(res)
    if n != (CITES_K if i else 0):
        raise Wrong("lookup_cites_out", f"paper {i}: {n} cites")
    return "lookup_cites_out"


def op_paper_citedby(ctx):
    i = ctx.rng.randrange(S["NP"])
    res = q(ctx, "MATCH (p:Paper {id: $id})<-[e:CITES]-(c) RETURN count(e)", {"id": i})
    if sk.scalar(res) != int(EXP["cites_in"][i]):
        raise Wrong("lookup_cited_by", f"paper {i}: {sk.scalar(res)} != {EXP['cites_in'][i]}")
    return "lookup_cited_by"


def op_author(ctx):
    i = ctx.rng.randrange(S["NA"])
    res = q(ctx, "MATCH (a:Author {id: $id})-[e:WROTE]->(p) RETURN count(e)", {"id": i})
    if sk.scalar(res) != int(EXP["auth_deg"][i]):
        raise Wrong("lookup_author", f"author {i}: {sk.scalar(res)} != {EXP['auth_deg'][i]}")
    return "lookup_author"


def op_junction(ctx):
    G = S["G"]
    x, y = ctx.rng.randrange(G - 1), ctx.rng.randrange(G - 1)
    res = q(ctx, "MATCH (a:Junction {id: $id})-[r:ROAD]->(b) RETURN b.id, r.w ORDER BY b.id", {"id": jid(x, y)})
    got = [(sk.cval(a[0]), sk.cval(a[1])) for a in sk.rows(res)]
    if got != [(jid(x + 1, y), 1), (jid(x, y + 1), 2)]:
        raise Wrong("lookup_junction", f"{x},{y}: {got}")
    return "lookup_junction"


def pick_lookup(ctx):
    r = ctx.rng.random()
    if r < 0.30:
        return op_paper_lookup(ctx)
    if r < 0.55:
        return op_paper_cites(ctx)
    if r < 0.75:
        return op_paper_citedby(ctx)
    if r < 0.90:
        return op_author(ctx)
    return op_junction(ctx)


def w_init(ctx):
    ctx.state["seq"] = 0
    ctx.state["mine"] = []


def op_write(ctx):
    r = ctx.rng.random()
    if r < 0.5:
        i = ctx.rng.randrange(S["NP"])
        q(ctx, "MATCH (p:Paper {id: $id}) SET p.views = coalesce(p.views, 0) + 1", {"id": i}, ro=False)
        return "write_set_views"
    if r < 0.85 or not ctx.state["mine"]:
        ctx.state["seq"] += 1
        did = 10_000_000_000 + ctx.wid * 10_000_000 + ctx.state["seq"]
        t = [ctx.rng.randrange(S["NP"]) for _ in range(3)]
        q(ctx, "CREATE (d:Draft {id: $id}) WITH d UNWIND $t AS t MATCH (p:Paper {id: t}) CREATE (d)-[:REFS]->(p)",
          {"id": did, "t": t}, ro=False)
        ctx.state["mine"].append(did)
        return "write_create_draft"
    did = ctx.state["mine"].pop(ctx.rng.randrange(len(ctx.state["mine"])))
    q(ctx, "MATCH (d:Draft {id: $id}) DETACH DELETE d", {"id": did}, ro=False)
    return "write_delete_draft"


# ---- analytics (each verifies against analytic ground truth)
def rand_pair(ctx, maxd):
    G = S["G"]
    dx, dy = ctx.rng.randint(0, maxd), ctx.rng.randint(0, maxd)
    dx, dy = min(dx, G - 1), min(dy, G - 1)
    x, y = ctx.rng.randint(0, G - 1 - dx), ctx.rng.randint(0, G - 1 - dy)
    return x, y, dx, dy


def an_sp_grid(ctx):
    """Undirected shortestPath between two random junctions anywhere on the lattice: length is the
    Manhattan distance. Explores a ball of radius d, so this is the long one at full scale."""
    G = S["G"]
    undirected = EXP.get("undirected_sp", True)
    if undirected:
        x1, y1, x2, y2 = (ctx.rng.randrange(G) for _ in range(4))
    else:  # engine without undirected shortestPath: monotone (right/down) pair, same Manhattan answer
        x1, y1, dx, dy = rand_pair(ctx, G)
        x2, y2 = x1 + dx, y1 + dy
    if (x1, y1) == (x2, y2):
        return "an_shortestPath"
    res = q(ctx, "MATCH (a:Junction {id:$a}), (b:Junction {id:$b}) "
                 "RETURN length(shortestPath((a)-[:ROAD%s]%s(b)))" % ("*", "-" if undirected else "->"),
            {"a": jid(x1, y1), "b": jid(x2, y2)}, timeout_ms=HEAVY_TO)
    got = sk.scalar(res)
    want = abs(x1 - x2) + abs(y1 - y2)
    if got != want:
        raise Wrong("an_shortestPath", f"({x1},{y1})->({x2},{y2}) len {got} != {want}")
    return "an_shortestPath"


def an_allsp_grid(ctx):
    while True:
        x, y, dx, dy = rand_pair(ctx, 8)
        if 0 < dx + dy <= 12:
            break
    res = q(ctx, "MATCH (a:Junction {id:$a}), (b:Junction {id:$b}) WITH a, b "
                 "MATCH p = allShortestPaths((a)-[:ROAD*]->(b)) RETURN count(p), min(length(p)), max(length(p))",
            {"a": jid(x, y), "b": jid(x + dx, y + dy)}, timeout_ms=HEAVY_TO)
    rs = sk.rows(res)
    got = tuple(sk.cval(c) for c in rs[0])
    want = (comb(dx + dy, dx), dx + dy, dx + dy)
    if got != want:
        raise Wrong("an_allShortestPaths", f"({x},{y})->+({dx},{dy}) {got} != {want}")
    return "an_allShortestPaths"


def an_sp_weighted(ctx):
    x, y, dx, dy = rand_pair(ctx, S["G"])
    if dx + dy == 0:
        return "an_SPpaths_weighted"
    res = q(ctx, "MATCH (a:Junction {id:$a}), (b:Junction {id:$b}) "
                 "CALL algo.SPpaths({sourceNode: a, targetNode: b, relTypes: ['ROAD'], weightProp: 'w'}) "
                 "YIELD path, pathWeight RETURN pathWeight, length(path)",
            {"a": jid(x, y), "b": jid(x + dx, y + dy)}, timeout_ms=HEAVY_TO)
    rs = sk.rows(res)
    if not rs:
        raise Wrong("an_SPpaths_weighted", f"no path ({x},{y})->+({dx},{dy})")
    pw, ln = sk.cval(rs[0][0]), sk.cval(rs[0][1])
    if pw != dx + 2 * dy or ln != dx + dy:
        raise Wrong("an_SPpaths_weighted", f"({x},{y})->+({dx},{dy}) w={pw} len={ln} want {dx + 2 * dy}/{dx + dy}")
    return "an_SPpaths_weighted"


def an_grid_varlen(ctx):
    G = S["G"]
    x, y = ctx.rng.randint(0, G - 5), ctx.rng.randint(0, G - 5)
    res = q(ctx, "MATCH (a:Junction {id:$a})-[:ROAD*1..4]->(b) RETURN count(b), count(DISTINCT b)",
            {"a": jid(x, y)}, timeout_ms=HEAVY_TO)
    got = tuple(sk.cval(c) for c in sk.rows(res)[0])
    if got != (30, 14):
        raise Wrong("an_grid_varlen", f"({x},{y}) {got} != (30, 14)")
    return "an_grid_varlen"


def an_grid_varlen_undirected(ctx):
    G = S["G"]
    x, y = ctx.rng.randint(4, G - 5), ctx.rng.randint(4, G - 5)
    res = q(ctx, "MATCH (a:Junction {id:$a})-[:ROAD*1..4]-(b) RETURN count(DISTINCT b)",
            {"a": jid(x, y)}, timeout_ms=HEAVY_TO)
    got = sk.scalar(res)
    # diamond of radius 4 around an interior point, minus the centre
    # (undirected walks may revisit nodes but the edge-isomorphic paths still reach exactly the
    # Manhattan ball except the start; the start is reachable too via a 4-cycle => included)
    if got not in (40, 41):
        raise Wrong("an_grid_varlen_undirected", f"({x},{y}) {got} not in (40, 41)")
    return "an_grid_varlen_undirected"


def an_bfs(ctx):
    G = S["G"]
    x, y = ctx.rng.randint(0, G - 4), ctx.rng.randint(0, G - 4)
    res = q(ctx, "MATCH (a:Junction {id:$a}) CALL algo.BFS(a, 3, 'ROAD') YIELD nodes RETURN size(nodes)",
            {"a": jid(x, y)}, timeout_ms=HEAVY_TO)
    got = sk.scalar(res)
    if got != EXP["bfs3"]:
        raise Wrong("an_BFS", f"({x},{y}) {got} != {EXP['bfs3']}")
    return "an_BFS"


def an_paper_varlen(ctx):
    # citation closure: 14^4 paths from a recent paper; result cross-checked for sanity only
    i = ctx.rng.randrange(S["NP"] // 2, S["NP"])
    res = q(ctx, "MATCH (p:Paper {id:$id})-[:CITES*1..5]->(q) RETURN count(q), count(DISTINCT q), sum(q.cat)",
            {"id": i}, timeout_ms=HEAVY_TO)
    c, d, _ = (sk.cval(x) for x in sk.rows(res)[0])
    if not (d <= c <= sum(CITES_K ** k for k in range(1, 6))) or d < CITES_K:
        raise Wrong("an_paper_varlen", f"paper {i}: count={c} distinct={d}")
    return "an_paper_varlen"


def an_hub_incoming(ctx):
    """Who (transitively) cites an old hub: huge frontier. Sanity bound only (>= its in-degree)."""
    hub = ctx.rng.randrange(0, 5)
    res = q(ctx, "MATCH (p:Paper {id:$id})<-[:CITES*1..2]-(q) RETURN count(DISTINCT q)", {"id": hub},
            timeout_ms=HEAVY_TO)
    d = sk.scalar(res)
    lo = 1 if EXP["cites_in"][hub] > 0 else 0
    if d < lo or d > S["NP"]:
        raise Wrong("an_hub_incoming", f"hub {hub}: distinct={d}")
    return "an_hub_incoming"


def an_paper_varlen_agg(ctx):
    # aggregation over var-length: citations-of-citations per venue
    i = ctx.rng.randrange(S["NP"] // 2, S["NP"])
    res = q(ctx, "MATCH (p:Paper {id:$id})-[:CITES*1..3]->(q) WITH q MATCH (q)-[:PUBLISHED_IN]->(v) "
                 "RETURN v.id, count(*) AS c ORDER BY c DESC LIMIT 10", {"id": i}, timeout_ms=HEAVY_TO)
    rs = sk.rows(res)
    tot = sum(sk.cval(r[1]) for r in rs)
    if not rs or tot < 1:
        raise Wrong("an_paper_varlen_agg", f"paper {i}: {rs}")
    return "an_paper_varlen_agg"


def an_label_counts(ctx):
    res = q(ctx, "MATCH (n) RETURN labels(n)[0] AS l, count(*) AS c", timeout_ms=HEAVY_TO)
    got = {sk.cval(r[0]): sk.cval(r[1]) for r in sk.rows(res)}
    want = {"Paper": S["NP"], "Author": S["NA"], "Venue": S["NV"], "Junction": S["G"] ** 2, "Island": S["NI"]}
    bad = {k: (got.get(k), v) for k, v in want.items() if got.get(k) != v}
    if bad:
        raise Wrong("an_label_counts", str(bad))
    return "an_label_counts"


def an_degree_dist(ctx):
    res = q(ctx, "MATCH (p:Paper)<-[e:CITES]-() WITH p, count(e) AS d RETURN d, count(*) AS c",
            timeout_ms=HEAVY_TO)
    got = {sk.cval(r[0]): sk.cval(r[1]) for r in sk.rows(res)}
    if got != EXP["indeg_hist"]:
        diff = [(k, got.get(k), EXP["indeg_hist"].get(k)) for k in set(got) | set(EXP["indeg_hist"])
                if got.get(k) != EXP["indeg_hist"].get(k)]
        raise Wrong("an_degree_dist", f"{len(diff)} bins differ, e.g. {diff[:3]}")
    return "an_degree_dist"


def an_sum_weights(ctx):
    res = q(ctx, "MATCH ()-[c:CITES]->() RETURN count(c), sum(c.w)", timeout_ms=HEAVY_TO)
    got = tuple(sk.cval(c) for c in sk.rows(res)[0])
    if got != (EXP["n_cites"], EXP["sum_w"]):
        raise Wrong("an_sum_weights", f"{got} != {(EXP['n_cites'], EXP['sum_w'])}")
    return "an_sum_weights"


def an_order_limit(ctx):
    res = q(ctx, "MATCH (p:Paper) RETURN p.id ORDER BY p.year DESC, p.id ASC LIMIT 5", timeout_ms=HEAVY_TO)
    got = [sk.cval(r[0]) for r in sk.rows(res)]
    ymax = 1990 + (S["NP"] - 1) * 35 // S["NP"]
    first = next(i for i in range(S["NP"]) if 1990 + i * 35 // S["NP"] == ymax)
    if got != list(range(first, first + 5)):
        raise Wrong("an_order_limit", f"{got} != {list(range(first, first + 5))}")
    return "an_order_limit"


def an_order_full(ctx):
    # full sort of every paper by (cat, id); probe a deep row
    per = S["NP"] // NCAT
    m = ctx.rng.randrange(per, S["NP"] - 3)
    res = q(ctx, "MATCH (p:Paper) RETURN p.id ORDER BY p.cat ASC, p.id ASC SKIP $m LIMIT 3",
            {"m": m}, timeout_ms=HEAVY_TO)
    got = [sk.cval(r[0]) for r in sk.rows(res)]
    want = [(k // per) + NCAT * (k % per) for k in range(m, m + 3)]
    if got != want:
        raise Wrong("an_order_full", f"skip {m}: {got} != {want}")
    return "an_order_full"


def an_pagerank_grid(ctx):
    res = q(ctx, "CALL algo.pageRank('Junction', 'ROAD') YIELD node, score RETURN count(node), max(score)",
            timeout_ms=HEAVY_TO)
    c = sk.cval(sk.rows(res)[0][0])
    if c != S["G"] ** 2:
        raise Wrong("an_pageRank_lattice", f"{c} != {S['G'] ** 2}")
    return "an_pageRank_lattice"


def an_pagerank_full(ctx):
    res = q(ctx, "CALL algo.pageRank(null, null) YIELD node, score RETURN count(node)", timeout_ms=HEAVY_TO)
    c = sk.cval(sk.rows(res)[0][0])
    # Draft nodes churn during the soak: only a lower bound
    if c < S["NN"]:
        raise Wrong("an_pageRank_full", f"{c} < {S['NN']}")
    return "an_pageRank_full"


def an_wcc(ctx):
    res = q(ctx, "CALL algo.WCC() YIELD node, componentId RETURN count(DISTINCT componentId)",
            timeout_ms=HEAVY_TO)
    c = sk.scalar(res)
    if c != EXP["wcc"]:
        raise Wrong("an_WCC", f"{c} != {EXP['wcc']}")
    return "an_WCC"


def an_wcc_islands(ctx):
    res = q(ctx, "CALL algo.WCC({nodeLabels: ['Island'], relationshipTypes: ['LINK']}) "
                 "YIELD node, componentId RETURN count(DISTINCT componentId)", timeout_ms=HEAVY_TO)
    c = sk.scalar(res)
    if c != S["K"]:
        raise Wrong("an_WCC_islands", f"{c} != {S['K']}")
    return "an_WCC_islands"


ANALYTICS = [  # (fn, weight)
    (an_sp_grid, 14), (an_allsp_grid, 8), (an_sp_weighted, 10), (an_grid_varlen, 10),
    (an_grid_varlen_undirected, 6), (an_bfs, 8), (an_paper_varlen, 10), (an_paper_varlen_agg, 8), (an_hub_incoming, 3),
    (an_label_counts, 4), (an_degree_dist, 3), (an_sum_weights, 3), (an_order_limit, 4),
    (an_order_full, 3), (an_pagerank_grid, 3), (an_wcc_islands, 2), (an_wcc, 1), (an_pagerank_full, 1),
]
_AW = [w for _, w in ANALYTICS]


def op_analytics(ctx):
    fn = ctx.rng.choices([f for f, _ in ANALYTICS], weights=_AW)[0]
    return fn(ctx)


def op_runaway(ctx):
    """Runaway query bounded by a client TIMEOUT; must fail with a timeout, promptly."""
    t0 = time.time()
    try:
        q(ctx, "MATCH (a:Paper)-[:CITES*1..4]->(b) RETURN count(b)", timeout_ms=EXP["rt_ms"])
    except redis.exceptions.ResponseError as e:
        el = time.time() - t0
        if "timed out" in str(e).lower() or "timeout" in str(e).lower():
            if el > EXP["rt_ms"] / 1000 + 10:
                raise Wrong("runaway_timeout", f"timed out only after {el:.1f}s (limit {EXP['rt_ms']} ms)")
            return "runaway_timeout"
        raise
    raise Wrong("runaway_timeout", f"runaway completed in {time.time() - t0:.1f}s without timing out")


# ---------------------------------------------------------------- main
def wait_indexes(r, tmo=600):
    t0 = time.time()
    while time.time() - t0 < tmo:
        try:
            res = sk.query(r, G_NAME, "CALL db.indexes() YIELD label, status RETURN label, status", ro=True)
            st = [sk.cval(x[1]) for x in sk.rows(res)]
            if st and all(str(s).upper().startswith("OPER") for s in st):
                return True
        except Exception:
            return None  # procedure shape differs: caller falls back to a sleep
        time.sleep(1)
    return False


def err_kind(e):
    s = str(e).lower()
    if "timed out" in s or "timeout" in s:
        return "timeout"
    if "mem" in s and ("capacity" in s or "exceed" in s):
        return "mem"
    return "other:" + str(e)[:120]


def main():
    ap = argparse.ArgumentParser()
    sk.add_engine_args(ap)  # --module PATH | --image REF
    ap.add_argument("--port", type=int, required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--duration", type=float, default=60)
    ap.add_argument("--scale", type=float, default=0.05)
    a = ap.parse_args()
    out = Path(a.out).resolve()
    out.mkdir(parents=True, exist_ok=True)
    setup_sizes(a.scale)

    # TIMEOUT_DEFAULT bounds every query that forgets a client timeout; TIMEOUT_MAX caps client ones
    srv = sk.Server(a.module, a.port, out, module_args=["TIMEOUT_DEFAULT", "300000", "TIMEOUT_MAX", "600000"])
    is_rust = srv.engine() == "rust"
    mon = sk.Monitor(srv, out)
    mon.start()
    checks = sk.Checks(out)
    r = srv.client()
    meta = {"scenario": "s06_large_analytics", "scale": a.scale, "engine": "rust" if is_rust else "c",
            "duration": a.duration, "sizes": {k: v for k, v in S.items() if k != "isl_sizes"}}

    # ---------------- ingest
    mon.mark("ingest")
    time.sleep(1.5)
    base = sk.Monitor  # noqa: F841
    info0 = r.info()
    rss0 = srv_rss_mb(srv)
    t = time.time()
    try:
        load_graph(r, EXP)
    except Exception as e:
        checks.add("bulk ingest", False, repr(e), "critical")
        mon.stop()
        srv.stop()
        sys.exit(1)
    t_bulk = time.time() - t
    for lab in ("Paper", "Author", "Junction", "Island"):
        sk.query(r, G_NAME, f"CREATE INDEX FOR (n:{lab}) ON (n.id)")
    ok = wait_indexes(r)
    if ok is None:
        time.sleep(20)
    ingest_s = time.time() - t
    meta.update(bulk_s=t_bulk, ingest_s=ingest_s, indexes_ready=ok)
    rss1 = srv_rss_mb(srv)
    um1 = r.info().get("used_memory", 0) / 2**20
    nodes_total, edges_total = EXP["nodes"], EXP["edges"]
    meta.update(nodes=nodes_total, edges=edges_total,
                rss_load_mb=rss1 - rss0, used_memory_load_mb=um1 - info0.get("used_memory", 0) / 2**20,
                bytes_per_edge_rss=(rss1 - rss0) * 2**20 / edges_total,
                bytes_per_edge_used_memory=(um1 - info0.get("used_memory", 0) / 2**20) * 2**20 / edges_total)
    print(f"ingest {ingest_s:.1f}s (bulk {t_bulk:.1f}s) nodes={nodes_total} edges={edges_total} "
          f"rss={rss1:.0f}MB B/edge rss={meta['bytes_per_edge_rss']:.1f} used_memory={meta['bytes_per_edge_used_memory']:.1f}",
          flush=True)

    nodes = sk.scalar(q(r, "MATCH (n) RETURN count(n)"))
    edges = sk.scalar(q(r, "MATCH ()-[e]->() RETURN count(e)"))
    checks.add("ingest node count", nodes == nodes_total, f"{nodes} != {nodes_total}", "critical")
    checks.add("ingest edge count", edges == edges_total, f"{edges} != {edges_total}", "critical")
    for rel, key in (("CITES", "n_cites"), ("WROTE", "n_wrote"), ("PUBLISHED_IN", "n_pub"),
                     ("ROAD", "n_road"), ("LINK", "n_link")):
        c = sk.scalar(q(r, f"MATCH ()-[e:{rel}]->() RETURN count(e)"))
        checks.add(f"ingest {rel} count", c == EXP[key], f"{c} != {EXP[key]}", "critical")

    # ---------------- quiet-phase exact verification of every analytic (also times each once)
    mon.mark("verify-pre")
    EXP["bfs3"] = 9
    EXP["rt_ms"] = 2000
    class _C:  # minimal ctx for running analytics from the parent
        pass
    import random
    c0 = _C()
    c0.r, c0.rng, c0.wid = r, random.Random(7), 0
    probe_times = {}
    try:
        q(r, "MATCH (a:Junction {id:0}), (b:Junction {id:5}) RETURN length(shortestPath((a)-[:ROAD*]-(b)))")
        EXP["undirected_sp"] = True
    except redis.exceptions.ResponseError as e:
        EXP["undirected_sp"] = False
        checks.add("undirected shortestPath supported (C lacks it: soak falls back to directed)", False,
                   str(e)[:120], "minor")
    bfs_n = sk.scalar(q(r, "MATCH (a:Junction {id:$a}) CALL algo.BFS(a, 3, 'ROAD') YIELD nodes RETURN size(nodes)",
                      {"a": jid(5, 5)}))
    meta["bfs3_observed"] = bfs_n
    EXP["bfs3"] = bfs_n  # first observation seeds the invariant; checked against 9/10 below
    checks.add("BFS depth-3 node count on lattice is analytic (9, +1 if source included)",
               bfs_n in (9, 10), f"{bfs_n}", "critical")
    for fn, _ in ANALYTICS:
        t = time.time()
        try:
            fn(c0)
            ok_, det = True, ""
        except Wrong as e:
            ok_, det = False, str(e)
        except redis.exceptions.ResponseError as e:
            ok_, det = False, f"error: {e}"
        dt_ = time.time() - t
        probe_times[fn.__name__] = round(dt_, 3)
        checks.add(f"verify {fn.__name__}", ok_, det, "critical")
    for fn in (op_paper_lookup, op_paper_cites, op_paper_citedby, op_author, op_junction):
        ok_ = True
        det = ""
        try:
            for _ in range(200):
                fn(c0)
        except Exception as e:
            ok_, det = False, str(e)
        checks.add(f"verify {fn.__name__} x200", ok_, det, "critical")
    meta["probe_times_s"] = probe_times
    print("probe times:", probe_times, flush=True)

    # cross-check var-length against sequential 1-hop expansion for a few papers
    bad = []
    for i in (S["NP"] - 1, S["NP"] // 2, S["NP"] // 3):
        a1 = {sk.cval(x[0]) for x in sk.rows(q(r, "MATCH (p:Paper {id:$i})-[:CITES]->(c) RETURN DISTINCT c.id", {"i": i}))}
        lvl, seen = a1, set()
        reach = set(a1)
        for _ in range(3):
            nxt = set()
            for src_id in sorted(lvl):
                for x in sk.rows(q(r, "MATCH (p:Paper {id:$i})-[:CITES]->(c) RETURN DISTINCT c.id", {"i": src_id})):
                    nxt.add(sk.cval(x[0]))
            reach |= nxt
            lvl = nxt
        got = sk.scalar(q(r, "MATCH (p:Paper {id:$i})-[:CITES*1..4]->(c) RETURN count(DISTINCT c)", {"i": i}))
        if got != len(reach):
            bad.append((i, got, len(reach)))
    checks.add("var-length *1..4 distinct reach == iterated 1-hop expansion", not bad, str(bad), "critical")

    # ---------------- guard rails: timeout, TIMEOUT_MAX, QUERY_MEM_CAPACITY (quiet, before soak)
    mon.mark("guards")
    guards = {}
    runaway = "MATCH (a:Paper)-[:CITES*1..4]->(b) RETURN count(b)"
    for tmo in (1000, 3000):
        t = time.time()
        try:
            q(r, runaway, timeout_ms=tmo)
            kind = "completed"
        except redis.exceptions.ResponseError as e:
            kind = err_kind(e)
        el = time.time() - t
        guards[f"timeout_{tmo}ms"] = {"result": kind, "elapsed_s": round(el, 2)}
        checks.add(f"runaway query with TIMEOUT {tmo} ms times out", kind == "timeout", f"{kind} after {el:.1f}s",
                   "major")
        checks.add(f"timeout {tmo} ms fires promptly (<= limit + 5s)", el <= tmo / 1000 + 5,
                   f"{el:.1f}s", "major")
    t = time.time()
    chk = sk.scalar(q(r, "MATCH (p:Paper {id: 1}) RETURN p.id"))
    guards["post_timeout_lookup_ms"] = round((time.time() - t) * 1000, 2)
    checks.add("server healthy after timeouts (lookup works)", chk == 1, str(chk), "critical")
    # timeout as a read-write (non-RO) query path as well
    t = time.time()
    try:
        q(r, runaway, ro=False, timeout_ms=1000)
        kind = "completed"
    except redis.exceptions.ResponseError as e:
        kind = err_kind(e)
    guards["timeout_graph_query_1000ms"] = {"result": kind, "elapsed_s": round(time.time() - t, 2)}
    checks.add("GRAPH.QUERY (non-RO) runaway also times out", kind == "timeout", kind, "major")
    # timeout above TIMEOUT_MAX
    try:
        q(r, "RETURN 1", timeout_ms=10_000_000)
        guards["timeout_above_max"] = "accepted"
    except redis.exceptions.ResponseError as e:
        guards["timeout_above_max"] = "rejected: " + str(e)[:100]
    # planner probe: var-length followed by a fixed hop (the WITH-separated form is what the soak runs)
    pq = "MATCH (p:Paper {id:90}) -[:CITES*1..3]->(q)-[:PUBLISHED_IN]->(v) RETURN count(v)"
    t = time.time()
    try:
        q(r, pq, timeout_ms=20000)
        kind = "completed"
    except redis.exceptions.ResponseError as e:
        kind = err_kind(e)
    el = time.time() - t
    guards["planner_varlen_then_hop"] = {"result": kind, "elapsed_s": round(el, 3)}
    checks.add("varlen-then-hop query anchored on index lookup stays fast (<2s)", kind == "completed" and el < 2,
               f"{kind} in {el:.1f}s (plan scans all nodes first if slow)", "major")
    # QUERY_MEM_CAPACITY
    try:
        cur = r.execute_command("GRAPH.CONFIG", "GET", "QUERY_MEM_CAPACITY")
        guards["mem_capacity_default"] = str(cur)
        cap = 256 * 2**20
        r.execute_command("GRAPH.CONFIG", "SET", "QUERY_MEM_CAPACITY", cap)
        rss_b = srv_rss_mb(srv)
        t = time.time()
        try:
            q(r, "MATCH (a:Paper)-[:CITES*1..3]->(b) RETURN collect(b.id)", timeout_ms=5000)
            kind = "completed"
        except redis.exceptions.ResponseError as e:
            kind = err_kind(e)
        guards["mem_capacity_256MB_collect"] = {"result": kind, "elapsed_s": round(time.time() - t, 2),
                                                "rss_peak_over_before_mb": round(srv_rss_mb(srv) - rss_b, 1)}
        checks.add("QUERY_MEM_CAPACITY=256MB aborts a memory-hog query", kind == "mem",
                   f"{kind} ({guards['mem_capacity_256MB_collect']})", "major")
        r.execute_command("GRAPH.CONFIG", "SET", "QUERY_MEM_CAPACITY", 0)
        small = sk.scalar(q(r, "MATCH (p:Paper {id: 7}) RETURN p.id"))
        checks.add("server healthy after memory-limit abort", small == 7, str(small), "critical")
    except redis.exceptions.ResponseError as e:
        guards["mem_capacity"] = "unsupported: " + str(e)[:100]
        checks.add("QUERY_MEM_CAPACITY supported", False, str(e)[:150], "minor")
    meta["guards"] = guards
    print("guards:", guards, flush=True)

    # ---------------- soak
    # worker counts: analytics streams, OLTP lookups (open loop), small write stream, runaway probe
    n_an = 4
    lookup_rate = 150
    specs = [
        sk.WorkerSpec(op_analytics, rate=ANALYST_RATE, count=n_an, name="analyst"),
        sk.WorkerSpec(pick_lookup, rate=lookup_rate, count=4, name="oltp"),
        sk.WorkerSpec(op_write, rate=12, count=2, init=w_init, name="writer"),
        sk.WorkerSpec(op_runaway, rate=1 / 45, count=1, name="runaway"),
    ]
    if os.environ.get("S06_NO_WRITERS"):  # diagnostic: isolate read-vs-read from read-vs-write stalls
        specs = [x for x in specs if x.name != "writer"]
    if os.environ.get("S06_NO_ANALYTICS"):
        specs = [x for x in specs if x.name not in ("analyst", "runaway")]
    mon.mark("soak")
    sk.run_load(srv, out, specs, a.duration)

    # ---------------- post
    mon.mark("verify")
    part = sk.analyze(out, srv)
    wrong = {op: d["errors"] for op, d in part["ops"].items() if d["errors"] and not op.startswith("_")}
    wrong_msgs = [(m, n) for m, n in part["top_errors"] if m.startswith("WRONG")]
    checks.add("no wrong analytic / lookup results during soak", not wrong_msgs, str(wrong_msgs[:5]), "critical")
    other_err = [(m, n) for m, n in part["top_errors"] if not m.startswith("WRONG")]
    checks.add("no unexpected query errors during soak", not other_err, str(other_err[:5]), "major")

    def okc(op):
        d = part["ops"].get(op, {})
        return d.get("count", 0) - d.get("errors", 0)

    created, deleted = okc("write_create_draft"), okc("write_delete_draft")
    drafts = sk.scalar(q(r, "MATCH (d:Draft) RETURN count(d)")) or 0
    refs = sk.scalar(q(r, "MATCH (:Draft)-[e:REFS]->(:Paper) RETURN count(e)")) or 0
    checks.add("acknowledged draft writes all visible (created - deleted)", drafts == created - deleted,
               f"drafts={drafts} created={created} deleted={deleted}", "critical")
    checks.add("REFS edges == 3 x drafts", refs == 3 * drafts, f"refs={refs} drafts={drafts}", "critical")
    views = sk.scalar(q(r, "MATCH (p:Paper) RETURN sum(coalesce(p.views, 0))"))
    checks.add("acknowledged SET views all visible", views == okc("write_set_views"),
               f"views={views} acked={okc('write_set_views')}", "critical")
    # analytics still exact after churn
    for fn in (an_degree_dist, an_sum_weights, an_wcc, an_label_counts, an_order_full):
        try:
            fn(c0)
            checks.add(f"post-soak {fn.__name__} exact", True)
        except Wrong as e:
            checks.add(f"post-soak {fn.__name__} exact", False, str(e), "critical")
        except redis.exceptions.ResponseError as e:
            checks.add(f"post-soak {fn.__name__} exact", False, f"error {e}", "critical")
    checks.add("no server crash markers", not srv.crashed(), str(srv.crashed()[:3]), "critical")
    checks.add("server alive at end", srv.alive(), "", "critical")

    # noisy-neighbour: lookup latency under analytics vs quiet baseline
    mon.mark("baseline-quiet")
    lat = []
    c1 = _C()
    c1.r, c1.rng, c1.wid = r, random.Random(9), 0
    for _ in range(1500):
        t = time.time()
        op_paper_lookup(c1)
        lat.append((time.time() - t) * 1000)
    lat.sort()
    meta["quiet_lookup_p50_ms"] = lat[len(lat) // 2]
    meta["quiet_lookup_p99_ms"] = lat[int(len(lat) * 0.99)]

    mon.stop()
    s = sk.analyze(out, srv)
    # noisy-neighbour numbers: lookup throughput collapse seconds + latency under load vs quiet
    try:
        import csv
        per = {}
        miss = 0
        for row in csv.DictReader(open(out / "latency_ts.csv")):
            if row["op"].startswith("lookup_"):
                per[int(row["sec"])] = per.get(int(row["sec"]), 0) + int(row["ok"])
            elif row["op"] == "_missed_schedule":
                miss += int(row["err"])
        vals = sorted(per.values())
        med = vals[len(vals) // 2] if vals else 0
        meta["lookup_median_ops_per_s"] = med
        meta["lookup_stalled_seconds_(<50%_of_median)"] = sum(1 for v in vals if v < med * 0.5)
        meta["lookup_seconds_total"] = len(vals)
        meta["missed_schedule_total"] = miss
        lk = [d for o, d in s["ops"].items() if o.startswith("lookup_")]
        if lk:
            meta["lookup_p99_ms_worst_op"] = max(d["p99_ms"] for d in lk)
            meta["lookup_p50_ms_worst_op"] = max(d["p50_ms"] for d in lk)
    except Exception as e:  # noqa: BLE001
        meta["noisy_neighbour_error"] = repr(e)
    s["meta"] = meta
    (out / "summary.json").write_text(sk.json.dumps(s, indent=1, default=str))
    srv.stop()
    sk.print_summary(s)
    print("META", sk.json.dumps(meta, default=str))


def srv_rss_mb(srv):
    import psutil
    try:
        return psutil.Process(srv.pid).memory_info().rss / 2**20
    except Exception:
        return float("nan")


if __name__ == "__main__":
    main()
