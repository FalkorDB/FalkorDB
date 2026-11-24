"""s03_graphrag_vector — GraphRAG / knowledge graph for an LLM app.

Doc -[:HAS]-> Chunk(text, vecf32 embedding) -[:MENTIONS]-> Entity -[:RELATED]-> Entity.
Vector index (cosine) on Chunk.embedding, full-text on Chunk.text + Entity.name, range on
Entity.name/eid, Chunk.cid, Doc.id.  Workload: KNN + 1/2-hop context expansion, hybrid
full-text + graph, entity lookups, continuous doc ingestion, chunk deletes and updates.

Ground truth: embeddings are a pure function emb(cid, ver) (planted clusters + noise), so the
script can brute-force exact top-k with numpy and verify graph expansion exactly (mentions and
RELATED edges are deterministic functions of the ids).

  python -I s03_graphrag_vector.py --module M --port P --out DIR --duration S --scale F
"""
import argparse
import json
import math
import multiprocessing as mp
import os
import sys
import time
import random
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "lib"))
import numpy as np  # noqa: E402
import soakkit as sk  # noqa: E402

ap = argparse.ArgumentParser()
sk.add_engine_args(ap)  # --module PATH | --image REF
ap.add_argument("--port", type=int, required=True)
ap.add_argument("--out", required=True)
ap.add_argument("--duration", type=float, default=60)
ap.add_argument("--scale", type=float, default=0.05)
ap.add_argument("--rate-mult", type=float, default=5.0, help="multiplier on all op rates")
a = ap.parse_args()
out = Path(a.out)
out.mkdir(parents=True, exist_ok=True)

G = "graphrag"
SEED = 20260
DIM = 384
N = max(3000, int(250_000 * a.scale))          # initial chunks
E = max(1500, int(100_000 * a.scale))          # entities
C = max(20, int(200 * a.scale))                # planted clusters (also entity pools)
CH_PER_DOC = 8
NDOC = N // CH_PER_DOC
K = 10
SIGMA = 1.0                                    # noise norm relative to unit centroid (cos ~0.5 inside cluster)
NEW_BASE = 10_000_000                          # cids of ingested chunks: NEW_BASE + wid*1e6 + n
POOL = E // C
RM = a.rate_mult

# ------------------------------------------------------------------ deterministic data model
_crng = np.random.default_rng(SEED)
CENT = _crng.standard_normal((C, DIM)).astype(np.float64)
CENT /= np.linalg.norm(CENT, axis=1, keepdims=True)
_vr = random.Random(SEED)
VOCAB = ["".join(_vr.choice("abcdefghijklmnopqrstuvwxyz") for _ in range(_vr.randint(4, 9)))
         for _ in range(3000)]
VOCAB = [w for w in VOCAB if w not in ("with", "this", "that", "from", "have", "they", "will")]


def letters(n: int, w: int) -> str:
    s = []
    for _ in range(w):
        s.append(chr(97 + n % 26))
        n //= 26
    return "".join(reversed(s))


def emb(cid: int, ver: int = 0):
    """Rounded (6 dp) unit-ish vector as list of python floats and float32 ndarray."""
    g = np.random.default_rng([SEED, cid, ver])
    v = CENT[cid % C] + g.standard_normal(DIM) * (SIGMA / math.sqrt(DIM))
    v /= np.linalg.norm(v)
    v = np.round(v, 6)
    return v.tolist(), v.astype(np.float32)


def query_vec(i: int):
    """Query near a random cluster (an LLM question embedding)."""
    return emb(10**9 + i, 0)


def ments(cid: int) -> list[int]:
    c = cid % C
    s = {c + C * ((cid * 13 + j * 7919) % POOL) for j in range(3)}
    s.add((cid * 104729 + 17) % E)          # one cross-cluster mention
    return sorted(x for x in s if x < E)


def rel(eid: int) -> list[int]:
    s = {(eid + m * C) % E for m in (1, 2, 3)}
    s.add((eid * 31 + 7) % E)
    s.discard(eid)
    return sorted(s)


def ename(eid: int) -> str:
    return "ent" + letters(eid, 7) + "x"   # trailing x: immune to stemmer suffix rules


def marker(cid: int) -> str:
    return "mk" + letters(cid, 6) + "x"


def vmark(ver: int) -> str:
    return "vr" + letters(ver, 7) + "x"


def ctext(cid: int, ver: int = 0) -> str:
    r = random.Random(f"{SEED}-{cid}-{ver}")
    words = [r.choice(VOCAB) for _ in range(24)]
    words += [ename(e) for e in ments(cid)]
    words.append(marker(cid))
    if ver:
        words.append(vmark(ver))
    r.shuffle(words)
    return " ".join(words)


def chunk_row(cid: int, ver: int = 0, doc: int | None = None) -> dict:
    return {"cid": cid, "doc": cid // CH_PER_DOC if doc is None else doc, "ver": ver,
            "text": ctext(cid, ver), "emb": emb(cid, ver)[0], "ents": ments(cid)}


CHUNK_CYPHER = ("CREATE (d)-[:HAS]->(c:Chunk {cid: r.cid, doc: r.doc, ver: r.ver, text: r.text, "
                "embedding: vecf32(r.emb)}) WITH c, r UNWIND r.ents AS eid "
                "MATCH (e:Entity {eid: eid}) CREATE (c)-[:MENTIONS]->(e)")


# ------------------------------------------------------------------ reply parsing
def flat(cell):
    if isinstance(cell, list) and len(cell) == 2 and isinstance(cell[0], int):
        if cell[0] == 6:
            return [flat(x) for x in cell[1]]
        return sk.cval(cell)
    return cell


def frows(res):
    return [[flat(c) for c in row] for row in sk.rows(res)]


def appendline(path: Path, line: str):
    fd = os.open(path, os.O_WRONLY | os.O_APPEND | os.O_CREAT, 0o644)
    try:
        os.write(fd, (line + "\n").encode())
    finally:
        os.close(fd)


VIOL = out / "viol.jsonl"
LAGF = out / "lag.jsonl"


def viol(kind: str, **kw):
    appendline(VIOL, json.dumps({"kind": kind, **kw}))


# ------------------------------------------------------------------ server + load
srv = sk.Server(a.module, a.port, out)
mon = sk.Monitor(srv, out)
mon.start()
checks = sk.Checks(out)
r = srv.client()
is_rust = srv.engine() == "rust"
print(f"N={N} E={E} C={C} dim={DIM} module={a.module}", flush=True)


def wait_indices(timeout=900):
    t = time.time()
    while time.time() - t < timeout:
        n = sk.scalar(sk.query(r, G, "CALL db.indexes() YIELD status WHERE status <> 'OPERATIONAL' "
                                     "RETURN count(1)", ro=True))
        if n == 0:
            return True
        time.sleep(0.5)
    return False


mon.mark("ingest")
t_ing = time.time()
# entities
EB = 5000
for i in range(0, E, EB):
    sk.query(r, G, "UNWIND $rows AS x CREATE (:Entity {eid: x[0], name: x[1]})",
             {"rows": [[e, ename(e)] for e in range(i, min(E, i + EB))]})
for lbl, prop in (("Entity", "eid"), ("Entity", "name"), ("Chunk", "cid"), ("Doc", "id")):
    sk.query(r, G, f"CREATE INDEX FOR (n:{lbl}) ON (n.{prop})")
for i in range(0, E, EB):
    rows_ = [[e, f, (e + f) % 10] for e in range(i, min(E, i + EB)) for f in rel(e)]
    sk.query(r, G, "UNWIND $rows AS x MATCH (a:Entity {eid: x[0]}), (b:Entity {eid: x[1]}) "
                   "CREATE (a)-[:RELATED {w: x[2]}]->(b)", {"rows": rows_})
for i in range(0, NDOC, 5000):
    sk.query(r, G, "UNWIND range($a, $b) AS i CREATE (:Doc {id: i, title: 'doc' + toString(i)})",
             {"a": i, "b": min(NDOC, i + 5000) - 1})
# indexes before chunks so ingest includes live index maintenance
sk.query(r, G, "CREATE FULLTEXT INDEX FOR (n:Entity) ON (n.name)")
sk.query(r, G, "CREATE FULLTEXT INDEX FOR (n:Chunk) ON (n.text)")
sk.query(r, G, "CREATE VECTOR INDEX FOR (n:Chunk) ON (n.embedding) OPTIONS {dimension: %d, "
               "similarityFunction: 'cosine', M: 16, efConstruction: 200, efRuntime: 100}" % DIM)
print(f"entities+indexes done {time.time()-t_ing:.1f}s", flush=True)

LB = 250  # chunks per load batch


def load_range(args):
    lo, hi = args
    c = sk.connect(a.port)
    for i in range(lo, hi, LB):
        rows_ = [chunk_row(cid) for cid in range(i, min(hi, i + LB))]
        sk.query(c, G, "UNWIND $rows AS r MATCH (d:Doc {id: r.doc}) " + CHUNK_CYPHER, {"rows": rows_})
    return hi - lo


NPROC = 4
step = (N + NPROC * 8 - 1) // (NPROC * 8)
parts = [(i, min(N, i + step)) for i in range(0, N, step)]
with mp.get_context("fork").Pool(NPROC) as pool:
    done = 0
    for n in pool.imap_unordered(load_range, parts):
        done += n
        if done % (step * NPROC) < step:
            print(f"  loaded {done}/{N} {time.time()-t_ing:.0f}s", flush=True)
load_s = time.time() - t_ing
idx_ok = wait_indices()
ingest_s = time.time() - t_ing
print(f"ingest {load_s:.0f}s, indexes operational after {ingest_s:.0f}s ok={idx_ok}", flush=True)
checks.add("indexes become operational after load", idx_ok, f"{ingest_s:.0f}s", "major")
nodes_c = sk.scalar(sk.query(r, G, "MATCH (c:Chunk) RETURN count(c)", ro=True))
nodes_e = sk.scalar(sk.query(r, G, "MATCH (e:Entity) RETURN count(e)", ro=True))
rel_c = sk.scalar(sk.query(r, G, "MATCH ()-[x:RELATED]->() RETURN count(x)", ro=True))
men_c = sk.scalar(sk.query(r, G, "MATCH ()-[x:MENTIONS]->() RETURN count(x)", ro=True))
checks.add("ingest chunk count", nodes_c == N, f"{nodes_c} != {N}", "critical")
checks.add("ingest entity count", nodes_e == E, f"{nodes_e} != {E}", "critical")
checks.add("ingest RELATED edge count", rel_c == sum(len(rel(e)) for e in range(E)), f"{rel_c}", "critical")
checks.add("ingest MENTIONS edge count", men_c == sum(len(ments(c)) for c in range(N)), f"{men_c}",
           "critical")

# ------------------------------------------------------------------ ground truth + recall
t = time.time()
X0 = np.empty((N, DIM), dtype=np.float32)
for cid in range(N):
    X0[cid] = emb(cid, 0)[1]
cids0 = np.arange(N)
print(f"truth matrix {time.time()-t:.1f}s", flush=True)

NQ = 150


def knn_query(c, vec, k=K, ro=True):
    res = sk.query(c, G, "CALL db.idx.vector.queryNodes('Chunk', 'embedding', $k, vecf32($q)) "
                         "YIELD node, score RETURN node.cid, score", {"k": k, "q": vec}, ro=ro)
    return [(x[0], x[1]) for x in frows(res)]


def truth_topk(X, cids, qv, k=K):
    q = qv / np.linalg.norm(qv)
    d = 1.0 - X @ q
    idx = np.argpartition(d, k)[:k]
    idx = idx[np.argsort(d[idx])]
    return [int(cids[i]) for i in idx], d[idx]


def measure_recall(X, cids, tag):
    rec, nres, bad_sorted = [], 0, 0
    t = time.time()
    for i in range(NQ):
        lst, qv = query_vec(i)
        got = knn_query(r, lst)
        tk, _ = truth_topk(X, cids, qv)
        nres += len(got)
        if [s for _, s in got] != sorted(s for _, s in got):
            bad_sorted += 1
        rec.append(len(set(c for c, _ in got) & set(tk)) / K)
    mean = sum(rec) / len(rec)
    print(f"recall@{K} {tag}: {mean:.4f} (min {min(rec):.2f}) in {time.time()-t:.1f}s", flush=True)
    return mean, min(rec), nres, bad_sorted


rec_early, rec_early_min, nres, bs = measure_recall(X0, cids0, "early")
checks.add("recall@10 early >= 0.90", rec_early >= 0.90, f"{rec_early:.4f}", "major")
checks.add("vector KNN returns k results early", nres == NQ * K, f"{nres}", "major")
checks.add("vector KNN scores ascending early", bs == 0, f"{bs} unsorted", "major")

# self-query sanity + stored embedding integrity (sample)
rng0 = random.Random(5)
bad_self = bad_emb = 0
for cid in rng0.sample(range(N), 100):
    got = knn_query(r, emb(cid, 0)[0], k=K)
    if not any(c == cid for c, _ in got):
        bad_self += 1
        print(f"  self-query miss cid={cid} got={got}", flush=True)
    st = frows(sk.query(r, G, "MATCH (c:Chunk {cid: $c}) RETURN c.embedding, c.text, c.ver", {"c": cid},
                        ro=True))
    if not st or np.max(np.abs(np.array(st[0][0], dtype=np.float32) - emb(cid, 0)[1])) > 1e-5 \
            or st[0][1] != ctext(cid, 0):
        bad_emb += 1
checks.add("exact self-query finds the chunk in top-10 (ANN: >=97%)", bad_self <= 3, f"{bad_self}/100", "critical")
checks.add("stored embedding/text round-trip", bad_emb == 0, f"{bad_emb}/100", "critical")

# range-index equality lookup must find every chunk, including cid 0 (a known Rust bug when the
# label has range+fulltext+vector indexes, see repro5.py), cid N-1 and a middle one
bad_rng = [c for c in (0, 1, N // 2, N - 1)
           if sk.scalar(sk.query(r, G, "MATCH (c:Chunk {cid: $c}) RETURN count(c)", {"c": c}, ro=True)) != 1]
checks.add("range index finds chunks by cid (0, 1, mid, last)", not bad_rng, f"not found: {bad_rng}", "critical")
rng_total = sum(sk.scalar(sk.query(r, G, "UNWIND range($a, $b) AS i MATCH (c:Chunk {cid: i}) RETURN count(c)",
                                   {"a": lo, "b": min(N, lo + 2000) - 1}, ro=True)) for lo in range(0, N, 2000))
checks.add("range index finds every initial chunk", rng_total == N, f"{rng_total} of {N}", "critical")

# fulltext capability probe: entity with ~12 chunks, default result limit
probe_e = ments(5)[0]
ftn = len(frows(sk.query(r, G, "CALL db.idx.fulltext.queryNodes('Chunk', $t) YIELD node RETURN node.cid",
                         {"t": ename(probe_e)}, ro=True)))
exp_ft = sum(1 for c in range(N) if probe_e in ments(c))
print(f"fulltext probe: entity {probe_e} returned {ftn}, expected {exp_ft}", flush=True)
FT_LIMIT = ftn if ftn < exp_ft else None  # server truncates results (e.g. default 10)
checks.add("fulltext returns all matches (no silent default limit)", ftn == exp_ft,
           f"returned {ftn} expected {exp_ft}", "minor")


# ------------------------------------------------------------------ workload ops
def poll(fn, timeout=10.0):
    """Returns (found, lag_s)."""
    t = time.time()
    while True:
        if fn():
            return True, time.time() - t
        if time.time() - t > timeout:
            return False, time.time() - t
        time.sleep(0.05)


def ft_has(c, cid, token=None):
    res = frows(sk.query(c, G, "CALL db.idx.fulltext.queryNodes('Chunk', $t) YIELD node RETURN node.cid",
                         {"t": token or marker(cid)}, ro=True))
    return any(x[0] == cid for x in res)


def vec_top(c, vec, k=3):
    return knn_query(c, vec, k=k)


def retrieve_op(ctx):
    qi = ctx.rng.randrange(1_000_000)
    lst, qv = query_vec(qi)
    two = ctx.rng.random() < 0.5
    if not two:
        res = sk.query(ctx.r, G, f"CALL db.idx.vector.queryNodes('Chunk', 'embedding', {K}, vecf32($q)) "
                                 "YIELD node, score MATCH (d:Doc)-[:HAS]->(node) "
                                 "OPTIONAL MATCH (node)-[:MENTIONS]->(e:Entity) "
                                 "RETURN node.cid, score, d.id, node.doc, collect(DISTINCT e.eid) "
                                 "ORDER BY score", {"q": lst}, ro=True)
        name = "retrieve_1hop"
    else:
        res = sk.query(ctx.r, G, f"CALL db.idx.vector.queryNodes('Chunk', 'embedding', {K}, vecf32($q)) "
                                 "YIELD node, score MATCH (node)-[:MENTIONS]->(e:Entity) "
                                 "OPTIONAL MATCH (e)-[:RELATED]->(f:Entity) "
                                 "RETURN node.cid, score, collect(DISTINCT e.eid), collect(DISTINCT f.eid) "
                                 "ORDER BY score", {"q": lst}, ro=True)
        name = "retrieve_2hop_context"
    rs = frows(res)
    if len(rs) != K:
        viol("knn_count", got=len(rs), q=qi)
    prev = -1.0
    for row in rs:
        cid, score = row[0], row[1]
        if score < prev - 1e-9:
            viol("knn_unsorted", q=qi)
        prev = score
        if not two:
            if row[2] != row[3]:
                viol("doc_mismatch", cid=cid, doc=row[2], prop=row[3])
            ents = row[4]
            exp = ments(cid)
            if sorted(ents) != exp:
                viol("mentions_mismatch", cid=cid, got=sorted(ents)[:8], exp=exp)
        else:
            exp = ments(cid)
            ents, fs = row[2], row[3]
            if sorted(ents) != exp:
                viol("mentions_mismatch", cid=cid, got=sorted(ents)[:8], exp=exp)
            expf = sorted({f for e in exp for f in rel(e)})
            if sorted(fs) != expf:
                viol("related_mismatch", cid=cid, got=sorted(fs)[:8], exp=expf[:8])
        # distance sanity for chunks that were never updated (ver==0 unknown here: only check cid%10!=1)
        if cid < N and cid % 10 != 1 and cid % 10 != 0:
            d = 1.0 - float(np.dot(qv / np.linalg.norm(qv), emb(cid, 0)[1] / np.linalg.norm(emb(cid, 0)[1])))
            if abs(d - score) > 2e-3:
                viol("score_wrong", cid=cid, score=score, exp=d)
    return name


def hybrid_op(ctx):
    cid = ctx.rng.randrange(N)
    if cid % 10 == 0:
        cid += 2                                   # not in the deleted class
    es = ments(cid)
    pick = ctx.rng.sample(es, 2)
    terms = ename(pick[0]) + "|" + ename(pick[1])
    res = frows(sk.query(ctx.r, G, "CALL db.idx.fulltext.queryNodes('Chunk', $t) YIELD node, score "
                                   "MATCH (node)-[:MENTIONS]->(e:Entity) "
                                   "RETURN node.cid, score, node.text, collect(e.name) "
                                   "ORDER BY score DESC LIMIT 25", {"t": terms}, ro=True))
    names = {ename(pick[0]), ename(pick[1])}
    for row in res:
        if not (set(row[2].split()) & names):
            viol("hybrid_irrelevant", cid=row[0], terms=terms)
            break
    if FT_LIMIT is None and len(res) < 25 and not any(x[0] == cid for x in res):
        viol("hybrid_missing", cid=cid, terms=terms, n=len(res))
    return "hybrid_fulltext_graph"


def entity_op(ctx):
    eid = ctx.rng.randrange(E)
    x = ctx.rng.random()
    if x < 0.6:
        res = frows(sk.query(ctx.r, G, "MATCH (e:Entity {name: $n})-[:RELATED]->(f:Entity) RETURN f.eid",
                             {"n": ename(eid)}, ro=True))
        if sorted(v[0] for v in res) != rel(eid):
            viol("entity_related_mismatch", eid=eid, got=sorted(v[0] for v in res), exp=rel(eid))
        return "entity_lookup_related"
    if x < 0.85:
        res = frows(sk.query(ctx.r, G, "CALL db.idx.fulltext.queryNodes('Entity', $n) YIELD node "
                                       "RETURN node.eid", {"n": ename(eid)}, ro=True))
        if [v[0] for v in res] != [eid]:
            viol("entity_fulltext_mismatch", eid=eid, got=[v[0] for v in res])
        return "entity_fulltext"
    res = frows(sk.query(ctx.r, G, "MATCH (e:Entity {eid: $e})-[:RELATED*1..2]->(f:Entity) "
                                   "RETURN count(DISTINCT f)", {"e": eid}, ro=True))
    exp = len({g for f in rel(eid) for g in [f] + rel(f)} - {eid} | set(rel(eid)))
    # variable-length 1..2 excludes the start only if it is not reachable via a cycle
    exp2 = {g for f in rel(eid) for g in [f] + rel(f)}
    if res[0][0] not in (len(exp2), len(exp2 - {eid})):
        viol("entity_2hop_mismatch", eid=eid, got=res[0][0], exp=len(exp2))
    return "entity_2hop_count"


def ingest_init(ctx):
    ctx.state["n"] = 0
    ctx.state["recent"] = []


def ingest_op(ctx):
    st = ctx.state
    if st["n"] % 6 == 5 and len(st["recent"]) > 20:        # retire an old ingested doc
        lo, hi = st["recent"].pop(0)
        sk.query(ctx.r, G, "MATCH (c:Chunk) WHERE c.cid >= $lo AND c.cid < $hi DETACH DELETE c",
                 {"lo": lo, "hi": hi})
        for cid in range(lo, hi):
            appendline(out / "acked_del.txt", str(cid))
        st["n"] += 1
        # one of them must be gone from fulltext + vector
        cid = lo + 1
        ok, lag = poll(lambda: not ft_has(ctx.r, cid), 10)
        if not ok:
            viol("deleted_in_fulltext", cid=cid)
        ok, lag = poll(lambda: all(c != cid for c, _ in vec_top(ctx.r, emb(cid, 0)[0])), 10)
        if not ok:
            viol("deleted_in_vector", cid=cid)
        return "retire_doc"
    base = NEW_BASE + ctx.wid * 1_000_000 + st["n"] * CH_PER_DOC
    docid = 50_000_000 + ctx.wid * 1_000_000 + st["n"]
    st["n"] += 1
    rows_ = [chunk_row(base + i, 0, docid) for i in range(CH_PER_DOC)]
    sk.query(ctx.r, G, "CREATE (d:Doc {id: $did, title: 'new'}) WITH d UNWIND $rows AS r " + CHUNK_CYPHER,
             {"did": docid, "rows": rows_})
    for x in rows_:
        appendline(out / "acked_ins.txt", str(x["cid"]))
    st["recent"].append((base, base + CH_PER_DOC))
    cid = base + 3
    t0 = time.time()
    ok1, lag1 = poll(lambda: ft_has(ctx.r, cid), 10)
    ok2, lag2 = poll(lambda: any(c == cid for c, _ in vec_top(ctx.r, emb(cid, 0)[0], 10)), 3)
    appendline(LAGF, json.dumps({"kind": "ingest_ft", "lag": lag1, "ok": ok1}))
    appendline(LAGF, json.dumps({"kind": "ingest_vec", "lag": lag2, "ok": ok2}))
    if not ok1:
        viol("ingested_not_in_fulltext", cid=cid)
    if not ok2:
        deep = any(c == cid for c, _ in vec_top(ctx.r, emb(cid, 0)[0], 200))
        viol("ingested_ann_miss_top10" if deep else "ingested_not_in_vector", cid=cid)
    return "ingest_doc_8chunks"


def delete_init(ctx):
    ctx.state["k"] = ctx.wid * 131


def delete_op(ctx):
    st = ctx.state
    m = max(1, N // 10)
    k = (st["k"] * 7919) % m
    st["k"] += 1
    if k == 0:
        k = m // 2 + 1
    cid = 10 * k                                          # class cid%10==0 is the deletable class (never cid 0)
    res = sk.query(ctx.r, G, "MATCH (c:Chunk {cid: $c}) DETACH DELETE c", {"c": cid})
    appendline(out / "acked_del.txt", str(cid))
    if ctx.rng.random() < 0.5:
        ok, _ = poll(lambda: not ft_has(ctx.r, cid), 10)
        if not ok:
            viol("deleted_in_fulltext", cid=cid)
        ok, _ = poll(lambda: all(c != cid for c, _ in vec_top(ctx.r, emb(cid, 0)[0], 5)), 10)
        if not ok:
            viol("deleted_in_vector", cid=cid)
    return "delete_chunk"


def update_init(ctx):
    ctx.state["uk"] = ctx.wid * 977 + 3
    ctx.state["ver"] = {}


def update_op(ctx):
    st = ctx.state
    m = max(1, N // 10)
    k = (st["uk"] * 6007) % m
    st["uk"] += 1
    cid = 10 * k + 1                                      # class cid%10==1 is the updatable class
    old = st["ver"].get(cid, 0)
    ver = ctx.rng.randrange(1, 26 ** 7)
    lst, nv = emb(cid, ver)
    sk.query(ctx.r, G, "MATCH (c:Chunk {cid: $c}) SET c.embedding = vecf32($e), c.ver = $v, c.text = $t",
             {"c": cid, "e": lst, "v": ver, "t": ctext(cid, ver)})
    st["ver"][cid] = ver
    if ctx.rng.random() < 0.5:
        ok, lag = poll(lambda: any(c == cid for c, _ in vec_top(ctx.r, lst, 10)), 3)
        appendline(LAGF, json.dumps({"kind": "update_vec", "lag": lag, "ok": ok}))
        if not ok:
            deep = any(c == cid for c, _ in vec_top(ctx.r, lst, 200))
            viol("updated_ann_miss_top10" if deep else "updated_not_in_vector", cid=cid)
        ok, lag = poll(lambda: ft_has(ctx.r, cid, vmark(ver)), 10)
        appendline(LAGF, json.dumps({"kind": "update_ft", "lag": lag, "ok": ok}))
        if not ok:
            viol("updated_not_in_fulltext", cid=cid)
        if old != ver:
            g = vec_top(ctx.r, emb(cid, old)[0], 3)
            if any(c == cid and s < 1e-3 for c, s in g):
                viol("update_stale_vector", cid=cid)
    return "update_chunk_embedding"


mon.mark("soak")
def mutate_init(ctx):
    delete_init(ctx)
    update_init(ctx)


def mutate_op(ctx):
    return delete_op(ctx) if ctx.rng.random() < 0.5 else update_op(ctx)


# 6 worker processes: 2 retrieval, 1 hybrid, 1 entity, 1 ingest, 1 delete/update
specs = [sk.WorkerSpec(retrieve_op, rate=10 * RM, count=2, name="retrieve"),
         sk.WorkerSpec(hybrid_op, rate=10 * RM, count=1, name="hybrid"),
         sk.WorkerSpec(entity_op, rate=40 * RM, count=1, name="entity"),
         sk.WorkerSpec(ingest_op, rate=1.5 * RM, count=1, name="ingest", init=ingest_init),
         sk.WorkerSpec(mutate_op, rate=6 * RM, count=1, name="mutate", init=mutate_init)]
t_load = time.time()
sk.run_load(srv, out, specs, a.duration)
print(f"load done {time.time()-t_load:.0f}s alive={srv.alive()}", flush=True)

# ------------------------------------------------------------------ post checks
mon.mark("verify")
if not srv.alive():
    checks.add("server alive after load", False, "", "critical")
else:
    r = srv.client()
    wait_indices(300)
    # violations from workers
    vc: dict[str, list] = {}
    if VIOL.exists():
        for line in VIOL.read_text().splitlines():
            d = json.loads(line)
            vc.setdefault(d["kind"], []).append(d)
    sev = {"deleted_in_fulltext": "critical", "deleted_in_vector": "critical",
           "ingested_not_in_fulltext": "critical", "ingested_not_in_vector": "critical",
           "updated_not_in_vector": "critical", "updated_not_in_fulltext": "critical",
           "update_stale_vector": "critical", "mentions_mismatch": "critical",
           "related_mismatch": "critical", "entity_related_mismatch": "critical",
           "entity_fulltext_mismatch": "critical", "entity_2hop_mismatch": "critical",
           "hybrid_irrelevant": "critical", "hybrid_missing": "major", "ingested_ann_miss_top10": "minor", "updated_ann_miss_top10": "minor", "score_wrong": "major",
           "knn_count": "major", "knn_unsorted": "major", "doc_mismatch": "critical"}
    for kind in sorted(set(sev) | set(vc)):
        l = vc.get(kind, [])
        checks.add(f"in-load: {kind} == 0", not l, f"{len(l)} e.g. {l[:2]}", sev.get(kind, "major"))
    # search lag
    lags: dict[str, list] = {}
    if LAGF.exists():
        for line in LAGF.read_text().splitlines():
            d = json.loads(line)
            lags.setdefault(d["kind"], []).append(d["lag"])
    lagstat = {k: {"n": len(v), "p50_s": sorted(v)[len(v) // 2], "max_s": max(v),
                   "pct_visible_immediately": sum(1 for x in v if x < 0.06) / len(v)}
               for k, v in lags.items() if v}
    print("search-lag:", json.dumps(lagstat), flush=True)

    def readset(path):
        return [int(x) for x in path.read_text().split()] if path.exists() else []
    ins, dele = readset(out / "acked_ins.txt"), set(readset(out / "acked_del.txt"))
    # live set from DB
    live = {}
    for lo in range(0, N, 20000):
        for cid, ver in frows(sk.query(r, G, "MATCH (c:Chunk) WHERE c.cid + 0 >= $a AND c.cid + 0 < $b "
                                             "RETURN c.cid, c.ver", {"a": lo, "b": min(N, lo + 20000)}, ro=True)):
            live[cid] = ver
    for cid, ver in frows(sk.query(r, G, "MATCH (c:Chunk) WHERE c.cid + 0 >= $a RETURN c.cid, c.ver",
                                   {"a": NEW_BASE}, ro=True)):
        live[cid] = ver
    exp_live = (set(range(N)) | set(ins)) - dele
    miss, extra = exp_live - set(live), set(live) - exp_live
    n_errs_mut = 0
    ssum = sk.analyze(out, srv)
    for op in ("ingest_doc_8chunks", "retire_doc", "delete_chunk"):
        n_errs_mut += ssum["ops"].get(op, {}).get("errors", 0)
    n_errs_mut += sum(d["errors"] for k, d in ssum["ops"].items()
                      if k in ("ingest", "mutate", "update_chunk_embedding"))
    checks.add("live chunk set == initial + acked inserts - acked deletes",
               not miss and not extra, f"missing={len(miss)} extra={len(extra)} (mutation errors={n_errs_mut})"
               f" e.g. {sorted(miss)[:3]} {sorted(extra)[:3]}", "critical" if n_errs_mut == 0 else "major")
    # deleted never come back
    rs = random.Random(7)
    sample = rs.sample(sorted(dele), min(300, len(dele))) if dele else []
    back_ft = back_vec = 0
    for cid in sample:
        if cid in live:
            continue
        v = 0 if cid < N or cid >= NEW_BASE else 0
        if ft_has(r, cid):
            back_ft += 1
        if any(c == cid for c, _ in knn_query(r, emb(cid, 0)[0], k=10)):
            back_vec += 1
    checks.add("deleted chunks absent from fulltext", back_ft == 0, f"{back_ft}/{len(sample)}", "critical")
    checks.add("deleted chunks absent from vector search", back_vec == 0, f"{back_vec}/{len(sample)}",
               "critical")
    checks.add("delete workload actually deleted chunks", len(dele) > 0, f"{len(dele)}", "minor")
    # new chunks searchable
    ins_s = rs.sample(ins, min(200, len(ins))) if ins else []
    nf = nv = 0
    for cid in ins_s:
        if cid not in live:
            continue
        if not ft_has(r, cid):
            nf += 1
        if not any(c == cid for c, _ in knn_query(r, emb(cid, 0)[0], k=10)):
            nv += 1
    checks.add("ingested chunks searchable via fulltext", nf == 0, f"{nf}/{len(ins_s)}", "critical")
    checks.add("ingested chunks searchable via vector (top-10, ANN: <=2%)", nv <= max(1, len(ins_s) // 50), f"{nv}/{len(ins_s)}", "critical")
    # late recall against exact brute force over the live set
    keep = [c for c in range(N) if c in live]
    X = np.empty((len(live), DIM), dtype=np.float32)
    cl = sorted(live)
    for i, cid in enumerate(cl):
        X[i] = X0[cid] if (cid < N and live[cid] == 0) else emb(cid, live[cid])[1]
    del X0
    cl = np.array(cl)
    rec_late, rec_late_min, nres, bs = measure_recall(X, cl, "late")
    checks.add("recall@10 late >= 0.90", rec_late >= 0.90, f"{rec_late:.4f}", "major")
    checks.add("recall late within 0.05 of early", rec_late >= rec_early - 0.05,
               f"early={rec_early:.4f} late={rec_late:.4f}", "major")
    checks.add("vector KNN returns k results late", nres == NQ * K, f"{nres}", "major")
    checks.add("vector KNN scores ascending late", bs == 0, f"{bs} unsorted", "major")
    # updated chunks carry the DB version's embedding
    upd = [c for c in live if live[c] != 0][:100]
    bad = 0
    for cid in upd:
        st = frows(sk.query(r, G, "MATCH (c:Chunk {cid: $c}) RETURN c.embedding, c.text", {"c": cid}, ro=True))
        if not st or np.max(np.abs(np.array(st[0][0], dtype=np.float32) - emb(cid, live[cid])[1])) > 1e-5 \
                or st[0][1] != ctext(cid, live[cid]):
            bad += 1
    checks.add("updated chunks hold last-written embedding/text", bad == 0, f"{bad}/{len(upd)}", "critical")
    # RDB round trip: recall preserved
    rst = None
    if len(live) <= 200_000 or a.scale < 0.5:
        try:
            rst = srv.restart(save=True)
            r = srv.client()
            ok = wait_indices(600)
            rec_rst = measure_recall(X, cl, "after-restart")[0]
            checks.add("restart: indexes operational", ok, f"restart {rst:.1f}s", "major")
            checks.add("restart: recall preserved", rec_rst >= rec_late - 0.05, f"{rec_rst:.4f} vs {rec_late:.4f}",
                       "major")
            cnt = sk.scalar(sk.query(r, G, "MATCH (c:Chunk) RETURN count(c)", ro=True))
            checks.add("restart: chunk count preserved", cnt == len(live), f"{cnt} vs {len(live)}", "critical")
        except Exception as e:
            checks.add("restart round-trip", False, f"{type(e).__name__}: {e}", "critical")
    checks.add("server alive at end", srv.alive(), "", "critical")

mon.stop()
s = sk.analyze(out, srv)
s["meta"] = {"scenario": "s03_graphrag_vector", "scale": a.scale, "chunks": N, "entities": E,
             "clusters": C, "dim": DIM, "docs": NDOC, "ingest_s": ingest_s, "load_s": load_s,
             "duration": a.duration, "recall_early": locals().get("rec_early"),
             "recall_late": locals().get("rec_late"), "lag": locals().get("lagstat"),
             "restart_s": locals().get("rst"), "live_chunks_end": len(locals().get("live", {})),
             "fulltext_default_limit_hit": FT_LIMIT, "module": os.path.basename(str(a.module))}
(out / "summary.json").write_text(json.dumps(s, indent=1, default=str))
srv.stop()
sk.print_summary(s)
