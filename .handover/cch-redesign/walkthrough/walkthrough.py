#!/usr/bin/env python3
"""Toy-scale executable model of the CCH redesign.

Spec: ../design/03-design.md (revision 3.1), amended by the "Change" list in
../prototypes/00-results.md, with the matching described in
../prototypes/m1-matching-quality.md.

It runs every step of the design on one small running example and writes
example.json next to this file. The walkthrough page renders only from that
JSON, so every array, matrix, rank, id, log entry and weight a reader sees is
computed here.

This is not FalkorDB code. Nothing in it is measured; it models the design's
rules at toy scale so the walkthrough's numbers are real outputs of those rules.

Run:  python3 -I walkthrough.py
"""

import heapq
import json
import os
from bisect import bisect_right
from collections import defaultdict

HERE = os.path.dirname(os.path.abspath(__file__))
OUT = os.path.join(HERE, "example.json")
INF = float("inf")
M32 = 0xFFFFFFFF
TYPES = ("ROAD", "FERRY")

# ======================================================================================
# Toy thresholds. THRESHOLDS (below) lists each with the real value and its source.
# ======================================================================================
NODE_CAP = 20
RESERVE = 2
SLABS = ((0, 10), (10, 20))
COARSEN_TO = 4          # stop a cell at <= 4 vertices; also CoarsenTo in the weight cap
MIN_SHRINK = 0.05       # stop a cell when a level shrinks it by less than 5 %
LEAF_MAX = 5            # cells of <= 5 vertices go to exact minimum degree
ROUND_CAP = 8           # matching rounds per level
TWO_HOP_BELOW = 0.15    # 2-hop pass when matching shrinks a cell by less than 15 %
CAP_FACTOR = 1.5        # vwgt[u] + vwgt[v] <= 1.5 * tvwgt / CoarsenTo
SEP_TRIALS = 7          # greedy-growing trials for the first separator
FM_UB = 1.2             # node FM balance: max side weight <= 1.2 * cell weight / 2
FM_MAX_BAD = 3          # node FM: stop a pass after this many moves without improvement
ARC_GUARD = 8           # max arcs per skeleton entry
SEP_GUARD = 4096        # top separator size that fails the build (not scaled)
N_MAX = 4096            # S2 new arcs per lock hold (not scaled)
B_GIL = 250             # work units per lock hold (toy; real: 25 ms)
VALVE_DIV = 8           # off the lock: a front over n/8 arcs takes the full path
FOLD_FLOOR = 0          # fold when delta entries > max(FOLD_FLOOR, sqrt(|base|))
CHUNK = 16              # metric slots per chunk
SALT_BASE = 1           # matching salt for (depth, level) = fmix32(SALT_BASE + 64·depth + level)
SALT_SCAN = 600         # B5.h re-runs the ordering under salt bases 0..599 to see if the 2-hop pass fires
LEAF_CAP = 4            # cow_btree entries per leaf page
BRANCH_CAP = 4          # cow_btree children per branch page

THRESHOLDS = [
    {"name": "node_cap", "toy": NODE_CAP, "real": "1.54M on California",
     "source": "03 §8 [U1 §1]"},
    {"name": "reserve (pool ranks)", "toy": RESERVE, "real": "max(1,024, node_cap/64)",
     "source": "03 §2"},
    {"name": "skeleton slab", "toy": "10 node ids per slab, 2 slabs", "real": "about 2M edges per slab",
     "source": "03 §5 Phases"},
    {"name": "coarsening stop: cell size", "toy": COARSEN_TO, "real": 100, "source": "03 §3 step 3"},
    {"name": "coarsening stop: shrink", "toy": "less than 5 %", "real": "less than 5 %",
     "source": "03 §3 step 3 (ometis.c:399-403)"},
    {"name": "leaf size for exact minimum degree", "toy": LEAF_MAX, "real": 120,
     "source": "03 §3 step 7 (MMDSWITCH)"},
    {"name": "matching round cap", "toy": ROUND_CAP, "real": 8, "source": "00-results Change 3"},
    {"name": "2-hop pass trigger", "toy": "matching shrinks a cell by less than 15 %",
     "real": "a level shrinks by less than 15 %", "source": "03 §3 step 3; m1-matching-quality"},
    {"name": "weight cap", "toy": "1.5 * tvwgt / CoarsenTo, CoarsenTo = 4",
     "real": "1.5 * tvwgt / CoarsenTo, CoarsenTo = the coarsening floor",
     "source": "00-results Change 1 (METIS coarsen.c:38)"},
    {"name": "first-separator trials", "toy": "up to 7 (one per coarse vertex here)", "real": 7,
     "source": "03 §3 step 4 (initpart.c:82-89)"},
    {"name": "node FM balance", "toy": FM_UB, "real": "not fixed by the design",
     "source": "toy choice"},
    {"name": "node FM pass stop", "toy": f"{FM_MAX_BAD} moves without improvement",
     "real": "not fixed by the design", "source": "toy choice"},
    {"name": "arc guard", "toy": f"{ARC_GUARD} arcs per skeleton entry", "real": "8 arcs per skeleton entry",
     "source": "03 §3 Guards (b)"},
    {"name": "separator guard", "toy": SEP_GUARD, "real": 4096, "source": "03 §3 Guards (a); not scaled"},
    {"name": "B_gil, the lock budget", "toy": f"{B_GIL} work units per lock hold",
     "real": "25 ms per lock hold, counted in work units plus the clock",
     "source": "03 §6b, decision D1; toy value chosen so that one later commit (v14) hands off"},
    {"name": "N_max, S2 new arcs per hold", "toy": N_MAX, "real": 4096,
     "source": "03 §6; not scaled, never reached here"},
    {"name": "scoped-recompute valve (off the lock only)", "toy": "front over n/8 arcs",
     "real": "front over n/8 arcs", "source": "03 §6b (partial.rs:131)"},
    {"name": "fold trigger", "toy": "Up.delta entries > sqrt(|base|), checked at every writer flush",
     "real": "delta entries > max(64k, sqrt(|base|))",
     "source": "03 §6 Decay, Built-entry jobs; the 64k floor is dropped at toy scale"},
    {"name": "S6 trigger (deletions)", "toy": "deletions ≥ n/8, n = the rank dimension; evaluated, not acted on",
     "real": "deletions ≥ max(1,024, n/8), or dead ids ≥ 12.5 %",
     "source": "03 §6 Decay; the 1,024 floor is dropped at toy scale, as the fold's 64k is; 'dead ids' is "
               "not defined in 03 and is not evaluated"},
    {"name": "S7 triggers", "toy": "the same four signals; evaluated, not acted on",
     "real": "arcs ≥ 1.25 × arcs at build; one commit adding > arcs/8; top ranks ≥ reserve; "
             "pool-row fill > arcs/16",
     "source": "03 §6 Decay. The toy reads 'arcs' in 'one commit adding > arcs/8' as the count before the "
               "commit, and 'pool-row fill' as LineageStats' 'arcs added by S2 at pool-ranked rows' (03 §2)"},
    {"name": "metric chunk", "toy": f"{CHUNK} slots", "real": "2,048 slots (CHUNK_BITS = 11)",
     "source": "customize.rs:126-127 @08c"},
    {"name": "cow_btree page", "toy": f"{LEAF_CAP} entries per leaf, {BRANCH_CAP} children per branch",
     "real": "256 entries per leaf, 256 children per branch", "source": "cow_btree/mod.rs:140-141 @main"},
]

# ======================================================================================
# Design figures and rules the page quotes. Real-scale numbers are 03's, each with its section; the
# page renders them from here so that no number is typed into the page.
# ======================================================================================
DESIGN_REFS = {
    "bytes": {
        "Up.base": "8 B per arc; 4 B per rank", "Down.base": "8 B per arc; 4 B per rank",
        "Up.delta": "≤ 16 B per delta entry", "Down.delta": "≤ 16 B per delta entry",
        "rank_of": "4 B per node", "node_of": "4 B per rank", "free_ranks": "small",
        "LineageStats": "O(1)", "added_since_fold": "16 B per delta entry, bounded by the fold trigger",
        "metric": "32 B per arc for the four arrays", "log": "16 B per retained pair + ≈40 B per node",
        "entry": "upkeep: O(1)",
        "source": "03 §2 table, Bytes column",
    },
    "against_c": {"text": "Per arc: 8 + 8 + 32 = 48 B, or 44 with an iso Down (decision D7). Per node: 4 (rank_of) + 4 "
                          "(node_of) + 8 (two row-pointer arrays) = 16 B. C: 48 B per arc and 64 B per node plus "
                          "row headers.",
                  "source": "03 §2 Against C [U1 §1]"},
    "real_scale": [
        {"what": "Dijkstra fallback while an entry is Building",
         "value": "222 ms p50 and 409 ms p95 on California", "source": "03 §5 Readers while Building "
         "(fallback.rs:28-31 [U2])"},
        {"what": "convergence limit", "value": "above roughly 1,250 SET/s, a Repair never adopts within budget",
         "source": "03 §1 What it gives up, §5 Convergence (derived)"},
        {"what": "one write", "value": "a single SET on a top arc can trigger a hand-off",
         "source": "03 §1 What it gives up, §6b Where it runs, and the bound"},
        {"what": "S7 fallback window", "value": "about one batched build, ≈ 4.1 s on California",
         "source": "03 §8 (derived)"},
    ],
    "bound": {"text": "If a copy's unconsumed pairs pass max(1M pairs, arcs/8), the writer replaces that copy's "
                      "whole unconsumed list with one Overflow{seq = its version} node, and resets pairs to 0. "
                      "An Overflow node is relinked, consumed and freed exactly like a segment. The phase that "
                      "consumes an Overflow rescans.",
              "built": "An Overflow is never created on a Built entry. Past the §5 bound, the writer clears upkeep "
                       "and the log instead of writing an Overflow: the job is abandoned, and a later commit "
                       "re-triggers it.",
              "source": "03 §5 Bound and Overflow, §6 Built-entry jobs (Retention)"},
    "rebase": {
        "rows": [
            ["Building", "Building (same generation)",
             "Relink the fork's own nodes (seq = fork.version, above every published seq) onto the published log. "
             "An own Overflow is relinked like a segment."],
            ["Building", "Built (the Seal or Repair adopted)",
             "Run the §6 flush over the own nodes, resolved against the fork graph, onto the published hierarchy, "
             "within the remaining B_gil. Over budget, or an own Overflow: Absorbed::Restart, which takes "
             "next_generation(), with Building{Repair, base = the published hierarchy, log = the own nodes}."],
            ["Built", "Built, upkeep job published (decision D6)",
             "Fold: re-apply the fold to the fork's copy. S6: replay the own nodes through §6 onto the new lineage, "
             "within B_gil; over budget, Restart as above."],
            ["Built", "Building, same generation",
             "Unreachable: a Built entry becomes Building only through a writer's commit (a hand-off or S7, both "
             "new generations), which makes this fork stale."],
        ],
        "note": "A writer always keeps its own nodes in its private version until it publishes, whatever the entry "
                "state, so every row above has them.",
        "source": "03 §5 Writer rebase, by published state"},
    "hooks": {"items": [
        "writer_rebase receives the fork's entry, and indexer_catch_up reads the published entry; neither gates on "
        "Written.",
        "overtaken also fires for a Built entry whose indexer_commits moved (decision D6).",
        "indexes_to_build_since also enqueues a Built entry with a new upkeep ticket (decision D6).",
        "a Built-entry job publish is fitted against generation and ticket (decision D6)."],
        "why": "The log is in CchIndex's own per-version state. CCH contributes no ids to the framework's Written, so "
               "since_last_indexer_step stays empty for CCH, and indexer_apply's clear never touches it.",
        "source": "03 §7 Framework changes, §5 The log"},
    "fold": {
        "overtaken": "At its publish the fold clears upkeep and the log, and counts the publish in the entry's "
                     "indexer_commits, so writers forked before it are overtaken. A writer that overtook a fold applies the same function to its "
                     "own copy: its bases become the folded ones, and its deltas become the arcs of its own "
                     "added_since_fold with seq > F. No segment is replayed, because no value changes.",
        "retention": "Without retention, a fold needs no log: metric-only writes after F stay in the published "
                     "metric, which the fold keeps. Retention serves S6, which replays the retained nodes with "
                     "seq > F onto the new lineage.",
        "source": "03 §6 Built-entry jobs (decision D6), Fold job"},
    "ranks": {
        "items": [
            "Ranks are lazy: isolated nodes and unused ids get no rank. A rank-less node has no indexed edge at that "
            "version (I1′).",
            "A rank-less node with a live edge takes the lowest free pool rank if its degree is ≤ 1 and free_ranks is "
            "not empty; otherwise it takes the next top rank n_r′ + 1. After the build that rank lives only in the "
            "deltas; at Supergraph it raises the base's dimension, since nothing lives in a delta yet (03 §4 step 1).",
            "Allocation is a phase of its own, run before classification by every resolver: writer flush, "
            "Supergraph, Seal, Repair, the Overflow paths and S6.",
            "A pool rank never has a holder (P: r ∈ free_ranks ⇔ node_of[r] = None), so taking one never strips a "
            "node. Within a lineage a rank never goes back to the pool: deleted nodes keep their rank; only S6 and "
            "S7 rebuild the pool."],
        "source": "03 §6 Rank allocation, §2 Contract changes (I1′, P)"},
    "contracts": {
        "items": [
            ["I1′", "rank_of is partial and injective. Readers: the query's endpoint lookup, which answers “no path” "
                    "unless source = target; and partial.rs:115-118, which must check for the rank's presence as "
                    "well as the bound."],
            ["I3′", "up_ids(r) yields ascending (hi, id) pairs from merging base and delta; arrays are sized by "
                    "arc_capacity. Readers: relax's x_slots_start + i (customize.rs:497-513), query.rs:113, "
                    "partial.rs:151,170 (the heap carries (lo, hi, id)), Cch::joins, and the full-seed pass and the "
                    "fold, which write by id."],
        ],
        "why_delta": "Why a base plus a delta, not VersionedMatrix: VersionedMatrix decides folds itself (its next "
                     "flush runs an O(|m|) eWiseAdd on the writer, under the lock), its set exists only for bool, and "
                     "a GraphBLAS delta read while it is written pays pending-tuple assembly. A cow_btree insert "
                     "costs O(log n), has no pending tuples, and can hold ranks above the base's dimension, so the "
                     "base is never resized.",
        "why_arrays": "Why the base exposes raw arrays: relax opens up(y) once per arc and steps once per triangle, "
                      "and a re-seeked GraphBLAS row iterator costs ≈298 instructions per seek (≈1.7G in seeks alone "
                      "on California, derived). So queries and customization read the base as flat slices, and "
                      "GraphBLAS reads the same memory through a read-only load for its algebra (decision D3).",
        "source": "03 §2 Contract changes; Why base plus delta; Why the base exposes raw arrays"},
    "replicas": {"text": "With non-integer weights, differing hierarchies can differ in the last bit",
                 "source": "03 §7 Determinism, derived"},
    "d5": {"text": "reusing depth 0's levels is one of the options for decision D5, how the ordering meets its time "
                   "bar", "source": "prototypes/00-results.md, decision D5 option b; prototypes/m1-optimize.md",
           "verdict": "Before the optimization pass, the prototype's verdict on that option was that reuse alone does not "
                      "meet the time bar at one builder thread, and that the cap-stop coarsening helps only as a cost cut "
                      "alongside more threads. The team has not decided D5",
           "verdict_source": "prototypes/d5-b-verdict.md",
           "after": "After the optimization pass, the cap-stop coarsening (GraphBLAS calls plus Rust passes over their "
                    "arrays) takes about a third of feral's whole ordering on one thread on SNAP's roadNet-CA (0.33×; "
                    "about 0.36× under the quality harness's floor, per the corrections note). The host steps "
                    "(separators, node FM, minimum degree, the tree) are not measured yet, so whether one thread is "
                    "enough is still open",
           "after_source": "prototypes/m1-optimize.md, read with its corrections note"},
}


WORK_UNITS = ("Work units, as 03 §6b counts them: one per apex-merge step, one per heap push, "
              "one per seed lookup (each direction), plus S2's fixed costs: one per row entry the "
              "overlay reads and one per delta insert (Up and Down each).")


# ======================================================================================
# Small utilities
# ======================================================================================
def fmix32(h):
    """MurmurHash3's 32-bit finalizer: a bijection on 32-bit words (the 'scramble')."""
    h &= M32
    h ^= h >> 16
    h = (h * 0x85EBCA6B) & M32
    h ^= h >> 13
    h = (h * 0xC2B2AE35) & M32
    h ^= h >> 16
    return h


_INV_A = pow(0x85EBCA6B, -1, 1 << 32)
_INV_B = pow(0xC2B2AE35, -1, 1 << 32)


def unfmix32(h):
    h &= M32
    h ^= h >> 16
    h = (h * _INV_B) & M32
    h ^= (h >> 13) ^ (h >> 26)
    h = (h * _INV_A) & M32
    h ^= h >> 16
    return h


def jw(x):
    """A weight for JSON: +inf is null."""
    return None if x == INF else x


def pr(u, v):
    return (u, v) if u < v else (v, u)


def ent(d):
    return [[i, j, v] for (i, j), v in sorted(d.items())]


def pat(s):
    return [[i, j] for (i, j) in sorted(s)]


def fmt_pairs(ps):
    return ", ".join(f"{a}–{b}" for a, b in ps)


def plural(n, word, many=None):
    return f"{n} {word if n == 1 else (many or word + 's')}"


def and_list(xs):
    xs = [str(x) for x in xs]
    if len(xs) <= 1:
        return "".join(xs)
    return ", ".join(xs[:-1]) + " and " + xs[-1]


# ======================================================================================
# The graph: nodes, relationships, and per type the tensor layers m / dp / dm
# (tensor.rs: "dp wins, else m unless its dm bit is set"). No pair has two edges of one
# type here, so the multi-edge layer me stays empty.
# ======================================================================================
class Graph:
    def __init__(self):
        self.nodes = set()
        self.rels = {}      # live relationship id -> (type, src, dst, weight)
        self.gone = {}      # deleted relationship id -> (type, src, dst, weight)
        self.layers = {t: {"m": {}, "dp": {}, "dm": set()} for t in TYPES}
        self.next_rel = 0
        self.next_node = 0
        self.free_ids = []
        self._w = None

    def clone(self):
        g = Graph()
        g.nodes = set(self.nodes)
        g.rels = dict(self.rels)
        g.gone = dict(self.gone)
        g.layers = {t: {"m": dict(L["m"]), "dp": dict(L["dp"]), "dm": set(L["dm"])}
                    for t, L in self.layers.items()}
        g.next_rel, g.next_node, g.free_ids = self.next_rel, self.next_node, list(self.free_ids)
        return g

    def create_node(self):
        if self.free_ids:
            nid = min(self.free_ids)       # a freed id is handed out again, lowest first
            self.free_ids.remove(nid)
        else:
            nid = self.next_node
            self.next_node += 1
        self.nodes.add(nid)
        self._w = None
        return nid

    def add_rel(self, t, s, d, w):
        e = self.next_rel
        self.next_rel += 1
        L, k = self.layers[t], (s, d)
        if k in L["m"]:
            assert k in L["dm"], "a second edge of one type on one pair is not in this example"
            L["dm"].discard(k)             # state G -> H: dp shadows m
            L["dp"][k] = e
        else:
            assert k not in L["dp"]
            L["dp"][k] = e                 # state A -> B
        self.rels[e] = (t, s, d, w)
        self._w = None
        return e

    def del_rel(self, e):
        t, s, d, w = self.rels.pop(e)
        self.gone[e] = (t, s, d, w)
        L, k = self.layers[t], (s, d)
        if L["dp"].get(k) == e:
            del L["dp"][k]
            if k in L["m"]:
                L["dm"].add(k)             # H -> G
        else:
            assert L["m"].get(k) == e and k not in L["dm"]
            L["dm"].add(k)                 # D -> G
        self._w = None

    def set_w(self, e, w):
        t, s, d, _ = self.rels[e]
        self.rels[e] = (t, s, d, w)
        self._w = None

    def delete_node(self, x):
        gone = sorted(e for e, (t, s, d, w) in self.rels.items() if x in (s, d))
        for e in gone:
            self.del_rel(e)
        self.nodes.discard(x)
        self.free_ids.append(x)
        self._w = None
        return gone

    def flush(self):
        """Fold every delta into m. Used only to set up the committed state before v5."""
        for L in self.layers.values():
            for k in L["dm"]:
                del L["m"][k]
            L["m"].update(L["dp"])
            L["dp"].clear()
            L["dm"].clear()

    def effective(self, t):
        L = self.layers[t]
        out = {k: e for k, e in L["m"].items() if k not in L["dm"] and k not in L["dp"]}
        out.update(L["dp"])
        return out

    def W(self):
        """W(u, v): the smallest weight of a live edge u -> v over the indexed types."""
        if self._w is None:
            W = {}
            for t in TYPES:
                for k, e in self.effective(t).items():
                    if k[0] != k[1]:
                        W[k] = min(W.get(k, INF), self.rels[e][3])
            self._w = W
        return self._w

    def pair_weights(self, u, v):
        if u is None or v is None:
            return INF, INF
        W = self.W()
        return W.get((u, v), INF), W.get((v, u), INF)

    def has_edge(self, u, v):
        W = self.W()
        return (u, v) in W or (v, u) in W

    def nbrs(self, x):
        out = set()
        for (u, v) in self.W():
            if u == x:
                out.add(v)
            elif v == x:
                out.add(u)
        return out

    def live_pairs(self):
        return {pr(u, v) for (u, v) in self.W()}

    def rel_of(self, t, s, d):
        for e, (tt, ss, dd, w) in self.rels.items():
            if (tt, ss, dd) == (t, s, d):
                return e
        raise KeyError((t, s, d))


def dijkstra_all(g):
    adj = defaultdict(list)
    for (u, v), w in g.W().items():
        adj[u].append((v, w))
    out = {}
    for s in sorted(g.nodes):
        dist = {s: 0}
        pq = [(0, s)]
        while pq:
            d, u = heapq.heappop(pq)
            if d > dist[u]:
                continue
            for v, w in adj[u]:
                nd = d + w
                if nd < dist.get(v, INF):
                    dist[v] = nd
                    heapq.heappush(pq, (nd, v))
        out[s] = dist
    return out


# ======================================================================================
# cow_btree (toy): a persistent B+ tree. An insert path-copies root to leaf.
# ======================================================================================
PAGES = {}
_page_seq = [0]


def bt_insert(root, key, doc):
    created = []

    def mk(p):
        pid = _page_seq[0]
        _page_seq[0] += 1
        PAGES[pid] = p
        created.append(pid)
        return pid

    def ins(pid):
        page = PAGES[pid]
        if page[0] == "leaf":
            es = sorted(page[1] + ((key, doc),))
            if len(es) <= LEAF_CAP:
                return mk(("leaf", tuple(es))), None
            h = (len(es) + 1) // 2
            return mk(("leaf", tuple(es[:h]))), (es[h][0], mk(("leaf", tuple(es[h:]))))
        _, kids, seps = page
        i = bisect_right(seps, key)
        left, split = ins(kids[i])
        kids, seps = list(kids), list(seps)
        kids[i] = left
        if split:
            seps.insert(i, split[0])
            kids.insert(i + 1, split[1])
        if len(kids) <= BRANCH_CAP:
            return mk(("branch", tuple(kids), tuple(seps))), None
        h = (len(kids) + 1) // 2
        return (mk(("branch", tuple(kids[:h]), tuple(seps[:h - 1]))),
                (seps[h - 1], mk(("branch", tuple(kids[h:]), tuple(seps[h:])))))

    if root is None:
        return mk(("leaf", ((key, doc),))), created
    left, split = ins(root)
    if split:
        return mk(("branch", (left, split[1]), (split[0],))), created
    return left, created


def bt_items(root):
    if root is None:
        return []
    page = PAGES[root]
    if page[0] == "leaf":
        return list(page[1])
    out = []
    for k in page[1]:
        out.extend(bt_items(k))
    return out


def bt_pages(root):
    if root is None:
        return set()
    page = PAGES[root]
    out = {root}
    if page[0] == "branch":
        for k in page[1]:
            out |= bt_pages(k)
    return out


def bt_path(root, key):
    """The pages from the root to the leaf that holds `key`."""
    out = []
    pid = root
    while pid is not None:
        out.append(pid)
        page = PAGES[pid]
        if page[0] == "leaf":
            break
        pid = page[1][bisect_right(page[2], key)]
    return out


def bt_json(root):
    out = []
    for pid in sorted(bt_pages(root)):
        p = PAGES[pid]
        if p[0] == "leaf":
            out.append({"page": pid, "kind": "leaf",
                        "entries": [[k >> 32, k & M32, d] for k, d in p[1]]})
        else:
            out.append({"page": pid, "kind": "branch", "children": list(p[1]),
                        "separators": [[k >> 32, k & M32] for k in p[2]]})
    return {"root": root, "pages": out}


# ======================================================================================
# Object identity (03 §2's versioning column, §6b MVCC). Every structure a version holds has an
# instance id. The id is given when the instance is created or copied, and carried by reference
# otherwise: a clone shares every id, and the first write to a structure (or to one of its chunks)
# in a clone gives it a new id. Bases (the Base object), delta pages (PAGES) and metric chunks
# (CHUNKS) already have ids; the containers below add them to the rank maps, the pool, the
# counters, added_since_fold and the metric's chunk tables. TopologyVersion and Cch ids are derived
# from their fields' ids at each publish (main(), "MVCC view").
# ======================================================================================
_IIDS = defaultdict(int)


def new_iid(kind):
    _IIDS[kind] += 1
    return _IIDS[kind]


def cow_clone(x):
    """A clone that shares identity, for the containers below; a plain copy for scratch state."""
    if hasattr(x, "clone"):
        return x.clone()
    return dict(x) if isinstance(x, dict) else list(x)


class CowList(list):
    """Whole-object copy-on-write: free_ranks (Arc<RoaringBitmap>, copied when changed) and
    added_since_fold (a persistent list: an append makes a new head)."""

    def __init__(self, kind, items=(), iid=None):
        super().__init__(items)
        self.kind = kind
        self.owned = iid is None
        self.iid = new_iid(kind) if iid is None else iid

    def _w(self):
        if not self.owned:
            self.iid = new_iid(self.kind)
            self.owned = True

    def clone(self):
        return CowList(self.kind, self, iid=self.iid)

    def append(self, x):
        self._w()
        super().append(x)

    def extend(self, xs):
        self._w()
        super().extend(xs)

    def insert(self, i, x):
        self._w()
        super().insert(i, x)

    def pop(self, *a):
        self._w()
        return super().pop(*a)

    def remove(self, x):
        self._w()
        super().remove(x)

    def clear(self):
        self._w()
        super().clear()

    def __setitem__(self, k, v):
        self._w()
        super().__setitem__(k, v)

    def __delitem__(self, k):
        self._w()
        super().__delitem__(k)


class CowDict(dict):
    """Whole-object copy-on-write: LineageStats (03 §2: copied)."""

    def __init__(self, kind, items=(), iid=None):
        super().__init__(items)
        self.kind = kind
        self.owned = iid is None
        self.iid = new_iid(kind) if iid is None else iid
        self.changed = []       # keys written in this instance

    def _w(self, k):
        if not self.owned:
            self.iid = new_iid(self.kind)
            self.owned = True
        self.changed.append(k)

    def clone(self):
        return CowDict(self.kind, self, iid=self.iid)

    def __setitem__(self, k, v):
        self._w(k)
        super().__setitem__(k, v)

    def __delitem__(self, k):
        self._w(k)
        super().__delitem__(k)


class CowMap(dict):
    """rank_of and node_of: 03 §2's ChunkedArr<Option<..>>, copy-on-write per chunk of CHUNK slots.
    The values stay in the dict; the chunk ids record which slot ranges a version wrote. Slot i
    holds key i + offset (node_of is over ranks, which start at 1)."""

    def __init__(self, kind, items=(), *, offset=0, span=0, chunks=None, iid=None):
        super().__init__(items)
        self.kind, self.offset = kind, offset
        self.owned_chunks = set()
        self.copied, self.created = [], []
        if chunks is None:
            self.iid = new_iid(kind)
            self.owned = True
            top = max([k - offset for k in self] + [span - 1, 0])
            self.chunks = {}
            for c in range(top // CHUNK + 1):
                self.chunks[c] = new_iid(kind + " chunk")
                self.owned_chunks.add(c)
        else:
            self.iid, self.owned, self.chunks = iid, False, dict(chunks)

    def clone(self):
        return CowMap(self.kind, self, offset=self.offset, chunks=self.chunks, iid=self.iid)

    def _w(self, k):
        if not self.owned:
            self.iid = new_iid(self.kind)
            self.owned = True
        c = (k - self.offset) // CHUNK
        if c not in self.owned_chunks:
            new = new_iid(self.kind + " chunk")
            if c in self.chunks:
                self.copied.append((c, self.chunks[c], new))
            else:
                self.created.append((c, new))
            self.chunks[c] = new
            self.owned_chunks.add(c)

    def __setitem__(self, k, v):
        self._w(k)
        super().__setitem__(k, v)

    def __delitem__(self, k):
        self._w(k)
        super().__delitem__(k)

    def chunk_values(self, c, missing=None):
        lo = c * CHUNK + self.offset
        return [self.get(k, missing) for k in range(lo, lo + CHUNK)]


# ======================================================================================
# Metric arrays: chunked, copy-on-write per chunk (customize.rs WeightArr), plus append.
# ======================================================================================
CHUNKS = {}
_chunk_seq = [0]


def _new_chunk(vals):
    cid = _chunk_seq[0]
    _chunk_seq[0] += 1
    CHUNKS[cid] = tuple(vals)
    return cid


class ChunkedArr:
    def __init__(self, ids, length, kind="metric", iid=None):
        self.ids = list(ids)
        self.length = length
        self.owned = set()
        self.copied = []        # (chunk index, old chunk id, new chunk id)
        self.created = []       # (chunk index, new chunk id): a chunk an append added
        self.kind = kind
        self.table_owned = iid is None
        self.iid = new_iid(kind) if iid is None else iid   # the chunk table's id
        self.copy_all = False   # G17's break only: the first write copies every chunk

    @staticmethod
    def from_list(vals, kind="metric"):
        ids = [_new_chunk(vals[i:i + CHUNK]) for i in range(0, len(vals), CHUNK)]
        a = ChunkedArr(ids, len(vals), kind)
        a.owned = set(ids)
        return a

    def clone(self):
        return ChunkedArr(self.ids, self.length, self.kind, iid=self.iid)

    def _table(self):
        if not self.table_owned:
            self.iid = new_iid(self.kind)
            self.table_owned = True
            if self.copy_all:
                for c in range(len(self.ids)):
                    self._own(c)

    def get(self, i):
        return CHUNKS[self.ids[i // CHUNK]][i % CHUNK]

    def _own(self, c):
        cid = self.ids[c]
        if cid not in self.owned:
            new = _new_chunk(CHUNKS[cid])
            self.copied.append((c, cid, new))
            self.ids[c] = new
            self.owned.add(new)
        return self.ids[c]

    def set(self, i, v):
        self._table()
        cid = self._own(i // CHUNK)
        vals = list(CHUNKS[cid])
        vals[i % CHUNK] = v
        CHUNKS[cid] = tuple(vals)

    def set_in_place(self, i, v):
        """G15's break only: write the shared chunk without copying it or the table."""
        cid = self.ids[i // CHUNK]
        vals = list(CHUNKS[cid])
        vals[i % CHUNK] = v
        CHUNKS[cid] = tuple(vals)

    def append(self, v):
        self._table()
        c = self.length // CHUNK
        if c == len(self.ids):
            cid = _new_chunk(())
            self.ids.append(cid)
            self.owned.add(cid)
            self.created.append((c, cid))
        cid = self._own(c)
        CHUNKS[cid] = CHUNKS[cid] + (v,)
        self.length += 1

    def tolist(self):
        return [self.get(i) for i in range(self.length)]


class FlatArr:
    """The builder's flat metric before the Seal chunks it."""

    def __init__(self, vals):
        self.vals = list(vals)
        self.length = len(vals)

    def get(self, i):
        return self.vals[i]

    def set(self, i, v):
        self.vals[i] = v

    def tolist(self):
        return list(self.vals)


# ======================================================================================
# The hierarchy: Up.base / Down.base as CSR arrays (value = arc id), Up.delta / Down.delta
# as cow_btrees, rank maps, the free pool, and the metric (4 arrays by arc id).
# ======================================================================================
class Base:
    count = 0

    def __init__(self, n, rows):
        Base.count += 1
        self.name = f"base {Base.count}"
        self.n = n
        self.Ap, self.Aj, self.Ax = [0], [], []
        for r in range(1, n + 1):
            for hi, i in rows.get(r, []):
                self.Aj.append(hi)
                self.Ax.append(i)
            self.Ap.append(len(self.Aj))
        cols = defaultdict(list)
        for r in range(1, n + 1):
            for k in range(self.Ap[r - 1], self.Ap[r]):
                cols[self.Aj[k]].append((r, self.Ax[k]))
        self.Dp, self.Dj, self.Dx = [0], [], []
        for r in range(1, n + 1):
            for lo, i in sorted(cols.get(r, [])):
                self.Dj.append(lo)
                self.Dx.append(i)
            self.Dp.append(len(self.Dj))

    def up(self, r):
        if r > self.n:
            return []
        s, e = self.Ap[r - 1], self.Ap[r]
        return list(zip(self.Aj[s:e], self.Ax[s:e]))

    def down(self, r):
        if r > self.n:
            return []
        s, e = self.Dp[r - 1], self.Dp[r]
        return list(zip(self.Dj[s:e], self.Dx[s:e]))

    def arcs(self):
        return {(r, hi): i for r in range(1, self.n + 1) for hi, i in self.up(r)}

    def json(self):
        return {"name": self.name, "rows": self.n, "row_of_rank": "row i holds rank i + 1",
                "Up": {"Ap": self.Ap, "Aj": self.Aj, "Ax": self.Ax},
                "Down": {"Ap": self.Dp, "Aj": self.Dj, "Ax": self.Dx}}


def key(r, c):
    return (r << 32) | c


class Hier:
    def __init__(self):
        self.lineage = 1
        self.base = None
        self.dup = None         # Up.delta root page
        self.ddn = None         # Down.delta root page
        self.n = 0              # rank dimension n_r'
        self.rank_of = {}
        self.node_of = {}
        self.free = []
        self.arc_capacity = 0
        self.metric = {}
        self.added = []         # added_since_fold: [(seq, [(lo, hi, id)])]
        self.stats = {}
        self._cache = None

    def clone(self):
        """A private version: every structure is shared (same instance ids) until written."""
        h = Hier()
        h.lineage, h.base, h.dup, h.ddn, h.n = self.lineage, self.base, self.dup, self.ddn, self.n
        h.rank_of, h.node_of, h.free = cow_clone(self.rank_of), cow_clone(self.node_of), cow_clone(self.free)
        h.arc_capacity = self.arc_capacity
        h.metric = {k: a.clone() for k, a in self.metric.items()}
        h.added = cow_clone(self.added)
        h.stats = cow_clone(self.stats)
        return h

    def invalidate(self):
        self._cache = None

    def _rows(self):
        if self._cache is None:
            up = defaultdict(list)
            for r in range(1, self.base.n + 1):
                up[r].extend(self.base.up(r))
            for k, d in bt_items(self.dup):
                up[k >> 32].append((k & M32, d))
            for r in up:
                up[r].sort()
            dn = defaultdict(list)
            for r in range(1, self.base.n + 1):
                dn[r].extend(self.base.down(r))
            for k, d in bt_items(self.ddn):
                dn[k >> 32].append((k & M32, d))
            for r in dn:
                dn[r].sort()
            arcs = {(r, hi): i for r, row in up.items() for hi, i in row}
            self._cache = (up, dn, arcs)
        return self._cache

    def up(self, r):
        return self._rows()[0].get(r, [])

    def down(self, r):
        return self._rows()[1].get(r, [])

    def arcs(self):
        return self._rows()[2]

    def m(self, name, i):
        return self.metric[name].get(i)

    def metric_by_id(self):
        L = {k: self.metric[k].tolist() for k in ("up", "dn", "mu", "md")}
        return {i: (L["up"][i], L["dn"][i], L["mu"][i], L["md"][i]) for i in range(len(L["up"]))}

    def json_maps(self):
        return {"n": self.n,
                "rank_of": [[x, r] for x, r in sorted(self.rank_of.items())],
                "node_of": [[r, self.node_of.get(r)] for r in range(1, self.n + 1)],
                "free_ranks": list(self.free),
                "arc_capacity": self.arc_capacity}

    def json_metric(self):
        L = {k: self.metric[k].tolist() for k in ("up", "dn", "mu", "md")}
        return {"up": [jw(x) for x in L["up"]], "down": [jw(x) for x in L["dn"]],
                "up_middle": L["mu"], "down_middle": L["md"]}

    def json_arcs(self):
        return [[lo, hi, i, self.node_of.get(lo), self.node_of.get(hi)]
                for (lo, hi), i in sorted(self.arcs().items())]

    def json_delta(self):
        return {"Up": bt_json(self.dup), "Down": bt_json(self.ddn)}


def pair_seeds(h, g):
    """Each arc's seed read pair by pair (PairWeights): the reference seeds, by arc id."""
    up, dn = [INF] * h.arc_capacity, [INF] * h.arc_capacity
    for (lo, hi), i in h.arcs().items():
        up[i], dn[i] = g.pair_weights(h.node_of.get(lo), h.node_of.get(hi))
    return up, dn


def full_customize(h, g, trace=None):
    """Seed every arc pair by pair, then relax every triangle in rank order. Returns
    {id: (up, down, up_middle, down_middle)}: the reference every recompute must equal."""
    return relax(h, *pair_seeds(h, g), trace=trace)


def relax(h, seeds_up, seeds_dn, trace=None):
    """The push sweep: start every arc at the seed held at its id, then relax every triangle in
    rank order. D2 and the full path run it from the full-seed pass's arrays."""
    arcs = h.arcs()
    up, dn, mu, md = {}, {}, {}, {}
    for (lo, hi), i in arcs.items():
        up[i], dn[i] = seeds_up[i], seeds_dn[i]
        mu[i] = md[i] = None
    for x in range(1, h.n + 1):
        row = h.up(x)
        for a in range(len(row)):
            for b in range(a + 1, len(row)):
                (y, iy), (z, iz) = row[a], row[b]
                top = arcs[(y, z)]
                cu = dn[iy] + up[iz]
                cd = dn[iz] + up[iy]
                was = (up[top], dn[top])
                if cu < up[top]:
                    up[top], mu[top] = cu, x
                if cd < dn[top]:
                    dn[top], md[top] = cd, x
                if trace is not None:
                    trace.append({"x": x, "y": y, "z": z, "arcs": [iy, iz, top],
                                  "try_up": jw(cu), "try_down": jw(cd),
                                  "was": [jw(was[0]), jw(was[1])],
                                  "now": [jw(up[top]), jw(dn[top])],
                                  "up_won": cu < was[0], "down_won": cd < was[1]})
    out = {i: (up[i], dn[i], mu[i], md[i]) for i in up}
    # ids without an arc (none in a lineage that never deletes arcs) stay at the preset
    for i in range(h.arc_capacity):
        out.setdefault(i, (INF, INF, None, None))
    return out


def diff_metric(a, b):
    bad = []
    for i in sorted(set(a) | set(b)):
        if a.get(i) != b.get(i):
            bad.append({"id": i, "got": [jw(x) if isinstance(x, (int, float)) else x for x in a.get(i, ())],
                        "want": [jw(x) if isinstance(x, (int, float)) else x for x in b.get(i, ())]})
    return bad


def recompute(h, g, front, *, propagate=True, known=None, trace=None, budget=None, spent=0):
    """03 §6b recustomize_seeded, generalized: the heap holds (lo, hi, id). `front` is a list
    of rank pairs that all have arcs. Writes only where values differ. Returns work counts.
    With a budget, the running total (spent + this pass) is checked after every pop, and the pass
    stops at the first pop that takes it over the budget (03 §6b: 'Aborting publishes nothing')."""
    arcs = h.arcs()
    M = h.metric
    work = {"merge_steps": 0, "pushes": 0, "seed_lookups": 0, "pops": 0}
    heap, queued = [], set()
    for (lo, hi) in sorted(set(front)):
        i = arcs[(lo, hi)]
        if i not in queued:
            queued.add(i)
            heapq.heappush(heap, (lo, hi, i))
            work["pushes"] += 1
    while heap:
        lo, hi, i = heapq.heappop(heap)
        work["pops"] += 1
        u, v = h.node_of.get(lo), h.node_of.get(hi)
        if known is not None and (lo, hi) in known:
            s_up, s_dn = known[(lo, hi)]
        else:
            s_up, s_dn = g.pair_weights(u, v)
        work["seed_lookups"] += 2
        best_up, best_dn, mid_up, mid_dn = s_up, s_dn, None, None
        dlo, dhi = h.down(lo), h.down(hi)
        a = b = 0
        apexes = []
        while a < len(dlo) and b < len(dhi):
            work["merge_steps"] += 1
            xa, xb = dlo[a][0], dhi[b][0]
            if xa < xb:
                a += 1
                continue
            if xa > xb:
                b += 1
                continue
            x, x_lo, x_hi = xa, dlo[a][1], dhi[b][1]
            cu = M["dn"].get(x_lo) + M["up"].get(x_hi)
            cd = M["dn"].get(x_hi) + M["up"].get(x_lo)
            wu, wd = cu < best_up, cd < best_dn
            if wu:
                best_up, mid_up = cu, x
            if wd:
                best_dn, mid_dn = cd, x
            apexes.append({"x": x, "try_up": jw(cu), "try_down": jw(cd), "up_won": wu, "down_won": wd})
            a += 1
            b += 1
        old = (M["up"].get(i), M["dn"].get(i), M["mu"].get(i), M["md"].get(i))
        changed = best_up != old[0] or best_dn != old[1]
        if best_up != old[0]:
            M["up"].set(i, best_up)
        if best_dn != old[1]:
            M["dn"].set(i, best_dn)
        if mid_up != old[2]:
            M["mu"].set(i, mid_up)
        if mid_dn != old[3]:
            M["md"].set(i, mid_dn)
        pushed = []
        if changed and propagate:
            for w, _ in h.up(lo):
                if w == hi:
                    continue
                d = pr(hi, w)
                j = arcs[d]
                if j not in queued:
                    queued.add(j)
                    heapq.heappush(heap, (d[0], d[1], j))
                    work["pushes"] += 1
                    pushed.append([d[0], d[1], j])
        running = spent + work["merge_steps"] + work["pushes"] + work["seed_lookups"]
        if trace is not None:
            trace.append({"arc": [lo, hi, i], "nodes": [u, v], "seed": [jw(s_up), jw(s_dn)],
                          "apexes": apexes, "old": [jw(old[0]), jw(old[1]), old[2], old[3]],
                          "new": [jw(best_up), jw(best_dn), mid_up, mid_dn],
                          "changed": changed, "pushed": pushed, "running_total": running})
        if budget is not None and running > budget:
            work["stopped_at_pop"] = work["pops"]
            work["left_in_heap"] = len(heap)
            break
    work["total"] = work["merge_steps"] + work["pushes"] + work["seed_lookups"]
    return work


def upward_search(h, r, which):
    dist = {r: 0}
    pq = [(0, r)]
    while pq:
        d, x = heapq.heappop(pq)
        if d > dist[x]:
            continue
        for y, i in h.up(x):
            nd = d + h.m(which, i)
            if nd < dist.get(y, INF):
                dist[y] = nd
                heapq.heappush(pq, (nd, y))
    return dist


def cch_query(h, s, t, trace=False):
    if s == t:
        return (0, None) if trace else 0
    rs, rt = h.rank_of.get(s), h.rank_of.get(t)
    if rs is None or rt is None:
        return (INF, None) if trace else INF
    f = upward_search(h, rs, "up")
    b = upward_search(h, rt, "dn")
    best, meet = INF, None
    for m in sorted(f):
        if m in b and f[m] + b[m] < best:
            best, meet = f[m] + b[m], m
    if trace:
        return best, {"forward": [[k, jw(v)] for k, v in sorted(f.items())],
                      "backward": [[k, jw(v)] for k, v in sorted(b.items())], "meet": meet}
    return best


def check_queries(h, g):
    ref = dijkstra_all(g)
    live = sorted(g.nodes)
    bad, n = [], 0
    for s in live:
        for t in live:
            n += 1
            want = ref[s].get(t, INF)
            got = cch_query(h, s, t)
            if got != want:
                bad.append([s, t, jw(got), jw(want)])
    return n, bad


def closure_violations(h):
    arcs = h.arcs()
    bad = []
    for r in range(1, h.n + 1):
        row = [y for y, _ in h.up(r)]
        for a in range(len(row)):
            for b in range(a + 1, len(row)):
                if (row[a], row[b]) not in arcs:
                    bad.append([r, row[a], row[b]])
    return bad


def coverage_violations(h, g):
    """pattern(S_now) ⊆ Up, and every node with an indexed edge has a rank."""
    arcs = h.arcs()
    bad = []
    for (u, v) in sorted(g.live_pairs()):
        ru, rv = h.rank_of.get(u), h.rank_of.get(v)
        if ru is None or rv is None:
            bad.append({"pair": [u, v], "why": "an endpoint has no rank"})
        elif pr(ru, rv) not in arcs:
            bad.append({"pair": [u, v], "why": f"no arc for ranks {pr(ru, rv)}"})
    return bad


def pool_violations(h):
    """P: r in free_ranks <=> node_of[r] is None, and then up(r) = down(r) = ∅."""
    bad = []
    free = set(h.free)
    for r in range(1, h.n + 1):
        holder = h.node_of.get(r)
        if (r in free) != (holder is None):
            bad.append({"rank": r, "holder": holder, "in_free": r in free})
        if (holder is None or r in free) and (h.up(r) or h.down(r)):
            bad.append({"rank": r, "holder": holder, "in_free": r in free, "has_arcs": True})
    inv = {}
    for x, r in h.rank_of.items():
        if h.node_of.get(r) != x:
            bad.append({"rank": r, "holder": h.node_of.get(r), "rank_of": x})
        inv.setdefault(r, []).append(x)
    for r, xs in inv.items():
        if len(xs) > 1:
            bad.append({"rank": r, "held_by": xs})
    return bad


def decay_signals(h, n_r, arcs_before, new_arcs):
    """03 §6 Decay, evaluated after a flush. Thresholds as THRESHOLDS lists them; nothing here acts."""
    arcs = len(h.arcs())
    dels = h.stats.get("deletions", 0)
    pool_arcs = h.stats.get("pool_row_arcs", 0)
    delta = len(bt_items(h.dup))
    fold_at = max(FOLD_FLOOR, len(h.base.Aj) ** 0.5)
    rows = [
        ("S6", "deletions ≥ n/8", dels, h.n / 8, dels >= h.n / 8),
        ("S7", "arcs ≥ 1.25 × arcs at build", arcs, 1.25 * h.stats["arcs_at_build"],
         arcs >= 1.25 * h.stats["arcs_at_build"]),
        ("S7", "one commit adding > arcs/8", new_arcs, arcs_before / 8, new_arcs > arcs_before / 8),
        ("S7", "top ranks ≥ reserve", h.n - n_r, RESERVE, h.n - n_r >= RESERVE),
        ("S7", "pool-row fill > arcs/16", pool_arcs, arcs / 16, pool_arcs > arcs / 16),
        ("fold", "delta entries > √|base|", delta, fold_at, delta > fold_at),
    ]
    return [{"job": j, "signal": s, "value": v, "threshold": round(t, 4), "met": m} for j, s, v, t, m in rows]


def down_mirror_violations(h):
    """I7: Down holds Up transposed, with the same ids."""
    up = {(lo, hi): i for (lo, hi), i in h.arcs().items()}
    dn = {}
    for r in range(1, h.n + 1):
        for lo, i in h.down(r):
            dn[(lo, r)] = i
    return [[k[0], k[1]] for k in sorted(set(up) ^ set(dn))] + \
        [[k[0], k[1]] for k in sorted(set(up) & set(dn)) if up[k] != dn[k]]


# ======================================================================================
# The running example
# ======================================================================================
XY = {i: [i % 4, i // 4] for i in range(16)}
XY.update({16: [4, 3], 17: [5, 3], 18: [5, 1], 19: [-1, 3], 20: [1.5, -1], 21: [-1, 1], 22: [1.5, 4]})


def road_w(u, v):
    u, v = pr(u, v)
    return 1 + (5 * u + 3 * v) % 9


GRID = sorted([(r * 4 + c, r * 4 + c + 1) for r in range(4) for c in range(3)] +
              [(r * 4 + c, (r + 1) * 4 + c) for r in range(3) for c in range(4)])
CHAIN = [(15, 16), (16, 17)]
ONE_WAY = (6, 7)            # ROAD 6 -> 7 only
DM_ROAD = (2, 7)            # pending deletion at v5 (both directions in dm)
DP_ROAD = (4, 5)            # pending addition at v5 (both directions in dp)
H_PAIR = (14, 13)           # replaced in one transaction: old id in m, new id in dp
H_OLD_W = 1                 # the old relationship's weight (deleted; stale in m)
H_NEW_W = 6                 # the new relationship's weight (live, in dp)
FERRY = [(10, 9, 2), (5, 10, 3)]

# Writers after the build (F). Each choice is asserted below to have the property its step shows.
F1_EDGE = ("ROAD", 8, 9)        # a weight SET whose change propagates
F2_ROAD = (1, 2)                # road 1–2 again: deleted at v6, dropped at Supergraph, so it has no arc
F4_NODE = 7                     # a node to delete
F5_ROAD = (1, 9)                # a road across the top separator: over budget
F15_SET = ("ROAD", 12, 13, 2)   # v15: the writer that sets the fold trigger again after the hand-off
F6_ROAD = (14, 17)              # an S2 insert while the fold runs
F8_ROAD = (7, 2)                # v17: a node reuses id 7 (deleted at v13); road 7–2 lands on a fill arc
V31_ROAD = (18, 17)             # G4's break: the road that gives 02's pool-ranked node 18 an arc
CTX = {}                      # key objects, for probing from outside
SNAP = {}                     # column key -> snapshot() of what that published version holds


def setup_example():
    g = Graph()
    for _ in range(19):
        g.create_node()             # 0..18; 18 stays isolated
    g.next_node = NODE_CAP
    g.free_ids = [19]               # id 19: a slot freed before the build
    for (u, v) in GRID + CHAIN + [DM_ROAD]:
        if (u, v) == DP_ROAD:
            continue
        w = road_w(u, v)
        if (u, v) == ONE_WAY:
            g.add_rel("ROAD", u, v, w)
        else:
            g.add_rel("ROAD", u, v, w)
            g.add_rel("ROAD", v, u, w)
    for s, d, w in FERRY:
        g.add_rel("FERRY", s, d, w)
    g.set_w(g.rel_of("ROAD", *H_PAIR), H_OLD_W)
    g.flush()
    # pending at v5
    g.del_rel(g.rel_of("ROAD", *DM_ROAD))
    g.del_rel(g.rel_of("ROAD", DM_ROAD[1], DM_ROAD[0]))
    g.add_rel("ROAD", DP_ROAD[0], DP_ROAD[1], road_w(*DP_ROAD))
    g.add_rel("ROAD", DP_ROAD[1], DP_ROAD[0], road_w(*DP_ROAD))
    g.del_rel(g.rel_of("ROAD", *H_PAIR))
    g.add_rel("ROAD", H_PAIR[0], H_PAIR[1], H_NEW_W)
    return g


def write(g, ops):
    """A writer's commit: apply ops to a private copy; record (min, max) pairs as record() does."""
    g = g.clone()
    pairs = set()
    journal = []
    for op in ops:
        kind = op[0]
        if kind == "node":
            nid = g.create_node()
            assert nid == op[1], (nid, op)
            journal.append(f"create node {nid}")
        elif kind == "add":
            _, t, s, d, w = op
            e = g.add_rel(t, s, d, w)
            if s != d:
                pairs.add(pr(s, d))
            journal.append(f"create {t} {s}→{d} (rel {e}, weight {w})")
        elif kind == "addroad":
            _, s, d, w = op
            for a, b in ((s, d), (d, s)):
                e = g.add_rel("ROAD", a, b, w)
                journal.append(f"create ROAD {a}→{b} (rel {e}, weight {w})")
            pairs.add(pr(s, d))
        elif kind == "del":
            _, t, s, d = op
            e = g.rel_of(t, s, d)
            g.del_rel(e)
            pairs.add(pr(s, d))
            journal.append(f"delete {t} {s}→{d} (rel {e})")
        elif kind == "set":
            _, t, s, d, w = op
            e = g.rel_of(t, s, d)
            old = g.rels[e][3]
            g.set_w(e, w)
            pairs.add(pr(s, d))
            journal.append(f"SET {t} {s}→{d} (rel {e}) weight {old} → {w}")
        elif kind == "delnode":
            _, x = op
            gone = [(e, g.rels[e]) for e in sorted(g.rels) if x in g.rels[e][1:3]]
            g.delete_node(x)
            for e, (t, s, d, w) in gone:
                pairs.add(pr(s, d))
                journal.append(f"cascade: delete {t} {s}→{d} (rel {e})")
            journal.append(f"delete node {x}")
        else:
            raise ValueError(op)
    return g, sorted(pairs), journal


def graph_json(g):
    rels = []
    for e, (t, s, d, w) in sorted(g.rels.items()):
        rels.append({"id": e, "type": t, "src": s, "dst": d, "weight": w})
    layers = {}
    for t in TYPES:
        L = g.layers[t]
        layers[t] = {
            "m": [[s, d, e] for (s, d), e in sorted(L["m"].items())],
            "dp": [[s, d, e] for (s, d), e in sorted(L["dp"].items())],
            "dm": [[s, d] for (s, d) in sorted(L["dm"])],
            "me": [],
        }
    return {"nodes": sorted(g.nodes), "free_ids": sorted(g.free_ids), "next_id": g.next_node,
            "relationships": rels, "tensors": layers}


def draw(nodes=None, edges=None, labels=None, space="node"):
    d = {"space": space}
    if nodes:
        d["nodes"] = {k: sorted(v) for k, v in nodes.items()}
    if edges:
        d["edges"] = {k: sorted([list(x) for x in v]) for k, v in edges.items()}
    if labels:
        d["labels"] = {str(k): v for k, v in sorted(labels.items())}
    return d


STEPS = []


def step(sid, part, title, design, what, calls=None, data=None, drw=None):
    STEPS.append({"id": sid, "part": part, "title": title, "design": design, "what": what,
                  "calls": calls or [], "data": data or {}, "draw": drw or {"space": "node"}})


def call(text, inputs=None, output=None, note=None):
    c = {"call": text}
    if inputs:
        c["in"] = inputs
    if output:
        c["out"] = output
    if note:
        c["note"] = note
    return c


# ======================================================================================
# B1: masked skeleton read, slab by slab
# ======================================================================================
def read_slab(g, lo, hi, mask_dm=True):
    P = set()
    per_type = {}
    for t in TYPES:
        L = g.layers[t]
        m_rows = {k: e for k, e in L["m"].items() if lo <= k[0] < hi}
        dm_rows = {k for k in L["dm"] if lo <= k[0] < hi}
        dp_rows = {k: e for k, e in L["dp"].items() if lo <= k[0] < hi}
        kept = {k for k in m_rows if not (mask_dm and k in dm_rows)}
        res = kept | set(dp_rows)
        per_type[t] = {"m": pat(m_rows), "dm": pat(dm_rows), "dp": pat(dp_rows),
                       "masked_out": pat(set(m_rows) - kept), "live": pat(res)}
        P |= {k for k in res if k[0] != k[1]}
    return P, per_type


def symmetrize(P):
    return {k for k in P | {(b, a) for (a, b) in P} if k[0] != k[1]}


# ======================================================================================
# B5..B10: multilevel nested dissection
# ======================================================================================
def adjacency(A, verts):
    nb = {v: set() for v in verts}
    for (i, j) in A:
        if i in nb and j in nb:
            nb[i].add(j)
    return nb


def side_weights(lab, w, vs):
    s = [0, 0, 0]
    for v in vs:
        s[lab[v]] += w[v]
    return s


def node_fm(vs, nb, w, lab, total):
    """Two-sided node FM seeded from the boundary (the separator vertices only)."""
    lab = dict(lab)
    limit = FM_UB * total / 2
    passes = []

    def cost(l):
        s = side_weights(l, w, vs)
        return (max(s[0], s[1]) > limit, s[2], abs(s[0] - s[1]))

    for _ in range(len(vs)):
        cur = dict(lab)
        locked = set()
        moves = []
        best, best_k, bad = cost(cur), 0, 0
        while True:
            s = side_weights(cur, w, vs)
            cands = []
            for v in sorted(x for x in vs if cur[x] == 2 and x not in locked):
                for side in (0, 1):
                    pulled = sorted(u for u in nb[v] if cur[u] == 1 - side)
                    pw = sum(w[u] for u in pulled)
                    gain = w[v] - pw
                    new = list(s)
                    new[side] += w[v]
                    new[1 - side] -= pw
                    new[2] += pw - w[v]
                    # within the balance limit, or at least shrinking the heavier side
                    if max(new[0], new[1]) > limit and max(new[0], new[1]) >= max(s[0], s[1]):
                        continue
                    cands.append((-gain, abs(new[0] - new[1]), v, side, pulled))
            if not cands:
                break
            cands.sort()
            ng, _, v, side, pulled = cands[0]
            cur[v] = side
            for u in pulled:
                cur[u] = 2
            locked.add(v)
            c = cost(cur)
            moves.append({"move": v, "to_side": side, "pulled_into_separator": pulled,
                          "gain": -ng, "separator_weight": c[1], "imbalance": c[2],
                          "over_balance_limit": c[0]})
            if c < best:
                best, best_k, bad = c, len(moves), 0
            else:
                bad += 1
                if bad >= FM_MAX_BAD:
                    break
        kept = moves[:best_k]
        for mv in kept:
            lab[mv["move"]] = mv["to_side"]
            for u in mv["pulled_into_separator"]:
                lab[u] = 2
        passes.append({"moves": moves, "kept": best_k})
        if best_k == 0:
            break
    return lab, passes


def anchor_grouping(un, K, w, cap_c, salt, mate):
    """The 2-hop pass (00-results Change 2): each unmatched vertex takes its heaviest neighbour as an
    anchor; vertices sharing an anchor pair up in scramble order, under the cap. Updates mate."""
    anchors = {}
    for v in un:
        row = [(x, j) for (i, j), x in K.items() if i == v]
        if row:
            anchors[v] = unfmix32(max(row)[0] & M32) ^ salt
    groups = defaultdict(list)
    for v, a in anchors.items():
        groups[a].append(v)
    made, glist = [], []
    for a in sorted(groups):
        vs = sorted(groups[a], key=lambda v: fmix32(v ^ salt))
        glist.append({"anchor": a, "members_in_scramble_order": vs})
        i = 0
        while i + 1 < len(vs):
            u, v = vs[i], vs[i + 1]
            if w[u] + w[v] <= cap_c:
                mate[u], mate[v] = v, u
                made.append(sorted([u, v]))
                i += 2
            else:
                i += 1
    return {"unmatched": list(un), "anchors": [[v, a] for v, a in sorted(anchors.items())], "groups": glist,
            "pairs": made}


def match_and_contract(L, active, tvw, depth, lvl, force_two_hop=False):
    salt = fmix32(SALT_BASE + 64 * depth + lvl)
    A, w, cell = L["A"], L["w"], L["cell"]
    act = [v for v in L["verts"] if cell[v] in active]
    K = {(i, j): (a << 32) | fmix32(j ^ salt) for (i, j), a in A.items() if cell[i] in active}
    cap = {c: CAP_FACTOR * tvw[c] / COARSEN_TO for c in sorted(active)}
    Kc = {k: x for k, x in K.items() if w[k[0]] + w[k[1]] <= cap[cell[k[0]]]}
    capped = sorted({pr(*k) for k in K if k not in Kc})
    unmatched = set(act)
    mate = {}
    rounds = []
    for r in range(1, ROUND_CAP + 1):
        y = {}
        for (i, j), x in Kc.items():
            if i in unmatched and j in unmatched and x > y.get(i, -1):
                y[i] = x
        p = {i: unfmix32(x & M32) ^ salt for i, x in y.items()}
        q = {i: p.get(t) for i, t in p.items()}
        mutual = sorted((i, t) for i, t in p.items() if q[i] == i and i < t)
        rounds.append({"round": r, "unmatched": sorted(unmatched),
                       "y": [[i, y[i] >> 32, y[i] & M32] for i in sorted(y)],
                       "proposal": [[i, p[i]] for i in sorted(p)],
                       "q": [[i, q[i]] for i in sorted(q)],
                       "mutual": [list(m) for m in mutual]})
        if not mutual:
            break
        for i, t in mutual:
            mate[i], mate[t] = t, i
            unmatched.discard(i)
            unmatched.discard(t)
    matched_rounds = sum(1 for r in rounds if r["mutual"])
    cap_stopped = len(rounds) == ROUND_CAP and bool(rounds[-1]["mutual"])
    two_hop = []
    for c in sorted(active):
        nb_ = sum(1 for v in act if cell[v] == c)
        pairs_c = sum(1 for v in act if cell[v] == c and v in mate) // 2
        shrink = 1 - (nb_ - pairs_c) / nb_
        rec = {"cell": c, "before": nb_, "after_matching": nb_ - pairs_c, "shrink": round(shrink, 4),
               "fires": shrink < TWO_HOP_BELOW}
        if rec["fires"] or force_two_hop:
            un = sorted(v for v in act if cell[v] == c and v not in mate)
            rec.update(anchor_grouping(un, K, w, cap[c], salt, mate))
            rec["forced"] = not rec["fires"]
        two_hop.append(rec)
    leader = {v: min(v, mate[v]) if v in mate else v for v in L["verts"]}
    leaders = sorted(set(leader.values()))
    cid = {l: k for k, l in enumerate(leaders)}
    c_of = {v: cid[leader[v]] for v in L["verts"]}
    CT = {(v, c_of[v]): 1 for v in L["verts"]}
    Ac, internal = {}, {}
    for (i, j), a in A.items():
        ci, cj = c_of[i], c_of[j]
        if ci != cj:
            Ac[(ci, cj)] = Ac.get((ci, cj), 0) + a
        else:
            internal[(ci, cj)] = internal.get((ci, cj), 0) + a
    wc = defaultdict(int)
    for v in L["verts"]:
        wc[c_of[v]] += w[v]
    nxt = {"verts": list(range(len(leaders))), "A": Ac, "w": dict(wc),
           "cell": {cid[l]: cell[l] for l in leaders},
           "members": {cid[l]: sorted(m for v in L["verts"] if leader[v] == l for m in L["members"][v])
                       for l in leaders}}
    n_before = defaultdict(int)
    n_after = defaultdict(int)
    for v in L["verts"]:
        n_before[cell[v]] += 1
    for v in nxt["verts"]:
        n_after[nxt["cell"][v]] += 1
    return {
        "salt": salt, "cap": cap, "K": [[i, j, x >> 32, x & M32] for (i, j), x in sorted(K.items())],
        "capped_out": [list(k) for k in capped], "rounds": rounds, "matched_rounds": matched_rounds,
        "round_cap_hit": cap_stopped, "two_hop": two_hop,
        "pairs": sorted([sorted([v, mate[v]]) for v in mate if v < mate[v]]),
        "leader": [[v, leader[v]] for v in L["verts"]], "c_of": c_of, "CT": sorted(CT),
        "internal_weight_dropped": ent(internal), "next": nxt,
        "n_before": dict(n_before), "n_after": dict(n_after),
    }


def initial_separator(top, cid, depth):
    vs = sorted(v for v in top["verts"] if top["cell"][v] == cid)
    A, w = top["A"], top["w"]
    nb = adjacency(A, vs)
    total = sum(w[v] for v in vs)
    salt = fmix32(0xC0FFEE + depth)
    seeds = sorted(vs, key=lambda v: fmix32(v ^ salt))[:SEP_TRIALS]
    trials = []
    for t, s in enumerate(seeds):
        side = {v: 1 for v in vs}
        side[s] = 0
        w0 = w[s]
        grown = [s]
        while w0 < total / 2:
            front = sorted({u for v in vs if side[v] == 0 for u in nb[v] if side[u] == 1})
            if not front:
                front = sorted(v for v in vs if side[v] == 1)
            if not front:
                break

            def gain(u):
                return (sum(A[(u, x)] for x in nb[u] if side[x] == 0) -
                        sum(A[(u, x)] for x in nb[u] if side[x] == 1))
            u = max(front, key=lambda u: (gain(u), -u))
            side[u] = 0
            w0 += w[u]
            grown.append(u)
        b0 = sorted(v for v in vs if side[v] == 0 and any(side[x] == 1 for x in nb[v]))
        b1 = sorted(v for v in vs if side[v] == 1 and any(side[x] == 0 for x in nb[v]))
        sep = b0 if sum(w[v] for v in b0) <= sum(w[v] for v in b1) else b1
        lab0 = {v: (2 if v in sep else side[v]) for v in vs}
        lab, passes = node_fm(vs, nb, w, lab0, total)
        s3 = side_weights(lab, w, vs)
        over = max(s3[0], s3[1]) > FM_UB * total / 2
        trials.append({"trial": t, "seed": s, "grown_side_0": grown, "boundary_0": b0, "boundary_1": b1,
                       "separator_before_fm": sep, "fm_passes": passes,
                       "labels": [[v, lab[v]] for v in vs], "weights": s3,
                       "over_balance_limit": over, "score": [int(over), s3[2], abs(s3[0] - s3[1])]})
    best = min(trials, key=lambda tr: (tr["score"], tr["trial"]))
    return {"cell": cid, "verts": vs, "total_weight": total, "trials": trials, "best": best["trial"],
            "labels": {v: l for v, l in best["labels"]}}


def dissect(big, A, depth):
    verts = sorted(v for c in big for v in c["verts"])
    cell_of = {v: c["cid"] for c in big for v in c["verts"]}
    tvw = {c["cid"]: len(c["verts"]) for c in big}
    lvl0 = {"verts": verts, "A": {k: a for k, a in A.items() if k[0] in cell_of}, "w": {v: 1 for v in verts},
            "cell": cell_of, "members": {v: [v] for v in verts}}
    levels, matchings = [lvl0], []
    active = set(tvw)
    stops = []
    while active:
        m = match_and_contract(levels[-1], active, tvw, depth, len(levels) - 1)
        stopped = []
        for c in sorted(active):
            nb_, na = m["n_before"][c], m["n_after"][c]
            why = None
            if na <= COARSEN_TO:
                why = f"{na} vertices, at or under {COARSEN_TO}"
            elif 1 - na / nb_ < MIN_SHRINK:
                why = f"shrank {nb_} → {na}, under 5 %"
            if why:
                stopped.append({"cell": c, "why": why})
        m["stopped"] = stopped
        matchings.append(m)
        levels.append(m["next"])
        active -= {s["cell"] for s in stopped}
        stops.append(stopped)
    top = levels[-1]
    lab = {}
    seps = []
    for c in big:
        r = initial_separator(top, c["cid"], depth)
        lab.update(r["labels"])
        seps.append(r)
    unc = []
    for li in range(len(levels) - 2, -1, -1):
        L = levels[li]
        c_of = matchings[li]["c_of"]
        proj = {v: lab[c_of[v]] for v in L["verts"]}
        nb = adjacency(L["A"], L["verts"])
        per_cell = []
        new = dict(proj)
        for c in big:
            vs = [v for v in L["verts"] if L["cell"][v] == c["cid"]]
            # mxv(A, [labels = s]·w): the vertex weight of each vertex's neighbours on side 0 and on side 1,
            # the weight a move pulls into the separator (node FM's gain is w(v) minus it)
            n0 = {v: sum(L["w"][u] for u in nb[v] if proj[u] == 0) for v in vs}
            n1 = {v: sum(L["w"][u] for u in nb[v] if proj[u] == 1) for v in vs}
            boundary = sorted(v for v in vs if proj[v] == 2)
            lab_c, passes = node_fm(vs, nb, L["w"], {v: proj[v] for v in vs}, len(c["verts"]))
            new.update(lab_c)
            per_cell.append({"cell": c["cid"], "boundary": boundary,
                             "side_0_neighbour_weight": [[v, n0[v]] for v in vs if n0[v]],
                             "side_1_neighbour_weight": [[v, n1[v]] for v in vs if n1[v]],
                             "fm_passes": passes,
                             "weights_before": side_weights(proj, L["w"], vs),
                             "weights_after": side_weights(lab_c, L["w"], vs)})
        unc.append({"level": li, "projected": [[v, proj[v]] for v in L["verts"]],
                    "cells": per_cell, "labels": [[v, new[v]] for v in L["verts"]]})
        lab = new
    return {"depth": depth, "cells": [{"cid": c["cid"], "name": c["name"], "verts": c["verts"],
                                       "lo": c["lo"], "hi": c["hi"]} for c in big],
            "levels": levels, "matchings": matchings, "separators": seps, "uncoarsen": unc,
            "labels": lab}


def split_cells(big, lab, A, rank):
    new_cells, Mrows, seps = [], [], []
    for c in big:
        vs = c["verts"]
        sep = sorted(v for v in vs if lab[v] == 2)
        parts = [sorted(v for v in vs if lab[v] == s) for s in (0, 1)]
        top0 = c["hi"] - len(sep) + 1
        sep_ranks = {}
        for k, v in enumerate(sep):
            rank[v] = top0 + k
            sep_ranks[v] = top0 + k
        lo = c["lo"]
        kids = []
        for s, part in enumerate(parts):
            if not part:
                continue
            nc = {"cid": len(new_cells), "name": c["name"] + str(s), "verts": part, "lo": lo,
                  "hi": lo + len(part) - 1, "parent": c["name"], "side": s}
            lo += len(part)
            new_cells.append(nc)
            kids.append(nc["name"])
            for v in part:
                Mrows.append((v, nc["cid"]))
        seps.append({"cell": c["name"], "separator": sep, "separator_ranks": [[v, r] for v, r in
                                                                             sorted(sep_ranks.items())],
                     "sides": parts, "children": kids})
    cell_of = dict(Mrows)
    A_next = {k: 1 for k in A if k[0] in cell_of and k[1] in cell_of and cell_of[k[0]] == cell_of[k[1]]}
    removed = sorted(k for k in A if k not in A_next)
    return new_cells, A_next, {"M": [[v, c] for v, c in sorted(Mrows)],
                               "cells": [{k: nc[k] for k in ("cid", "name", "verts", "lo", "hi", "parent", "side")}
                                         for nc in new_cells],
                               "separators": seps, "A_next": pat(A_next), "removed": pat(removed)}


def mmd_leaf(c, A, rank, depth):
    vs = c["verts"]
    S = set(vs)
    adj = {v: {u for (x, u) in A if x == v and u in S} for v in vs}
    steps = []
    r = c["lo"]
    while adj:
        v = min(adj, key=lambda x: (len(adj[x]), x))
        nbv = adj.pop(v)
        for a in nbv:
            adj[a].discard(v)
        fill = []
        for a in sorted(nbv):
            for b in sorted(nbv):
                if a < b and b not in adj[a]:
                    fill.append([a, b])
                    adj[a].add(b)
                    adj[b].add(a)
        steps.append({"node": v, "degree": len(nbv), "neighbours": sorted(nbv), "fill": fill, "rank": r})
        rank[v] = r
        r += 1
    return {"cell": c["name"], "depth": depth, "verts": vs, "lo": c["lo"], "hi": c["hi"], "steps": steps,
            "edges": pat({k for k in A if k[0] in S and k[1] in S and k[0] < k[1]})}


def nested_dissection(A0, comps):
    rank = {}
    A = dict(A0)
    cells = [{"cid": i, "name": "C" if len(comps) == 1 else f"C{i}", "verts": sorted(c["verts"]),
              "lo": c["lo"], "hi": c["hi"]} for i, c in enumerate(comps)]
    depth = 0
    depths, leaves = [], []
    while cells:
        small = [c for c in cells if len(c["verts"]) <= LEAF_MAX]
        big = [c for c in cells if len(c["verts"]) > LEAF_MAX]
        for c in small:
            leaves.append(mmd_leaf(c, A, rank, depth))
        if not big:
            break
        rec = dissect(big, A, depth)
        cells, A_next, sp = split_cells(big, rec["labels"], A, rank)
        rec["split"] = sp
        rec["A"] = pat(A)
        depths.append(rec)
        A = A_next
        depth += 1
    return rank, depths, leaves


# ======================================================================================
# B11, C: elimination tree, postorder, counts, the game
# ======================================================================================
def liu_etree(n, adj):
    parent = {r: None for r in range(1, n + 1)}
    anc = {r: None for r in range(1, n + 1)}
    for k in range(1, n + 1):
        for i in sorted(x for x in adj.get(k, ()) if x < k):
            while i is not None and i < k:
                nxt = anc[i]
                anc[i] = k
                if nxt is None:
                    parent[i] = k
                i = nxt
    return parent


def postorder(n, parent):
    kids = defaultdict(list)
    roots = []
    for r in range(1, n + 1):
        (kids[parent[r]] if parent[r] is not None else roots).append(r)
    order = []

    def visit(r):
        for c in sorted(kids[r]):
            visit(c)
        order.append(r)
    for r in sorted(roots):
        visit(r)
    return {old: new for new, old in enumerate(order, 1)}


def col_counts(n, adj, parent):
    counts = {r: 0 for r in range(1, n + 1)}
    for i in range(1, n + 1):
        seen = set()
        for k in sorted(x for x in adj.get(i, ()) if x < i):
            j = k
            while j is not None and j < i and j not in seen:
                seen.add(j)
                j = parent[j]
        for j in seen:
            counts[j] += 1
    return counts


def game(n, U0, drop=None):
    up, parent, kids, steps = {}, {}, defaultdict(list), []
    for r in range(1, n + 1):
        own = sorted(U0.get(r, ()))
        row = set(own)
        ks = sorted(kids[r])
        for c in ks:
            row |= set(up[c]) - {r}
        if drop and drop[0] == r:
            row.discard(drop[1])
        up[r] = sorted(row)
        p = up[r][0] if up[r] else None
        parent[r] = p
        if p is not None:
            kids[p].append(r)
        steps.append({"rank": r, "own": own, "kids": ks, "up": up[r], "parent": p,
                      "fill": sorted(row - set(own))})
    return up, parent, steps


def adj_of(pairs):
    adj = defaultdict(set)
    for a, b in pairs:
        adj[a].add(b)
        adj[b].add(a)
    return adj


# ======================================================================================
# Rank allocation (03 §6), S2 (Davis–Hager with a host overlay), the §6 flush
# ======================================================================================
def allocate(h, g, pairs, top_dim_in_delta=True):
    E = set()
    for (u, v) in pairs:
        if g.has_edge(u, v):
            E |= {u, v}
    out = []
    for x in sorted(E):
        if x in h.rank_of:
            continue
        d = len(g.nbrs(x))
        if d <= 1 and h.free:
            r = h.free.pop(0)
            kind = "pool"
        else:
            h.n += 1
            r = h.n
            kind = "top"
        h.rank_of[x] = r
        h.node_of[r] = x
        if kind == "top":
            h.stats["top_ranks_used"] = h.stats.get("top_ranks_used", 0) + 1   # LineageStats (03 §2)
        out.append({"node": x, "degree": d, "rank": r, "kind": kind})
    return out


def allocate_02(h, g, pairs, top_dim_in_delta=True):
    """02 §6's rule, used only by G4's break (03 App. B V3.1): a rank-less endpoint takes the lowest
    rank in free_ranks whatever its degree, and that rank's holder becomes rank-less. Nothing takes a
    rank out of free_ranks when its holder gains arcs."""
    E = set()
    for (u, v) in pairs:
        if g.has_edge(u, v):
            E |= {u, v}
    out = []
    for x in sorted(E):
        if x in h.rank_of:
            continue
        stripped = None
        if h.free:
            r = h.free.pop(0)
            stripped = h.node_of.get(r)
            if stripped is not None:
                del h.rank_of[stripped]
            kind = "pool"
        else:
            h.n += 1
            r = h.n
            kind = "top"
        h.rank_of[x] = r
        h.node_of[r] = x
        out.append({"node": x, "rank": r, "kind": kind, "stripped": stripped})
    return out


def s2(h, inserts, seq, *, push=True, renumber=False, down_too=True):
    over = defaultdict(set)
    heap, done = [], set()
    rows_read = 0
    for lo, hi in sorted(set(inserts)):
        over[lo].add(hi)
        heapq.heappush(heap, lo)
    trace = []
    while heap:
        j = heapq.heappop(heap)
        if j in done:
            continue
        done.add(j)
        row = sorted({hi for hi, _ in h.up(j)} | over[j])
        rows_read += len(row)
        p = row[0]
        need = []
        if push:
            prow = {hi for hi, _ in h.up(p)} | over[p]
            rows_read += len(prow)
            need = sorted(set(row[1:]) - prow)
            if need:
                over[p] |= set(need)
                heapq.heappush(heap, p)
        trace.append({"row": j, "up": row, "parent": p, "pushed_into_parent": need})
    new = sorted((r, c) for r in over for c in over[r])
    ids = {}
    for k, (lo, hi) in enumerate(new):
        ids[(lo, hi)] = h.arc_capacity + k
    copied_pages = []
    for (lo, hi) in new:
        h.dup, made = bt_insert(h.dup, key(lo, hi), ids[(lo, hi)])
        copied_pages.append(made)
        if down_too:
            h.ddn, made = bt_insert(h.ddn, key(hi, lo), ids[(lo, hi)])
            copied_pages.append(made)
    h.arc_capacity += len(new)
    pool_new = sum(1 for lo, _ in new if lo <= RESERVE)
    if pool_new:            # LineageStats is copied only when a counter changes
        h.stats["pool_row_arcs"] = h.stats.get("pool_row_arcs", 0) + pool_new
    h.added.append((seq, [(lo, hi, ids[(lo, hi)]) for (lo, hi) in new]))
    for name, v in (("up", INF), ("dn", INF), ("mu", None), ("md", None)):
        for _ in new:
            h.metric[name].append(v)
    h.invalidate()
    if renumber:
        # the break: ids renumbered row-major after the pass (a positional write)
        rows = {}
        k = 0
        remap = {}
        for r in range(1, h.n + 1):
            rr = []
            for hi, i in h.up(r):
                remap[i] = k
                rr.append((hi, k))
                k += 1
            rows[r] = rr
        h.base = Base(h.n, rows)
        h.dup = h.ddn = None
        h.invalidate()
    work = rows_read + len(new) * (2 if down_too else 1)
    return {"inserted_pairs": [list(p) for p in sorted(set(inserts))], "overlay_trace": trace,
            "new_arcs": [[lo, hi, ids[(lo, hi)]] for (lo, hi) in new], "work": work}


def flush(h, g, pairs, seq, *, budget=None, offlock=False, front_seed=True, propagate=True,
          s2_push=True, s2_renumber=False, s2_down=True, record=True, topology_only=False,
          alloc=allocate, seed_delta="by_id", metric_from_vec=True):
    """03 §6 on a private hierarchy h: resolve, allocate, classify, S2, recompute. With a budget
    (a writer under the lock), the work stops at the first point where it passes the budget."""
    rec = {"seq": seq, "pairs": [list(p) for p in pairs]}
    rec["resolved"] = [{"pair": [u, v], "live_edge": g.has_edge(u, v),
                        "weights": [jw(x) for x in g.pair_weights(u, v)]} for (u, v) in pairs]
    rec["allocation"] = alloc(h, g, pairs)
    arcs = h.arcs()
    metric_only, inserts, nothing = [], [], []
    for (u, v) in pairs:
        ru, rv = h.rank_of.get(u), h.rank_of.get(v)
        if ru is None or rv is None:
            nothing.append({"pair": [u, v], "why": "an endpoint has no rank and no edge"})
            continue
        lo, hi = pr(ru, rv)
        if (lo, hi) in arcs:
            metric_only.append((lo, hi))
        elif g.has_edge(u, v):
            inserts.append((lo, hi))
        else:
            nothing.append({"pair": [u, v], "why": "no arc and no edge"})
    rec["classify"] = {"metric_only": [list(p) for p in metric_only],
                       "s2_insert": [list(p) for p in inserts], "nothing": nothing}
    work = 0
    N = []
    if inserts:
        r2 = s2(h, inserts, seq, push=s2_push, renumber=s2_renumber, down_too=s2_down)
        rec["s2"] = r2
        work += r2["work"]
        N = [(lo, hi) for lo, hi, _ in r2["new_arcs"]]
    if topology_only:
        return rec
    if budget is not None and work > budget:
        rec["budget"] = budget
        rec["over_budget"] = True
        rec["stopped"] = {"in": "S2", "work": work}
        rec["work"] = {"total": work, "s2": work}
        return rec
    front = sorted(set(metric_only + inserts + N))
    if not front_seed:
        front = sorted(set(N) - set(inserts))
    rec["front"] = [[lo, hi, h.arcs()[(lo, hi)]] for (lo, hi) in front]
    known = None
    if not front_seed:
        # the break: the changed pairs' seeds are not re-read; their arcs keep the old seed
        known = {}
    if offlock and len(front) > h.n / VALVE_DIV:
        rec["path"] = "full"
        rec["valve"] = f"front of {len(front)} arcs > n/{VALVE_DIV} = {h.n / VALVE_DIV:g}"
        # the full path: seeds written by arc id (base from Ax, delta from its entries), then one
        # relax sweep that starts from those arrays
        full = full_seed_by_id(h, g, delta_mode=seed_delta)
        rec["full_seed"] = full["record"]
        trace = []
        got = relax(h, full["up"], full["dn"], trace)
        if metric_from_vec:
            # 03 §5 Phases: a Repair runs "as Seal, from base", and the Seal chunks the metric "one array
            # at a time (WeightArr::from_vec)". So the full path builds flat arrays and chunks them: every
            # chunk of every array is new, and none is shared with the base.
            flat_ = {"up": [], "dn": [], "mu": [], "md": []}
            for i in range(h.arc_capacity):
                for name, val in zip(("up", "dn", "mu", "md"), got[i]):
                    flat_[name].append(val)
            h.metric = {name: ChunkedArr.from_list(vals, "metric " + name) for name, vals in flat_.items()}
            rec["metric_rebuilt"] = True
        else:       # G17's break only: write the changed values into the base's chunks
            for i, (a, b, c, d) in got.items():
                for name, val in (("up", a), ("dn", b), ("mu", c), ("md", d)):
                    if h.metric[name].get(i) != val:
                        h.metric[name].set(i, val)
        rec["sweep_triangles"] = len(trace)
        rec["work"] = {"total": work + len(trace), "s2": work, "sweep_triangles": len(trace)}
        return rec
    rec["path"] = "scoped"
    trace = [] if record else None
    wk = recompute(h, g, front, propagate=propagate, trace=trace, budget=budget, spent=work)
    if record:
        rec["recompute"] = trace
    wk["s2"] = work
    wk["total"] += work
    rec["work"] = wk
    if budget is not None:
        rec["budget"] = budget
        rec["over_budget"] = wk["total"] > budget
        if "stopped_at_pop" in wk:
            rec["stopped"] = {"in": "recompute", "pop": wk["stopped_at_pop"], "work": wk["total"]}
    return rec


# ======================================================================================
# The full-seed pass (03 §5): m<¬struct(dm) ∧ ¬struct(dp)> ∪ dp, MIN across types,
# both directions, scattered by arc id from Ax and from the delta's entries
# ======================================================================================
def full_seed_by_id(h, g, mask_dp=True, both_dirs=True, slabs=SLABS, delta_mode="by_id"):
    """delta_mode: 'by_id' (the design); 'skip' (02's bug, App. B V3.4: seeds aligned to the base only,
    so delta arcs stay +∞); 'by_position' (a positional write: the j-th delta entry in key order is
    written at slot |base| + j instead of at its id). The last two exist only for G's breaks."""
    per = []
    W = {}
    for (lo, hi) in slabs + (((NODE_CAP, 1 << 30),) if g.next_node > NODE_CAP else ()):
        for t in TYPES:
            L = g.layers[t]
            m = {k: e for k, e in L["m"].items() if lo <= k[0] < hi}
            dm = {k for k in L["dm"] if lo <= k[0] < hi}
            dp = {k: e for k, e in L["dp"].items() if lo <= k[0] < hi}
            kept = {k: e for k, e in m.items() if k not in dm and (k not in dp or not mask_dp)}
            X = {}
            for k, e in kept.items():
                X[k] = g.rels[e][3] if e in g.rels else g.gone[e][3]
            for k, e in dp.items():
                if k in X:
                    X[k] = min(X[k], g.rels[e][3])   # only reached by the no-dp-mask break
                else:
                    X[k] = g.rels[e][3]
            per.append({"rows": [lo, min(hi, g.next_node) - 1], "type": t,
                        "m": [[s, d, e, g.rels[e][3] if e in g.rels else g.gone[e][3]] for (s, d), e in sorted(m.items())],
                        "masked_by_dm": pat(dm),
                        "masked_by_dp": pat(set(m) & set(dp)),
                        "dp": [[s, d, e, g.rels[e][3]] for (s, d), e in sorted(dp.items())],
                        "X": ent(X)})
            for k, x in X.items():
                if k[0] != k[1]:
                    W[k] = min(W.get(k, INF), x)
    Wp = {}
    for (u, v), x in W.items():
        ru, rv = h.rank_of.get(u), h.rank_of.get(v)
        if ru is not None and rv is not None:
            Wp[(ru, rv)] = x
    upv = {(a, b): x for (a, b), x in Wp.items() if a < b}
    dnv = {(b, a): x for (a, b), x in Wp.items() if a > b} if both_dirs else {}
    seeds_up = [INF] * h.arc_capacity
    seeds_dn = [INF] * h.arc_capacity
    aligned = []
    for r in range(1, h.base.n + 1):
        for k in range(h.base.Ap[r - 1], h.base.Ap[r]):
            hi, i = h.base.Aj[k], h.base.Ax[k]
            a, b = upv.get((r, hi), INF), dnv.get((r, hi), INF)
            aligned.append([k, r, hi, i, jw(a), jw(b)])
            seeds_up[i], seeds_dn[i] = a, b
    delta = []
    for j, (kk, i) in enumerate(bt_items(h.dup)):
        r, hi = kk >> 32, kk & M32
        a, b = upv.get((r, hi), INF), dnv.get((r, hi), INF)
        delta.append([r, hi, i, jw(a), jw(b)])
        if delta_mode == "by_id":
            seeds_up[i], seeds_dn[i] = a, b
        elif delta_mode == "by_position":
            seeds_up[len(h.base.Aj) + j], seeds_dn[len(h.base.Aj) + j] = a, b
    rec = {"per_slab_and_type": per, "W": ent({k: v for k, v in W.items()}),
           "W_rank": ent(Wp), "triu_up": ent(upv), "triu_down": ent(dnv),
           "aligned_base": aligned, "delta_entries": delta,
           "seeds_up": [jw(x) for x in seeds_up], "seeds_down": [jw(x) for x in seeds_dn]}
    return {"up": seeds_up, "dn": seeds_dn, "record": rec}


# ======================================================================================
# MVCC snapshots: what one published version holds, taken at its publish
# ======================================================================================
TOPOLOGY_FIELDS = ("Up.base", "Down.base", "Up.delta", "Down.delta", "rank_of", "node_of", "free_ranks",
                   "LineageStats", "added_since_fold")
METRIC_FIELDS = ("metric up", "metric down", "metric up_middle", "metric down_middle")
METRIC_KEY = {"metric up": "up", "metric down": "dn", "metric up_middle": "mu", "metric down_middle": "md"}


def snapshot(h):
    """Every structure of a published hierarchy: its instance id and a copy of its contents, taken at
    publish time. Pages and chunks are listed by id, with their contents, so a later check can compare
    two versions that share an id (G15)."""
    s = {}
    b = h.base.json()
    s["Up.base"] = {"iid": h.base.name, "contents": b["Up"]}
    s["Down.base"] = {"iid": h.base.name, "contents": b["Down"]}
    for kind, root in (("Up.delta", h.dup), ("Down.delta", h.ddn)):
        s[kind] = {"iid": root, "contents": bt_json(root),
                   "entries": [[k >> 32, k & M32, d] for k, d in bt_items(root)]}
    for kind, m in (("rank_of", h.rank_of), ("node_of", h.node_of)):
        chunks = {c: cid for c, cid in sorted(m.chunks.items())}
        s[kind] = {"iid": m.iid, "chunks": [[c, cid] for c, cid in chunks.items()],
                   "chunk_values": {str(cid): m.chunk_values(c) for c, cid in chunks.items()},
                   "contents": [[k, v] for k, v in sorted(m.items())]}
    s["free_ranks"] = {"iid": h.free.iid, "contents": sorted(h.free)}
    s["LineageStats"] = {"iid": h.stats.iid, "contents": dict(sorted(h.stats.items()))}
    s["added_since_fold"] = {"iid": h.added.iid, "contents": [[q, [list(x) for x in a]] for q, a in h.added]}
    for kind in METRIC_FIELDS:
        a = h.metric[METRIC_KEY[kind]]
        conv = jw if kind in ("metric up", "metric down") else (lambda v: v)
        s[kind] = {"iid": a.iid, "chunks": list(a.ids), "length": a.length,
                   "chunk_values": {str(cid): [conv(v) for v in CHUNKS[cid]] for cid in a.ids}}
    s["scalars"] = {"n": h.n, "arc_capacity": h.arc_capacity, "lineage": h.lineage}
    return s


def snapshot_values(s, kind):
    """A metric array's values by arc id, read back from a snapshot's chunks."""
    out = []
    for cid in s[kind]["chunks"]:
        out.extend(s[kind]["chunk_values"][str(cid)])
    return out[:s[kind]["length"]]


def delta_page_split(pages_old, pages_new, root_new, page_of, keys, root_old):
    """The pages a writer's inserts created: the root-to-leaf path of each inserted key, plus the new
    half of each leaf that split (a new leaf off every path, whose parent is on one, holding only keys
    of one old leaf that an insert reached)."""
    made = pages_new - pages_old
    path = set()
    for k in keys:
        pid = root_new
        while pid is not None:
            path.add(pid)
            p = page_of[pid]
            if p[0] == "leaf":
                break
            pid = p[1][bisect_right(p[2], k)]
    parent = {}
    for pid in pages_new:
        p = page_of[pid]
        if p[0] == "branch":
            for c in p[1]:
                parent[c] = pid
    off = made - path
    # a split's new half holds only old keys, all from one old leaf that some insert reached in the old
    # tree; a copy of a leaf no insert reached is not a split half
    reached = set()
    for k in keys:
        pid = root_old
        while pid is not None and page_of[pid][0] == "branch":
            pid = page_of[pid][1][bisect_right(page_of[pid][2], k)]
        if pid is not None:
            reached.add(pid)
    reached_keys = [{k for k, _ in page_of[q][1]} for q in reached]

    def split_half(p):
        ks = {k for k, _ in page_of[p][1]}
        return any(ks <= o for o in reached_keys)
    split = {p for p in off if parent.get(p) in path and page_of[p][0] == "leaf" and split_half(p)}
    return {"made": made, "path": path, "split": split, "stray": off - split, "path_not_made": path - made}


def retain_log(upkeep, log, node, *, keep_always=False):
    """03 §7 Retention at a writer's publish on a Built entry: with upkeep = Some{since}, the log keeps
    nodes with seq > since; with upkeep = None, the log is dropped. keep_always is G18's break."""
    if upkeep is not None:
        if node["seq"] > upkeep["since"]:
            return [node] + log, "kept"
        return log, "not above since"
    if keep_always:
        return [node] + log, "kept"
    return [], "dropped"


# ======================================================================================
# The log (03 §5): a persistent list of nodes, each with a seq
# ======================================================================================
_lognode = [0]


def log_node(seq, pairs, kind="Pairs"):
    _lognode[0] += 1
    return {"uid": _lognode[0], "seq": seq, "kind": kind, "pairs": [list(p) for p in pairs]}


def log_json(log):
    return [{"node": n["uid"], "seq": n["seq"], "kind": n["kind"], "pairs": n["pairs"]} for n in log]


def relink_above(batch_log, published_log, fork):
    """CatchUp: the published copy's nodes with seq > fork, walked from the head and stopping at
    the first seq <= fork, re-created (≈40 B each, pairs shared) on top of the batch's copy."""
    above = []
    for n in published_log:
        if n["seq"] <= fork:
            break
        above.append(n)
    relinked = [dict(n, uid=None) for n in above]
    for n in relinked:
        _lognode[0] += 1
        n["uid"] = _lognode[0]
    return relinked + batch_log, above


# ======================================================================================
# The walkthrough
# ======================================================================================
def main():
    assert all(unfmix32(fmix32(x)) == x for x in (0, 1, 7, 12345, M32, 0xDEADBEEF))
    G = {}
    g5 = setup_example()
    G[5] = g5
    notes = []
    events = []
    gen = [1]

    # ---------------------------------------------------------------- A. Storage
    gj = graph_json(g5)
    stale = {}
    for t in TYPES:
        L = g5.layers[t]
        for k in L["dp"]:
            if k in L["m"]:
                e = L["m"][k]
                stale[f"{t} {k[0]}→{k[1]}"] = {"m_id": e, "m_weight": g5.gone[e][3], "dp_id": L["dp"][k],
                                               "dp_weight": g5.rels[L["dp"][k]][3]}
    step("A", "A", "Storage at v5: two relationship types, each with m, dp and dm", "03 §3, §5; tensor.rs:52-108",
         f"Each relationship type keeps its edges in three layers: m (committed), dp (added since the last "
         f"flush) and dm (deleted since the last flush). The live value of a pair is dp if present, else m "
         f"unless dm marks it. ROAD has {len(g5.layers['ROAD']['m'])} entries in m, {len(g5.layers['ROAD']['dp'])} "
         f"in dp and {len(g5.layers['ROAD']['dm'])} in dm; FERRY has {len(g5.layers['FERRY']['m'])} in m.",
         [],
         {"graph": gj, "coords": {str(k): v for k, v in XY.items()},
          "pending": {"dm_road": list(DM_ROAD), "dp_road": list(DP_ROAD), "replaced": list(H_PAIR),
                      "one_way": list(ONE_WAY)},
          "stale_m_under_dp": stale,
          "multi_edge_layer": "empty: no pair has two edges of one type in this example",
          "unused_ids": sorted(g5.free_ids), "isolated": [x for x in sorted(g5.nodes) if not g5.nbrs(x)]},
         draw(nodes={"node": set(g5.nodes), "unused": {19}, "isolated": {18}},
              edges={"road": {pr(*k) for k in g5.effective("ROAD") if pr(*k) not in {pr(*DP_ROAD), ONE_WAY}},
                     "road_one_way": {ONE_WAY}, "pending_add": {DP_ROAD}, "pending_delete": {DM_ROAD},
                     "replaced": {pr(*H_PAIR)}, "ferry": {pr(s, d) for s, d, _ in FERRY}}))

    # ---------------------------------------------------------------- E bookkeeping
    pub = {"version": 5, "state": "Building", "phase": "ReadSkeleton", "gen": 1, "log": []}

    def ev(**kw):
        kw["published_log"] = log_json(pub["log"])
        kw["entry"] = {"state": pub["state"], "phase": pub["phase"], "generation": pub["gen"]}
        events.append(kw)

    events.append({"actor": "writer", "version": 5, "what": "CREATE the CCH index (ROAD and FERRY, weight w)",
                   "entry": {"state": "Building", "phase": "ReadSkeleton", "generation": 1},
                   "published_log": [], "note": "a new generation enqueues the build job"})

    # ---------------------------------------------------------------- B1 slab 0 at v5
    P0, slab0 = read_slab(G[5], *SLABS[0])
    ev(actor="batch", name="ReadSkeleton slab 0", fork=5, reads=f"rows {SLABS[0][0]}–{SLABS[0][1] - 1} at v5",
       fit="Same", why="no writer committed since the fork", consumed=[])

    # writer v6: delete road 1–2 (both directions), rows already read
    g6, pairs6, j6 = write(G[5], [("del", "ROAD", 1, 2), ("del", "ROAD", 2, 1)])
    G[6] = g6
    n6 = log_node(6, pairs6)
    pub["log"] = [n6] + pub["log"]
    pub["version"] = 6
    ev(actor="writer", version=6, what="; ".join(j6), log_node=log_json([n6])[0],
       note="the entry is Building, so the writer appends its pairs to the log")

    P1, slab1 = read_slab(G[6], *SLABS[1])
    ev(actor="batch", name="ReadSkeleton slab 1", fork=6, reads=f"rows {SLABS[1][0]}–{SLABS[1][1] - 1} at v6",
       fit="Same", why="no writer committed since the fork", consumed=[],
       note="the batch's copy already holds the log node with seq 6, so Same keeps it")
    pub["phase"] = "Order"

    P = P0 | P1
    S = symmetrize(P)
    nnzS = len(S)
    P0_nomask, _ = read_slab(G[5], *SLABS[0], mask_dm=False)
    S_nomask = symmetrize(P0_nomask | P1)
    step("B1", "B", f"Masked skeleton read: 2 slabs → S with {nnzS} entries", "03 §3 Input skeleton, §5 Phases",
         f"Each slab reads the live pattern of its rows: pattern(m) without the dm entries, plus pattern(dp). "
         f"Slab 0 (rows 0–9) is read at v5 and slab 1 (rows 10–19) at v6. The dm mask drops road "
         f"{DM_ROAD[0]}–{DM_ROAD[1]}; without it S would have {len(S_nomask)} entries instead of {nnzS}. "
         f"The road 1–2 deleted at v6 is still in S, because its rows were read at v5.",
         [call("P_t = pattern(m_t)⟨¬struct(dm_t)⟩ ∪ pattern(dp_t), per type t and slab", ["m", "dm", "dp"], "P_t"),
          call("P = ∪ over types and slabs", ["P_ROAD", "P_FERRY"], "P"),
          call("S = select(OFFDIAG, eWiseAdd(ONEB, P, Pᵀ))", ["P"], "S")],
         {"slabs": [{"rows": [SLABS[0][0], SLABS[0][1] - 1], "version": 5, "per_type": slab0, "P": pat(P0)},
                    {"rows": [SLABS[1][0], SLABS[1][1] - 1], "version": 6, "per_type": slab1, "P": pat(P1)}],
          "P": pat(P), "S": pat(S), "S_without_dm_mask": pat(S_nomask), "nnz": nnzS},
         draw(nodes={"slab0": set(range(0, 10)) & g5.nodes, "slab1": set(range(10, 20)) & g5.nodes},
              edges={"skeleton": {pr(*k) for k in S}, "masked_out": {DM_ROAD}, "pending_add": {DP_ROAD},
                     "deleted_after_read": {(1, 2)}}))

    # ---------------------------------------------------------------- B2 degrees
    deg = defaultdict(int)
    for (i, j) in S:
        deg[i] += 1
    ids = list(range(NODE_CAP))
    no_rank = [i for i in ids if deg.get(i, 0) == 0]
    ranked = [i for i in ids if deg.get(i, 0) > 0]
    n_r = len(ranked) + RESERVE
    step("B2", "B", f"Degrees: {len(ranked)} nodes get ranks, {len(no_rank)} do not, {RESERVE} pool ranks",
         "03 §3 step 0, §2 Rank space",
         f"d = reduce_rows(PLUS, S) counts each node's neighbours. Nodes "
         f"{and_list(no_rank)} have no entry: {no_rank[0]} exists but has no edge, and "
         f"{no_rank[1]} is an unused id. They get no rank. Ranks 1–{RESERVE} are created with no holder; they "
         f"form the pool, and the {len(ranked)} ranked nodes take ranks {RESERVE + 1}–{n_r}. So n_r = "
         f"{len(ranked)} + {RESERVE} = {n_r} counts only nodes with an edge in the skeleton S. A node whose "
         f"first edge comes through the log gets its rank later, at Supergraph (03 §3 step 0, §4 step 1; "
         f"step C2).",
         [call("d = reduce_rows(PLUS, S)", ["S"], "d")],
         {"d": [[i, deg[i]] for i in ranked], "no_rank": no_rank, "pool_ranks": list(range(1, RESERVE + 1)),
          "n_r": n_r, "ranked": len(ranked), "reserve": RESERVE},
         draw(nodes={"ranked": set(ranked), "no_rank": set(no_rank)},
              edges={"skeleton": {pr(*k) for k in S}}, labels={i: str(deg[i]) for i in ranked}))

    # ---------------------------------------------------------------- B3 peeling
    dd = {i: deg[i] for i in ranked}
    peeled = {}
    rounds = []
    f = sorted(i for i in ranked if dd[i] == 1)
    nxt_rank = RESERVE + 1
    rnd = 0
    while f:
        rnd += 1
        got = {}
        for i in f:
            peeled[i] = nxt_rank
            got[i] = nxt_rank
            nxt_rank += 1
        t = defaultdict(int)
        for (i, j) in S:
            if j in got and i not in peeled:
                t[i] += 1
        for i, c in t.items():
            dd[i] -= c
        f_new = sorted(i for i in t if dd[i] <= 1)
        rounds.append({"round": rnd, "frontier": sorted(got), "ranks": [[i, r] for i, r in sorted(got.items())],
                       "t": [[i, c] for i, c in sorted(t.items())], "d_after": [[i, dd[i]] for i in sorted(t)],
                       "next_frontier": f_new})
        f = f_new
    core = sorted(i for i in ranked if i not in peeled)
    step("B3", "B", f"Hanging trees: {len(peeled)} nodes peeled in {len(rounds)} rounds",
         "03 §3 step 1",
         f"Nodes of degree 1 hang off the rest and add no shortcut, so they take the lowest ranks above the pool, "
         f"in (round, id) order. Round 1 peels node {and_list(rounds[0]['frontier'])}; that drops node "
         f"{and_list(rounds[0]['next_frontier'])} to degree 1, so round 2 peels it. Then no node has degree 1 "
         f"and {len(core)} nodes remain.",
         [call("f = select(VALUEEQ, d, 1)", ["d"], "f"),
          call("t⟨¬peeled⟩ = S PLUS_PAIR f   (mxv: frontier neighbours per node)", ["S", "f"], "t"),
          call("d⟨¬peeled⟩ = d − t;  f = select(VALUEEQ, d⟨¬peeled⟩, 1)", ["d", "t"], "f")],
         {"rounds": rounds, "peeled_ranks": [[i, r] for i, r in sorted(peeled.items())], "core": core},
         draw(nodes={"peeled": set(peeled), "core": set(core)}, edges={"skeleton": {pr(*k) for k in S}},
              labels={i: f"r{r}" for i, r in peeled.items()}))

    # ---------------------------------------------------------------- B4 components
    core_set = set(core)
    S_core = {k for k in S if k[0] in core_set and k[1] in core_set}
    comp = {}
    for s in core:
        if s in comp:
            continue
        stack = [s]
        comp[s] = s
        while stack:
            x = stack.pop()
            for (a, b) in S_core:
                if a == x and b not in comp:
                    comp[b] = s
                    stack.append(b)
    labels = sorted(set(comp.values()))
    comps = []
    lo = nxt_rank
    for l in labels:
        vs = sorted(v for v in core if comp[v] == l)
        comps.append({"label": l, "verts": vs, "lo": lo, "hi": lo + len(vs) - 1})
        lo += len(vs)
    step("B4", "B", f"Components: {len(comps)} component of {len(comps[0]['verts'])} nodes", "03 §3 step 2",
         f"Connected components of what is left after peeling. Each component is ordered on its own and owns a "
         f"contiguous range of ranks. Here one component holds nodes {core[0]}–{core[-1]} and owns ranks "
         f"{comps[0]['lo']}–{comps[0]['hi']}.",
         [call("LAGr_ConnectedComponents(G) on a private copy of S⟨core⟩", ["S"], "component",
               note="it unloads G->A (FastSV7.c:42-46), so it gets a copy; the label is the smallest id")],
         {"component": [[v, comp[v]] for v in core], "cells": comps},
         draw(nodes={"component": core_set, "peeled": set(peeled)}, edges={"skeleton": {pr(*k) for k in S_core}}))

    # ---------------------------------------------------------------- B5..B10 nested dissection
    A0 = {k: 1 for k in S_core}
    nd_rank, depths, leaves = nested_dissection(A0, comps)

    def coarse_step(sid, title_prefix, d0, li):
        m = d0["matchings"][li]
        L = d0["levels"][li]
        nxt_ = d0["levels"][li + 1]
        nb_, na = len(L["verts"]), len(nxt_["verts"])
        th = [x for x in m["two_hop"] if x["fires"]]
        caps = and_list(sorted({f"{v:g}" for v in m["cap"].values()}))
        what = (f"Every vertex proposes to its heaviest unmatched neighbour, with ties broken by a scrambled "
                f"id; two vertices that propose to each other match. "
                f"{plural(m['matched_rounds'], 'round')} matched {plural(len(m['pairs']), 'pair')}")
        if th:
            what += (f"; matching alone shrank a cell by under 15 %, so the 2-hop pass grouped vertices "
                     f"that share an anchor")
        what += (f". Merging each pair gives {na} vertices from {nb_}, with edge weights summed. "
                 + (f"The weight cap ({caps}) blocked {plural(len(m['capped_out']), 'candidate pair')}."
                    if m["capped_out"] else f"The weight cap ({caps}) blocked no pair."))
        calls = [
            call("K = apply(A_ℓ, key(i,j) = (A_ℓ(i,j) ≪ 32) | fmix32(j ^ salt))", ["A_ℓ"], "K",
                 note="a positional user op, named for the JIT (00-results Change 5)"),
            call("K_cap = K⟨w(i) + w(j) ≤ cap(cell)⟩, cap = 1.5·tvwgt/CoarsenTo", ["K", "w"], "K_cap"),
            call("per round: y⟨u⟩ = K_cap MAX_FIRST u", ["K_cap", "u"], "y"),
            call("per round: p = target(y), the column whose scrambled id is in y's low 32 bits", ["y"], "p"),
            call("per round: q = GxB_Vector_extract_Vector(p, p), so q(i) = p(p(i)); mutual where q(i) = i",
                 ["p"], "q"),
        ]
        if th:
            calls.append(call("2-hop: a⟨unmatched⟩ = K MAX_FIRST 1 (anchor); build (anchor, scramble(v)) and "
                              "sort rows; pair neighbours in a row under the cap", ["K"], "pairs"))
        calls += [
            call("Cᵀ = GrB_Matrix_build(I = v, J = coarse id of v's leader, X = 1)", ["leader"], "Cᵀ",
                 note="Cᵀ is built directly, not C (00-results Change 6)"),
            call("A_ℓ₊₁ = select(OFFDIAG, (Cᵀ)ᵀ · A_ℓ · Cᵀ)   (PLUS_TIMES)", ["Cᵀ", "A_ℓ"], "A_ℓ₊₁"),
            call("w_ℓ₊₁ = (Cᵀ)ᵀ · w_ℓ", ["Cᵀ", "w_ℓ"], "w_ℓ₊₁"),
        ]
        data = {"depth": d0["depth"], "level": li, "verts": L["verts"], "A": ent(L["A"]),
                "w": [[v, L["w"][v]] for v in L["verts"]], "cell": [[v, L["cell"][v]] for v in L["verts"]],
                "members": [[v, L["members"][v]] for v in L["verts"]], "salt": m["salt"],
                "cap": [[c, v] for c, v in m["cap"].items()], "K": m["K"], "capped_out": m["capped_out"],
                "rounds": m["rounds"], "matched_rounds": m["matched_rounds"], "round_cap_hit": m["round_cap_hit"],
                "two_hop": m["two_hop"], "pairs": m["pairs"], "leader": m["leader"], "CT": [list(x) for x in m["CT"]],
                "internal_weight_dropped": m["internal_weight_dropped"],
                "next": {"verts": nxt_["verts"], "A": ent(nxt_["A"]), "w": [[v, nxt_["w"][v]] for v in nxt_["verts"]],
                         "cell": [[v, nxt_["cell"][v]] for v in nxt_["verts"]],
                         "members": [[v, nxt_["members"][v]] for v in nxt_["verts"]]},
                "stopped": m["stopped"]}
        groups = {}
        for a, b in m["pairs"]:
            for x in L["members"][a] + L["members"][b]:
                groups[x] = f"{a}+{b}"
        pair_edges = set()
        for a, b in m["pairs"]:
            pair_edges |= {pr(x, y) for x in L["members"][a] for y in L["members"][b] if (x, y) in A0}
        title = f"{title_prefix} level {li}: {nb_} → {na} vertices in {plural(m['matched_rounds'], 'matching round')}"
        if len(m["rounds"]) > m["matched_rounds"]:
            title += f" (round {len(m['rounds'])} matches nothing)"
        if th:
            title += " + 2-hop"
        step(sid, "B", title, "03 §3 step 3; 00-results Change 1–3, 6; m1-matching-quality", what, calls, data,
             draw(space=f"coarse:{d0['depth']}:{li}",
                  nodes={"in_play": {x for v in L["verts"] for x in L["members"][v]}},
                  edges={"matched": pair_edges}))
        return title

    d0 = depths[0]
    b5_titles = []
    for li in range(len(d0["matchings"])):
        b5_titles.append(coarse_step(f"B5.{li}", "Coarsen depth 0,", d0, li))
    fired = [(d["depth"], li) for d in depths for li, m in enumerate(d["matchings"]) for x in m["two_hop"]
             if x["fires"]]
    # Forcing the trigger on every level of the running example: what would the pass pair?
    forced = []
    for d in depths:
        tvw_d = {c["cid"]: len(c["verts"]) for c in d["cells"]}
        act = set(tvw_d)
        for li in range(len(d["matchings"])):
            mf = match_and_contract(d["levels"][li], act, tvw_d, d["depth"], li, force_two_hop=True)
            for x in mf["two_hop"]:
                forced.append({"depth": d["depth"], "level": li, "cell": x["cell"], "unmatched": x.get("unmatched", []),
                               "anchors": x.get("anchors", []), "groups": x.get("groups", []),
                               "pairs": x.get("pairs", [])})
            act -= {st["cell"] for st in d["matchings"][li]["stopped"]}
    forced_pairs = sum(len(x["pairs"]) for x in forced)
    shared = [(x["depth"], x["level"], g["anchor"], g["members_in_scramble_order"]) for x in forced
              for g in x["groups"] if len(g["members_in_scramble_order"]) > 1]
    forced_why = ("the leftover vertices share no anchor" if not shared else
                  "the leftover vertices share an anchor only at " +
                  and_list(f"depth {a} level {b} (vertices {and_list(m)} under anchor {c})" for a, b, c, m in shared) +
                  ", where the weight cap blocks the pair")
    # The same ordering under other fixed salt bases: does the pass ever fire?
    saved = globals()["SALT_BASE"]
    scan_fired = []
    for base in range(SALT_SCAN):
        globals()["SALT_BASE"] = base
        _, dd_, _ = nested_dissection(A0, comps)
        if any(x["fires"] for d in dd_ for m in d["matchings"] for x in m["two_hop"]):
            scan_fired.append(base)
    globals()["SALT_BASE"] = saved
    star_n = 6
    star = {"verts": list(range(star_n + 1)), "A": {}, "w": {v: 1 for v in range(star_n + 1)},
            "cell": {v: 0 for v in range(star_n + 1)}, "members": {v: [v] for v in range(star_n + 1)}}
    for i in range(1, star_n + 1):
        star["A"][(0, i)] = star["A"][(i, 0)] = 1
    sm = match_and_contract(star, {0}, {0: star_n + 1}, 0, 0)
    th = sm["two_hop"][0]
    assert th["fires"] and th["pairs"]
    assert not fired and not scan_fired, (fired, scan_fired)
    star_xy = {0: [0, 0], 1: [0, -1], 2: [0.87, -0.5], 3: [0.87, 0.5], 4: [0, 1], 5: [-0.87, 0.5], 6: [-0.87, -0.5]}
    step("B5.h", "B", f"The 2-hop pass, on a side example: a star of {star_n + 1} vertices shrinks "
                      f"{star_n + 1} → {len(sm['next']['verts'])}",
         "03 §3 step 3; 00-results Change 2; m1-matching-quality",
         f"On the running example every level shrinks by at least 15 %, so this pass never fires there "
         f"(nor under any of the {SALT_SCAN} salt bases tried; forced on every level, it pairs "
         f"{'no vertex' if not forced_pairs else plural(forced_pairs, 'vertex', 'vertices')}). "
         f"A star shows why it exists: every leaf proposes to the hub, the hub "
         f"accepts one, and the round matches {len(sm['rounds'][0]['mutual'])} pair, a shrink of "
         f"{th['shrink'] * 100:.1f} %. Each unmatched leaf then takes its heaviest neighbour as an anchor; leaves "
         f"sharing an anchor pair up in scramble order under the cap, giving {len(th['pairs'])} more pairs.",
         [call("y⟨u⟩ = K_cap MAX_FIRST u; p = target(y); q = p(p); mutual where q(i) = i", ["K_cap"], "pairs"),
          call("a⟨unmatched⟩ = K MAX_FIRST 1, decoded: each unmatched vertex's anchor", ["K"], "a"),
          call("build (anchor, scramble(v)) for the unmatched v and sort each row", ["a"], "groups"),
          call("pair neighbours within a row while w(u) + w(v) ≤ cap", ["groups", "w"], "pairs")],
         {"side_example": True, "coords": {str(k): v for k, v in star_xy.items()},
          "verts": star["verts"], "A": ent(star["A"]), "w": [[v, 1] for v in star["verts"]],
          "salt": sm["salt"], "cap": [[c, v] for c, v in sm["cap"].items()], "K": sm["K"],
          "rounds": sm["rounds"], "two_hop": sm["two_hop"], "pairs": sm["pairs"], "leader": sm["leader"],
          "next": {"verts": sm["next"]["verts"], "A": ent(sm["next"]["A"]),
                   "w": [[v, sm["next"]["w"][v]] for v in sm["next"]["verts"]],
                   "members": [[v, sm["next"]["members"][v]] for v in sm["next"]["verts"]]},
          "fires_on_running_example": [list(x) for x in fired],
          "forced_on_running_example": forced, "salt_scan": {"bases": [0, SALT_SCAN - 1], "fired": scan_fired}},
         draw(space="side:star", edges={"matched": {tuple(p) for p in sm["rounds"][0]["mutual"]},
                                        "two_hop": {tuple(p) for p in th["pairs"]}}))

    # B6 first separator
    sp0 = d0["separators"][0]
    best = sp0["trials"][sp0["best"]]
    top = d0["levels"][-1]
    nbt = adjacency(top["A"], top["verts"])
    complete = all(len(nbt[v]) == len(top["verts"]) - 1 for v in top["verts"])
    empty_side = 0 in best["weights"][:2]
    step("B6", "B", f"First separator on the coarsest graph ({len(top['verts'])} vertices): "
                    f"weight {best['weights'][2]}, sides {best['weights'][0]} and {best['weights'][1]}",
         "03 §3 step 4",
         f"Each trial grows side 0 from one seed vertex until it holds half the weight, takes the lighter "
         f"boundary as the separator, and improves it with node FM (Fiduccia–Mattheyses refinement: move one "
         f"separator vertex at a time to the side where it pulls the least weight into the separator, choosing only "
         f"among moves that keep both sides under the balance limit or at least shrink the heavier side; then "
         f"keep the best state seen). {plural(len(sp0['trials']), 'trial')} ran, one per coarse vertex; trial "
         f"{sp0['best']} (seed {best['seed']}) wins with separator weight {best['weights'][2]}. "
         + ("Every coarse vertex here touches every other one, so no separator can leave both sides non-empty; "
            "uncoarsening (B7) moves weight back to the empty side." if complete and empty_side else ""),
         [call("host: greedy growing from each seed, then boundary → vertex separator, then node FM",
               ["A_top", "w_top"], "labels")],
         {"coarsest": {"verts": top["verts"], "A": ent(top["A"]), "w": [[v, top["w"][v]] for v in top["verts"]],
                       "members": [[v, top["members"][v]] for v in top["verts"]]},
          "complete_graph": complete, "balance_limit": FM_UB * sp0["total_weight"] / 2,
          "trials": sp0["trials"], "best": sp0["best"], "labels": [[v, l] for v, l in sorted(sp0["labels"].items())]},
         draw(space=f"coarse:0:{len(d0['levels']) - 1}",
              nodes={k: {x for v in top["verts"] if sp0["labels"][v] == s for x in top["members"][v]}
                     for k, s in (("side0", 0), ("side1", 1), ("separator", 2))}))

    # B7 uncoarsen
    final0 = d0["labels"]
    sepw = sum(1 for v in final0 if final0[v] == 2)
    step("B7", "B", f"Uncoarsen depth 0: project and refine over {plural(len(d0['uncoarsen']), 'level')} → "
                    f"separator of {sepw}",
         "03 §3 step 5",
         f"Each level takes its labels from the coarser one through Cᵀ, then runs node FM starting from the "
         f"separator vertices only. At the finest level the separator is nodes "
         f"{and_list(sorted(v for v in final0 if final0[v] == 2))}, with "
         f"{and_list(sorted(len([v for v in final0 if final0[v] == s_]) for s_ in (0, 1)))} nodes on the sides.",
         [call("labels_ℓ = Cᵀ_ℓ · labels_ℓ₊₁   (mxv)", ["Cᵀ", "labels"], "labels_ℓ"),
          call("n_s = mxv(A_ℓ, [labels = s]·w)   (PLUS_SECOND) per side s: the summed vertex weight of each vertex's "
               "neighbours on side s", ["A_ℓ", "labels_ℓ", "w"], "n_0, n_1"),
          call("host: node FM over a private unload, seeded from the separator", ["A_ℓ", "labels_ℓ"], "labels_ℓ")],
         {"levels": d0["uncoarsen"]},
         draw(nodes={k: {v for v in final0 if final0[v] == s} for k, s in (("side0", 0), ("side1", 1),
                                                                          ("separator", 2))},
              edges={"skeleton": {pr(*k) for k in S_core}}))

    # B8 split
    sp = d0["split"]
    step("B8", "B", f"Split depth 0: separator takes ranks "
                    f"{sp['separators'][0]['separator_ranks'][0][1]}–{sp['separators'][0]['separator_ranks'][-1][1]}; "
                    f"{len(sp['cells'])} cells of {', '.join(str(len(c['verts'])) for c in sp['cells'])}",
         "03 §3 step 6",
         f"M has one row per non-separator node, with a 1 in its new cell's column. M ANY_PAIR Mᵀ, masked by "
         f"A_0, keeps exactly the edges inside one cell; {len(sp['removed'])} entries touching the separator are "
         f"dropped. The separator nodes take the top ranks of the cell, lowest id first.",
         [call("M = GxB_Matrix_build_Scalar_Vector(I = node, J = new cell, 1)", ["labels"], "M"),
          call("A_1⟨A_0, struct⟩ = M ANY_PAIR Mᵀ", ["M", "A_0"], "A_1")],
         sp,
         draw(nodes={**{f"cell_{c['name']}": set(c["verts"]) for c in sp["cells"]},
                     "separator": set(sp["separators"][0]["separator"])},
              edges={"kept": {pr(*k) for k in [tuple(x) for x in sp["A_next"]]},
                     "dropped": {pr(*k) for k in [tuple(x) for x in sp["removed"]]}},
              labels={v: f"r{r}" for v, r in sp["separators"][0]["separator_ranks"]}))

    # B9 deeper depths
    for d in depths[1:]:
        lv = []
        for li in range(len(d["matchings"])):
            m = d["matchings"][li]
            lv.append({"level": li, "verts": d["levels"][li]["verts"], "A": ent(d["levels"][li]["A"]),
                       "w": [[v, d["levels"][li]["w"][v]] for v in d["levels"][li]["verts"]],
                       "cell": [[v, d["levels"][li]["cell"][v]] for v in d["levels"][li]["verts"]],
                       "members": [[v, d["levels"][li]["members"][v]] for v in d["levels"][li]["verts"]],
                       "salt": m["salt"], "cap": [[c, v] for c, v in m["cap"].items()], "K": m["K"],
                       "capped_out": m["capped_out"], "rounds": m["rounds"], "matched_rounds": m["matched_rounds"],
                       "two_hop": m["two_hop"], "pairs": m["pairs"], "CT": [list(x) for x in m["CT"]],
                       "stopped": m["stopped"]})
        topd = d["levels"][-1]
        lab = d["labels"]
        spd = d["split"]
        cells_txt = and_list(f"{c['name']} with {len(c['verts'])} nodes" for c in d["cells"])
        seps_txt = "; ".join(f"cell {s['cell']} by {'node' if len(s['separator']) == 1 else 'nodes'} "
                             f"{and_list(s['separator'])}" for s in spd["separators"])
        th_any = any(x["fires"] for m in d["matchings"] for x in m["two_hop"])
        step(f"B9.{d['depth']}", "B",
             f"Depth {d['depth']}: {plural(len(d['cells']), 'cell')} in one block-diagonal matrix, "
             f"{plural(len(d['matchings']), 'coarsening level')}{' with a 2-hop pass' if th_any else ''}",
             "03 §3 steps 3–6",
             f"All cells of depth {d['depth']} with more than {LEAF_MAX} nodes ({cells_txt}) are coarsened, "
             f"separated, refined and split together; their edges sit in one matrix with one block per cell. "
             f"The separators split {seps_txt}.",
             [call("the same calls as B5–B8, once per level for all cells", ["A_d"], "A_d+1")],
             {"depth": d["depth"], "cells": d["cells"], "A": d["A"], "levels": lv,
              "coarsest": {"verts": topd["verts"], "A": ent(topd["A"]), "w": [[v, topd["w"][v]] for v in topd["verts"]],
                           "cell": [[v, topd["cell"][v]] for v in topd["verts"]],
                           "members": [[v, topd["members"][v]] for v in topd["verts"]]},
              "separators": d["separators"], "uncoarsen": d["uncoarsen"], "split": spd},
             draw(nodes={**{f"cell_{c['name']}": set(c["verts"]) for c in spd["cells"]},
                         "separator": {v for s in spd["separators"] for v in s["separator"]}},
                  edges={"kept": {pr(*k) for k in [tuple(x) for x in spd["A_next"]]}},
                  labels={v: f"r{r}" for s in spd["separators"] for v, r in s["separator_ranks"]}))

    # B10 leaves
    step("B10", "B", f"Leaves: {plural(len(leaves), 'cell')} of at most {LEAF_MAX} nodes, by exact minimum degree",
         "03 §3 step 7",
         "A cell this small is not dissected. Its nodes are eliminated one at a time, always the one with the "
         "fewest remaining neighbours inside the cell, lowest id on a tie; each elimination joins that node's "
         "neighbours. The order fills the cell's rank range from the bottom.",
         [call("host: exact minimum degree on 128-bit bitsets; ties to the lowest id", ["A_leaf"], "ranks")],
         {"leaves": leaves},
         draw(nodes={f"leaf_{l['cell']}": set(l["verts"]) for l in leaves},
              labels={s["node"]: f"r{s['rank']}" for l in leaves for s in l["steps"]}))

    # B11 final order (ND ranks, then 03 §3 step 8's etree postorder relabel)
    rank_nd = dict(peeled)
    rank_nd.update(nd_rank)
    assert sorted(rank_nd.values()) == list(range(RESERVE + 1, n_r + 1))
    Bnd = {(rank_nd[a], rank_nd[b]) for (a, b) in S}
    par_nd = liu_etree(n_r, adj_of(Bnd))
    q = postorder(n_r, par_nd)
    identity = all(q[r] == r for r in q)
    rank_of = {x: q[r] for x, r in rank_nd.items()}
    rank_to_node = {r: x for x, r in rank_of.items()}
    step("B11", "B", f"Final order: rank_to_node over ranks 1–{n_r}"
                     f"{' (postorder relabel is the identity)' if identity else ' after the postorder relabel'}",
         "03 §3 step 8, §2",
         f"Pool ranks 1–{RESERVE} have no node. The two peeled nodes sit just above them, then the component's "
         f"leaves and separators. Order's last step extracts B = S(π,π) once and builds its elimination tree "
         f"(a rank's parent is its lowest higher neighbour once lower ranks are eliminated). It then renumbers "
         f"the ranks in postorder, children before parents, so each subtree holds consecutive ranks; "
         + ("here the postorder already equals the order, so π is unchanged." if identity else
            f"{plural(sum(1 for r in q if q[r] != r), 'rank')} move."),
         [call("B = S(π,π)   (one private extract)", ["S", "π"], "B"),
          call("host: Liu's elimination tree over B, postorder q, relabel B's arrays by q", ["B"], "q")],
         {"rank_nd": [[x, r] for x, r in sorted(rank_nd.items())], "etree_nd": [[r, p] for r, p in par_nd.items()],
          "q": [[r, q[r]] for r in range(1, n_r + 1)], "q_is_identity": identity,
          "rank_of": [[x, rank_of.get(x)] for x in range(NODE_CAP)],
          "rank_to_node": [[r, rank_to_node.get(r)] for r in range(1, n_r + 1)], "n_r": n_r},
         draw(nodes={"pool_holderless": set(), "ranked": set(rank_of), "no_rank": set(no_rank)},
              edges={"skeleton": {pr(*k) for k in S}},
              labels={x: f"r{r}" for x, r in rank_of.items()}))

    # order batches (E)
    for d in depths:
        ev(actor="batch", name=f"Order depth {d['depth']}", fork=6, fit="Same",
           why="no writer committed since the fork", consumed=[], note="private state only; no snapshot read")

    # writer v7 commits while the last Order batch runs
    g7, pairs7, j7 = write(G[6], [("node", 19), ("addroad", 12, 19, 4), ("node", 20),
                                  ("addroad", 0, 20, 3), ("addroad", 3, 20, 6)])
    G[7] = g7
    n7 = log_node(7, pairs7)
    pub["log"] = [n7] + pub["log"]
    pub["version"] = 7
    ev(actor="writer", version=7, what="; ".join(j7), log_node=log_json([n7])[0],
       note="commits while the last Order batch runs")
    batch_log = [n6]                      # the Order batch forked at v6
    caught, above = relink_above(batch_log, pub["log"], 6)
    pub["log"] = caught
    ev(actor="batch", name="Order: leaves and final π", fork=6, fit="CatchUp",
       why="a writer (v7) committed since the fork",
       relinked=[n["seq"] for n in above], consumed=[],
       note="the batch's copy gets the published nodes with seq > 6 relinked on top")
    pub["phase"] = "Supergraph"

    # ---------------------------------------------------------------- C1 B = S(π,π)
    B = {(rank_of[a], rank_of[b]) for (a, b) in S}
    par_B = liu_etree(n_r, adj_of(B))
    cnt_B = col_counts(n_r, adj_of(B), par_B)
    arcs_B = sum(cnt_B.values())
    guard = ARC_GUARD * nnzS
    step("C1", "C", f"B = S(π,π) in rank space: {len(B)} entries; counts give {arcs_B} arcs (guard {guard})",
         "03 §3 step 8, Guards (b)",
         f"B is the matrix B11 extracted once and relabelled by q; there is no second extract. The column "
         f"counts (how many arcs each rank will have, read off the elimination tree) give the exact number of "
         f"arcs the game would make, {arcs_B}, before any arc is built. The guard allows {ARC_GUARD} arcs per "
         f"skeleton entry, {guard} here, so the build goes on.",
         [call("B, relabelled by q (from B11; no new extract)", ["B11"], "B"),
          call("host: Liu's etree over B, then column counts. The toy walks each row's subtree, standing in "
               "for the design's Gilbert–Ng–Peyton counts; the numbers are the same", ["B"], "parent, counts")],
         {"B": pat(B), "etree": [[r, par_B[r]] for r in range(1, n_r + 1)],
          "col_counts": [[r, cnt_B[r]] for r in range(1, n_r + 1)], "arcs": arcs_B, "guard": guard},
         draw(space="rank", edges={"B": {pr(*k) for k in B}}))

    # ---------------------------------------------------------------- C2 Supergraph's log
    sg = Hier()
    sg.n = n_r
    sg.rank_of = CowMap("rank_of", rank_of, span=NODE_CAP)
    sg.node_of = CowMap("node_of", {r: rank_to_node.get(r) for r in range(1, n_r + 1)}, offset=1, span=n_r)
    sg.free = CowList("free_ranks", range(1, RESERVE + 1))
    sg.added = CowList("added_since_fold")
    log_sg = [n for n in pub["log"] if n["seq"] <= 7]
    sg_pairs = sorted({tuple(p) for n in log_sg for p in n["pairs"]})
    resolved = [{"pair": list(p), "live_edge": g7.has_edge(*p)} for p in sg_pairs]

    class _H:   # allocation needs only the maps
        pass
    alloc = allocate(sg, g7, sg_pairs)
    n_r1 = sg.n
    Aset, Dset = set(), set()
    for (u, v) in sg_pairs:
        ru, rv = sg.rank_of.get(u), sg.rank_of.get(v)
        if g7.has_edge(u, v):
            Aset |= {(ru, rv), (rv, ru)}
        elif ru is not None and rv is not None:
            Dset |= {(ru, rv), (rv, ru)}
    B1 = (B - Dset) | Aset
    log_nodes_live = sorted({x for p in sg_pairs if g7.has_edge(*p) for x in p} - set(ranked))
    n_r_sec2 = len(ranked) + len(log_nodes_live) + RESERVE
    step("C2", "C", f"Supergraph forks at v7: {len(log_sg)} log nodes, {len(alloc)} new ranks, "
                    f"B′ = (B ∖ D) ∪ A with {len(Dset) // 2} pair dropped and {len(Aset) // 2} added",
         "03 §4 steps 1–2 (revision 3.1), §6 Rank allocation",
         "Every logged pair is checked against the graph at v7. Rank allocation runs first: a new node with "
         "one neighbour takes the lowest pool rank, one with more takes a new top rank. Pairs with a live edge "
         "go into A; pairs with none go into D and leave B, so B′ is exactly the live pattern at v7. "
         f"Nodes {and_list(log_nodes_live)} get their ranks here: 03 §3 step 0 ranks only nodes with an edge "
         f"in S, and §4 step 1 ranks the log's nodes. "
         f"03 §2 defines n_r differently: it also counts nodes with an edge in the log at V_sg, which would "
         f"give n_r = {n_r_sec2} instead of {n_r}. The toy follows §3 and §4.",
         [call("host: resolve each logged pair at V_sg; allocate ranks; extend π", ["log", "graph@v7"], "π′"),
          call("B′ = (B ∖ D) ∪ A", ["B", "D", "A"], "B′")],
         {"log": log_json(log_sg), "resolved": resolved, "allocation": alloc,
          "free_ranks_after": list(sg.free), "n_r_prime": n_r1,
          "n_r": {"toy_and_03_s3_s4": n_r, "if_03_s2_counted_the_log": n_r_sec2,
                  "log_nodes_with_an_edge": log_nodes_live},
          "A": pat(Aset), "D": pat(Dset), "B_prime": pat(B1)},
         draw(edges={"added": {tuple(p) for p in sg_pairs if g7.has_edge(*p)},
                     "dropped": {tuple(p) for p in sg_pairs if not g7.has_edge(*p)}},
              nodes={"new_pool_rank": {a["node"] for a in alloc if a["kind"] == "pool"},
                     "new_top_rank": {a["node"] for a in alloc if a["kind"] == "top"}},
              labels={a["node"]: f"r{a['rank']}" for a in alloc}))
    pub["log"] = []
    ev(actor="batch", name="Supergraph", fork=7, fit="Same", why="no writer committed since the fork",
       consumed=sorted(n["seq"] for n in log_sg), note="its copy sets head = None; consumed nodes are freed")
    pub["phase"] = "Seed"

    # ---------------------------------------------------------------- C3 etree over B′
    adjB1 = adj_of(B1)
    par = liu_etree(n_r1, adjB1)
    cnt = col_counts(n_r1, adjB1, par)
    arcs_n = sum(cnt.values())
    # contiguity of subtrees for the ranks Order assigned (S5 checks this before relying on it)
    kids = defaultdict(list)
    for r, p in par.items():
        if p is not None:
            kids[p].append(r)

    def subtree(r):
        out = [r]
        for c in kids[r]:
            out += subtree(c)
        return out
    def noncontiguous(parent, n_dim, ranks):
        ks = defaultdict(list)
        for r_, p_ in parent.items():
            if p_ is not None:
                ks[p_].append(r_)

        def sub(r_):
            out = [r_]
            for c in ks[r_]:
                out += sub(c)
            return out
        bad_ = []
        for r_ in ranks:
            st_ = sorted(sub(r_))
            if st_ != list(range(st_[0], st_[0] + len(st_))):
                bad_.append(r_)
        return bad_
    order_ranks = range(1, n_r + 1)                      # the ranks Order assigned (and pool ranks)
    noncontig = noncontiguous(par, n_r1, order_ranks)
    # Why each subtree broke: the same tree over B′'s two halves alone
    par_B21 = liu_etree(n_r1, adj_of(B))                 # B, in the n_r′ dimension
    par_drop = liu_etree(n_r1, adj_of(B - Dset))         # only D's drop
    par_add = liu_etree(n_r1, adj_of(B | Aset))          # only A's additions
    by_add = [r for r in noncontig if r in noncontiguous(par_add, n_r1, order_ranks)]
    by_drop = [r for r in noncontig if r in noncontiguous(par_drop, n_r1, order_ranks)]
    assert set(by_add) | set(by_drop) == set(noncontig)
    moved_by_drop = [(r, par_B21[r], par_drop[r]) for r in range(1, n_r1 + 1) if par_B21[r] != par_drop[r]]
    new_low = [r for r in range(1, n_r + 1) if par_B21[r] is None and not adj_of(B).get(r) and par[r] is not None]
    dropped_roads = sorted({tuple(sorted((sg.node_of[a], sg.node_of[b]))) for (a, b) in Dset})
    top_new = list(range(n_r + 1, n_r1 + 1))
    why = []
    if by_add:
        why.append(f"{plural(len(by_add), 'subtree')} ({and_list(by_add)}) now contain"
                   f"{'s' if len(by_add) == 1 else ''} pool rank{'s' if len(new_low) > 1 else ''} "
                   f"{and_list(new_low)}, which node{'s' if len(new_low) > 1 else ''} "
                   f"{and_list(sg.node_of[r] for r in new_low)} took at this Supergraph and which hang"
                   f"{'s' if len(new_low) == 1 else ''} under rank{'s' if len(new_low) > 1 else ''} "
                   f"{and_list(par[r] for r in new_low)}")
    if by_drop:
        why.append(f"dropping road {fmt_pairs(dropped_roads)} moves "
                   + and_list(f"rank {r} from under {a} to under {b}" for r, a, b in moved_by_drop)
                   + f", which breaks {and_list(by_drop)}")
    both = sorted(set(by_add) & set(by_drop))
    why_txt = "; ".join(why)
    if both:
        why_txt += f" ({and_list(both)} for both reasons)"
    if top_new:
        why_txt += (f". Top rank{'s' if len(top_new) > 1 else ''} {and_list(top_new)} "
                    f"{'are' if len(top_new) > 1 else 'is'} new and outside the ranks Order assigned, so "
                    f"{'they are' if len(top_new) > 1 else 'it is'} not counted")
    step("C3", "C", f"Elimination tree over B′, counts: exactly {arcs_n} arcs (guard {guard})",
         "03 §4 step 2, §3 step 8, Guards (b)",
         f"B′ differs from B, so Liu's tree and the column counts are computed again. They give {arcs_n} arcs, "
         f"still under the guard. The postorder relabel is not redone; "
         + (f"{plural(len(noncontig), 'subtree')} of ranks 1–{n_r} now hold non-consecutive ranks. Of these, "
            f"{why_txt}. Only S5 (a later PR) relies on consecutive subtrees, and it checks first." if noncontig else
            "every subtree still has contiguous ranks."),
         [call("host: Liu's etree over B′", ["B′"], "parent"),
          call("host: column counts over B′ → exact arc count; check guard (b) again", ["B′", "parent"], "counts")],
         {"etree": [[r, par[r]] for r in range(1, n_r1 + 1)],
          "col_counts": [[r, cnt[r]] for r in range(1, n_r1 + 1)], "arcs": arcs_n, "guard": guard,
          "noncontiguous_subtrees": noncontig, "postorder_redone": False,
          "why_noncontiguous": {"contain_a_new_pool_rank": by_add, "broken_by_the_drop": by_drop,
                                "new_pool_ranks": [[r, sg.node_of[r], par[r]] for r in new_low],
                                "moved_by_the_drop": [[r, a, b] for r, a, b in moved_by_drop],
                                "dropped_roads": [list(p) for p in dropped_roads],
                                "top_ranks_not_counted": top_new}},
         draw(space="rank", edges={"etree": {(r, p) for r, p in par.items() if p is not None}}))

    # ---------------------------------------------------------------- C4 game
    U0 = defaultdict(set)
    for (a, b) in B1:
        if a < b:
            U0[a].add(b)
    up_rows, gpar, gsteps = game(n_r1, U0)
    for s_ in gsteps:
        s_["node"] = sg.node_of.get(s_["rank"])
    n_arcs = sum(len(v) for v in up_rows.values())
    assert n_arcs == arcs_n, (n_arcs, arcs_n)
    fill_total = sum(len(s_["fill"]) for s_ in gsteps)
    step("C4", "C", f"The elimination game: {n_arcs} arcs, {fill_total} of them fill",
         "03 §4 step 3",
         "Ranks are taken in increasing order. A rank's up-row is its own higher neighbours in B′ plus every "
         "child's up-row without itself. A child is a rank whose lowest up-entry is this rank. Entries that "
         "came from children and not from B′ are fill: shortcuts no edge supplies.",
         [call("U0 = triu(B′, 1)", ["B′"], "U0"),
          call("host: row(r) = U0(r,:) ∪ ⋃ children (row(c) ∖ {r}), in increasing r", ["U0"], "Up rows")],
         {"steps": gsteps, "arcs": n_arcs, "fill": fill_total},
         draw(space="rank", edges={"original": {(a, b) for (a, b) in B1 if a < b},
                                   "fill": {(s_["rank"], f) for s_ in gsteps for f in s_["fill"]}}))

    # ---------------------------------------------------------------- C5 CSR
    rows = {}
    k = 0
    for r in range(1, n_r1 + 1):
        rows[r] = []
        for hi in up_rows[r]:
            rows[r].append((hi, k))
            k += 1
    base1 = Base(n_r1, rows)
    sg.base = base1
    sg.arc_capacity = k
    sg.stats = CowDict("LineageStats", {"arcs_at_build": k, "deletions": 0, "top_ranks_used": n_r1 - n_r,
                                        "pool_row_arcs": 0})
    sg.invalidate()
    step("C5", "C", f"Up.base and Down.base: CSR arrays over {n_r1} rows, {k} arc ids",
         "03 §2, §4 step 4",
         f"Ids are assigned row-major, so arc id = Ap[row] + position. Down.base is the transpose and keeps the "
         f"same ids. Both deltas start empty; pool rank {and_list(sg.free)} has no entries.",
         [call("ids row-major into Ap, Aj, Ax", ["Up rows"], "Up.base"),
          call("Down.base = GrB_transpose of a read-only load of Up.base, unloaded into Rust-owned arrays",
               ["Up.base"], "Down.base")],
         {"base": base1.json(), **sg.json_maps(), "delta": {"Up": None, "Down": None}},
         draw(space="rank", edges={"arc": set(base1.arcs())}))

    # ---------------------------------------------------------------- the matrix track
    # The page's matrix track draws the construct frame by frame: S in node ids, B = S(π,π) with the
    # dissection's cells, Up over B′ with its fill, and Up.base as stored. Each frame here is computed
    # from the frame before it and from the steps' recorded outputs (the cells' pre-relabel ranks, q, the
    # log's pairs), not copied from C1–C5. G19 then checks the track against those steps.
    def track_order(use_q=True):
        """π from the dissection's records: peeled ranks, each separator's ranks and each leaf's
        elimination steps give the order's ranks; the postorder relabel q then renumbers them."""
        pre = dict(peeled)
        for d_ in depths:
            for s_ in d_["split"]["separators"]:
                pre.update({x: r for x, r in s_["separator_ranks"]})
        for lf in leaves:
            pre.update({st["node"]: st["rank"] for st in lf["steps"]})
        return {x: (q[r] if use_q else r) for x, r in pre.items()}

    def track_up(pi, with_log=True):
        """The one extract B = S(π,π), then B′ = (B ∖ D) ∪ A from the logged pairs, then the game."""
        Bt = {(pi[a], pi[b]) for (a, b) in S}
        pi1 = dict(pi)
        for a_ in alloc:
            pi1[a_["node"]] = a_["rank"]
        if with_log:
            add_ = {(pi1[u], pi1[v]) for (u, v) in sg_pairs if g7.has_edge(u, v)}
            drop = {(pi1[u], pi1[v]) for (u, v) in sg_pairs if not g7.has_edge(u, v) and u in pi1 and v in pi1}
            Bp = (Bt - {(b, a) for a, b in drop} - drop) | add_ | {(b, a) for a, b in add_}
        else:
            Bp = Bt
        U = defaultdict(set)
        for (a, b) in Bp:
            if a < b:
                U[a].add(b)
        rows_t, _, steps_t = game(n_r1, U)
        return Bt, Bp, rows_t, {(s_["rank"], f) for s_ in steps_t for f in s_["fill"]}

    def track_ids(rows_t, row_major=True):
        cells_ = sorted(((r, hi) for r in rows_t for hi in rows_t[r]), key=(lambda x: x) if row_major else
                        (lambda x: (x[1], x[0])))
        return {c: i for i, c in enumerate(cells_)}

    pi_t = track_order()
    B_t, Bp_t, up_t, fill_t = track_up(pi_t)
    ids_t = track_ids(up_t)
    cells_t = [{"name": "pool", "kind": "pool", "ranks": list(range(1, RESERVE + 1)), "depth": None,
                "parent": None, "nodes": []},
               {"name": "hanging tree", "kind": "peeled", "ranks": sorted(pi_t[x] for x in peeled), "depth": None,
                "parent": None, "nodes": sorted(peeled)}]
    tree_t = [{"name": "C" if len(comps) == 1 else f"C{i}", "parent": None,
               "ranks": sorted(pi_t[x] for x in c["verts"])} for i, c in enumerate(comps)]
    for rec_ in depths:
        for c in rec_["split"]["cells"]:
            tree_t.append({"name": c["name"], "parent": c["parent"], "ranks": sorted(pi_t[x] for x in c["verts"])})
    for t_ in tree_t:
        s_ = next((s_ for rec_ in depths for s_ in rec_["split"]["separators"] if s_["cell"] == t_["name"]), None)
        t_["separator"] = sorted(pi_t[x] for x in s_["separator"]) if s_ else []
        t_["leaf"] = s_ is None
    for d_, rec_ in enumerate(depths):
        for s_ in rec_["split"]["separators"]:
            cells_t.append({"name": f"separator of {s_['cell']}", "kind": "separator", "depth": d_,
                            "cell": s_["cell"], "parent": s_["cell"], "children": s_["children"],
                            "nodes": sorted(s_["separator"]), "ranks": sorted(pi_t[x] for x in s_["separator"])})
    for lf in leaves:
        cells_t.append({"name": lf["cell"], "kind": "leaf", "depth": lf["depth"], "parent": None,
                        "nodes": sorted(lf["verts"]), "ranks": sorted(pi_t[x] for x in lf["verts"])})
    track = {
        "S": {"n": NODE_CAP, "entries": pat(S), "masked_by_dm": pat(S_nomask - S),
              "slabs": [list(sl) for sl in SLABS]},
        "order": {"pi": [[x, r] for x, r in sorted(pi_t.items())],
                  "before_relabel": [[x, r] for x, r in sorted(track_order(False).items())],
                  "moved_by_relabel": [[r, q[r]] for r in range(1, n_r + 1) if q[r] != r]},
        "B": {"n": n_r, "entries": pat(B_t), "cells": cells_t, "tree": tree_t,
              "arcs_by_counts": arcs_B},
        "Up": {"n": n_r1, "B_prime": pat(Bp_t), "added": pat(Aset), "dropped": pat(Dset),
               "arcs": sorted([r, hi] for r in up_t for hi in up_t[r]), "fill": sorted([a, b] for a, b in fill_t),
               "new_ranks": [[a_["rank"], a_["node"], a_["kind"]] for a_ in alloc]},
        "base": {"ids": sorted([lo, hi, i] for (lo, hi), i in ids_t.items())},
    }

    # ---------------------------------------------------------------- D1 seed
    seedrec = full_seed_by_id(sg, g7)
    ev(actor="batch", name="Seed (V_c = V_sg = 7)", fork=7, fit="Same", why="no writer committed since the fork",
       consumed=[])
    pub["phase"] = "Relax"
    # check: the full seed equals per-pair seeds
    pairseed_ok = True
    for (lo, hi), i in sg.arcs().items():
        a, b = g7.pair_weights(sg.node_of.get(lo), sg.node_of.get(hi))
        if (seedrec["up"][i], seedrec["dn"][i]) != (a, b):
            pairseed_ok = False
    finite = sum(1 for x in seedrec["up"] if x != INF) + sum(1 for x in seedrec["dn"] if x != INF)
    step("D1", "D", f"The full-seed pass at v7: {finite} finite seeds over {k} arcs × 2 directions",
         "03 §5 The full-seed pass",
         f"Per type and slab, the live values are m without the dm and dp entries, plus dp; so the stale m value "
         f"under the replaced {H_PAIR[0]}→{H_PAIR[1]} is never read. The minimum across types gives W. Up seeds "
         f"come from triu(W(π,π)) and down seeds from triu(W(π,π)ᵀ), then each is written at the id in Ax.",
         [call("X_t = apply(weight, m_t⟨¬struct(dm_t) ∧ ¬struct(dp_t)⟩ ∪ dp_t), per type t and slab",
               ["m", "dm", "dp"], "X_t"),
          call("W = MIN over types of X_t", ["X_ROAD", "X_FERRY"], "W"),
          call("Wπ = W(π,π) over n_r′", ["W", "π"], "Wπ"),
          call("up = triu(Wπ);  down = triu(Wπᵀ)", ["Wπ"], "up, down"),
          call("eWiseAdd(MIN, up, iso +∞ pattern of Up.base)⟨struct(Up.base)⟩, then seeds[Ax[k]] = val[k]",
               ["up", "Up.base"], "seeds_up"),
          call("the same for down", ["down", "Up.base"], "seeds_down")],
         {**seedrec["record"], "equals_per_pair_seeds": pairseed_ok},
         draw(edges={"seeded": {pr(*k) for k in g7.W()}}))

    # ---------------------------------------------------------------- D2 relax
    flat = {"up": FlatArr(seedrec["up"]), "dn": FlatArr(seedrec["dn"]),
            "mu": FlatArr([None] * k), "md": FlatArr([None] * k)}
    sg.metric = flat
    tri = []
    ref7 = relax(sg, seedrec["up"], seedrec["dn"], tri)      # the sweep starts from D1's arrays
    for i, (a, b, c, d) in ref7.items():
        flat["up"].set(i, a)
        flat["dn"].set(i, b)
        flat["mu"].set(i, c)
        flat["md"].set(i, d)
    d2_ok = not diff_metric(ref7, full_customize(sg, g7))     # reference: seeds read pair by pair
    assert d2_ok
    # two relax batches by apex rank
    per_x = defaultdict(int)
    for t_ in tri:
        per_x[t_["x"]] += 1
    half, acc, cut = len(tri) / 2, 0, n_r1 + 1
    for x in range(1, n_r1 + 1):
        acc += per_x[x]
        if acc >= half:
            cut = x + 1
            break
    improved = sum(1 for t_ in tri if t_["up_won"] or t_["down_won"])
    step("D2", "D", f"Relax: {len(tri)} triangles in rank order, {improved} improve an arc",
         "03 §5 Phases (Relax); customize.rs relax",
         "For each rank x in increasing order and each pair y < z in up(x), the detours y→x→z and z→x→y are tried "
         "for the arc y–z. A strictly smaller sum is written, with x as the middle. Inputs below x are final when "
         "x is processed, so one pass is enough. The sweep starts from D1's seed arrays, by arc id; the result "
         "equals a full customization whose seeds are read pair by pair, bit for bit.",
         [call("host: start each arc at seeds_up[id], seeds_down[id] (D1); for x ascending, for y < z in up(x): "
               "up(y,z) = min(up(y,z), down(x,y) + up(x,z)); "
               "down(y,z) = min(down(y,z), down(x,z) + up(x,y))", ["seeds_up", "seeds_down"], "metric")],
         {"equals_full": d2_ok, "triangles": tri, "batches": [{"apex_ranks": [1, cut - 1], "triangles": sum(per_x[x] for x in range(1, cut))},
                                        {"apex_ranks": [cut, n_r1], "triangles": sum(per_x[x] for x in range(cut, n_r1 + 1))}],
          "metric_v7": {"up": [jw(ref7[i][0]) for i in range(k)], "down": [jw(ref7[i][1]) for i in range(k)],
                        "up_middle": [ref7[i][2] for i in range(k)], "down_middle": [ref7[i][3] for i in range(k)]}},
         draw(space="rank", edges={"arc": set(sg.arcs())}))
    ev(actor="batch", name=f"Relax apex ranks {1}–{cut - 1}", fork=7, fit="Same",
       why="no writer committed since the fork", consumed=[], note="metric frozen at V_c = 7")

    # writer v8 during the second relax batch: SET ROAD 5->6
    g8, pairs8, j8 = write(G[7], [("set", "ROAD", 5, 6, 1)])
    G[8] = g8
    n8 = log_node(8, pairs8)
    pub["log"] = [n8] + pub["log"]
    pub["version"] = 8
    ev(actor="writer", version=8, what="; ".join(j8), log_node=log_json([n8])[0],
       note="during Relax; the sweep is never patched mid-way")
    caught, above = relink_above([], pub["log"], 7)
    pub["log"] = caught
    ev(actor="batch", name=f"Relax apex ranks {cut}–{n_r1}", fork=7, fit="CatchUp",
       why="a writer (v8) committed since the fork", relinked=[n["seq"] for n in above], consumed=[])
    pub["phase"] = "Seal"

    # ---------------------------------------------------------------- D3 Seal
    seal = sg            # the builder's private hierarchy; the Seal works on its flat metric
    seal_log = list(pub["log"])
    seal_pairs = sorted({tuple(p) for n in seal_log for p in n["pairs"]})
    sealrec = flush(seal, g8, seal_pairs, 8, offlock=True)
    ref8 = full_customize(seal, g8)
    seal_ok = not diff_metric(seal.metric_by_id(), ref8)
    # chunk the metric (WeightArr::from_vec), one array at a time
    seal.metric = {kk: ChunkedArr.from_list(a.tolist(), "metric " + kk) for kk, a in seal.metric.items()}
    chunk_ids_seal = {kk: list(a.ids) for kk, a in seal.metric.items()}
    # writer v9 commits during the Seal
    g9, pairs9, j9 = write(G[8], [("del", "ROAD", 9, 10), ("del", "ROAD", 10, 9)])
    G[9] = g9
    n9 = log_node(9, pairs9)
    pub["log"] = [n9] + pub["log"]
    pub["version"] = 9
    ev(actor="writer", version=9, what="; ".join(j9), log_node=log_json([n9])[0], note="during the Seal")
    # Seal publish: CatchUp, relink seq 9 onto the Seal's consumed copy; then adoption
    caught, above = relink_above([], pub["log"], 8)
    built9 = seal.clone()
    adopt = flush(built9, g9, sorted({tuple(p) for n in caught for p in n["pairs"]}), 9, budget=B_GIL)
    assert not adopt["over_budget"], adopt["work"]
    ref9 = full_customize(built9, g9)
    adopt_ok = not diff_metric(built9.metric_by_id(), ref9)
    pub["log"] = []
    pub["state"] = "Built"
    pub["phase"] = None
    ev(actor="batch", name="Seal (next = None: adoption)", fork=8, fit="CatchUp",
       why="a writer (v9) committed since the fork", consumed=[n["seq"] for n in seal_log],
       relinked=[n["seq"] for n in above],
       absorbed={"seqs": [n["seq"] for n in above], "work": adopt["work"]["total"], "budget": B_GIL},
       note="the relinked node is absorbed under the lock within B_gil, so the entry becomes Built with no log")
    assert seal_ok and adopt_ok
    step("D3", "D", f"The Seal at v8: one logged SET; the scoped recompute pops "
                    f"{plural(len(sealrec['recompute']), 'arc')} and matches a full customization",
         "03 §5 Phases (Seal), §6b",
         f"The Seal forks at v8 and resolves the log node written during Relax: {j8[0]}. The pair already has an "
         f"arc, so only the metric changes. The front (the arcs to recompute first) is that arc with its new "
         f"seeds; an arc whose weight changes pushes the arcs above it. The result equals a full customization "
         f"at v8, bit for bit. At its publish, the v9 commit is absorbed the same way, under the lock, in "
         f"{adopt['work']['total']} of the {B_GIL} work units B_gil allows.",
         [call("host: resolve log ≤ 8 at v8; allocate; classify; recompute (03 §6b), off the lock",
               ["log", "graph@v8"], "metric"),
          call("WeightArr::from_vec, one array at a time", ["flat metric"], "chunked metric")],
         {"log": log_json(seal_log), "flush": sealrec, "equals_full": seal_ok,
          "chunks": chunk_ids_seal,
          "adoption": {"log": log_json(caught), "flush": adopt, "equals_full": adopt_ok}},
         draw(edges={"changed": {tuple(p) for p in seal_pairs}}))

    # ---------------------------------------------------------------- E timeline step
    step("E", "E", f"The batched build, v5 → v9: {sum(1 for e in events if e['actor'] == 'batch')} batches, "
                   f"{sum(1 for e in events if e['actor'] == 'writer')} writer commits",
         "03 §5",
         "Batches fork at the published version, work off the lock, and publish. A batch publishes Same when no "
         "writer committed since its fork, and CatchUp when one did; CatchUp relinks the published log nodes "
         "with a higher seq on top of the batch's copy. Supergraph and the Seal consume the nodes they resolve.",
         [],
         {"events": events, "swap": "never here: Swap needs another index's batch, and this example has one index",
          "generations": [{"generation": 1, "from": 5, "why": "the build"}]})

    # ---------------------------------------------------------------- F: after the build
    hist = {9: built9}          # published hierarchies by version
    SNAP["9"] = snapshot(built9)
    retention = []              # each writer's publish on a Built entry: its upkeep and its log node
    after = []                  # every F-part event, in order
    entry = {"state": "Built", "phase": None, "generation": gen[0], "upkeep": None, "log": []}
    decay = {}                  # version -> decay signals after its flush

    def entry_json():
        e = {"state": entry["state"], "generation": entry["generation"]}
        if entry["state"] == "Built":
            e["upkeep"] = dict(entry["upkeep"]) if entry["upkeep"] else None
        else:
            e["phase"] = entry["phase"]
        return e

    def aev(**kw):
        kw["entry"] = entry_json()
        kw["published_log"] = log_json(entry["log"])
        after.append(kw)

    def commit(v, ops, h_prev, *, budget=B_GIL):
        g_new, pairs, journal = write(G[v - 1], ops)
        G[v] = g_new
        h = h_prev.clone()
        rec = flush(h, g_new, pairs, v, budget=budget)
        return g_new, pairs, journal, h, rec

    def metric_copies(h_old, h_new):
        out = {}
        for kk in ("up", "dn", "mu", "md"):
            out[kk] = [[c, old, new] for (c, old, new) in h_new.metric[kk].copied]
        return out

    def newly_met(v, prev):
        """S6/S7 signals met after v's flush that were not met after prev's."""
        was = {s["signal"] for s in decay.get(prev, []) if s["met"]}
        return [s for s in decay[v] if s["met"] and s["job"] != "fold" and s["signal"] not in was]

    def decay_txt(sigs):
        if not sigs:
            return ""
        return (" Decay: " + and_list(f"{s['job']}'s '{s['signal']}' (value {s['value']}, threshold "
                                      f"{s['threshold']:g})" for s in sigs)
                + f" {'is' if len(sigs) == 1 else 'are'} met. The toy records "
                + ("it" if len(sigs) == 1 else "them") + " and runs neither S6 nor S7 (see the notes).")

    def publish_writer(v, pairs, journal, rec, h, h_prev):
        """A writer's publish on a Built entry. 03 §6 Built-entry jobs: the writer whose flush finds the
        fold trigger met, with no upkeep set, sets upkeep. 03 §7 Retention: with upkeep = Some{since},
        the log keeps nodes with seq > since; with upkeep = None, the log is dropped at publish."""
        new_arcs = len(rec.get("s2", {}).get("new_arcs", []))
        decay[v] = decay_signals(h, n_r, len(h_prev.arcs()), new_arcs)
        fold = next(s for s in decay[v] if s["job"] == "fold")
        node = log_node(v, pairs)
        outcome = [f"within B_gil ({rec['work']['total']} of {B_GIL} units)"]
        started = None
        if entry["upkeep"] is None and fold["met"]:
            gen[0] += 1
            entry["upkeep"] = {"kind": "Fold", "ticket": gen[0], "since": v}
            started = dict(entry["upkeep"])
            outcome.append(f"Up.delta holds {fold['value']} entries > √|base| = {fold['threshold']:.2f} and no "
                           f"upkeep is set, so it sets upkeep Fold, ticket {gen[0]}, since {v}")
        retention.append({"version": v, "upkeep": dict(entry["upkeep"]) if entry["upkeep"] else None,
                          "node": node})
        entry["log"], how = retain_log(entry["upkeep"], entry["log"], node)
        retention[-1]["how"] = how
        if how == "kept":
            outcome.append(f"upkeep is set, so the log keeps its node (seq {v} > since {entry['upkeep']['since']})")
        elif how == "not above since":
            outcome.append(f"its own node is not kept (seq {v} is not above since {entry['upkeep']['since']})")
        else:
            outcome.append("no upkeep, so its log node is dropped at publish")
        aev(actor="writer", version=v, what="; ".join(journal), work=rec["work"]["total"], budget=B_GIL,
            outcome="; ".join(outcome),
            decay_met=[f"{s['job']}: {s['signal']}" for s in decay[v] if s["met"]], col=str(v))
        hist[v] = h
        SNAP[str(v)] = snapshot(h)
        return started

    # F1: a weight SET
    i0 = len(after)
    g10, p10, j10, h10, r10 = commit(10, [("set", *F1_EDGE, 1)], built9)
    ref10 = full_customize(h10, g10)
    f1_ok = not diff_metric(h10.metric_by_id(), ref10)
    assert f1_ok
    assert not r10["over_budget"]
    publish_writer(10, p10, j10, r10, h10, built9)
    CTX.update(built9=built9, h10=h10, G=G, sp=sp)
    changed10 = [t_ for t_ in r10["recompute"] if t_["changed"]]
    step("F1", "F", f"A weight SET at v10: {len(r10['recompute'])} arcs popped, {len(changed10)} changed, "
                    f"{r10['work']['total']} work units",
         "03 §6 (metric only, S1), §6b",
         f"{j10[0]}. The pair has an arc, so the topology does not change. The front is that arc; each arc "
         f"whose weight changes pushes the arcs above it (the top arc of every triangle it is a bottom arc of). "
         f"The result equals a full customization bit for bit, in {r10['work']['total']} of {B_GIL} work units. "
         f"No upkeep job is set, so the writer's log node is dropped at publish (03 §7)."
         + decay_txt(newly_met(10, None)),
         [call("host: recustomize_seeded with front {arc of the pair}, heap key (lo, hi, id)", ["front"], "metric")],
         {"journal": j10, "flush": r10, "equals_full": f1_ok, "chunks_copied": metric_copies(built9, h10),
          "decay": decay[10], "events": after[i0:]},
         draw(edges={"set": {pr(F1_EDGE[1], F1_EDGE[2])}}))

    # F2: an edge insert between two nodes with no arc
    i0 = len(after)
    assert pr(h10.rank_of[F2_ROAD[0]], h10.rank_of[F2_ROAD[1]]) not in h10.arcs()
    g11, p11, j11, h11, r11 = commit(11, [("addroad", *F2_ROAD, 2)], h10)
    ref11 = full_customize(h11, g11)
    f2_ok = not diff_metric(h11.metric_by_id(), ref11)
    assert f2_ok
    assert not r11["over_budget"]
    publish_writer(11, p11, j11, r11, h11, h10)

    def s2_equals_game(h_old, h_new, rec_):
        rows_ = defaultdict(set)
        for (lo_, hi_) in h_old.arcs():
            rows_[lo_].add(hi_)
        for lo_, hi_ in rec_["classify"]["s2_insert"]:
            rows_[lo_].add(hi_)
        g_up_, _, _ = game(h_new.n, rows_)
        return {(r_, c_) for r_ in g_up_ for c_ in g_up_[r_]} == set(h_new.arcs())
    s2_game_ok = s2_equals_game(h10, h11, r11)
    fold11 = next(s for s in decay[11] if s["job"] == "fold")
    step("F2", "F", f"An edge insert at v11 on a pair with no arc: S2 adds {len(r11['s2']['new_arcs'])} arcs "
                    f"(ids {r11['s2']['new_arcs'][0][2]}–{r11['s2']['new_arcs'][-1][2]})",
         "03 §6 S2 (Davis–Hager), §6b Lemma A",
         f"Road {F2_ROAD[0]}–{F2_ROAD[1]} comes back: it was deleted at v6 and dropped at Supergraph, so its "
         f"ranks have no arc. S2 (fill repair) puts the pair in a host overlay and pushes each changed row into "
         f"its parent until nothing changes. The new arcs are sorted by (lo, hi), get ids from arc_capacity up, "
         f"go into both deltas, and start at +∞. The scoped recompute equals a full customization of the new "
         f"topology bit for bit, in {r11['work']['total']} of {B_GIL} work units. Up.delta now holds "
         f"{fold11['value']} entries, not over the fold trigger √|base| = {fold11['threshold']:.2f}."
         + decay_txt(newly_met(11, 10)),
         [call("host overlay: up(lo) += {hi}; pop rows by rank reading merge(base, delta, overlay); "
               "push U(j) ∖ {p} into up(p) until fixpoint", ["Up.base", "Up.delta"], "new arcs"),
          call("cow_btree insert (lo≪32 | hi) → id into Up.delta, (hi≪32 | lo) → id into Down.delta",
               ["new arcs"], "deltas"),
          call("metric append (+∞, +∞, none, none) per new id", [], "metric"),
          call("host: recustomize_seeded with front = inserted arc ∪ N", ["front"], "metric")],
         {"journal": j11, "flush": r11, "equals_full": f2_ok, "s2_equals_game": s2_game_ok,
          "delta": h11.json_delta(), "added_since_fold": [[s, [list(x) for x in a]] for s, a in h11.added],
          "arc_capacity": h11.arc_capacity, "decay": decay[11], "events": after[i0:]},
         draw(edges={"inserted": {F2_ROAD},
                     "new_arc": {pr(h11.node_of[lo], h11.node_of[hi]) for lo, hi, _ in r11["s2"]["new_arcs"]}}))

    # F3: new nodes: degree 1 -> pool rank; degree 2 -> top rank in the delta. The flush crosses the fold
    # trigger, so the writer sets upkeep and a fold job starts.
    i0 = len(after)
    g12, p12, j12, h12, r12 = commit(12, [("node", 21), ("addroad", 4, 21, 5), ("node", 22),
                                          ("addroad", 12, 22, 2), ("addroad", 15, 22, 7)], h11)
    ref12 = full_customize(h12, g12)
    f3_ok = not diff_metric(h12.metric_by_id(), ref12)
    assert f3_ok
    assert not r12["over_budget"]
    s2_game_ok12 = s2_equals_game(h11, h12, r12)
    fold12 = publish_writer(12, p12, j12, r12, h12, h11)
    assert fold12, "the v12 flush is expected to cross the fold trigger"
    aev(actor="job", name=f"Fold (ticket {fold12['ticket']})", fork=12,
        note="forks at v12 and builds base ⊕ delta off the lock; it is still running at v14")
    f12 = next(s for s in decay[12] if s["job"] == "fold")
    step("F3", "F", "New nodes at v12: " + ", ".join(
        f"node {a['node']} (degree {a['degree']}) → {a['kind']} rank {a['rank']}" for a in r12["allocation"])
        + f"; the fold trigger is crossed",
         "03 §6 Rank allocation, Decay, Built-entry jobs (D6)",
         f"Allocation runs before classification, in id order. A degree-1 node takes the lowest holderless "
         f"pool rank, which costs one arc and no fill. A node with more neighbours takes a new top rank "
         f"{h12.n}, above the base's {h12.base.n} rows, so its arcs live only in the deltas. S2 then adds "
         f"{len(r12['s2']['new_arcs'])} arcs, and the recompute matches a full customization. After the flush, "
         f"Up.delta holds {f12['value']} entries, over the fold trigger √|base| = {f12['threshold']:.2f}, and no "
         f"upkeep is set. So this writer sets upkeep: a fold job with ticket {fold12['ticket']} (tickets come "
         f"from the generation counter), which forks at v12. Its own log node is not kept, because the log "
         f"keeps only nodes with seq above since = {fold12['since']}."
         + decay_txt(newly_met(12, 11)),
         [call("host: allocate (pool if degree ≤ 1 and free_ranks not empty, else top), then classify",
               ["resolved pairs"], "ranks"),
          call("S2 and recompute as in F2", ["front"], "metric"),
          call("decay check: delta entries > max(64k, √|base|) → upkeep = Some{Fold, ticket, since = 12}",
               ["Up.delta", "Up.base"], "upkeep")],
         {"journal": j12, "flush": r12, "equals_full": f3_ok, "s2_equals_game": s2_game_ok12,
          "maps": h12.json_maps(), "delta": h12.json_delta(), "decay": decay[12], "upkeep": fold12,
          "events": after[i0:]},
         draw(nodes={"pool": {a["node"] for a in r12["allocation"] if a["kind"] == "pool"},
                     "top": {a["node"] for a in r12["allocation"] if a["kind"] == "top"}},
              edges={"new_arc": {pr(h12.node_of[lo], h12.node_of[hi]) for lo, hi, _ in r12["s2"]["new_arcs"]}},
              labels={a["node"]: f"r{a['rank']}" for a in r12["allocation"]}))

    # F4: node delete; upkeep is set, so the writer's log node is retained
    i0 = len(after)
    g13, p13, j13, h13, r13 = commit(13, [("delnode", F4_NODE)], h12)
    h13.stats["deletions"] = h13.stats.get("deletions", 0) + len(p13)
    ref13 = full_customize(h13, g13)
    f4_ok = not diff_metric(h13.metric_by_id(), ref13)
    assert f4_ok
    assert not r13["over_budget"]
    publish_writer(13, p13, j13, r13, h13, h12)
    assert [n["seq"] for n in entry["log"]] == [13]
    rr = h13.rank_of[F4_NODE]
    at_rank = [[lo, hi, i, jw(h13.m("up", i)), jw(h13.m("dn", i))] for (lo, hi), i in sorted(h13.arcs().items())
               if rr in (lo, hi)]
    all_inf = all(x[3] is None and x[4] is None for x in at_rank)
    assert all_inf
    step("F4", "F", f"Node {F4_NODE} deleted at v13: {plural(len(p13), 'pair')} cascade; its rank {rr} and "
                    f"{len(at_rank)} arcs stay, all now +∞",
         "03 §6 Node delete, §7 Retention",
         f"The cascade deletes {plural(sum(1 for x in j13 if x.startswith('cascade: delete')), 'relationship')} "
         f"and logs one (min, max) pair per neighbour: both directions of a road give the same pair, so they "
         f"log {plural(len(p13), 'pair')}. The node keeps rank {rr} and every arc; the front is the "
         f"arcs of its pairs, which reseed to +∞, and every other arc at its rank reaches +∞ through the "
         f"triangles. It matches a full customization bit for bit. The fold from v12 is still running, so "
         f"upkeep is set and this writer keeps its log node: the published log is [seq 13]."
         + decay_txt(newly_met(13, 12)),
         [call("host: recompute with front = arcs of the cascaded pairs", ["front"], "metric")],
         {"journal": j13, "flush": r13, "equals_full": f4_ok, "arcs_at_rank": at_rank, "all_inf": all_inf,
          "decay": decay[13], "events": after[i0:]},
         draw(nodes={"deleted": {F4_NODE}}, edges={"cascade": set(p13)}))

    # F5: a commit over budget -> stop at the pop that passes B_gil -> hand-off (new generation; the
    # in-flight fold is Gone) -> Repair (full path) -> adoption
    i0 = len(after)
    g14, p14, j14 = write(G[13], [("addroad", *F5_ROAD, 3)])
    G[14] = g14
    attempt = h13.clone()
    r14 = flush(attempt, g14, p14, 14, budget=B_GIL)
    assert r14["over_budget"] and r14.get("stopped", {}).get("in") == "recompute", r14["work"]
    r14_full = flush(h13.clone(), g14, p14, 14, record=False)    # finishing it: never run under the lock
    assert r14_full["work"]["total"] > B_GIL
    stop_pop, stop_work = r14["stopped"]["pop"], r14["stopped"]["work"]
    decay["14"] = decay_signals(attempt, n_r, len(h13.arcs()), len(r14["s2"]["new_arcs"]))
    gen[0] += 1
    handoff_gen = gen[0]
    old_gen = entry["generation"]
    gone = entry["upkeep"]
    log14 = [log_node(14, p14)]
    entry.update(state="Building", phase="Repair", generation=handoff_gen, upkeep=None, log=log14)
    aev(actor="writer", version=14, what="; ".join(j14), work=stop_work, budget=B_GIL,
        stopped_at_pop=stop_pop, full_cost={"pops": r14_full["work"]["pops"], "work": r14_full["work"]["total"]},
        outcome=f"the running total passes B_gil at pop {stop_pop} of the recompute ({stop_work} units): the "
                f"writer drops its attempt and hands off to Building{{Repair}} under generation {handoff_gen}",
        decay_met=[f"{s['job']}: {s['signal']}" for s in decay["14"] if s["met"]], col="14w")
    SNAP["14w"] = snapshot(h13)     # Building{Repair, base}: base is the hierarchy at the abort, v13's
    assert gone, "the fold from v12 is expected to be in flight at v14"
    aev(actor="job", name=f"Fold (ticket {gone['ticket']})", fork=gone["since"], fit="Gone",
        why=f"the v14 hand-off took generation {handoff_gen} (it was {old_gen})",
        outcome="discarded; it publishes nothing")
    # the Repair forks at v14
    rep = h13.clone()
    rrep = flush(rep, g14, sorted({tuple(p) for n in log14 for p in n["pairs"]}), 14, offlock=True)
    assert rrep["path"] == "full"
    ref14 = full_customize(rep, g14)
    f5_ok = not diff_metric(rep.metric_by_id(), ref14)
    assert f5_ok
    full14 = full_seed_by_id(rep, g14)
    seeds14_ok = (full14["up"], full14["dn"]) == pair_seeds(rep, g14)
    assert seeds14_ok
    s2_game_ok14 = s2_equals_game(h13, rep, rrep)
    hist[14] = rep
    entry.update(state="Built", phase=None, upkeep=None, log=[])
    aev(actor="job", name="Repair", fork=14, fit="Same", why="no writer committed since the fork", consumed=[14],
        path=rrep["path"], outcome=f"adopts: Built under generation {handoff_gen}, with no upkeep and no log",
        col="14")
    SNAP["14"] = snapshot(rep)
    top_sep = sorted(sp["separators"][0]["separator"])
    step("F5", "F", f"An insert across the top separator at v14 passes B_gil {B_GIL} at pop {stop_pop}: "
                    f"hand-off, the fold is Gone, Repair, adoption",
         "03 §6 Hand-off, §5 Phases (Repair), §6b valves and lock budget",
         f"{'; '.join(j14)}. Nodes {F5_ROAD[0]} and {F5_ROAD[1]} lie on opposite sides of the top separator, so "
         f"S2 adds {len(r14['s2']['new_arcs'])} arcs ({r14['work']['s2']} work units) and the cone (the arcs a "
         f"change reaches) is large. The writer checks its running total after every pop. At pop {stop_pop} of "
         f"the recompute it reaches {stop_work} units, over B_gil = {B_GIL}, so it stops there; finishing would "
         f"have taken {r14_full['work']['pops']} pops and {r14_full['work']['total']} units. The writer drops "
         f"its attempt and publishes Building{{Repair}} under a new generation {handoff_gen}; readers at v14 "
         f"use the Dijkstra fallback. The new generation also discards the fold that v12 started (ticket "
         f"{gone['ticket']}) as Gone, and the log node v13 had kept is dropped: the new entry's log is [seq 14] "
         f"alone. The Repair runs off "
         f"the lock. Its front of {plural(len(rrep['front']), 'arc')} is over n/{VALVE_DIV} = "
         f"{rep.n / VALVE_DIV:g}, so the valve sends it down the full path: the full-seed pass writes every seed by "
         f"arc id, from Ax for the base and from the entries for the delta, then one relax sweep starts from "
         f"those seeds. It publishes Same and adopts. The seeds equal the per-pair seeds, and the metric "
         f"equals a full customization."
         + decay_txt(newly_met("14", 13)),
         [call("writer: running total > B_gil after a pop → abort; entry = Building{Repair, base = hierarchy at "
               "the abort, log = [seq 14]} under next_generation()", ["attempt"], "entry"),
          call("the new generation discards the in-flight fold (ticket mismatch → Gone)", ["upkeep"], "—"),
          call("Repair: resolve log ≤ 14 at v14; allocate; S2 (off the lock)", ["base", "log"], "topology"),
          call("valve: front > n/8 → full path: seeds[Ax[k]] for the base, seeds[id] for each delta entry; "
               "then relax from those arrays", ["graph@v14", "Up.base", "Up.delta"], "metric")],
         {"journal": j14,
          "attempt": {"work": r14["work"], "stopped": r14["stopped"], "new_arcs": len(r14["s2"]["new_arcs"]),
                      "recompute_until_stop": r14["recompute"],
                      "full_cost": {"pops": r14_full["work"]["pops"], "work": r14_full["work"]}},
          "generation": handoff_gen, "fold_discarded": gone, "top_separator": top_sep, "repair": rrep,
          "equals_full": f5_ok, "full_seed_equals_per_pair": seeds14_ok, "s2_equals_game": s2_game_ok14,
          "decay": {"attempt": decay["14"]}, "events": after[i0:], "delta": rep.json_delta()},
         draw(nodes={"separator": set(top_sep)}, edges={"inserted": {F5_ROAD},
                                                       "new_arc": {pr(rep.node_of[lo], rep.node_of[hi])
                                                                   for lo, hi, _ in rrep["s2"]["new_arcs"]}}))

    # F6: the next writer finds the fold trigger met with no upkeep and sets it again; the fold runs
    i0 = len(after)
    g15, p15, j15, h15, r15 = commit(15, [("set", *F15_SET)], rep)
    assert not r15["over_budget"]
    fold15 = publish_writer(15, p15, j15, r15, h15, rep)
    assert fold15, "the v15 writer is expected to set upkeep again"
    ticket = fold15["ticket"]
    trig15 = next(s for s in decay[15] if s["job"] == "fold")
    assert entry["log"] == []
    F_fork = 15
    fold_src = h15.clone()             # the job's fork
    aev(actor="job", name=f"Fold (ticket {ticket})", fork=F_fork, note="builds base ⊕ delta off the lock")
    # writer v16 inserts while the fold runs: S2 arcs added after F
    g16, p16, j16, h16, r16 = commit(16, [("addroad", *F6_ROAD, 4)], h15)
    assert not r16["over_budget"] and r16.get("s2")
    publish_writer(16, p16, j16, r16, h16, h15)
    assert [n["seq"] for n in entry["log"]] == [16]
    ref16 = full_customize(h16, g16)
    v16_ok = not diff_metric(h16.metric_by_id(), ref16)
    s2_game_ok16 = s2_equals_game(h15, h16, r16)
    # the fold builds new bases from base ⊕ delta_F
    rows_f = defaultdict(list)
    for r in range(1, fold_src.n + 1):
        rows_f[r] = list(fold_src.up(r))
    newbase = Base(fold_src.n, rows_f)
    # publish under the lock: fitted against generation and ticket
    assert entry["generation"] == handoff_gen and entry["upkeep"]["ticket"] == ticket
    folded = h16.clone()               # metric, maps, arc_capacity and stats from the published copy
    folded.base = newbase
    folded.dup = folded.ddn = None
    keep = [(s, a) for s, a in h16.added if s > F_fork]
    for s, a in keep:
        for lo, hi, i in a:
            folded.dup, _ = bt_insert(folded.dup, key(lo, hi), i)
            folded.ddn, _ = bt_insert(folded.ddn, key(hi, lo), i)
    folded.added = CowList("added_since_fold", keep)      # rebuilt: the kept groups, relinked
    folded.invalidate()
    hist["16f"] = folded
    fold_ids_same = folded.arcs() == h16.arcs()
    fold_metric_carried = folded.metric_by_id() == h16.metric_by_id()
    fold_metric_ok = not diff_metric(folded.metric_by_id(), full_customize(folded, g16))
    assert fold_ids_same and fold_metric_carried and fold_metric_ok
    entry.update(upkeep=None, log=[])
    aev(actor="job", name=f"Fold publish (ticket {ticket})", fork=F_fork,
        fit=f"generation {handoff_gen} and ticket {ticket} unchanged",
        outcome="replace the bases, the deltas and added_since_fold; clear upkeep and the log", col="16f")
    SNAP["16f"] = snapshot(folded)
    step("F6", "F", f"A fold: base ⊕ delta → new base with {len(newbase.Aj)} arcs; the new delta holds "
                    f"{len(bt_items(folded.dup))} arcs added after the fork",
         "03 §6 Fold job, Built-entry jobs (D6), §7 Retention",
         f"At v15 a writer commits: {'; '.join(j15)}. After its flush the delta holds {trig15['value']} entries, "
         f"over √|base| = "
         f"{trig15['threshold']:.2f}, and no upkeep is set: the v14 hand-off discarded the fold v12 started. So "
         f"this writer sets upkeep again, a fold with ticket {ticket}, since = 15. Its own log node is not kept "
         f"(the log keeps nodes with seq above since). The job forks at v15 and builds new bases from base ⊕ "
         f"delta with the ids carried as values. A writer at v16 adds {len(r16['s2']['new_arcs'])} arcs "
         f"meanwhile; upkeep is set, so its log node is kept. At publish, generation {handoff_gen} and ticket "
         f"{ticket} are unchanged, so the result is the published v16 hierarchy with only the bases and deltas "
         f"replaced; the new deltas hold exactly the arcs added after v15. Every arc keeps its id. The metric "
         f"is the published copy's, and it equals a full customization of the folded topology.",
         [call("eWiseAdd over read-only loads of Up.base and a matrix of Up.delta at F, then unload", ["Up.base", "Up.delta@F"],
               "new Up.base"),
          call("the same for Down", ["Down.base", "Down.delta@F"], "new Down.base"),
          call("under the lock: fit against generation and ticket; new deltas = added_since_fold entries with "
               "seq > F", ["added_since_fold"], "deltas")],
         {"trigger": {"delta_entries": trig15["value"], "threshold": trig15["threshold"], "fires": trig15["met"]},
          "upkeep": fold15, "fork": F_fork, "v15": {"journal": j15, "flush": r15, "decay": decay[15]},
          "v16": {"journal": j16, "flush": r16, "equals_full": v16_ok, "s2_equals_game": s2_game_ok16,
                  "decay": decay[16]},
          "old_base": h16.base.json(), "new_base": newbase.json(), "new_delta": folded.json_delta(),
          "added_since_fold_after": [[s, [list(x) for x in a]] for s, a in folded.added],
          "ids_unchanged": fold_ids_same, "metric_carried": fold_metric_carried,
          "metric_equals_full": fold_metric_ok, "events": after[i0:]},
         draw(edges={"new_arc_after_fork": {pr(h16.node_of[lo], h16.node_of[hi]) for lo, hi, _ in r16["s2"]["new_arcs"]}}))

    # F7: MVCC: v15 and v16 share one base, each with its own delta
    p15_pages = bt_pages(h15.dup) | bt_pages(h15.ddn)
    p16_pages = bt_pages(h16.dup) | bt_pages(h16.ddn)
    shared = sorted(p15_pages & p16_pages)
    copied = sorted(p16_pages - p15_pages)

    def page_split(root_old, root_new, keys):
        made = bt_pages(root_new) - bt_pages(root_old)
        path = set()
        for kk in keys:
            path |= set(bt_path(root_new, kk))
        assert path <= made, "every page on an inserted key's path is a fresh copy"
        off = made - path
        assert all(PAGES[p][0] == "leaf" for p in off), "an off-path new page is a split-off leaf"
        return sorted(path), sorted(off)
    up_path, up_split = page_split(h15.dup, h16.dup, [key(lo, hi) for lo, hi, _ in r16["s2"]["new_arcs"]])
    dn_path, dn_split = page_split(h15.ddn, h16.ddn, [key(hi, lo) for lo, hi, _ in r16["s2"]["new_arcs"]])
    assert sorted(up_path + up_split + dn_path + dn_split) == copied
    chunk_copies = {kk: [[c, old, new] for (c, old, new) in h16.metric[kk].copied] for kk in h16.metric}

    def split_txt(n):
        return "no leaf split" if n == 0 else f"{plural(n, 'new leaf', 'new leaves')} where an insert split a full leaf"
    step("F7", "F", f"MVCC: v15 and v16 share {h15.base.name}; v16's writer copied {len(copied)} delta pages "
                    f"and shares {len(shared)} with v15",
         "03 §6b MVCC, §7",
         f"Every write goes to a private version. The base is never written, so both versions read the same "
         f"{h15.base.name}. The v16 writer's {plural(len(r16['s2']['new_arcs']), 'insert')} per delta "
         f"path-copied the pages from the root to the leaf each key lands in: {len(up_path)} pages in Up.delta "
         f"and {len(dn_path)} in Down.delta. Each delta also has {split_txt(len(up_split))}"
         + ("" if len(up_split) == len(dn_split) else f" in Up.delta and {split_txt(len(dn_split))} in Down.delta")
         + f". The other {len(shared)} pages are shared with v15. Its metric writes copied only the chunks "
           f"they touched.",
         [call("cow_btree insert: path-copy root → leaf; a full leaf splits into two new leaves", ["Up.delta@15"],
               "Up.delta@16"),
          call("WeightArr::set copies its chunk", ["metric@15"], "metric@16")],
         {"base_shared": h15.base is h16.base, "base": h15.base.name,
          "v15": {"delta": h15.json_delta(), "chunks": {kk: list(a.ids) for kk, a in h15.metric.items()}},
          "v16": {"delta": h16.json_delta(), "chunks": {kk: list(a.ids) for kk, a in h16.metric.items()}},
          "pages_shared": shared, "pages_copied_by_v16": copied,
          "copied_on_paths": {"Up": up_path, "Down": dn_path}, "copied_split_leaves": {"Up": up_split, "Down": dn_split},
          "chunks_copied_by_v16": chunk_copies})

    # F8: a reused id keeps its old rank and arcs; its road lands on a fill arc, so it is metric only (S1)
    i0 = len(after)
    x7 = F8_ROAD[0]
    r7 = folded.rank_of[x7]
    assert x7 not in G[16].nodes and x7 in G[16].free_ids and folded.node_of[r7] == x7
    fill_arcs = {(s_["rank"], f) for s_ in gsteps for f in s_["fill"]}
    target = pr(r7, folded.rank_of[F8_ROAD[1]])
    assert target in folded.arcs() and target in fill_arcs, target
    # each row: lo, hi, id, the node at the arc's other end, up, down, up_middle, down_middle
    arcs7_before = [[lo, hi, i, folded.node_of.get(hi if lo == r7 else lo), jw(folded.m("up", i)),
                     jw(folded.m("dn", i)), folded.metric["mu"].get(i), folded.metric["md"].get(i)]
                    for (lo, hi), i in sorted(folded.arcs().items()) if r7 in (lo, hi)]
    g17, p17, j17, h17, r17 = commit(17, [("node", x7), ("addroad", *F8_ROAD, road_w(*F8_ROAD))], folded)
    assert not r17["over_budget"] and "s2" not in r17 and r17["classify"]["metric_only"] == [list(target)]
    assert not r17["allocation"]
    ref17 = full_customize(h17, g17)
    f8_ok = not diff_metric(h17.metric_by_id(), ref17)
    assert f8_ok
    publish_writer(17, p17, j17, r17, h17, folded)
    arcs7_after = [[lo, hi, i, h17.node_of.get(hi if lo == r7 else lo), jw(h17.m("up", i)), jw(h17.m("dn", i)),
                    h17.metric["mu"].get(i), h17.metric["md"].get(i)]
                   for (lo, hi), i in sorted(h17.arcs().items()) if r7 in (lo, hi)]
    fill_src = next(s_ for s_ in gsteps if s_["rank"] == target[0])
    step("F8", "F", f"v17: a new node reuses id {x7} and keeps rank {r7} and its {len(arcs7_before)} arcs; "
                    f"road {F8_ROAD[0]}–{F8_ROAD[1]} lands on a fill arc, so it is metric only",
         "03 §6 table: 'Node created, or id reused', 'insert on a pair with an arc'",
         f"{'; '.join(j17)}. A freed id is handed out again, lowest first, so the new node gets id {x7}. Node "
         f"{x7} was deleted at v13, but rank {r7} and its {len(arcs7_before)} arcs stayed, all at +∞, and the "
         f"fold carried them; the new node takes them over. Its first edge is road {F8_ROAD[0]}–{F8_ROAD[1]}. "
         f"Ranks {target[0]} and {target[1]} already have arc {folded.arcs()[target]}, a fill arc the game made "
         f"at rank {fill_src['rank']} (C4)"
         + (f": road {DM_ROAD[0]}–{DM_ROAD[1]} was a pending deletion at v5, so the skeleton read masked it out "
            f"(B1), and the arc came from elimination instead" if pr(*F8_ROAD) == pr(*DM_ROAD) else "")
         + f". So allocation gives no rank, S2 "
         f"adds nothing, and the change is metric only (S1): the front is that arc with its new seed. The result "
         f"equals a full customization bit for bit, in {r17['work']['total']} of {B_GIL} work units."
         + decay_txt(newly_met(17, 16)),
         [call("create node: the lowest free id (id_space.rs:75); rank_of[7] is still 12", ["free ids"], "node 7"),
          call("host: resolve; allocate (nothing: 7 and 2 have ranks); classify → metric only", ["pairs"], "front"),
          call("host: recustomize_seeded with front {arc of the pair}", ["front"], "metric")],
         {"journal": j17, "flush": r17, "equals_full": f8_ok, "rank": r7, "fill_arc": list(target),
          "arc_id": folded.arcs()[target], "arcs_at_rank_before": arcs7_before, "arcs_at_rank_after": arcs7_after,
          "decay": decay[17], "events": after[i0:]},
         draw(nodes={"reused": {x7}}, edges={"inserted": {pr(*F8_ROAD)}}, labels={x7: f"r{r7}"}))

    # ================================================================ MVCC view: what each version holds
    # Columns are the published states after the build, in order. A job's publish keeps the data version
    # (mvcc_graph.rs:304), so v14 and v16 each have two states. "pred" is the state the column's hierarchy
    # was cloned from; v14's Repair forks from the hand-off's Building entry, whose base is v13's hierarchy.
    COLS = [("9", 9, None, "the Seal's adoption"), ("10", 10, "9", "a writer"), ("11", 11, "10", "a writer"),
            ("12", 12, "11", "a writer"), ("13", 13, "12", "a writer"), ("14w", 14, "13", "a writer: hand-off"),
            ("14", 14, "14w", "the Repair job"), ("15", 15, "14", "a writer"), ("16", 16, "15", "a writer"),
            ("16f", 16, "16", "the fold job"), ("17", 17, "16f", "a writer")]
    LABEL = {"9": "v9", "10": "v10", "11": "v11", "12": "v12", "13": "v13", "14w": "v14 hand-off",
             "14": "v14 Repair", "15": "v15", "16": "v16", "16f": "v16 fold", "17": "v17"}
    ev_of = {e["col"]: e for e in after if e.get("col")}
    entry_of = {"9": {"state": "Built", "generation": 1, "upkeep": None}}
    log_of = {"9": []}
    for kcol, e in ev_of.items():
        entry_of[kcol], log_of[kcol] = e["entry"], e["published_log"]
    ret_how = {r_["version"]: r_ for r_ in retention}

    def log_why(kcol):
        """Why a state's log holds what it holds, from who published it (03 §5, §7 Retention)."""
        seqs = ", ".join(f"seq {n_['seq']}" for n_ in log_of[kcol])
        if kcol == "9":
            return "empty: the Seal consumed its log and the adoption absorbed the relinked node"
        if kcol == "14w":
            return f"kept: {seqs}; a Building entry keeps its log for the Repair"
        if kcol == "14":
            return "empty: the Repair consumed " + ", ".join(f"seq {q_}" for q_ in ev_of["14"]["consumed"])
        if kcol == "16f":
            return "empty: cleared by the fold's publish"
        r_ = ret_how[int(kcol)]
        if r_["how"] == "kept":
            return f"kept: {seqs} (upkeep set, seq > since)"
        if r_["how"] == "not above since":
            return "empty: its own node is not above since" if not log_of[kcol] else f"kept: {seqs}"
        return "empty: dropped at publish (no upkeep)"
    TOPO, CCHI = {}, {}
    REBUILT = {"14"} if rrep.get("metric_rebuilt") else set()     # full-path Repair: metric from from_vec
    for kcol, v, pred, by in COLS:
        cur, prev = SNAP[kcol], (SNAP[pred] if pred else None)
        tch = [kk for kk in TOPOLOGY_FIELDS if prev is None or cur[kk]["iid"] != prev[kk]["iid"]]
        if prev is not None and cur["scalars"] != prev["scalars"]:
            tch.append("scalars")
        mch = [kk for kk in METRIC_FIELDS if prev is None or cur[kk]["iid"] != prev[kk]["iid"]]
        TOPO[kcol] = new_iid("TopologyVersion") if (prev is None or tch) else TOPO[pred]
        CCHI[kcol] = new_iid("Cch") if (prev is None or tch or mch) else CCHI[pred]

    def chunk_diff(cur_chunks, prev_chunks):
        """(index, 'copied' | 'added') for each chunk index whose id is new in cur."""
        prev_at = dict(prev_chunks)
        return [[c, "copied" if c in prev_at else "added"] for c, cid in cur_chunks if prev_at.get(c) != cid]

    def entry_keys(entries):
        return {key(r, c) for r, c, _ in entries}

    def delta_split(kcol, pred, kind):
        cur, prev = SNAP[kcol][kind], SNAP[pred][kind]
        keys = sorted(entry_keys(cur["entries"]) - entry_keys(prev["entries"]))
        pg_new = {p["page"] for p in cur["contents"]["pages"]}
        pg_old = {p["page"] for p in prev["contents"]["pages"]}
        return delta_page_split(pg_old, pg_new, cur["iid"], PAGES, keys, prev["iid"]), keys

    def how_text(kind, kcol, pred):
        """What a new instance copied or rebuilt, against the state it was cloned from."""
        cur = SNAP[kcol][kind]
        if pred is None:
            return {"Up.base": "built at Supergraph (v7)", "Down.base": "the transpose of Up.base, same ids",
                    "Up.delta": "empty", "Down.delta": "empty"}.get(kind, "built by the batched build")
        prev = SNAP[pred][kind]
        if kind in ("Up.base", "Down.base"):
            return f"{cur['iid']}: built by the fold from {prev['iid']} ⊕ the delta at its fork"
        if kind in ("Up.delta", "Down.delta"):
            if kcol == "16f":
                return (f"rebuilt by the fold: {plural(len(cur['entries']), 'entry', 'entries')} in "
                        f"{plural(len(cur['contents']['pages']), 'new page')}")
            sp_, ks = delta_split(kcol, pred, kind)
            out = (f"{plural(len(ks), 'insert')}: {plural(len(sp_['made']), 'new page')}, "
                   f"{len(sp_['path'])} on the insert paths")
            if sp_["split"]:
                out += f" and {len(sp_['split'])} split {'half' if len(sp_['split']) == 1 else 'halves'}"
            if prev["iid"] is None:
                out += " (the tree was empty)"
            return out
        if kind in METRIC_FIELDS and kcol in REBUILT:
            return (f"rebuilt from flat arrays (WeightArr::from_vec): all {len(cur['chunks'])} chunks new, "
                    f"none shared with the base")
        if kind in ("rank_of", "node_of") or kind in METRIC_FIELDS:
            if kind in METRIC_FIELDS:      # a metric array's chunk table: chunk index -> chunk id
                cc, pc = list(enumerate(cur["chunks"])), list(enumerate(prev["chunks"]))
            else:                          # a rank map: [[chunk index, chunk id]]
                cc, pc = cur["chunks"], prev["chunks"]
            parts = defaultdict(list)
            for c, how in chunk_diff(cc, pc):
                parts[how].append(c)
            return "; ".join(f"{'chunk' if len(cs) == 1 else 'chunks'} {and_list(cs)} {how}"
                             for how, cs in sorted(parts.items(), key=lambda x: x[0] != "copied"))
        if kind == "free_ranks":
            taken = sorted(set(prev["contents"]) - set(cur["contents"]))
            back = sorted(set(cur["contents"]) - set(prev["contents"]))
            return "copied: " + "; ".join(([f"rank {and_list(taken)} taken"] if taken else []) +
                                          ([f"rank {and_list(back)} returned"] if back else []))
        if kind == "LineageStats":
            ch = [f"{kk.replace('_', ' ')} {prev['contents'].get(kk, 0)} → {vv}"
                  for kk, vv in cur["contents"].items() if prev["contents"].get(kk) != vv]
            return "copied: " + "; ".join(ch)
        if kind == "added_since_fold":
            if kcol == "16f":
                return "rebuilt by the fold: keeps " + and_list(
                    f"seq {q_} ({plural(len(a_), 'arc')})" for q_, a_ in cur["contents"])
            old = {q_ for q_, _ in prev["contents"]}
            return "new head: " + and_list(f"seq {q_} appended ({plural(len(a_), 'arc')})"
                                           for q_, a_ in cur["contents"] if q_ not in old)
        return "new"

    # display labels: bases by name; every other row numbers its instances in order of first appearance
    ordinals = defaultdict(dict)

    def label_of(kind, iid):
        if kind in ("Up.base", "Down.base"):
            return iid
        if kind in ("Up.delta", "Down.delta") and iid is None:
            return "empty"
        o = ordinals[kind]
        if iid not in o:
            o[iid] = len(o) + 1
        return f"#{o[iid]}"

    instances = defaultdict(dict)
    pages_out, mchunks_out, map_chunks_out = {}, {}, {}
    columns = []
    for kcol, v, pred, by in COLS:
        s_ = SNAP[kcol]
        objs = {}
        for kind in TOPOLOGY_FIELDS + METRIC_FIELDS:
            iid = s_[kind]["iid"]
            new = pred is None or iid != SNAP[pred][kind]["iid"]
            objs[kind] = {"iid": iid, "label": label_of(kind, iid), "new": new,
                          "how": how_text(kind, kcol, pred) if new else None}
            if new and pred and kind in ("Up.delta", "Down.delta") and kcol != "16f":
                sp_, ks_ = delta_split(kcol, pred, kind)
                objs[kind]["pages"] = {"new": len(sp_["made"]), "path": len(sp_["path"]),
                                       "split": len(sp_["split"]), "inserts": len(ks_),
                                       "tree_was_empty": SNAP[pred][kind]["iid"] is None}
            body = {k_: v_ for k_, v_ in s_[kind].items() if k_ not in ("iid", "chunk_values")}
            if kind in ("Up.delta", "Down.delta"):
                body = {"root": iid, "pages": [p["page"] for p in s_[kind]["contents"]["pages"]],
                        "entries": s_[kind]["entries"]}
                for p in s_[kind]["contents"]["pages"]:
                    pages_out[str(p["page"])] = p
            if kind in METRIC_FIELDS:
                mchunks_out.update(s_[kind]["chunk_values"])
            if kind in ("rank_of", "node_of"):
                map_chunks_out.update(s_[kind]["chunk_values"])
            instances[kind]["null" if iid is None else str(iid)] = body     # the key the page looks up
        tp_new = pred is None or TOPO[kcol] != TOPO[pred]
        cc_new = pred is None or CCHI[kcol] != CCHI[pred]
        objs["TopologyVersion"] = {
            "iid": TOPO[kcol], "label": label_of("TopologyVersion", TOPO[kcol]), "new": tp_new,
            "how": (None if not tp_new else "built by the batched build" if pred is None else
                    "new struct: " + and_list(kk for kk in TOPOLOGY_FIELDS if objs[kk]["new"]) +
                    " new; every other field shared")}
        objs["Cch"] = {
            "iid": CCHI[kcol], "label": label_of("Cch", CCHI[kcol]), "new": cc_new,
            "how": (None if not cc_new else "built by the batched build" if pred is None else
                    "new: " + and_list((["topology"] if tp_new else []) +
                                       [kk for kk in METRIC_FIELDS if objs[kk]["new"]]))}
        instances["TopologyVersion"][str(TOPO[kcol])] = {kk: objs[kk]["label"] for kk in TOPOLOGY_FIELDS} | {
            "n": s_["scalars"]["n"], "arc_capacity": s_["scalars"]["arc_capacity"]}
        instances["Cch"][str(CCHI[kcol])] = {"topology": objs["TopologyVersion"]["label"],
                                             **{kk: objs[kk]["label"] for kk in METRIC_FIELDS}}
        e = entry_of[kcol]
        columns.append({"key": kcol, "label": LABEL[kcol], "version": v, "by": by, "pred": pred,
                        "entry": e, "log": log_of[kcol], "log_why": log_why(kcol),
                        "serves": "Cch" if e["state"] == "Built" else "fallback", "objects": objs,
                        "scalars": s_["scalars"], "metric_rebuilt": kcol in REBUILT,
                        "s7_met": [x["signal"] for x in decay.get(v if kcol != "14w" else "14", [])
                                   if x["job"] == "S7" and x["met"]] if kcol not in ("9", "14", "16f") else []})

    # what each state holds, and what only it holds
    def held(kcol):
        c = next(x for x in columns if x["key"] == kcol)
        s_ = SNAP[kcol]
        out = {("Cch", c["objects"]["Cch"]["iid"]), ("TopologyVersion", c["objects"]["TopologyVersion"]["iid"])}
        for kind in TOPOLOGY_FIELDS + METRIC_FIELDS:
            if s_[kind]["iid"] is not None:          # an empty delta is no instance
                out.add((kind, s_[kind]["iid"]))
        for kind in ("Up.delta", "Down.delta"):
            out |= {("delta page", p["page"]) for p in s_[kind]["contents"]["pages"]}
        for kind in METRIC_FIELDS:
            out |= {("metric chunk", cid) for cid in s_[kind]["chunks"]}
        for kind in ("rank_of", "node_of"):
            out |= {(kind + " chunk", cid) for _, cid in s_[kind]["chunks"]}
        out |= {("log node", f"seq {n_['seq']}") for n_ in c["log"]}
        return out
    order_ = [kc for kc, _, _, _ in COLS]
    for i_, c in enumerate(columns):
        # what no other published state holds: with Arc, it is freed when this state's last holder
        # (the entry, a reader or a job) lets go, whatever order the states drop in
        others = set()
        for kc in order_:
            if kc != c["key"]:
                others |= held(kc)
        gone_ = sorted(held(c["key"]) - others, key=lambda x: (x[0], str(x[1])))
        by_kind = defaultdict(list)
        for kind, iid in gone_:
            by_kind[kind].append(iid if kind in ("delta page", "metric chunk", "log node", "rank_of chunk",
                                                 "node_of chunk") else
                                 (c["objects"][kind]["label"] if kind in c["objects"] else iid))
        c["held_only"] = dict(by_kind)
    # jobs hold the state they forked from until they publish or are discarded
    pins = [{"job": f"fold, ticket {fold12['ticket']}", "forked": "12", "until": "14w", "inclusive": True,
             "end": "is discarded as Gone by the hand-off's new generation"},
            {"job": "Repair", "forked": "14w", "until": "14", "inclusive": False, "end": "adopts"},
            {"job": f"fold, ticket {ticket}", "forked": "15", "until": "16f", "inclusive": False,
             "end": "publishes"}]
    # the v14 attempt: what the writer had copied when it passed B_gil (all of it is dropped)
    SNAP["14 attempt"] = snapshot(attempt)     # not a published state: never in COLS
    attempt_copies = {kind: how_text(kind, "14 attempt", "13") for kind in TOPOLOGY_FIELDS + METRIC_FIELDS
                      if SNAP["14 attempt"][kind]["iid"] != SNAP["13"][kind]["iid"]}
    mvcc = {"columns": columns, "pins": pins, "instances": instances, "pages": pages_out,
            "metric_chunks": mchunks_out, "map_chunks": map_chunks_out,
            "fields": {"topology": list(TOPOLOGY_FIELDS), "metric": list(METRIC_FIELDS)}, "chunk": CHUNK,
            "attempt_v14": attempt_copies}

    # one card per commit or job after the build (03 §6): change, classification, objects written, work
    FLUSH = {10: r10, 11: r11, 12: r12, 13: r13, 14: r14, 15: r15, 16: r16, 17: r17}
    cards = []
    for i_, e in enumerate(after):
        c = {"event": i_, "actor": e["actor"], "col": e.get("col"), "outcome": e.get("outcome"),
             "entry": e["entry"], "published_log": e["published_log"]}
        if e["actor"] == "writer":
            f_ = FLUSH[e["version"]]
            c.update(version=e["version"], what=e["what"], work=e["work"], budget=e["budget"],
                     classify=f_["classify"], allocation=f_["allocation"],
                     new_arcs=f_.get("s2", {}).get("new_arcs", []), path=f_.get("path"),
                     front=len(f_.get("front", [])), stopped=f_.get("stopped"),
                     full_cost=e.get("full_cost"), decay_met=e.get("decay_met", []))
        else:
            c.update(name=e["name"], fork=e.get("fork"), fit=e.get("fit"), why=e.get("why"), note=e.get("note"),
                     path=e.get("path"))
        if c["col"] and c["col"] != "14w":
            col_ = next(x for x in columns if x["key"] == c["col"])
            c["written"] = [{"kind": kk, "label": o_["label"], "how": o_["how"]}
                            for kk, o_ in col_["objects"].items() if o_["new"]]
        cards.append(c)
    mvcc["cards"] = cards

    # ================================================================ G. invariants + breaks
    published = [(9, built9, G[9]), (10, h10, G[10]), (11, h11, G[11]), (12, h12, G[12]), (13, h13, G[13]),
                 (14, rep, G[14]), (15, h15, G[15]), (16, h16, G[16]), ("16 fold", folded, G[16]),
                 (17, h17, G[17])]
    every_published = "every published version: " + ", ".join(f"v{v}" for v, _, _ in published)
    inv = []

    def add_inv(name, design, where, passed, detail, breaks):
        assert passed, (name, detail)
        for b in breaks:
            assert not b["passed"], (name, b)
        inv.append({"invariant": name, "design": design, "checked_at": where, "passed": passed,
                    "detail": detail, "breaks": breaks})

    # G1 coverage
    cov = {str(v): coverage_violations(h, g) for v, h, g in published}
    cov_sg = coverage_violations(sg, g7)
    # break: Supergraph adds no logged pair (only drops), so v7's new edges have no arc
    brk_B = B - Dset
    U0b = defaultdict(set)
    for (a, b) in brk_B:
        if a < b:
            U0b[a].add(b)
    upb, _, _ = game(n_r1, U0b)
    rows_b = {r: [(hi, i) for i, hi in enumerate(upb[r])] for r in upb}
    hb = Hier()
    hb.base = Base(n_r1, rows_b)
    hb.n = n_r1
    hb.rank_of, hb.node_of = dict(sg.rank_of), dict(sg.node_of)
    cov_brk = coverage_violations(hb, g7)
    # break 2: F2 classified as metric-only (no S2)
    hb2 = h10.clone()
    cov_brk2 = coverage_violations(hb2, g11)
    add_inv("Every live edge has an arc after every publish: pattern(S_now) ⊆ Up, and every node with an edge has a rank",
            "03 §2 M5 debug checks, §4 Contract, §6b Order", "Supergraph (v7) and " + every_published,
            not cov_sg and all(not v for v in cov.values()),
            {"versions": list(cov), "violations": cov, "at_supergraph_v7": cov_sg},
            [{"break": "Supergraph adds no logged pair (B′ = B ∖ D)", "passed": not cov_brk, "violations": cov_brk},
             {"break": "the v11 insert skips S2 (treated as metric-only)", "passed": not cov_brk2,
              "violations": cov_brk2}])

    # G2 closure
    clo = {str(v): closure_violations(h) for v, h, g in published}
    clo["supergraph"] = closure_violations(sg)
    drop = next((s_["rank"], s_["fill"][0]) for s_ in gsteps if s_["fill"])
    upd, _, _ = game(n_r1, U0, drop=drop)
    hd = Hier()
    hd.base = Base(n_r1, {r: [(hi, i) for i, hi in enumerate(upd[r])] for r in upd})
    hd.n = n_r1
    clo_brk = closure_violations(hd)
    # break 2: S2 without the push step
    hs = h10.clone()
    flush(hs, g11, p11, 11, s2_push=False, record=False, topology_only=True)
    clo_brk2 = closure_violations(hs)
    add_inv("Closure (I5): for every rank r and y < z in up(r), the arc (y, z) exists",
            "03 §4 Contract, §6 S2", "Supergraph (v7) and " + every_published,
            all(not v for v in clo.values()), {"violations": clo},
            [{"break": f"drop one fill arc from the game (rank {drop[0]} → {drop[1]})", "passed": not clo_brk,
              "violations": clo_brk[:5]},
             {"break": "S2 adds the inserted arc but never pushes into the parent", "passed": not clo_brk2,
              "violations": clo_brk2[:5]}])

    # G3 parent(r) = up(r)[0]
    par_ok = all(par[r] == gpar[r] for r in range(1, n_r1 + 1))
    par_old = liu_etree(n_r1, adj_of(B))
    par_brk = [r for r in range(1, n_r1 + 1) if par_old.get(r) != gpar[r]]
    add_inv("parent(r) = up(r)[0]: Liu's tree over B′ equals the game's lowest up-entry",
            "03 §4 step 2; I6", "Supergraph (v7) only, where the tree is computed", par_ok, {"ranks": n_r1},
            [{"break": "the elimination tree is not recomputed over B′ (B's tree is kept)", "passed": not par_brk,
              "ranks_that_differ": par_brk}])

    # G4 P. The break models 02's pool (03 App. B V3.1) on this example: 02's step 0 put isolated nodes on
    # pool ranks that stay in free_ranks, and 02's allocation pops the lowest free rank and strips its holder.
    pool = {str(v): pool_violations(h) for v, h, g in published}
    pool["supergraph"] = pool_violations(sg)
    free_at_11 = list(h11.free)
    assert free_at_11 and h11.node_of.get(free_at_11[0]) is None
    pr_ = free_at_11[0]
    hp = h11.clone()
    hp.rank_of[18] = pr_                       # 02 §3 step 0: isolated node 18 holds a pool rank ...
    hp.node_of[pr_] = 18                       # ... which stays in free_ranks
    gb1, pb1, jb1 = write(G[11], [("addroad", *V31_ROAD, road_w(*V31_ROAD))])
    rb1 = flush(hp, gb1, pb1, 12, alloc=allocate_02, record=False, topology_only=True)
    p_after_road = pool_violations(hp)         # node 18 now has an arc at a rank still in free_ranks
    gb2, pb2, jb2 = write(gb1, [("node", 21), ("addroad", 4, 21, 5)])   # F3's degree-1 node 21
    rb2 = flush(hp, gb2, pb2, 13, alloc=allocate_02, record=False, topology_only=True)
    stripped = [a for a in rb2["allocation"] if a.get("stripped") is not None]
    cov_after_pop = coverage_violations(hp, gb2)
    add_inv("P: a rank is in free_ranks exactly when it has no holder, and then it has no arcs",
            "03 §2 P", "Supergraph (v7) and " + every_published, all(not v for v in pool.values()),
            {"violations": pool},
            [{"break": f"02's pool rule (03 App. B V3.1): isolated node 18 holds pool rank {pr_}, which stays in "
                       f"free_ranks; road {V31_ROAD[0]}–{V31_ROAD[1]} gives node 18 an arc there (P fails), then "
                       f"v12's node 21 pops rank {pr_} and strips node 18, whose road loses its arc (coverage fails)",
              "passed": not p_after_road and not cov_after_pop,
              "s2_arcs_at_the_pool_rank": [x for x in rb1.get("s2", {}).get("new_arcs", []) if pr_ in x[:2]],
              "P_violations_after_the_road": p_after_road, "pop": rb2["allocation"], "stripped": stripped,
              "coverage_violations_after_the_pop": cov_after_pop}])

    # G5 ids unique, never reused within a lineage
    seen = {}
    id_bad = []
    for v, h, g in published:
        a = h.arcs()
        if len(set(a.values())) != len(a):
            id_bad.append({"version": str(v), "why": "duplicate ids"})
        for kk, i in a.items():
            if seen.setdefault(i, kk) != kk:
                id_bad.append({"version": str(v), "id": i, "was": list(seen[i]), "now": list(kk)})
    hr = h10.clone()
    flush(hr, g11, p11, 11, s2_renumber=True, record=False, topology_only=True)
    seen2 = {}
    id_brk = []
    for v, h in ((9, built9), (10, h10), (11, hr)):
        for kk, i in h.arcs().items():
            if seen2.setdefault(i, kk) != kk:
                id_brk.append({"version": v, "id": i, "was": list(seen2[i]), "now": list(kk)})
    add_inv("Arc ids are unique and never reused within a lineage", "03 §2 Arc identity", every_published,
            not id_bad, {"versions": [str(v) for v, _, _ in published], "violations": id_bad},
            [{"break": "S2 renumbers ids row-major after inserting (a positional write)", "passed": not id_brk,
              "violations": id_brk[:5], "count": len(id_brk)}])

    # G6 scoped = full (the full path is G14)
    eq = {"step D3 Seal (v8)": seal_ok, "adoption v9": adopt_ok, "v10": f1_ok, "v11": f2_ok, "v12": f3_ok,
          "v13": f4_ok, "v15": not diff_metric(h15.metric_by_id(), full_customize(h15, g15)),
          "v16": v16_ok, "v17": f8_ok}
    assert r15["path"] == r16["path"] == r17["path"] == "scoped"
    hb1 = built9.clone()
    flush(hb1, g10, p10, 10, propagate=False, record=False)
    brk1 = diff_metric(hb1.metric_by_id(), ref10)
    hb2_ = h10.clone()
    flush(hb2_, g11, p11, 11, front_seed=False, record=False)
    brk2 = diff_metric(hb2_.metric_by_id(), ref11)
    hb4 = h12.clone()
    flush(hb4, g13, [], 13, record=False)   # the cascade's pairs are not logged
    brk4 = diff_metric(hb4.metric_by_id(), ref13)
    add_inv("A scoped recompute equals a full customization of the same topology, bit for bit",
            "03 §6b Lemma A", "every scoped recompute: " + ", ".join(eq), all(eq.values()), {"cases": eq},
            [{"break": "v10 without propagation (a changed arc pushes nothing)", "passed": not brk1,
              "differing_ids": [d["id"] for d in brk1]},
             {"break": "v11 without the inserted pair's new seed in the front", "passed": not brk2,
              "differing_ids": [d["id"] for d in brk2]},
             {"break": "v13 with the cascade's pairs not logged", "passed": not brk4,
              "differing_ids": [d["id"] for d in brk4]}])

    # G7 queries = Dijkstra
    qres = {}
    for v, h, g in published:
        n_, bad = check_queries(h, g)
        qres[str(v)] = {"pairs": n_, "mismatches": bad}
    # break: the Seal's CatchUp does not relink seq 9, so v9's delete is never absorbed
    lost = seal.clone()
    n_, bad_lost = check_queries(lost, g9)
    add_inv("CCH distances equal Dijkstra for every pair of live nodes after every published version",
            "03 §7 (equal distances)", every_published, all(not r["mismatches"] for r in qres.values()),
            {"versions": qres, "building_versions": {"5–8": "Building: Dijkstra fallback",
                                                     "14 (before the Repair)": "Building{Repair}: Dijkstra fallback"}},
            [{"break": "the Seal's CatchUp drops seq 9 (no relink), so the v9 delete is never absorbed",
              "passed": not bad_lost, "mismatches": bad_lost}])

    # G8 B′ is the exact live pattern at V_sg
    live7 = {(a, b) for (u, v) in g7.live_pairs() for (a, b) in ((u, v), (v, u))}
    live7r = {(sg.rank_of[a], sg.rank_of[b]) for (a, b) in live7}
    exact = B1 == live7r
    S_nm = {(rank_of[a], rank_of[b]) for (a, b) in S_nomask}
    brk_mask = ((S_nm - Dset) | Aset) == live7r
    brk_drop = (B | Aset) == live7r
    add_inv("B′ is exactly the live pattern at V_sg (no dead arc survives the build)",
            "03 revision 3.1: §3 Input skeleton, §4 step 2; M5 Deletions", "Supergraph (v7) only", exact,
            {"B_prime": len(B1), "live_at_v7": len(live7r)},
            [{"break": "ReadSkeleton ignores dm", "passed": brk_mask,
              "extra": pat(((S_nm - Dset) | Aset) - live7r)},
             {"break": "Supergraph adds live pairs but drops none", "passed": brk_drop,
              "extra": pat((B | Aset) - live7r)}])

    # G9 full seed = per-pair seeds, wherever the full-seed pass runs: D1 at v7 (base only) and F5's
    # Repair at v14 (base and delta)
    def seeds_match(h, g, rec_):
        return (rec_["up"], rec_["dn"]) == pair_seeds(h, g)

    def seeds_wrong(h, g, rec_):
        want_up, want_dn = pair_seeds(h, g)
        return [i for i in range(h.arc_capacity) if (rec_["up"][i], rec_["dn"][i]) != (want_up[i], want_dn[i])]
    s_nodp = full_seed_by_id(sg, g7, mask_dp=False)
    s_one = full_seed_by_id(sg, g7, both_dirs=False)
    s_pos = full_seed_by_id(rep, g14, delta_mode="by_position")
    s_skip = full_seed_by_id(rep, g14, delta_mode="skip")
    g9_cases = {"step D1 (v7)": seeds_match(sg, g7, seedrec), "v14 Repair": seeds_match(rep, g14, full14)}
    add_inv("The full-seed pass gives each arc its pair's minimum live weight in each direction, written by arc id",
            "03 §5 The full-seed pass (V2.6, V3.4)", "where the pass runs: step D1 (v7, base only) and v14's Repair "
            "(v14, base and delta)", all(g9_cases.values()),
            {"cases": g9_cases, "arcs": {"v7": sg.arc_capacity, "v14": rep.arc_capacity},
             "delta_entries_v14": len(bt_items(rep.dup))},
            [{"break": "v7: no ¬struct(dp) mask: the stale m value under the replaced 14→13 is read too",
              "passed": seeds_match(sg, g7, s_nodp), "wrong_ids": seeds_wrong(sg, g7, s_nodp)},
             {"break": "v7: only the upward direction is seeded", "passed": seeds_match(sg, g7, s_one),
              "wrong_ids": seeds_wrong(sg, g7, s_one)},
             {"break": "v14: the delta's seeds are written by position (slot |base| + j) instead of by id",
              "passed": seeds_match(rep, g14, s_pos), "wrong_ids": seeds_wrong(rep, g14, s_pos)},
             {"break": "v14: the delta is not scattered at all, so its arcs stay +∞ (02's bug, App. B V3.4)",
              "passed": seeds_match(rep, g14, s_skip), "wrong_ids": seeds_wrong(rep, g14, s_skip)}])

    # G10 Down mirrors Up with the same ids
    mir = {str(v): down_mirror_violations(h) for v, h, g in published}
    hm = h10.clone()
    flush(hm, g11, p11, 11, s2_down=False, record=False, topology_only=True)
    mir_brk = down_mirror_violations(hm)
    add_inv("Down holds Up transposed, with the same ids (I7)", "03 §2 I7", every_published,
            all(not v for v in mir.values()), {"violations": mir},
            [{"break": "S2 inserts into Up.delta only", "passed": not mir_brk, "violations": mir_brk}])

    # G11 S2 equals the game over (old arcs ∪ new pairs), at every commit that runs S2
    s2_cases = {"v11": s2_game_ok, "v12": s2_game_ok12, "v14 Repair": s2_game_ok14, "v16": s2_game_ok16}
    hs2 = h10.clone()
    r_ = flush(hs2, g11, p11, 11, s2_push=False, record=False, topology_only=True)
    s2_brk_ok = s2_equals_game(h10, hs2, r_)
    add_inv("S2's arcs equal the elimination game over the old arcs plus the inserted pairs",
            "03 §6 S2 (DH01 Thm 4.2)", "every commit that runs S2: " + ", ".join(s2_cases),
            all(s2_cases.values()), {"cases": s2_cases},
            [{"break": "v11: S2 without the push step", "passed": s2_brk_ok}])

    # G12 the fold changes representation, not values
    stale_fold = h16.clone()
    stale_fold.base = newbase
    stale_fold.dup = stale_fold.ddn = None      # break: deltas rebuilt from the fork's copy (empty after the fold)
    stale_fold.invalidate()
    fork_metric = folded.clone()                # break: the metric taken from the fork (v15), not the published copy
    fork_metric.metric = {kk: a.clone() for kk, a in h15.metric.items()}
    fork_metric_diff = diff_metric(fork_metric.metric_by_id(), full_customize(fork_metric, g16))
    add_inv("A fold keeps every arc and id, and its metric equals a full customization of the folded topology",
            "03 §6 Fold job (V1.4)", "the fold's publish (v16)",
            fold_ids_same and fold_metric_ok,
            {"ids": fold_ids_same, "metric_equals_full": fold_metric_ok,
             "metric_carried_from_the_published_copy": fold_metric_carried},
            [{"break": "the new deltas come from the fork's copy, so the v16 arcs are lost",
              "passed": stale_fold.arcs() == h16.arcs(),
              "missing": [list(x) for x in sorted(set(h16.arcs()) - set(stale_fold.arcs()))]},
             {"break": "the fold takes the metric from its fork (v15), not from the published copy at publish",
              "passed": not fork_metric_diff, "differing_ids": [d["id"] for d in fork_metric_diff]}])

    # G13 exact counts behind guard (b)
    add_inv("Guard (b) is exact: the column counts equal the arcs the game makes, and are under the guard",
            "03 §3 Guards (b), §4 step 2", "Supergraph (v7) only", arcs_n == n_arcs and arcs_n <= guard,
            {"counts": arcs_n, "game": n_arcs, "guard": guard},
            [{"break": "the counts are not recomputed over B′ (B's counts are kept)", "passed": arcs_B == n_arcs,
              "counts": arcs_B, "game": n_arcs}])

    # G14 the full path (seeds by id, then one sweep from them) equals a full customization
    full_cases = {"step D2 build (v7)": d2_ok, "v14 Repair": f5_ok}
    hskip = h13.clone()
    flush(hskip, g14, p14, 14, offlock=True, record=False, seed_delta="skip")
    skip_diff = diff_metric(hskip.metric_by_id(), ref14)
    add_inv("The full path, a relax sweep from seeds written by arc id, equals a full customization",
            "03 §6b valves, §5 The full-seed pass (V3.4)", "every full path: " + ", ".join(full_cases),
            all(full_cases.values()), {"cases": full_cases},
            [{"break": "v14's Repair with the delta not scattered, so its arcs start at +∞ (02's bug, App. B V3.4)",
              "passed": not skip_diff, "differing_ids": [d["id"] for d in skip_diff]}])

    # ---------------------------------------------------------------- MVCC invariants (the design view)
    col_keys = [kc for kc, _, _, _ in COLS]
    pred_of = {kc: pr_ for kc, _, pr_, _ in COLS}

    def shared_contents(keys):
        """For every object, page and chunk id two states share: are the contents copied at their
        publishes equal? Returns the mismatches and the number of distinct ids seen."""
        seen, bad = {}, []
        for kc in keys:
            s_ = SNAP[kc]
            items = [((kind, s_[kind]["iid"]), {k_: v_ for k_, v_ in s_[kind].items() if k_ != "iid"})
                     for kind in TOPOLOGY_FIELDS + METRIC_FIELDS]
            for kind in ("Up.delta", "Down.delta"):
                items += [(("delta page", p["page"]), p) for p in s_[kind]["contents"]["pages"]]
            for kind in METRIC_FIELDS:
                items += [(("metric chunk", cid), v_) for cid, v_ in s_[kind]["chunk_values"].items()]
            for kind in ("rank_of", "node_of"):
                items += [((kind + " chunk", cid), v_) for cid, v_ in s_[kind]["chunk_values"].items()]
            for k_, body in items:
                js = json.dumps(body, sort_keys=True)
                if k_ in seen and seen[k_][1] != js:
                    bad.append({"object": k_[0], "id": k_[1], "states": [seen[k_][0], kc]})
                seen.setdefault(k_, (kc, js))
        return bad, len(seen)
    sh_bad, sh_n = shared_contents(col_keys)
    # break: a writer writes a metric chunk in place, without copying it (or the chunk table)
    hx = h16.clone()
    cid0 = hx.metric["up"].ids[0]
    vals0 = CHUNKS[cid0]
    hx.metric["up"].set_in_place(0, -1)
    SNAP["G15 break"] = snapshot(hx)
    sh_brk, _ = shared_contents(col_keys + ["G15 break"])
    CHUNKS[cid0] = vals0
    del SNAP["G15 break"]
    add_inv("An object, page or chunk id that two versions share has the same contents in both",
            "03 §6b MVCC (every write goes to a private version), §2 versioning column",
            "every published state: " + ", ".join(LABEL[k_] for k_ in col_keys),
            not sh_bad, {"ids_compared": sh_n, "mismatches": sh_bad},
            [{"break": "a writer forked from v16 writes metric up at arc id 0 in place: no chunk copy, no table copy",
              "passed": not sh_brk, "mismatches": sh_brk}])

    # pages a state's inserts created = the inserted keys' root-to-leaf paths, plus split siblings
    pg_cases, pg_bad = {}, []
    for kc in col_keys:
        pr_ = pred_of[kc]
        if pr_ is None or kc == "16f":
            continue
        for kind in ("Up.delta", "Down.delta"):
            if SNAP[kc][kind]["iid"] == SNAP[pr_][kind]["iid"]:
                continue
            sp_, ks_ = delta_split(kc, pr_, kind)
            pg_cases[f"{LABEL[kc]} {kind}"] = {"inserts": len(ks_), "on_paths": len(sp_["path"]),
                                               "split_leaves": len(sp_["split"]), "made": len(sp_["made"])}
            if sp_["stray"] or sp_["path_not_made"]:
                pg_bad.append({"state": LABEL[kc], "delta": kind, "stray": sorted(sp_["stray"]),
                               "path_not_copied": sorted(sp_["path_not_made"])})

    def bt_deep_copy(root):
        if root is None:
            return None
        page = PAGES[root]
        new = page if page[0] == "leaf" else ("branch", tuple(bt_deep_copy(c) for c in page[1]), page[2])
        pid = _page_seq[0]
        _page_seq[0] += 1
        PAGES[pid] = new
        return pid
    root_x = bt_deep_copy(h15.dup)
    keys16 = [key(lo, hi) for lo, hi, _ in r16["s2"]["new_arcs"]]
    for (lo, hi, i) in r16["s2"]["new_arcs"]:
        root_x, _ = bt_insert(root_x, key(lo, hi), i)
    sp_x = delta_page_split(bt_pages(h15.dup), bt_pages(root_x), root_x, PAGES, keys16, h15.dup)
    # break 2: the v16 writer also copies one leaf its inserts never reached (an untouched sibling of a
    # path leaf, under a path branch)
    pg15 = bt_pages(h15.dup)
    path16 = set()
    for k_ in keys16:
        path16 |= set(bt_path(h16.dup, k_))
    par_, kid_ = next((p_, c_) for p_ in sorted(path16) if PAGES[p_][0] == "branch" for c_ in PAGES[p_][1]
                      if c_ in pg15 and PAGES[c_][0] == "leaf")
    cp_ = _page_seq[0]
    _page_seq[0] += 1
    PAGES[cp_] = PAGES[kid_]                        # the same entries, a new page
    np_ = _page_seq[0]
    _page_seq[0] += 1
    PAGES[np_] = ("branch", tuple(cp_ if c_ == kid_ else c_ for c_ in PAGES[par_][1]), PAGES[par_][2])

    def swap_root(pid):
        if pid == par_:
            return np_
        pg_ = PAGES[pid]
        if pg_[0] == "leaf" or par_ not in bt_pages(pid):
            return pid
        nid = _page_seq[0]
        _page_seq[0] += 1
        PAGES[nid] = ("branch", tuple(swap_root(c_) for c_ in pg_[1]), pg_[2])
        return nid
    root_y = swap_root(h16.dup)
    sp_y = delta_page_split(pg15, bt_pages(root_y), root_y, PAGES, keys16, h15.dup)
    add_inv("The delta pages a version's inserts create are exactly the inserted keys' root-to-leaf paths, "
            "plus the new half of each leaf that split; every other page is shared",
            "03 §2 (cow_btree: each insert path-copies O(log n) pages), §6b MVCC",
            "every state whose deltas gained entries by insertion: " + ", ".join(pg_cases),
            not pg_bad, {"cases": pg_cases, "violations": pg_bad},
            [{"break": "v16's inserts go into a deep copy of v15's Up.delta (every page copied)",
              "passed": not (sp_x["stray"] or sp_x["path_not_made"]), "stray_pages": len(sp_x["stray"])},
             {"break": "v16's writer also copies one leaf its inserts never reached, under a branch on a path",
              "passed": not (sp_y["stray"] or sp_y["path_not_made"]), "stray_pages": sorted(sp_y["stray"])}])

    # metric chunks a state copied = the chunks holding a value it changed or appended; a full-path
    # Repair rebuilds every chunk (03 §5 Phases: "as Seal", which chunks with WeightArr::from_vec)
    def chunk_mismatch(cur_s, prev_s, rebuilt=False):
        bad = []
        for kind in METRIC_FIELDS:
            pc, cc = prev_s[kind]["chunks"], cur_s[kind]["chunks"]
            new_ix = {c for c, cid in enumerate(cc) if c >= len(pc) or pc[c] != cid}
            a_, b_ = snapshot_values(cur_s, kind), snapshot_values(prev_s, kind)
            wrote = {i // CHUNK for i in range(len(a_)) if i >= len(b_) or a_[i] != b_[i]}
            want = set(range(len(cc))) if rebuilt else wrote
            if new_ix != want or (rebuilt and set(cc) & set(pc)):
                bad.append({"array": kind, "chunks_copied": sorted(new_ix), "expected": sorted(want)})
        return bad
    rebuilt_cols = {"14"} if rrep.get("metric_rebuilt") else set()
    assert rebuilt_cols, "the v14 Repair takes the full path"
    ch_cases = {LABEL[kc]: chunk_mismatch(SNAP[kc], SNAP[pred_of[kc]], kc in rebuilt_cols)
                for kc in col_keys if pred_of[kc]}
    hcx = built9.clone()
    for a_ in hcx.metric.values():
        a_.copy_all = True
    flush(hcx, g10, p10, 10, record=False)
    ch_brk = chunk_mismatch(snapshot(hcx), SNAP["9"])
    hrx = h13.clone()                     # break 2: the full-path Repair writes into its base's chunks
    flush(hrx, g14, sorted({tuple(p) for n in log14 for p in n["pairs"]}), 14, offlock=True, record=False,
          metric_from_vec=False)
    ch_brk2 = chunk_mismatch(snapshot(hrx), SNAP["14w"], rebuilt=True)
    add_inv("The metric chunks a version copies are exactly the chunks that hold a value it changed or appended; "
            "a full-path Repair builds every chunk new",
            "03 §2 Metric (#3774's chunks, plus append), §6b MVCC (WeightArr::set copies its chunk), "
            "§5 Phases (Repair runs as Seal: WeightArr::from_vec)",
            "every published state against the one it was cloned from: " + ", ".join(ch_cases),
            all(not x for x in ch_cases.values()), {"violations": {k_: v_ for k_, v_ in ch_cases.items() if v_},
                                                     "rebuilt": [LABEL[k_] for k_ in sorted(rebuilt_cols)]},
            [{"break": "v10 rerun with WeightArr::set copying every chunk of an array on its first write",
              "passed": not ch_brk, "violations": len(ch_brk)},
             {"break": "the v14 Repair's full path writes the changed values into its base's chunks "
                       "instead of chunking a flat metric",
              "passed": not ch_brk2, "violations": len(ch_brk2)}])

    # retention: a Built entry keeps a log only while an upkeep job is in flight, and only seq > since
    def replay_logs(keep_always):
        log_, out, rv = [], {}, {r_["version"]: r_ for r_ in retention}
        for e in after:
            kc = e.get("col")
            if not kc:
                continue
            if e["actor"] == "writer" and e["version"] in rv:
                r_ = rv[e["version"]]
                log_, _ = retain_log(r_["upkeep"], log_, r_["node"], keep_always=keep_always)
            elif kc == "14w":
                log_ = list(log14)
            else:
                log_ = []          # the Repair's adoption and the fold's publish clear the log
            out[kc] = [n_["seq"] for n_ in log_]
        return out

    def retention_bad(logs):
        bad = []
        for c in columns:
            if c["key"] not in logs or c["entry"]["state"] != "Built":
                continue
            up_ = c["entry"].get("upkeep")
            seqs = logs[c["key"]]
            if (up_ is None and seqs) or (up_ is not None and any(q_ <= up_["since"] for q_ in seqs)):
                bad.append({"state": c["label"], "upkeep": up_, "log": seqs})
        return bad
    logs_real = replay_logs(False)
    replay_ok = all(logs_real[c["key"]] == [n_["seq"] for n_ in c["log"]] for c in columns if c["key"] in logs_real)
    ret_bad = retention_bad({c["key"]: [n_["seq"] for n_ in c["log"]] for c in columns})
    ret_brk = retention_bad(replay_logs(True))
    add_inv("A Built entry keeps a log only while an upkeep job is in flight, and then only nodes with seq > since",
            "03 §7 Retention, §6 Built-entry jobs",
            "every Built state from v9 to v17; the replay of the publish rule reproduces every published log",
            not ret_bad and replay_ok, {"violations": ret_bad, "replay_reproduces_the_logs": replay_ok},
            [{"break": "a writer keeps its log node at publish even with no upkeep set", "passed": not ret_brk,
              "violations": ret_brk}])

    # the matrix track against the steps it draws
    up_c4 = {(r, hi) for r in up_rows for hi in up_rows[r]}
    fill_c4 = {(s_["rank"], f) for s_ in gsteps for f in s_["fill"]}
    tr_order = pi_t == rank_of
    tr_up = {(r, hi) for r in up_t for hi in up_t[r]} == up_c4 and fill_t == fill_c4
    tr_ids = ids_t == base1.arcs()
    _, _, up_nolog, fill_nolog = track_up(pi_t, with_log=False)
    add_inv("The matrix track draws the build: its order is π, its supergraph is step C4's arcs and fill, and its "
            "stored ids are step C5's Ax",
            "03 §3 step 8 (one extract B = S(π,π)), §4 steps 1–4",
            "the track's frames: B (order), Up (pattern), Up.base (ids)",
            tr_order and tr_up and tr_ids,
            {"order": tr_order, "supergraph": tr_up, "ids": tr_ids, "arcs": len(up_c4), "fill": len(fill_c4)},
            [{"break": "the track orders by Order's ranks without the postorder relabel q",
              "passed": track_order(False) == rank_of,
              "nodes_that_move": sorted(x for x, r in track_order(False).items() if rank_of[x] != r)},
             {"break": "the track runs the game over B, not over B′ (the log is ignored)",
              "passed": {(r, hi) for r in up_nolog for hi in up_nolog[r]} == up_c4,
              "arcs": sum(len(v_) for v_ in up_nolog.values())},
             {"break": "the track numbers the arcs column by column instead of row by row",
              "passed": track_ids(up_t, row_major=False) == base1.arcs(),
              "ids_that_differ": sum(1 for c_, i in track_ids(up_t, row_major=False).items()
                                     if base1.arcs()[c_] != i)}])

    # a fold replaces only the bases, the deltas and added_since_fold; every other structure is the
    # published copy's instance (03 §6 Fold job: "the published Cch with only the four structures replaced")
    REPLACED = ("Up.base", "Down.base", "Up.delta", "Down.delta", "added_since_fold")

    def fold_sharing(s_new, s_pub):
        return ([kk for kk in TOPOLOGY_FIELDS + METRIC_FIELDS if kk not in REPLACED and
                 s_new[kk]["iid"] != s_pub[kk]["iid"]],
                [kk for kk in REPLACED if s_new[kk]["iid"] == s_pub[kk]["iid"]])
    fs_bad = fold_sharing(SNAP["16f"], SNAP["16"])
    fx = fold_src.clone()                   # the break: the fold's result built on its fork's copy (v15)
    fx.base = newbase
    fx.dup = fx.ddn = None
    for s_q, a_q in keep:
        for lo, hi, i in a_q:
            fx.dup, _ = bt_insert(fx.dup, key(lo, hi), i)
            fx.ddn, _ = bt_insert(fx.ddn, key(hi, lo), i)
    fx.added = CowList("added_since_fold", keep)
    fx.invalidate()
    fs_brk = fold_sharing(snapshot(fx), SNAP["16"])
    add_inv("A fold shares every structure with the published copy except the bases, the deltas and "
            "added_since_fold, which it replaces",
            "03 §6 Fold job (V1.4), decision D6", "the fold's publish (v16 fold against v16)",
            not fs_bad[0] and not fs_bad[1], {"not_shared": fs_bad[0], "not_replaced": fs_bad[1]},
            [{"break": "the fold's result is built on its fork's copy (v15), not on the published v16",
              "passed": not fs_brk[0] and not fs_brk[1], "not_shared": fs_brk[0]}])

    step("G", "G", f"Invariants: {sum(1 for x in inv if x['passed'])} of {len(inv)} pass; "
                   f"{sum(len(x['breaks']) for x in inv)} deliberate breaks, all fail",
         "03 §9 M5",
         f"Each invariant is checked where the step it guards runs; its 'checked at' line says where. "
         f"{sum(1 for x in inv if every_published in x['checked_at'])} of them run on every published version, "
         f"from v9 to v17, plus the fold's publish. Each is then run once more with one fix removed or one input "
         f"corrupted, and the check must fail; a check that cannot fail proves nothing.",
         [], {"invariants": inv})

    # ================================================================ example metadata + output
    sep_ranks = sp["separators"][0]["separator_ranks"]
    # every decay signal, and the versions after whose flush it is met
    dv = sorted(decay, key=lambda k: int(k))
    met_at = {}
    for k in dv:
        for s_ in decay[k]:
            if s_["met"]:
                met_at.setdefault((s_["job"], s_["signal"]), []).append(int(k))

    def when(vs):
        allv = [int(k) for k in dv]
        a = allv.index(vs[0])
        if len(vs) > 1 and allv[a:a + len(vs)] == vs:
            return f"from v{vs[0]} on" if a + len(vs) == len(allv) else f"from v{vs[0]} to v{vs[-1]}"
        return "at " + and_list(f"v{x}" for x in vs)
    decay_lines = [f"{j}'s '{s}' {when(vs)}" for (j, s), vs in met_at.items() if j != "fold"]
    fold_lines = met_at.get(("fold", "delta entries > √|base|"), [])
    first_s7 = min(vs[0] for (j, _), vs in met_at.items() if j == "S7")
    fits = [r10, r11, r12, r13, r15, r16, r17, adopt]
    notes += [
        "Node ids: 0–15 form the 4×4 grid (row-major, x = column, y = row); 16 and 17 hang off corner 15; 18 "
        "exists with no edge; 19 is a slot freed before the build. node_cap is 20 at the build; nodes created "
        "later take id 19 first (the engine hands out the lowest free id, graph/src/graph/id_space.rs:75) and "
        f"then 20, 21, 22. Node {F4_NODE} is deleted at v13, and its id is handed out again at v17.",
        "A two-way road is two relationships, one per direction. So the pending deletion of road 2–7 is two dm "
        "entries and the pending addition of road 4–5 is two dp entries.",
        f"Added beyond the brief: ROAD 14→13 was replaced in one transaction (tensor state H: the old id stays in "
        f"m with weight {H_OLD_W}, the new id is in dp with weight {H_NEW_W}). Without it the seed pass's "
        f"¬struct(dp) mask would remove nothing, and D1 would not show why it is there. The old weight is the "
        f"smaller one on purpose: a pass that read the stale m value would then seed the dead, faster road, and "
        f"the check in G can catch it (with a larger stale value, MIN would hide the bug).",
        f"FERRY has two relationships, both one-way: {FERRY[0][0]}→{FERRY[0][1]} (weight {FERRY[0][2]}, parallel to "
        f"ROAD {FERRY[0][1]}–{FERRY[0][0]} at {road_w(FERRY[0][0], FERRY[0][1])}) and {FERRY[1][0]}→{FERRY[1][1]} "
        f"(weight {FERRY[1][2]}, across the cell 5-6-9-10).",
        f"ROAD weights come from a fixed formula, 1 + (5·lo + 3·hi) mod 9 for the road lo–hi, so they are not "
        f"random and the script can be re-run to the same numbers. Writers set their own weights.",
        f"Toy B_gil is {B_GIL} work units. Every writer commit fits except v14: its running total passes "
        f"{B_GIL} at pop {stop_pop} of the recompute ({stop_work} units), and it stops there; finishing would "
        f"have cost {r14_full['work']['total']}. The largest that fits is "
        f"{max(r_['work']['total'] for r_ in fits)} units.",
        f"The 2-hop pass never fires on the running example: every matching shrinks its cell by at least 15 %, "
        f"and the same holds for all {SALT_SCAN} salt bases from 0 to {SALT_SCAN - 1} (B5.h re-runs the ordering "
        f"under each). Forcing the trigger on every level pairs "
        f"{'no vertex' if not forced_pairs else plural(forced_pairs, 'vertex', 'vertices')}: {forced_why}. "
        f"So B5.h runs the same matching code on a side "
        f"example, a star of 7 vertices, the shape that stalls mutual matching. It is not part of the running "
        f"example's order.",
        "The coarsest graph at depth 0 is complete (4 coarse vertices that all touch), so the first separator "
        "leaves one side empty. Node FM at the finer levels restores the balance (B7).",
        f"Decay signals (03 §6) are evaluated after every writer flush from v10 to v17 (at v14, on the "
        f"writer's attempt before it hands off), with the toy thresholds in the Toy scale table. Met: "
        f"{'; '.join(decay_lines)}. The fold trigger is met "
        f"{when(fold_lines)}; the toy acts on it (the upkeep set at v12 and v15, and the fold published at "
        f"v16). It does not act on S6 or S7. Under the design, "
        f"the v{first_s7} writer would start S7, a re-order as a new generation through the batched build, "
        f"with readers on the Dijkstra fallback, and the v14 commit would be S7 too. So the states after "
        f"v{first_s7} show their rules on a "
        f"hierarchy the design would already be replacing. The toy suppresses S6 and S7 so that the node "
        f"delete, the hand-off, the fold and id reuse can be shown on one lineage; running the build "
        f"once more would show nothing new.",
    ]
    build_tops = and_list(f"node {a_['node']} top rank {a_['rank']}" for a_ in alloc if a_["kind"] == "top")
    first_top_met = min(int(k) for k in decay if any(x["signal"] == "top ranks ≥ reserve" and x["met"]
                                                     for x in decay[k]))
    top_used_v12 = next(x["value"] for x in decay[first_top_met] if x["signal"] == "top ranks ≥ reserve")
    design_questions = [
        {"where": "03 §2 vs §3 step 0 and §4 step 1",
         "question": f"§2 defines n_r = ranked + reserve, with ranked counting nodes with an edge in the skeleton "
                     f"or the log at V_sg; that gives {n_r_sec2} here. §3 step 0 ranks only the skeleton's nodes "
                     f"and §4 step 1 ranks the log's nodes at Supergraph, which gives n_r = {n_r} and n_r′ = "
                     f"{n_r1}. The toy follows §3 and §4."},
        {"where": "03 §6 Built-entry jobs vs §7 Retention",
         "question": "§6 says that while upkeep is set every writer keeps its nodes; §7 says a Built entry with "
                     "upkeep = Some{since} keeps its nodes with seq > since. The writer that sets upkeep has "
                     "seq = since, so the two differ on that writer's node. The toy follows §7 (the job forks at "
                     "F ≥ since and needs only seq > F)."},
        {"where": "03 §6 Built-entry jobs, Start",
         "question": "'The writer whose flush crosses a trigger sets upkeep' does not say what a writer does when "
                     "the trigger is already met and no upkeep is set, as after a hand-off discards the job. "
                     "§6 Retention says that after an abandoned job 'a later commit re-triggers it'. The toy "
                     "reads both as: a writer whose flush ends with the trigger met and no upkeep sets it (v12 "
                     "and v15)."},
        {"where": "03 §6 Decay",
         "question": "The S6 signal's n and 'dead ids' are not defined, and 'arcs' in 'one commit adding > "
                     "arcs/8' could be the count before or after the commit. The toy uses n = the rank "
                     "dimension, the count before the commit, and does not evaluate dead ids."},
        {"where": "03 §2 LineageStats and §6 Decay vs §4 step 1",
         "question": f"§2 lists 'top ranks used' among LineageStats' counters, and §6's S7 signal reads 'top ranks "
                     f"≥ reserve'. Neither says whether a top rank given at build time counts. Supergraph (§4 step 1) "
                     f"gives {build_tops} "
                     f"at v7. The toy counts it: 'top ranks used' is {n_r1 - n_r} at the build, so the signal "
                     f"({top_used_v12} ≥ reserve {RESERVE}) is met at v{first_top_met}, when one more top rank is "
                     f"taken. If build-time top ranks did not count, it would read {top_used_v12 - (n_r1 - n_r)} there "
                     f"and not be met (SIMPLIFICATIONS #41)."},
    ]
    example = {
        "title": "CCH redesign: one running example",
        "spec": ["design/03-design.md (revision 3.1)",
                 "prototypes/00-results.md (its Change list: proposed, not yet merged into 03)",
                 "prototypes/m1-matching-quality.md"],
        "conventions": {
            "ranks": "Ranks are 1-based, as in 03 §2: pool ranks are 1..reserve. CSR rows are 0-based: row i holds rank i + 1.",
            "weights": "null in a weight array means +∞. null in a middle array means no middle.",
            "arcs": "An arc is (lo, hi) in rank space, lo < hi. Upward is node_of(lo) → node_of(hi); downward is the reverse.",
            "matrices": "Sparse matrices are lists of [row, col] or [row, col, value].",
            "draw": "Each step's draw names node and edge sets by role; edges are node-id pairs (space 'node') or "
                    "rank pairs (space 'rank'); 'coarse:d:l' draws the fine graph with level-l groups of depth d.",
            "work_units": WORK_UNITS,
        },
        "example": {
            "node_cap": NODE_CAP, "coords": {str(k): v for k, v in XY.items()},
            "types": list(TYPES), "graph_v5": graph_json(g5), "notes": notes,
            "road_weight_formula": "1 + (5·lo + 3·hi) mod 9",
        },
        "thresholds": THRESHOLDS,
        "design_refs": DESIGN_REFS,
        "design_questions": design_questions,
        "steps": STEPS,
        "after_build_events": after,
        "decay": {str(k): decay[k] for k in dv},
        "versions": {str(v): {"graph": graph_json(G[v])} for v in sorted(G)},
        "hierarchies": {str(v): {"maps": h.json_maps(), "arcs": h.json_arcs(), "metric": h.json_metric(),
                                 "base": h.base.name, "delta": h.json_delta(),
                                 "added_since_fold": [[s, [list(x) for x in a]] for s, a in h.added]}
                        for v, h, g in published},
        "bases": {b.name: b.json() for b in {id(h.base): h.base for _, h, _ in published}.values()},
        "mvcc": mvcc,
        "track": track,
    }
    # distances at v9 and at the last version, for a "try any pair" explorer
    for label, h, g in ((9, built9, G[9]), (17, h17, G[17])):
        ref = dijkstra_all(g)
        example["hierarchies"][str(label)]["distances"] = [
            [s, t, jw(ref[s].get(t, INF))] for s in sorted(g.nodes) for t in sorted(g.nodes) if s != t]
    with open(OUT, "w", encoding="utf-8") as fh:
        json.dump(example, fh, ensure_ascii=False, allow_nan=False, separators=(",", ":"))
    size = os.path.getsize(OUT)
    print(f"wrote {OUT} ({size:,} bytes, {len(STEPS)} steps)")
    for s_ in STEPS:
        print(f"  {s_['id']:<6} {s_['title']}")
    for x in inv:
        print(f"  G: {'PASS' if x['passed'] else 'FAIL'}  {x['invariant']}")
        for b in x["breaks"]:
            print(f"       break -> {'PASS (useless!)' if b['passed'] else 'FAIL (good)'}: {b['break']}")


if __name__ == "__main__":
    main()
