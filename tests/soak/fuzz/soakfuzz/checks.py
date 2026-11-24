"""What an epoch must satisfy, judged from the drivers' records, the PING log and the ledger.

Pure: no server and no redis-py. `Ledger.check` takes any object with `execute_command`.

Records (lists, as the driver processes dump them to JSON):
  read   [graph, op, scheduled, sent, replied | None, outcome, pool wait]
  write  [graph, op, scheduled, sent, replied | None, outcome, error message, keys, pool wait]
  long   [graph, op, sent, replied, outcome, error message]
Pool wait: seconds the request waited for a free connection in the driver (absent: 0).
Outcome: 0 ok, 1 timed out, 2 error reply, 3 unknown (still open when the epoch ended),
4 connection or other error. Times are time.monotonic() of the driver host.
"""

from __future__ import annotations

import bisect
import collections
import json
import math
import statistics

from . import model

PING_INTERVAL_S = 0.010  # the prober sends a PING to each server this often
PING_COVERAGE = 0.5  # no_freeze needs at least half the PINGs the epoch should have had, on each server
DRIVER_LAG_S = 0.05  # a request sent, or given a connection, this late is driver-limited: not judged on timing
FREEZE_PING_MS = 500  # a FalkorDB PING this late ...
CONTROL_OK_MS = 100  # ... while the plain server beside it answered within this, is a freeze
STARVE_S = 1.0  # a write may finish at most this long after the long reads it waited behind
DRAIN_S = 5.0  # a read may finish at most this long (plus its backlog) after them
QUIET_SLACK_MS = 100  # a read on a quiet graph may take at most this much over its cost alone

#: checks that depend on timing (a repeat is needed before a break counts as a regression); error
#: replies are here too, since an overloaded server can refuse a request on one run and not another
TIMING_CHECKS = ("no_freeze", "no_timeouts_while_waiting", "writers_dont_starve", "quiet_graphs_stay_fast",
                 "requests_drain", "no_error_replies")
#: checks on what the graph holds, or whether the server stayed up (a break counts at once)
DATA_CHECKS = ("writes_add_up", "upsert_edges_complete")


def pct(v, p):
    if not v:
        return 0.0
    v = sorted(v)
    return v[min(len(v) - 1, int(math.ceil(p / 100 * len(v))) - 1)]


def outcome_of(exc: Exception | None, response_error: type = Exception) -> tuple[int, str]:
    """0 ok, 1 timeout, 2 error reply, 4 connection/other."""
    if exc is None:
        return 0, ""
    msg = str(exc)
    if isinstance(exc, response_error):
        return (1, "") if "timed out" in msg.lower() else (2, msg[:200])
    return 4, f"{type(exc).__name__}: {msg[:180]}"


def read_pings(path, t0, t1):
    out = {"falkor": [], "control": []}
    downs = []
    try:
        with open(path) as f:
            for line in f:
                name, sched, lat = line.rstrip("\n").split(",")
                sched = float(sched)
                if not t0 <= sched <= t1:
                    continue
                if lat == "down":
                    downs.append((name, sched))
                else:
                    out[name].append((sched, float(lat)))
    except FileNotFoundError:
        pass  # the prober wrote nothing: no pings, which the checks report as zero pings
    return out, downs


# --- ledger -------------------------------------------------------------------------------------


PAIR_BATCH = 500  # acknowledged pairs looked up per query


class Ledger:
    """What the graphs must contain, from the upserts' outcomes.

    Problems come back as (kind, message):
      count     a label's node count, or a pair type's edge count above what was written
      missing   a pair type has fewer edges than the committed upserts created (Rust #3082 loses them)
      duplicate a key appears on two nodes
      timed-out a timed-out write left a new node or edge behind
    """

    def __init__(self, sizes: list[dict]):
        self.n0 = sizes
        self.ok = [collections.defaultdict(set) for _ in sizes]       # label/type -> keys committed
        self.timeout = [collections.defaultdict(set) for _ in sizes]  # ... of timed-out writes
        self.unknown = [collections.defaultdict(set) for _ in sizes]  # ... with unknown outcome
        self.extra = [collections.Counter() for _ in sizes]           # accepted constraint violations
        self.incomplete = False  # a driver result file was lost: counts can no longer be predicted

    def add(self, g, keys, oc):
        bucket = self.ok if oc == 0 else self.timeout if oc == 1 else self.unknown if oc in (3, 4) else None
        if bucket is None:  # error reply: nothing committed
            return
        for k in keys:
            if len(k) == 2:
                bucket[g][k[0]].add(k[1])
            else:
                bucket[g][k[0]].add((k[1], k[2]))

    def check(self, r) -> list[tuple[str, str]]:
        def one(q):
            return r.execute_command("GRAPH.RO_QUERY", graph, q)[1][0][0]

        problems = []
        for g, n in enumerate(self.n0):
            graph = gname(g)
            for label in model.UPSERT_LABELS:
                new_ok = {k for k in self.ok[g][label] if k >= n[label]}
                new_unk = {k for k in self.unknown[g][label] if k >= n[label]} - new_ok
                lo = n[label] + len(new_ok) + self.extra[g][label]
                hi = lo + len(new_unk)
                c = one(f"MATCH (x:{label}) RETURN count(x)")
                if not self.incomplete and not lo <= c <= hi:
                    problems.append(("count", f"g{g} {label}: {c} nodes, expected {lo}" + (f"..{hi}" if hi != lo else "")))
                dup = one(f"MATCH (x:{label}) WITH x.k AS k, count(*) AS c WHERE c > 1 RETURN count(k)")
                if dup - self.extra[g][label] > 0:
                    problems.append(("duplicate", f"g{g} {label}: {dup} duplicated keys"))
                gone = sorted({k for k in self.timeout[g][label] if k >= n[label]} - self.ok[g][label]
                              - self.unknown[g][label])
                if gone and not self.incomplete:
                    c2 = one(f"CYPHER keys={json.dumps(gone)} MATCH (x:{label}) WHERE x.k IN $keys RETURN count(x)")
                    if c2:
                        problems.append(("timed-out", f"g{g} {label}: {c2} of {len(gone)} timed-out new keys exist"))
            for src, typ, dst in model.PAIRS:
                base = n[dst]
                new_ok = {p for p in self.ok[g][typ] if not model.initial_pair(n, src, dst, *p)}
                new_unk = {p for p in self.unknown[g][typ] if not model.initial_pair(n, src, dst, *p)} - new_ok
                lo, hi = base + len(new_ok), base + len(new_ok) + len(new_unk)
                c = one(f"MATCH ()-[e:{typ}]->() RETURN count(e)")
                if not self.incomplete:
                    if c > hi:
                        problems.append(("count", f"g{g} {typ}: {c} edges, expected at most {hi}"))
                    elif c < lo:
                        problems.append(("missing", f"g{g} {typ}: {c} edges, expected at least {lo}"))
                    # the count can hide a lost edge behind one an unknown-outcome write added: look
                    # each acknowledged new pair up
                    acked = sorted(new_ok)
                    found = 0
                    for b0 in range(0, len(acked), PAIR_BATCH):
                        batch = [list(p) for p in acked[b0:b0 + PAIR_BATCH]]
                        found += one(f"CYPHER pairs={json.dumps(batch)} UNWIND $pairs AS p "
                                     f"MATCH (a:{src} {{k: p[0]}})-[:{typ}]->(b:{dst} {{k: p[1]}}) "
                                     "WITH DISTINCT p RETURN count(p)")
                    if found < len(acked):
                        problems.append(("missing", f"g{g} {typ}: {len(acked) - found} of {len(acked)} "
                                                    "acknowledged new pairs have no edge"))
                # a timed-out write between existing nodes leaves no new node behind, only its edge
                gone = sorted({p for p in self.timeout[g][typ] if not model.initial_pair(n, src, dst, *p)}
                              - self.ok[g][typ] - self.unknown[g][typ])
                if gone and not self.incomplete:
                    pairs = [list(p) for p in gone]
                    c2 = one(f"CYPHER pairs={json.dumps(pairs)} UNWIND $pairs AS p "
                             f"MATCH (a:{src} {{k: p[0]}})-[e:{typ}]->(b:{dst} {{k: p[1]}}) RETURN count(e)")
                    if c2:
                        problems.append(("timed-out", f"g{g} {typ}: {c2} of {len(gone)} timed-out new pairs exist"))
        return problems


def gname(g: int) -> str:
    return f"soak_g{g}"


# --- the checks ----------------------------------------------------------------------------------


def _pool_wait(rec, index):
    return rec[index] if len(rec) > index else 0.0


def analyse(ep, hp, cal, data, pings, downs, events, constraint_info, ledger_problems) -> dict:
    """Every check of one epoch: {"checks": {name: {"pass": bool | None, ...}}, "info": {...}}.
    data["window_s"], when given, is how long the prober ran for the epoch (for no_freeze's coverage)."""
    reads, writes, longs = data["reads"], data["writes"], data["longs"]
    ng = hp["graphs"]
    checks = {}

    # freeze: a FalkorDB PING later than 500 ms while the plain server answered within 100 ms (a
    # reply from it within 1 s either side is needed: without one the late PING is not attributed)
    ctl = sorted(pings["control"])
    ctl_t = [x[0] for x in ctl]
    late = [(s_, l_) for s_, l_ in pings["falkor"] if l_ > FREEZE_PING_MS]
    freeze, unattributed = [], 0
    for s_, l_ in late:
        i, j = bisect.bisect_left(ctl_t, s_ - 1), bisect.bisect_right(ctl_t, s_ + 1)
        if i == j:
            unattributed += 1
        elif max(ctl[k][1] for k in range(i, j)) < CONTROL_OK_MS:
            freeze.append((s_, l_))
    fl = [l_ for _, l_ in pings["falkor"]]
    cl = [l_ for _, l_ in pings["control"]]
    need = PING_COVERAGE * data["window_s"] / PING_INTERVAL_S if data.get("window_s") else 1
    if len(fl) < need or len(cl) < need:
        frozen = None  # the prober did not measure enough of the epoch to say
    elif freeze:
        frozen = False
    else:
        frozen = None if unattributed else True
    checks["no_freeze"] = {"pass": frozen, "freeze_pings": len(freeze), "unattributed_late_pings": unattributed,
                           "pings_needed": int(need),
                           "worst_ping_ms": round(max(fl, default=0), 1),
                           "worst_control_ms": round(max(cl, default=0), 1),
                           "p99_ping_ms": round(pct(fl, 99), 2), "pings": len(fl), "control_pings": len(cl)}

    # timeouts while waiting: a read whose cost alone is under 5% of its budget must never time out
    cheap = {(g, op) for g in range(ng) for op, (_, tmo) in model.READ_OPS.items()
             if cal[g].get(op, 1e9) < model.CHEAP_SHARE * tmo * 1000}
    tos = collections.Counter((r_[0], r_[1]) for r_ in reads if r_[5] == 1)
    pool_limited_reads = [r_ for r_ in reads if _pool_wait(r_, 6) > DRIVER_LAG_S]
    spurious = sum(c for k, c in tos.items() if k in cheap)
    nreads = len(reads)
    checks["no_timeouts_while_waiting"] = {
        "pass": spurious == 0, "spurious_timeouts": spurious,
        "reads": nreads, "timeout_rate": round(sum(tos.values()) / nreads, 5) if nreads else 0,
        "all_read_timeouts": sum(tos.values()),
        "not_judged": sorted(f"g{g}:{op}" for g in range(ng) for op in model.READ_OPS if (g, op) not in cheap)}

    # writer starvation: a write completes within 1 s after the long reads it waited behind
    by_graph = collections.defaultdict(list)
    for g, op, a, b, oc, _ in longs:
        by_graph[g].append((a, b))
    for g in by_graph:
        by_graph[g].sort()

    def wait_bound(g, send, reply):
        """Latest end among long reads on g that started before send+50ms and overlap the wait."""
        end = send
        for a, b in by_graph.get(g, ()):
            if a > send + 0.05:
                break
            if b > send and a < reply:
                end = max(end, b)
        return end

    # C runs a graph's writes one at a time, each taking the lock on its own, so a write first
    # waits for the writes queued ahead of it. Starvation is judged from the moment it reaches the
    # head of its graph's queue: max(send, last reply of the writes sent before it).
    starve = []
    wlat, qwait = [], []
    head_of = {}
    run_max = collections.defaultdict(float)
    for i, w in sorted(enumerate(writes), key=lambda x: x[1][3]):
        g, send, reply = w[0], w[3], w[4]
        head_of[i] = max(send, run_max[g])
        if reply is not None:
            run_max[g] = max(run_max[g], reply)
    for i, w in enumerate(writes):
        g, op, sched, send, reply, oc = w[:6]
        if oc != 0 or _pool_wait(w, 8) > DRIVER_LAG_S:  # a write that waited for a connection: driver-limited
            continue
        wlat.append((reply - send) * 1000)
        head = head_of[i]
        qwait.append(head - send)
        excess = reply - wait_bound(g, head, reply)
        if excess > STARVE_S:
            starve.append(round(excess, 2))
    acked = sum(1 for w in writes if w[5] == 0)
    # every acknowledged write driver-limited: nothing was judged, so the check did not measure
    checks["writers_dont_starve"] = {"pass": None if acked and not wlat else not starve, "slow_writes": len(starve),
                                     "worst_excess_s": max(starve, default=0), "writes": len(wlat),
                                     "write_p99_ms": round(pct(wlat, 99), 1), "write_max_ms": round(max(wlat, default=0), 1),
                                     "queue_wait_max_s": round(max(qwait, default=0), 2)}

    # quiet graphs: graphs with no long reads and no writes this epoch
    quiet = [g for g in range(ng) if g not in ep["busy"]]
    qviol = 0
    qlat = []
    quiet_ok = sum(1 for r_ in reads if r_[0] in quiet and r_[5] == 0)
    for r_ in reads:
        g, op, sched, send, reply, oc = r_[:6]
        if g in quiet and oc == 0 and (send - sched) < DRIVER_LAG_S and _pool_wait(r_, 6) <= DRIVER_LAG_S:
            lat = (reply - sched) * 1000
            qlat.append(lat)
            if lat > cal[g].get(op, 0) + QUIET_SLACK_MS:
                qviol += 1
    checks["quiet_graphs_stay_fast"] = {"pass": None if quiet_ok and not qlat else qviol == 0,
                                        "applies": bool(quiet), "quiet_graphs": quiet,
                                        "slow_reads": qviol, "reads": len(qlat),
                                        "p99_ms": round(pct(qlat, 99), 1), "max_ms": round(max(qlat, default=0), 1)}

    # requests drain: every read completes within 5 s (+ backlog) after the long reads it waited
    # behind (writes are judged by writers_dont_starve, which accounts for their queue)
    tc = hp["thread_count"]
    all_req = [(r_[0], r_[1], r_[3], r_[4], r_[5]) for r_ in reads if _pool_wait(r_, 6) <= DRIVER_LAG_S]
    sends = sorted((x[2], cal[x[0]].get(x[1], 1.0) / 1000) for x in all_req)
    st = [x[0] for x in sends]
    cum = [0.0]
    for _, c in sends:
        cum.append(cum[-1] + c)
    jam = []
    for g, op, send, reply, oc in all_req:
        if oc not in (0, 1):
            continue
        bound = wait_bound(g, send, reply)
        i, j = bisect.bisect_left(st, send), bisect.bisect_right(st, reply)
        backlog = (cum[j] - cum[i]) / tc
        excess = reply - bound
        if excess > DRAIN_S + backlog:
            jam.append(round(excess, 2))
    checks["requests_drain"] = {"pass": not jam and data["unfinished"] == 0, "late_requests": len(jam),
                                "pool_limited_reads": len(pool_limited_reads),
                                "worst_excess_s": max(jam, default=0), "unfinished_at_end": data["unfinished"],
                                "pending_at_stop": data["pending_at_stop"], "drain_s": round(data["drain_s"], 1)}

    # what the graphs hold, from the ledger
    by_kind = collections.defaultdict(list)
    for kind, msg in ledger_problems:
        by_kind[kind].append(msg)
    # "harness": the end-of-epoch queries themselves failed, so nothing was counted (pass None)
    unknown = by_kind["harness"]
    # a duplicated key is wrong whether or not a constraint is installed, so it counts here in every epoch
    checks["writes_add_up"] = {"pass": None if unknown else not (by_kind["count"] or by_kind["duplicate"]),
                               "problems": by_kind["count"] + by_kind["duplicate"] + unknown}
    checks["upsert_edges_complete"] = {"pass": None if unknown else not by_kind["missing"],
                                       "problems": by_kind["missing"] + unknown}
    wto = sum(1 for w in writes if w[5] == 1)
    checks["timed_out_writes_leave_nothing"] = {"pass": not by_kind["timed-out"], "applies": ep["write_timeout_ms"] > 0,
                                                "timed_out_writes": wto, "problems": by_kind["timed-out"]}

    # constraints
    dups = by_kind["duplicate"]
    if ep["constraints"]:
        states = constraint_info.get("states", {})  # the graphs they were created on
        bad_state = {g: st for g, st in states.items() if not st or any(x != "OPERATIONAL" for x in st)}
        acc = constraint_info.get("violations_accepted", [])
        held = not bad_state and not acc and not dups
        checks["constraints_hold"] = {"pass": held if states or bad_state or acc or dups else None, "applies": True,
                                      "not_operational": bad_state, "seconds": constraint_info.get("seconds"),
                                      "violations_rejected": constraint_info.get("violations_rejected", 0),
                                      "violations_accepted": acc, "duplicates": dups}
    else:
        checks["constraints_hold"] = {"pass": not dups, "applies": False, "duplicates": dups}

    # error replies and connection errors on any request (an engine that rejects or drops a query)
    by_op = collections.Counter(r_[1] for r_ in reads if r_[5] in (2, 4))
    by_op.update(w[1] for w in writes if w[5] in (2, 4))
    by_op.update(x[1] for x in longs if x[4] in (2, 4))
    checks["no_error_replies"] = {"pass": not by_op, "by_op": dict(by_op), "first_messages": dict(data["errors"]),
                                  "long_read_messages": sorted({x[5] for x in longs if x[4] in (2, 4)})[:5]}

    # unexpected errors (long mode): error replies, driver failures and failed epoch events
    errs = dict(data["errors"])
    died = data["died"]
    other_long = [x for x in longs if x[4] in (2, 4)]
    err_reads = sum(1 for r_ in reads if r_[5] in (2, 4))
    err_writes = sum(1 for w in writes if w[5] in (2, 4))
    ev_bad = [e for e in events if e["event"] in ("index_error", "constraint_error", "events_died",
                                                    "bgsave_error", "violation_odd_error", "indexes_stuck",
                                                    "events_stuck", "cleanup_incomplete")]
    checks["no_unexpected_errors"] = {"pass": not (errs or died or other_long or err_reads or err_writes or ev_bad),
                                      "error_messages": errs, "died": died, "long_read_errors": len(other_long),
                                      "read_errors": err_reads, "write_errors": err_writes, "event_errors": ev_bad}

    long_d = collections.defaultdict(list)
    for g, op, a, b, oc, _ in longs:
        long_d[op].append(b - a)
    rl = collections.defaultdict(list)
    for r_ in reads:
        g, op, sched, send, reply, oc = r_[:6]
        if oc == 0:
            rl[op].append((reply - sched) * 1000)
    info = {
        "reads_by_op": {op: {"n": len(v), "p50_ms": round(pct(v, 50), 1), "p99_ms": round(pct(v, 99), 1),
                             "max_ms": round(max(v, default=0), 1)} for op, v in rl.items()},
        "long_reads": {op: {"n": len(v), "mean_s": round(statistics.mean(v), 2) if v else 0,
                            "max_s": round(max(v, default=0), 2),
                            "timed_out": sum(1 for x in longs if x[1] == op and x[4] == 1)} for op, v in long_d.items()},
        "harness": data["harness"], "server_downs": len(downs),
    }
    return {"checks": checks, "info": info}
