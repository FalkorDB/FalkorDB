"""Reproductions of open S1 (critical) issues that affect only the Rust engine.

One class per issue. Each docstring links the issue, names the trigger and the
behaviour the asserts expect, so every test fails on the current Rust engine
and passes once its issue is fixed. Several of them crash or hang the server,
so each class should be run on its own (see the PR description for how).

This file is deliberately NOT listed in flow_tests_done.txt: CI does not run
it. When an issue is fixed, move its class into the suite that covers the
feature (and into CI), and delete it from here.
"""

from common import *

import base64
import os
import socket
import struct
import time
import zlib

import redis
from redis import ResponseError


def _expect_error(env, graph, query, *fragments):
    """Run `query`, expect a ResponseError whose message contains every
    fragment, then check the server is still up."""
    try:
        graph.query(query)
        env.assertTrue(False, message=f"expected an error for: {query[:80]}")
    except ResponseError as e:
        for f in fragments:
            env.assertContains(f, str(e))
    env.assertTrue(env.getConnection().ping())


class testIssue3304():
    """https://github.com/FalkorDB/FalkorDB/issues/3304
    A long chain of clauses (2000 UNWINDs, 300 OPTIONAL MATCH ... WHERE) builds
    a plan as tall as the query and overflows a worker stack, crashing the
    server. C runs both; the query must run (or fail cleanly) and the server
    must stay up."""

    def __init__(self):
        self.env, self.db = Env()
        self.graph = self.db.select_graph("issue_3304")

    def test01_long_unwind_chain_does_not_crash(self):
        q = " ".join(f"UNWIND [1] AS x{i}" for i in range(2000)) + " RETURN count(*)"
        res = self.graph.query(q)
        self.env.assertEqual(res.result_set, [[1]])
        self.env.assertTrue(self.env.getConnection().ping())

    def test02_long_optional_match_where_chain_does_not_crash(self):
        self.graph.query("CREATE (:N)-[:R]->(:N)")
        q = ("MATCH (a) WITH a LIMIT 1 " +
             " ".join(f"OPTIONAL MATCH (a)-->(b{i}) WHERE b{i}.v > 0"
                      for i in range(300)) +
             " RETURN count(*)")
        res = self.graph.query(q)
        self.env.assertEqual(res.result_set, [[1]])
        self.env.assertTrue(self.env.getConnection().ping())


class testIssue3279():
    """https://github.com/FalkorDB/FalkorDB/issues/3279
    Clauses after CALL ... YIELD are not validated: a trailing MATCH panics the
    planner and crashes the server, a trailing UNWIND/OPTIONAL MATCH runs, and
    CREATE/MERGE accept two relationship types. All must be rejected as C does."""

    def __init__(self):
        self.env, self.db = Env()
        self.graph = self.db.select_graph("issue_3279")
        self.graph.query("CREATE (:N)")

    def test01_trailing_unwind_or_optional_match_is_rejected(self):
        _expect_error(self.env, self.graph,
                      "CALL db.labels() YIELD label UNWIND [1] AS x",
                      "Query cannot conclude with UNWIND")
        _expect_error(self.env, self.graph,
                      "CALL db.labels() YIELD label OPTIONAL MATCH (n)",
                      "Query cannot conclude with MATCH")

    def test02_create_merge_with_two_relationship_types_is_rejected(self):
        _expect_error(self.env, self.graph,
                      "CALL db.labels() YIELD label CREATE (a)-[:R|S]->(b)",
                      "Exactly one relationship type must be specified",
                      "CREATE")
        _expect_error(self.env, self.graph,
                      "CALL db.labels() YIELD label MERGE (a)-[:R|S]->(b)",
                      "Exactly one relationship type must be specified",
                      "MERGE")
        # nothing may have been written
        res = self.graph.query("MATCH (n) RETURN count(n)")
        self.env.assertEqual(res.result_set, [[1]])
        res = self.graph.query("MATCH ()-[r]->() RETURN count(r)")
        self.env.assertEqual(res.result_set, [[0]])

    def test03_trailing_match_is_rejected_without_crash(self):
        _expect_error(self.env, self.graph,
                      "CALL db.labels() YIELD label MATCH (n)",
                      "Query cannot conclude with MATCH")
        _expect_error(self.env, self.graph,
                      "CALL db.labels() YIELD label WITH label MATCH (n)",
                      "Query cannot conclude with MATCH")


class testIssue3003():
    """https://github.com/FalkorDB/FalkorDB/issues/3003
    An unterminated relationship filter inside shortestPath/allShortestPaths
    sends the parser into an infinite loop, so the query never returns. It
    must come back promptly with a syntax error, as in C."""

    QUERIES = [
        "RETURN shortestPath((a)-[{",
        "RETURN shortestPath((a)-[$p",
        "RETURN shortestPath((a)-[{x: [1, {y: 2",
        "MATCH (a), (b) RETURN shortestPath((a)-[* {x: 1",
        "MATCH (a), (b) RETURN allShortestPaths((a)-[:R {x: 1",
    ]

    def __init__(self):
        self.env, self.db = Env()
        self.graph_id = "issue_3003"

    def test01_unterminated_edge_filter_is_a_syntax_error(self):
        for q in self.QUERIES:
            # a short deadline bounds the hang; each hung query pins a worker
            conn = redis.Redis(host=self.env.host, port=self.env.port,
                               decode_responses=True, socket_timeout=30,
                               retry=NO_RETRY)
            try:
                conn.execute_command("GRAPH.QUERY", self.graph_id, q)
                self.env.assertTrue(False, message=f"expected a syntax error: {q}")
            except ResponseError as e:
                self.env.assertContains("Invalid input", str(e))
            except redis.TimeoutError:
                self.env.assertTrue(False, message=f"query hung: {q}")
                return  # don't pin more workers
            finally:
                conn.close()
        self.env.assertTrue(self.env.getConnection().ping())


class testIssue2895():
    """https://github.com/FalkorDB/FalkorDB/issues/2895
    SET/REMOVE property chains (n.a.a.a... 5000 deep) and 500 nested FOREACH
    bypass the parser depth guard and overflow the stack, crashing the server.
    They must be rejected with a clean error (C rejects the chains and runs the
    FOREACH) and the server must stay up."""

    def __init__(self):
        self.env, self.db = Env()
        self.graph = self.db.select_graph("issue_2895")
        self.graph.query("CREATE (:N)")

    def test01_deep_set_property_chain_does_not_crash(self):
        _expect_error(self.env, self.graph,
                      "MATCH (n) SET n" + ".a" * 5000 + " = 1")

    def test02_deep_remove_property_chain_does_not_crash(self):
        _expect_error(self.env, self.graph,
                      "MATCH (n) REMOVE n" + ".a" * 5000)

    def test03_deeply_nested_foreach_does_not_crash(self):
        q = "FOREACH (x IN [1] | " * 500 + "CREATE ()" + ")" * 500
        try:
            # C runs it; a clean nesting error is acceptable too
            res = self.graph.query(q)
            self.env.assertEqual(res.nodes_created, 1)
        except ResponseError as e:
            self.env.assertContains("nesting", str(e))
        self.env.assertTrue(self.env.getConnection().ping())


class testIssue2930():
    """https://github.com/FalkorDB/FalkorDB/issues/2930
    date()/localdatetime() panic on an out-of-range week or dayOfQuarter and on
    a multi-byte week string, taking the server down. Each must return an
    error (as C does) and the server must stay up."""

    def __init__(self):
        self.env, self.db = Env()
        self.graph = self.db.select_graph("issue_2930")

    def test01_day_of_quarter_out_of_range_is_rejected(self):
        # no crash, but Rust returns 2021-07-19 instead of an error
        _expect_error(self.env, self.graph,
                      "RETURN date({year:2021, quarter:1, dayOfQuarter:200})",
                      "dayOfQuarter")

    def test02_huge_week_is_rejected(self):
        _expect_error(self.env, self.graph,
                      "RETURN date({year:2020, week:10000000000})", "week")

    def test03_huge_day_of_quarter_is_rejected(self):
        _expect_error(self.env, self.graph,
                      "RETURN date({year:2020, quarter:2, dayOfQuarter:9223372036854775807})",
                      "dayOfQuarter")

    def test04_multibyte_week_string_is_rejected(self):
        _expect_error(self.env, self.graph, "RETURN date('2020W1é')")

    def test05_huge_negative_week_does_not_crash(self):
        # C returns a value here rather than an error; either is fine as long
        # as the server survives
        try:
            self.graph.query("RETURN localdatetime({year:2020, week:-9223372036854775807})")
        except ResponseError:
            pass
        self.env.assertTrue(self.env.getConnection().ping())


class testIssue3150():
    """https://github.com/FalkorDB/FalkorDB/issues/3150
    A RETURN that projects a null grouping key before an aggregate
    (e.g. `RETURN null AS x, count(1) AS n`) panics the server in reply.rs.
    Expected: one row with the null key and the aggregate value; the server stays up."""

    def __init__(self):
        self.env, self.db = Env()
        self.conn = self.env.getConnection()
        self.conn.delete("issue3150")
        self.g = self.db.select_graph("issue3150")

    def test01_null_literal_before_count(self):
        res = self.g.query("RETURN null AS x, count(1) AS n")
        self.env.assertEqual(res.result_set, [[None, 1]])
        self.env.assertTrue(self.conn.ping())

    def test02_null_literal_before_collect(self):
        res = self.g.query("RETURN null AS x, collect(1) AS c")
        self.env.assertEqual(res.result_set, [[None, [1]]])
        self.env.assertTrue(self.conn.ping())

    def test03_missing_property_before_count(self):
        self.g.query("CREATE (:B {id:'b'})")
        res = self.g.query("MATCH (b:B) RETURN b.id AS k, b.missing AS x, count(b) AS n")
        self.env.assertEqual(res.result_set, [['b', None, 1]])
        self.env.assertTrue(self.conn.ping())


class testIssue2972():
    """https://github.com/FalkorDB/FalkorDB/issues/2972
    A `MATCH ... WHERE` followed by a MATCH that traverses from it loses the
    earlier WHERE (and clauses planned below it), so unmatched rows flow on, up to DETACH DELETE.
    Expected: the earlier clause is kept as outer context, as in C."""

    def __init__(self):
        self.env, self.db = Env()
        self.conn = self.env.getConnection()
        self.conn.delete("issue2972", "issue2972_varlen")
        self.g = self.db.select_graph("issue2972")
        self.g.query("CREATE (a:A {v:1})-[:R {w:2}]->(b:B {v:0}), (:B {v:5}), "
                     "(a)-[:R {w:3}]->(b), (b)-[:S {w:1}]->(:C {v:7})")
        self.gv = self.db.select_graph("issue2972_varlen")
        self.gv.query("CREATE (a1 {v:1})-[:R]->(:B {w:2}), (a2 {v:0})-[:R]->(:B {w:2}), "
                      "(a3 {v:0})-[:R]->(:X)-[:R]->(:B {w:2})")

    def test01_where_of_previous_match_is_kept(self):
        res = self.g.query("MATCH (a) WHERE a.v = 99 MATCH (a)-[r:R]->(b) RETURN count(r)")
        self.env.assertEqual(res.result_set, [[0]])
        res = self.g.query("MATCH (a:A) WHERE a.v = 2 MATCH (a)-[r:R]->(b:B) RETURN count(r)")
        self.env.assertEqual(res.result_set, [[0]])
        res = self.g.query("MATCH (a) WHERE a.v = 0 MATCH (x)-[:S]->(a) RETURN count(x)")
        self.env.assertEqual(res.result_set, [[0]])

    def test02_unwind_below_previous_match_is_kept(self):
        res = self.g.query("UNWIND [1,2] AS i MATCH (a) WHERE a.v = 1 "
                           "MATCH (a)-[r:R]->(b) RETURN count(r)")
        self.env.assertEqual(res.result_set, [[4]])

    def test03_variable_of_previous_match_stays_bound(self):
        res = self.g.query("MATCH (x:C) WHERE x.v = 7 MATCH (a)-[r:R]->(b) RETURN x.v, count(r)")
        self.env.assertEqual(res.result_set, [[7, 2]])

    def test04_var_len_where_of_previous_match_is_kept(self):
        res = self.gv.query("MATCH (a) WHERE a.v = 1 MATCH (a)-[*1..2]->(b:B {w:2}) RETURN a.v, b.w")
        self.env.assertEqual(res.result_set, [[1, 2]])
        res = self.gv.query("UNWIND [1,2] AS i MATCH (a) WHERE a.v = 1 "
                            "MATCH (a)-[*1..2]->(b:B {w:2}) RETURN count(*)")
        self.env.assertEqual(res.result_set, [[2]])

    def test05_detach_delete_does_not_remove_unmatched_data(self):
        res = self.g.query("MATCH (a) WHERE a.v = 99 MATCH (a)-[:R]->(b) DETACH DELETE b")
        self.env.assertEqual(res.nodes_deleted, 0)
        self.env.assertEqual(res.relationships_deleted, 0)
        res = self.gv.query("MATCH (a) WHERE a.v = 99 MATCH (a)-[*1..2]->(b:B {w:2}) DETACH DELETE b")
        self.env.assertEqual(res.nodes_deleted, 0)
        self.env.assertEqual(res.relationships_deleted, 0)
        self.env.assertEqual(self.gv.query("MATCH (n) RETURN count(n)").result_set, [[7]])


class testIssue2894():
    """https://github.com/FalkorDB/FalkorDB/issues/2894
    `UNWIND range(a, b, step)` whose range ends within one step of i64::MAX/MIN never
    terminates (the iterator's step past the end wraps around). Expected: exactly the
    range's elements, like the eager range(). Queries carry a timeout so a hang fails cleanly."""

    TIMEOUT_MS = 10000

    def __init__(self):
        self.env, self.db = Env()
        self.conn = self.env.getConnection()
        self.conn.delete("issue2894")
        self.g = self.db.select_graph("issue2894")

    def _q(self, q):
        return self.g.query(q, timeout=self.TIMEOUT_MS).result_set

    def test01_ascending_range_ending_at_i64_max(self):
        self.env.assertEqual(
            self._q("UNWIND range(9223372036854775806, 9223372036854775807) AS x RETURN count(x)"),
            [[2]])
        self.env.assertTrue(self.conn.ping())

    def test02_descending_range_ending_at_i64_min(self):
        self.env.assertEqual(
            self._q("UNWIND range(-9223372036854775807, -9223372036854775807 - 1, -1) AS x RETURN count(x)"),
            [[2]])
        self.env.assertTrue(self.conn.ping())

    def test03_stepped_range_near_i64_max(self):
        self.env.assertEqual(
            self._q("UNWIND range(9223372036854775800, 9223372036854775807, 3) AS x RETURN collect(x)"),
            [[[9223372036854775800, 9223372036854775803, 9223372036854775806]]])
        self.env.assertTrue(self.conn.ping())

    def test04_step_i64_min(self):
        self.env.assertEqual(
            self._q("UNWIND range(0, -10, -9223372036854775807 - 1) AS x RETURN x"),
            [[0]])
        self.env.assertTrue(self.conn.ping())


class testIssue2769():
    """https://github.com/FalkorDB/FalkorDB/issues/2769
    outdegree()/indegree() evaluated after an edge was created and deleted in the same
    query panics ("relationship N not found") and kills the server.
    Expected: the degree is 0 and the server stays up."""

    def __init__(self):
        self.env, self.db = Env()
        self.conn = self.env.getConnection()
        self.conn.delete("issue2769")
        self.g = self.db.select_graph("issue2769")

    def test01_outdegree_after_deleting_pending_edge(self):
        res = self.g.query("CREATE (a:A)-[r:R]->(b:B) DELETE r SET a.d = outdegree(a) RETURN a.d")
        self.env.assertEqual(res.result_set, [[0]])
        self.env.assertTrue(self.conn.ping())

    def test02_indegree_after_deleting_pending_edge(self):
        res = self.g.query("CREATE (a:A)-[r:R]->(b:B) DELETE r SET b.d = indegree(b) RETURN b.d")
        self.env.assertEqual(res.result_set, [[0]])
        self.env.assertTrue(self.conn.ping())


# A genuine GRAPH.RESTORE payload for the graph
#     CREATE (a:A)-[:R]->(b:B), (a)-[:R]->(b)
# (two parallel A->B edges, i.e. one multi-edge pair), captured from
# `GRAPH.COPY`'s AOF propagation on falkordb/falkordb-server:edge (Rust engine,
# module ver 999999) on 2026-10-13, then zlib-compressed and base64-encoded so
# the tests can reconstruct the exact bytes. It round-trips cleanly through a
# healthy GRAPH.RESTORE and yields two edges.
#
# The payload is in the Rust engine's v19 BufferedWriter framing. The C engine
# uses a different on-disk format and crashes on this payload regardless of any
# corruption, so these tests are Rust-only; the C column is recorded n/a.
_GOOD_PAYLOAD_ZB64 = (
    "eNrtmsFOAjEQhmd2Da5Igj6Db4DGeCEqF2JiJCF41pt3n4AL8Ao+gwcw8WE8+DBIYneBabt0FUK"
    "3+89lSnen2y39dv42pZh+7aVFB5EqLwukFbJLbC+QtR1Ka25NbbO8q5N3l95o33R7TYvLehVrNT"
    "Wt5lALT7SaI/sIkMMlKjKAz2nVzbpPY+fKsrHckf27/e7153y+8BPlx8qPlJ8qH6ix7x1MZEfTW"
    "d597Tx1er17AxKxFp21UF+JvnsYnLeKhR+vhD+q+Mgtnoo/nkzvvAi7MlAbb0STBJreT62pQHEs"
    "UO2GjWZ5TEcyZ3q6MEEFkDIh6cJU3mMJKCJLAsWAUFyK+mCUa8XTY0WUK+9XuTKUK9Il0iWUKxa"
    "RMP9Q5Aoo141Mlk652tLhTPkPMOmFclXzbajhcXnhLD7/IB6hXfcJ50RAOBOQjgCnH2aC0iFl9q"
    "FeywPjpgw5AYyA0XP9qh/VIFlD38Lz6eIl399SUJssgOJ1hZL9Tix/z4kG5LBd/zprN9XYsfook"
    "PAsfGJtb9mX9X7qy5lwz07YJEOghrMTUPG+b3vZBAO2vcq87eWUv6HhkSVhu0aRK4hiJBRvimKE"
    "rAjB6rdgzRGcDREe2bDMzus3tt2F7eKJTAmDaPUiU2L9CPMMRQ4bxR+srYMm"
)

# Offsets into the decompressed payload, from a TLV walk of these exact bytes.
# The tensor section for the one multi-edge pair is:
#   8667 UNSIGNED 2   total edge_count
#   8676 UNSIGNED 1   base group count (number of multi-edge pairs)
#   8685 UNSIGNED 0   src node id
#   8694 UNSIGNED 1   dst node id
#   8703 BYTES  219   GxB-serialized bool id-vector for this pair
#   8931 UNSIGNED 0   delta-plus group count
# TYPE tags: UNSIGNED=4, BYTES=0. Every method asserts the tags it edits so a
# format drift fails loudly rather than silently editing the wrong byte.
_OFF_GROUP_COUNT = 8676   # tag at 8676, u64 value at 8677..8685
_OFF_ENTRY = 8685         # first tensor entry: src(9)+dst(9)+blob(9+219)
_OFF_BLOB = 8703          # BYTES tag; u64 length at 8704..8712; 219 data bytes
_OFF_AFTER_ENTRY = 8931   # delta-plus group count onward


def _good_payload():
    return zlib.decompress(base64.b64decode(_GOOD_PAYLOAD_ZB64))


def _assert_server_alive(env):
    # A fresh connection: if the module panicked the process is gone and this
    # raises, which is the failure these tests guard against.
    env.assertTrue(env.getConnection().ping())


class testIssue2537():
    """https://github.com/FalkorDB/FalkorDB/issues/2537

       GRAPH.RESTORE with a zero-length payload reaches BufferedReader's
       stream-refill path with a null RedisModuleIO and SIGSEGVs the server.
       Expected: a clean client error and the server stays up."""

    def __init__(self):
        self.env, self.db = Env()
        self.con = self.env.getConnection()

    def test01_empty_payload_is_rejected_not_fatal(self):
        try:
            self.con.execute_command("GRAPH.RESTORE", "empty_restore", b"")
            raised = False
        except (ResponseError, redis.ConnectionError) as e:
            raised = True
            # A ResponseError is the correct outcome; a dropped connection means
            # the process crashed, which is the bug.
            self.env.assertIsInstance(
                e, ResponseError,
                message=f"empty GRAPH.RESTORE must be a clean error, got {e!r}")
        self.env.assertTrue(raised, message="empty GRAPH.RESTORE must be refused")
        _assert_server_alive(self.env)


class testIssue2994():
    """https://github.com/FalkorDB/FalkorDB/issues/2994

       GRAPH.RESTORE of a payload whose multi-edge id blob is corrupt makes
       Vector<bool>::decode_blob assert on GxB_Vector_deserialize returning
       GrB_INVALID_OBJECT, and the panic hook kills the server. Expected: the
       load is refused with an error and the server stays up."""

    def __init__(self):
        self.env, self.db = Env()
        self.con = self.env.getConnection()

    def test01_corrupt_multi_edge_blob_is_rejected_not_fatal(self):
        b = _good_payload()
        # Truncate the multi-edge id blob to its first 8 bytes (rewrite the BYTES
        # record's length to 8) so GxB_Vector_deserialize gets an invalid blob.
        self.env.assertEqual(b[_OFF_BLOB], 0)  # BYTES tag
        self.env.assertEqual(
            int.from_bytes(b[_OFF_BLOB + 1:_OFF_BLOB + 9], "little"), 219)
        blob = b[_OFF_BLOB + 9:_OFF_BLOB + 9 + 219]
        tail = b[_OFF_BLOB + 9 + 219:]
        corrupt = (b[:_OFF_BLOB]
                   + bytes([0]) + (8).to_bytes(8, "little") + blob[:8]
                   + tail)

        try:
            self.con.execute_command("GRAPH.RESTORE", "corrupt_blob", corrupt)
            raised = False
        except (ResponseError, redis.ConnectionError) as e:
            raised = True
            self.env.assertIsInstance(
                e, ResponseError,
                message=f"corrupt-blob RESTORE must be a clean error, got {e!r}")
        self.env.assertTrue(raised,
            message="RESTORE of a corrupt multi-edge blob must be refused")
        _assert_server_alive(self.env)


class testIssue3031():
    """https://github.com/FalkorDB/FalkorDB/issues/3031

       GRAPH.RESTORE accepts a tensor section with more multi-edge id rows than
       there are MULTI_EDGE markers (an orphan 'me' row for a pair that is not
       multi-edge). That breaks the invariant edge_count relies on: edge_count
       panics ('multi exceeds the effective pattern'), crashing the server on
       the restore / next save. Expected: the load is refused and the server
       stays up."""

    def __init__(self):
        self.env, self.db = Env()
        self.con = self.env.getConnection()

    def test01_orphan_me_row_is_rejected_not_fatal(self):
        b = _good_payload()
        # Bump the base group count 1 -> 2 and append a second tensor entry for a
        # pair (7, 9) that has no MULTI_EDGE marker in the forward matrix, reusing
        # the captured id blob. The decoder then builds an 'me' row with no
        # backing marker.
        self.env.assertEqual(b[_OFF_GROUP_COUNT], 4)  # UNSIGNED tag
        self.env.assertEqual(
            int.from_bytes(b[_OFF_GROUP_COUNT + 1:_OFF_GROUP_COUNT + 9], "little"), 1)
        entry1 = b[_OFF_ENTRY:_OFF_AFTER_ENTRY]        # src(9)+dst(9)+blob(9+219)
        self.env.assertEqual(entry1[0], 4)             # src UNSIGNED tag
        self.env.assertEqual(entry1[9], 4)             # dst UNSIGNED tag
        self.env.assertEqual(entry1[18], 0)            # blob BYTES tag
        entry2 = (entry1[0:1] + (7).to_bytes(8, "little")
                  + entry1[9:10] + (9).to_bytes(8, "little")
                  + entry1[18:])
        corrupt = (b[:_OFF_GROUP_COUNT + 1] + (2).to_bytes(8, "little")
                   + entry1 + entry2 + b[_OFF_AFTER_ENTRY:])

        try:
            self.con.execute_command("GRAPH.RESTORE", "orphan_me", corrupt)
            # If the restore succeeded, a save must not panic either.
            self.con.execute_command("GRAPH.QUERY", "orphan_me",
                                     "MATCH ()-[e]->() RETURN count(e)")
            raised = False
        except (ResponseError, redis.ConnectionError) as e:
            raised = True
            self.env.assertIsInstance(
                e, ResponseError,
                message=f"orphan-row RESTORE must be a clean error, got {e!r}")
        self.env.assertTrue(raised,
            message="RESTORE of an orphan multi-edge row must be refused")
        _assert_server_alive(self.env)


class testIssue3266():
    """https://github.com/FalkorDB/FalkorDB/issues/3266

    A nested-list SIValue in a client GRAPH.EFFECT payload makes the v3 decoder
    reserve memory quadratic in the payload size before it rejects it. Expected:
    the malformed payload is refused immediately with a bounded amount of memory
    and the server stays alive. Scaled down to a 128 KB payload: on the unfixed
    decoder that already drives `used_memory_peak` to ~4.4 GB; the test asserts
    the peak barely moves. Reserved address space only, so safe on the laptop.
    """

    def __init__(self):
        self.env, self.db = Env()
        self.conn = self.env.getConnection()

    @staticmethod
    def _payload(levels):
        # One CREATE_NODE record with one attribute whose value is `levels`
        # nested T_ARRAY headers; level k claims the most children the trailing
        # bytes could hold, so every level passes the per-level width check.
        T_ARRAY = 1 << 3
        value = b''.join(
            struct.pack('<II', T_ARRAY, 2 * (levels - k - 1))
            for k in range(levels))
        rec = struct.pack('<IIHHHI', 3, 1, 0, 1, 0, 1) + bytes([0, 0, 1])
        return bytes([3, 0]) + rec + value

    def test01_nested_list_effect_payload_is_bounded(self):
        # 128 KB payload (16000 nested levels). On the unfixed Rust decoder this
        # alone pushes `used_memory_peak` to ~4.4 GB before the buffer is
        # rejected; the correct decoder — and C, which has no v3 decoder and
        # refuses the version byte outright — reserves only a few MB.
        levels = 16000
        payload = self._payload(levels)

        peak_before = self.conn.info("memory")["used_memory_peak"]

        rejected = True
        try:
            self.conn.execute_command("GRAPH.EFFECT", "effbomb", payload)
            rejected = False  # a malformed buffer must never be accepted
        except redis.exceptions.ResponseError:
            rejected = True

        peak_after = self.conn.info("memory")["used_memory_peak"]
        growth = peak_after - peak_before

        # The server must still be answering after the payload.
        self.env.assertTrue(self.conn.ping())

        # The payload must have been refused, not accepted.
        self.env.assertTrue(
            rejected, message="malformed GRAPH.EFFECT payload was not refused")

        # Decoding a 128 KB buffer must not spike peak memory into the GBs. A
        # linear decoder stays well under 256 MB here; the quadratic one blows
        # past 4 GB.
        self.env.assertLess(
            growth, 256 * 1024 * 1024,
            message=f"GRAPH.EFFECT decode reserved {growth / 2**30:.2f} GB of "
                    f"peak memory for a 128 KB payload (quadratic reservation)")


class testIssue2946():
    """https://github.com/FalkorDB/FalkorDB/issues/2946

    A write whose entities are all created and cancelled in one query (e.g.
    `CREATE (a:A {p:1}) DELETE a`) still grows the primary's schema/id space but
    puts nothing on the replication stream. The next normal write's ADD_SCHEMA
    is then refused on the replica and forces a full resync. Expected: the
    replica converges with the primary and no forced resync happens. Run with
    REPLICA=1.
    """

    def __init__(self):
        if SANITIZER:
            Environment.skip(None)
        self.env, self.db = Env(env='oss', useSlaves=True)
        self.master = self.env.getConnection()
        self.replica = self.env.getSlaveConnection()
        self.graph_id = "cancel_repl"

    def _wait_offset(self, timeout=60):
        target = self.master.info("replication")["master_repl_offset"]
        deadline = time.time() + timeout
        while time.time() < deadline:
            if self.replica.info("replication").get("slave_repl_offset", -1) >= target:
                return
            time.sleep(0.01)
        raise AssertionError("replica did not catch up to primary offset")

    def test01_cancelled_write_then_next_write_does_not_diverge(self):
        m = self.db.select_graph(self.graph_id)

        full_before = self.master.info("stats")["sync_full"]

        # A write whose every entity is cancelled in the same query: it still
        # registers label A and attribute p on the primary, but replicates
        # nothing on the unfixed build.
        m.query("CREATE (a:A {p:1}) DELETE a")
        # (The issue's `OPTIONAL MATCH (n:Nope) SET n:L6` schema-only variant is
        # omitted: it crashes the C engine on a separate, unrelated C-side bug
        # — blocked.c — which would mask the replication comparison here.)
        # The next ordinary write carries ADD_SCHEMA for label B; on the unfixed
        # build the replica assigns a different id and is forced to resync.
        m.query("CREATE (:B {q:2})")

        self._wait_offset()

        # The replica must agree with the primary. Compare the node count only
        # (the real divergence signal); numeric scalars come back typed
        # differently across the two engines' raw replies.
        res_m = m.ro_query("MATCH (n:B) RETURN count(n)").result_set
        res_r = self.replica.execute_command(
            "GRAPH.RO_QUERY", self.graph_id, "MATCH (n:B) RETURN count(n)")
        # res_r is the raw reply; the rows live in res_r[1].
        self.env.assertEqual(res_m, [[1]])
        self.env.assertEqual(
            res_r[1], [[1]],
            message="replica does not have the B node the primary created "
                    "(replication divergence)")

        # No forced resync: the cancelled/schema-only writes must have been
        # replicated, so the ADD_SCHEMA that followed lined up on the replica.
        full_after = self.master.info("stats")["sync_full"]
        self.env.assertEqual(
            full_after, full_before,
            message="a cancelled / schema-only write forced a full resync "
                    "(replication divergence)")


class testIssue2653():
    """https://github.com/FalkorDB/FalkorDB/issues/2653

    A graph whose Redis key is not valid UTF-8 replicates to a *different* key
    on the replica: the name is lossily derived through to_string_lossy, so the
    bad bytes become U+FFFD before being replicated. Expected: the replica holds
    the byte-exact key and the node is queryable there, matching C. Run with
    REPLICA=1.
    """

    def __init__(self):
        if SANITIZER:
            Environment.skip(None)
        self.env, self.db = Env(env='oss', useSlaves=True)
        self.master = self.env.getConnection()
        self.replica = self.env.getSlaveConnection()
        self.key = b"\xc3\x28bin_2653"  # invalid UTF-8

    def _wait_offset(self, timeout=60):
        target = self.master.info("replication")["master_repl_offset"]
        deadline = time.time() + timeout
        while time.time() < deadline:
            if self.replica.info("replication").get("slave_repl_offset", -1) >= target:
                return
            time.sleep(0.01)
        raise AssertionError("replica did not catch up to primary offset")

    def test01_non_utf8_key_replicates_byte_exact(self):
        # Write a node into a graph under a non-UTF-8 key on the master.
        self.master.execute_command("GRAPH.QUERY", self.key, "CREATE (:A)")
        self._wait_offset()

        # The master owns exactly this byte sequence.
        self.env.assertEqual(self.master.execute_command("EXISTS", self.key), 1)

        # The replica must hold the identical key, not a mangled U+FFFD variant.
        self.env.assertEqual(
            self.replica.execute_command("EXISTS", self.key), 1,
            message="non-UTF-8 graph key did not replicate byte-exact")

        # And the node must be readable on the replica through the real key.
        res = self.replica.execute_command(
            "GRAPH.RO_QUERY", self.key, "MATCH (n:A) RETURN count(n)")
        self.env.assertEqual(res[1], [[1]])


class testIssue2782():
    """https://github.com/FalkorDB/FalkorDB/issues/2782

    GRAPH_REGISTRY is keyed by graph name only, ignoring the Redis logical db,
    so the same key name in two databases makes the second registration displace
    the first. On a release build the first graph is then permanently
    write-locked. Expected: `shared` in db0 and db1 are independent graphs, both
    readable and writable, and the server stays up.
    """

    def __init__(self):
        self.env, self.db = Env()
        self.c0 = self.env.getConnection()  # db 0
        self.c1 = redis.Redis(self.env.host, self.env.port, db=1,
                              decode_responses=True)

    def test01_same_name_two_dbs_stay_independent(self):
        name = "shared_2782"

        # Create the same-named graph in db0 and db1.
        self.c0.execute_command("GRAPH.QUERY", name, "CREATE (:InDB0 {n:'zero'})")
        self.c1.execute_command("GRAPH.QUERY", name, "CREATE (:InDB1 {n:'one'})")

        # The server must still be alive (a debug build would have panicked).
        self.env.assertTrue(self.c0.ping())

        # db0's graph must still be writable — on the unfixed build it has been
        # displaced from the registry and every write aborts with "graph was
        # deleted or replaced while the query was running".
        err = None
        try:
            self.c0.execute_command("GRAPH.QUERY", name, "CREATE (:After)")
        except redis.exceptions.ResponseError as e:
            err = str(e)
        self.env.assertTrue(
            err is None,
            message=f"db0 graph became write-locked after a same-named db1 "
                    f"graph was created: {err!r}")

        # Both graphs keep their own contents.
        r0 = self.c0.execute_command(
            "GRAPH.RO_QUERY", name, "MATCH (n:InDB0) RETURN n.n")
        r1 = self.c1.execute_command(
            "GRAPH.RO_QUERY", name, "MATCH (n:InDB1) RETURN n.n")
        self.env.assertEqual(r0[1], [["zero"]])
        self.env.assertEqual(r1[1], [["one"]])


class testIssue3013():
    """https://github.com/FalkorDB/FalkorDB/issues/3013

    Flooding the server with more queued queries than a fixed 1024-slot channel
    holds deadlocks it permanently: PING never answers and only a restart
    recovers. Expected: the server completes the load (or caps it with
    MAX_QUEUED_QUERIES) and stays responsive, matching C. Bound the hang with
    TEST_TIMEOUT; the flood is spread over 16 graphs (<1024 writes each), the
    variant C completes.
    """

    def __init__(self):
        self.env, self.db = Env()
        self.conn = self.env.getConnection()

    @staticmethod
    def _resp(*args):
        out = f"*{len(args)}\r\n".encode()
        for a in args:
            a = a if isinstance(a, bytes) else a.encode()
            out += f"${len(a)}\r\n".encode() + a + b"\r\n"
        return out

    def test01_flood_does_not_hang_the_server(self):
        host = self.env.host or "127.0.0.1"
        port = int(self.env.port)
        n_queries = 1200   # > the 1024-slot pool queue
        n_graphs = 16      # < 1024 writes per graph: the variant C completes
        # A long-running write per query keeps every pool worker busy, so the
        # bounded queue fills while the main thread is still dispatching — the
        # condition the deadlock needs. Fire all of them from raw sockets
        # without reading replies, so >1024 queries are in flight at once.
        query = "UNWIND range(1,5000) AS x CREATE (:N)"

        socks = []
        try:
            for i in range(n_queries):
                s = socket.socket()
                s.settimeout(30)
                s.connect((host, port))
                socks.append(s)
            for i, s in enumerate(socks):
                s.sendall(self._resp("GRAPH.QUERY", f"flood_{i % n_graphs}", query))

            # The deadlock shows up as the Redis main thread never answering
            # again: a fresh PING on its own connection times out. Poll for a
            # PONG — the server must recover (or never have blocked). On the
            # unfixed build this stays dead until restart.
            pong = False
            deadline = time.time() + 30
            while time.time() < deadline:
                p = socket.socket()
                p.settimeout(4)
                try:
                    p.connect((host, port))
                    p.sendall(self._resp("PING"))
                    if b"PONG" in p.recv(64):
                        pong = True
                        break
                except OSError:
                    pass
                finally:
                    p.close()
                time.sleep(1)

            self.env.assertTrue(
                pong,
                message="server never answered PING after the flood; the main "
                        "thread is deadlocked on a bounded queue")
        finally:
            for s in socks:
                try:
                    s.close()
                except OSError:
                    pass


FORGE_LIB = """
function fn(n) { return {__falkor_type: 'node', __falkor_node_id: n}; }
function fe(n) { return {__falkor_type: 'edge', __falkor_edge_id: n}; }
falkor.register('fn', fn);
falkor.register('fe', fe);
"""


class testIssue2974():
    """https://github.com/FalkorDB/FalkorDB/issues/2974
    A JS UDF that returns a plain object carrying `__falkor_type`/`__falkor_*_id`
    must not be turned into a graph entity. Such a look-alike is a Map: using it
    as a node in CREATE or as an edge in startNode() must raise a type error, not
    create a dangling edge or crash the server.
    """

    def __init__(self):
        self.env, self.db = Env()
        self.conn = self.env.getConnection()
        self.graph = self.db.select_graph("issue_2974")

    def setUp(self):
        self.conn.delete("issue_2974")
        self.db.udf_flush()
        self.db.udf_load("forge", FORGE_LIB, True)
        self.graph.query("CREATE (:A {id: 0})-[:R]->(:A {id: 1})")

    def tearDown(self):
        self.db.udf_flush()
        self.conn.delete("issue_2974")

    def test01_forged_node_is_rejected_by_create(self):
        # node 7 does not exist; a forged reference must not become a CREATE endpoint
        try:
            self.graph.query("WITH forge.fn(7) AS n MATCH (a {id: 0}) "
                             "CREATE (a)-[r:R]->(n) RETURN id(r)")
            self.env.assertTrue(False, message="CREATE accepted a forged node reference")
        except ResponseError:
            pass

        # no dangling edge was created
        res = self.graph.query("MATCH ()-[r]->() RETURN count(r)").result_set
        self.env.assertEqual(res[0][0], 1)
        res = self.graph.ro_query("MATCH (n) RETURN count(n)").result_set
        self.env.assertEqual(res[0][0], 2)

    def test02_forged_edge_is_rejected_by_startNode(self):
        # edge 1000 does not exist; a forged edge must be a Map -> type error
        try:
            self.graph.query("RETURN startNode(forge.fe(1000))")
            self.env.assertTrue(False, message="startNode accepted a forged edge reference")
        except ResponseError:
            pass

        # the server must survive
        self.env.assertTrue(self.conn.ping())
