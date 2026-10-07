from common import *
from rdb_payload import (GraphValue, reframe, crc64, rdb_module_values,
                         T_INT64, T_NULL, T_VECTOR_F32)

import os
import time
import shutil
import tempfile
import subprocess

GRAPH_ID = "g"
FAILED = "Failed loading graph key"


# a small graph touching every part of the encoding the tests below corrupt:
# indexes (range + vector), constraints, properties of several types, a
# multi-edge (tensor), and a deleted node and edge
def _build_graph(conn):
    q = lambda s: conn.execute_command("GRAPH.QUERY", GRAPH_ID, s)
    q("CREATE INDEX FOR (n:N) ON (n.v)")
    q("CREATE VECTOR INDEX FOR (m:V) ON (m.e) "
      "OPTIONS {dimension: 4, similarityFunction: 'euclidean'}")
    q("CREATE (a:N {v: 1, s: 'one'}), (b:N {v: 2}), (c:N {v: 3}), "
      "(:V {e: vecf32([1, 2, 3, 4])}), "
      "(a)-[:R {w: 1}]->(b), (a)-[:R {w: 2}]->(b), (b)-[:R {w: 3}]->(c)")
    q("CREATE (:N {v: 4})-[:R {w: 4}]->(:N {v: 5})")
    q("MATCH (n:N {v: 4}) DETACH DELETE n")
    for kind in ("UNIQUE", "MANDATORY"):
        conn.execute_command("GRAPH.CONSTRAINT", "CREATE", GRAPH_ID, kind,
                             "NODE", "N", "PROPERTIES", 1, "v")

    # only active constraints are encoded
    for _ in range(100):
        res = conn.execute_command("GRAPH.RO_QUERY", GRAPH_ID,
            "CALL db.constraints() YIELD status RETURN status")
        if res[1] and all(row[0] == "OPERATIONAL" for row in res[1]):
            return
        time.sleep(0.1)
    raise AssertionError("constraints did not become operational")


def _log_path(env):
    return os.path.join(env.logDir,
                        env.envRunner._getFileName("master", ".log"))


# server log from byte 'offset' on, None when there is no log file (RLTest
# runs without one when output capturing is disabled)
def _read_log(env, offset=0):
    try:
        with open(_log_path(env), "rb") as f:
            f.seek(offset)
            return f.read().decode(errors="replace")
    except FileNotFoundError:
        return None


def _log_size(env):
    try:
        return os.path.getsize(_log_path(env))
    except FileNotFoundError:
        return 0


# the schema prefix ("node_schema[k]") of the label named 'label'
def _schema(gv, label):
    names = [n[:-len(".name")] for n in gv.find(".name", label.encode() + b"\0")
             if n.startswith("node_schema[") and n.count(".") == 1]
    assert len(names) == 1, names
    return names[0]


# the property prefix ("node[k].property[p]") of the first property of type 't'
def _property_of_type(gv, t):
    names = [n[:-len(".type")] for n in gv.find(".type", t)
             if ".property[" in n]
    assert names, t
    return names[0]


#-------------------------------------------------------------------------------
# a payload with one field corrupted must be rejected, leave no key behind and
# log exactly one line naming the key and what was wrong
#-------------------------------------------------------------------------------

class testRdbMalformed():
    def __init__(self):
        self.env, self.db = Env()
        self.conn = self.env.getConnection()

        self.conn.flushall()
        _build_graph(self.conn)

        # uncompressed, so the walker can read the serializer buffers
        self.conn.config_set("rdbcompression", "no")
        try:
            kw = self.conn.connection_pool.connection_kwargs
            raw = redis.Redis(host=kw.get("host", "localhost"),
                              port=kw["port"], decode_responses=False)
            payload = raw.execute_command("DUMP", GRAPH_ID)
        finally:
            self.conn.config_set("rdbcompression", "yes")

        self.body, self.version = payload[:-10], payload[-10:-8]
        self.conn.flushall()

    # a fresh, parsed copy of the healthy value (offset 1: after the type byte)
    def _value(self):
        return GraphValue(self.body, 1)

    def _assert_rejected(self, gv, *reason):
        log_start = _log_size(self.env)

        failed = False
        try:
            self.conn.restore("bad", 0, reframe(gv.buf, self.version))
        except ResponseError:
            failed = True
        self.env.assertTrue(failed)
        self.env.assertTrue(self.conn.ping())
        self.env.assertEqual(self.conn.keys("*"), [])

        log = _read_log(self.env, log_start)
        if log is None:
            return
        lines = [l for l in log.splitlines() if FAILED in l]
        self.env.assertEqual(len(lines), 1)
        if lines:
            self.env.assertContains(f"{FAILED} 'bad': ", lines[0])
            for r in reason:
                self.env.assertContains(r, lines[0])

    def test00_healthy_payload_restores(self):
        # the walker accounts for every byte of the healthy value, and the
        # value restores: the corruptions below start from a valid payload
        gv = self._value()
        self.env.assertEqual(gv.fields["graph_name"].value, b"g\0")
        self.conn.restore("ok", 0, reframe(gv.buf, self.version))
        res = self.conn.execute_command("GRAPH.RO_QUERY", "ok",
                                        "MATCH (n) RETURN count(n)")
        self.env.assertEqual(res[1], [[5]])
        self.conn.flushall()

    #---------------------------------------------------------------------------
    # entities
    #---------------------------------------------------------------------------

    def test_too_many_properties(self):
        gv = self._value()
        self.env.assertGreater(gv.fields["node[0].property_count"].value, 0)
        gv.set_uint("node[0].property_count", 70000)
        self._assert_rejected(gv, "entity has too many properties")

    def test_invalid_property_type(self):
        # an integer property re-typed as NULL, its value removed
        gv = self._value()
        p = _property_of_type(gv, T_INT64)
        gv.set_uint(f"{p}.type", T_NULL)
        gv.remove_value(f"{p}.value")
        self._assert_rejected(gv, "invalid property value type")

    def test_vector_size(self):
        gv = self._value()
        p = _property_of_type(gv, T_VECTOR_F32)
        size = gv.fields[f"{p}.value"].size
        self.env.assertEqual(size % 4, 0)
        gv.set_length(f"{p}.value", size - 1)
        self._assert_rejected(gv, "vector size is not a multiple of float")

    def test_node_id_out_of_range(self):
        gv = self._value()
        gv.set_uint("node[0].id", 1 << 40)
        self._assert_rejected(gv, "node id out of range")

    def test_edge_id_out_of_range(self):
        gv = self._value()
        gv.set_uint("edge[0].id", 1 << 40)
        self._assert_rejected(gv, "edge id out of range")

    def test_deleted_nodes_buffer_size(self):
        gv = self._value()
        size = gv.fields["deleted_nodes"].size
        self.env.assertEqual(size, 8)
        gv.set_length("deleted_nodes", size - 1)
        self._assert_rejected(gv, "deleted nodes buffer size mismatch")

    def test_deleted_node_id_out_of_range(self):
        gv = self._value()
        gv.set_bytes(gv.fields["deleted_nodes"].off,
                     (1 << 40).to_bytes(8, "little"))
        self._assert_rejected(gv, "deleted node id out of range")

    #---------------------------------------------------------------------------
    # schema
    #---------------------------------------------------------------------------

    def test_schema_out_of_order(self):
        gv = self._value()
        s = _schema(gv, "N")
        self.env.assertEqual(gv.fields[f"{s}.id"].value, 0)
        gv.set_uint(f"{s}.id", 7)
        self._assert_rejected(gv, "duplicate or out of order schema")

    def test_unknown_constraint_type(self):
        gv = self._value()
        s = _schema(gv, "N")
        gv.set_uint(f"{s}.constraint[0].type", 9)
        self._assert_rejected(gv, "unknown constraint type")

    def test_constraint_unknown_attribute(self):
        gv = self._value()
        s = _schema(gv, "N")
        gv.set_uint(f"{s}.constraint[0].attribute[0]", 99)
        self._assert_rejected(gv, "constraint on an unknown attribute")

    def test_duplicate_constraint(self):
        # turn the mandatory constraint into a second unique one
        gv = self._value()
        s = _schema(gv, "N")
        t0 = gv.fields[f"{s}.constraint[0].type"].value
        t1 = gv.fields[f"{s}.constraint[1].type"].value
        self.env.assertNotEqual(t0, t1)
        gv.set_uint(f"{s}.constraint[1].type", t0)
        self._assert_rejected(gv, "duplicate constraint")

    def test_unsupported_index_language(self):
        gv = self._value()
        f = gv.fields[f"{_schema(gv, 'N')}.index.language"]
        self.env.assertEqual(f.value, b"english\0")
        gv.set_bytes(f.off + 6, b"X")
        self._assert_rejected(gv, "unsupported index language")

    def test_invalid_index_field_type(self):
        gv = self._value()
        gv.set_uint(f"{_schema(gv, 'N')}.index.field[0].type", 0x80)
        self._assert_rejected(gv, "invalid index field")

    def test_invalid_vector_metric(self):
        gv = self._value()
        name = f"{_schema(gv, 'V')}.index.field[0].similarity"
        self.env.assertEqual(gv.fields[name].value, 0)   # euclidean
        gv.set_uint(name, 9)
        self._assert_rejected(gv, "invalid index field")

    #---------------------------------------------------------------------------
    # matrices
    #---------------------------------------------------------------------------

    def test_label_matrix_unknown_label(self):
        gv = self._value()
        gv.set_uint("label_matrix[0].label", 99)
        self._assert_rejected(gv, "label matrix for an unknown label")

    def test_relation_matrix_out_of_order(self):
        gv = self._value()
        self.env.assertEqual(gv.fields["relation_matrix[0].relation"].value, 0)
        gv.set_uint("relation_matrix[0].relation", 5)
        self._assert_rejected(gv, "relation matrix out of order")

    def test_tensor_out_of_range(self):
        gv = self._value()
        gv.set_uint("tensor[0].i", 1 << 40)
        self._assert_rejected(gv, "GraphBLAS error", "setting tensor")

    def test_tensor_blob(self):
        gv = self._value()
        gv.set_bytes(gv.fields["tensor[0].blob"].off, b"\xff" * 8)
        self._assert_rejected(gv, "GraphBLAS error", "deserializing tensor")

    def test_matrix_vector_size(self):
        gv = self._value()
        name = "label_matrix[0].M.x.bytes"
        gv.set_uint(name, gv.fields[name].value + 8)
        self._assert_rejected(gv, "matrix vector holds")

    def test_matrix_container_size(self):
        gv = self._value()
        name = "label_matrix[0].M.container"
        gv.set_length(name, gv.fields[name].size - 8)
        self._assert_rejected(gv, "matrix container holds")


#-------------------------------------------------------------------------------
# a damaged dump.rdb: loading it at startup must end in Redis' own exit, with
# the reason in the log, not in a crash
#
# the file is loaded by a second server started from the test server's own
# command line; DEBUG RELOAD can't stand in for a startup load, since Redis
# treats a load run from a client like a RESTORE (logs, replies, keeps going)
#-------------------------------------------------------------------------------

# keep only graph keys (the RDB walker parses module values only), SAVE and
# return the file's bytes
def _save_graph_rdb(conn):
    for k in conn.keys("*"):
        if conn.type(k) != "graphdata":
            conn.delete(k)
    conn.config_set("rdbcompression", "no")
    conn.save()
    path = os.path.join(conn.config_get("dir")["dir"],
                        conn.config_get("dbfilename")["dbfilename"])
    with open(path, "rb") as f:
        return f.read()


def _set_arg(args, name, value):
    if name in args:
        args[args.index(name) + 1] = value
    else:
        args += [name, value]


# start a server on 'rdb' and wait for it to exit; returns (exit code, log)
def _start_on_rdb(env, rdb):
    d = tempfile.mkdtemp()
    try:
        with open(os.path.join(d, "dump.rdb"), "wb") as f:
            f.write(rdb)

        # no TCP port (nothing to collide with), a unix socket in its place:
        # Redis refuses to start without any listener
        args = list(env.envRunner.masterProcess.args)
        _set_arg(args, "--port", "0")
        _set_arg(args, "--unixsocket", os.path.join(d, "redis.sock"))
        _set_arg(args, "--dir", d)
        _set_arg(args, "--dbfilename", "dump.rdb")
        _set_arg(args, "--logfile", os.path.join(d, "redis.log"))

        p = subprocess.Popen(args, cwd=d, stdout=subprocess.DEVNULL,
                             stderr=subprocess.DEVNULL)
        try:
            exit_code = p.wait(timeout=60)
        except subprocess.TimeoutExpired:
            p.kill()
            p.wait()
            exit_code = None
        with open(os.path.join(d, "redis.log"), errors="replace") as f:
            return exit_code, f.read()
    finally:
        shutil.rmtree(d, ignore_errors=True)


# a short read: Redis reports an unexpected EOF, the module adds nothing
def _assert_truncated_load(env, conn, module_type):
    rdb = _save_graph_rdb(conn)
    values = [v for v in rdb_module_values(rdb) if v[1] == module_type]
    env.assertGreater(len(values), 0)
    _, _, start, end = values[0]

    exit_code, log = _start_on_rdb(env, rdb[:(start + end) // 2])
    tail = log[-2000:]
    env.assertEqual(exit_code, 1, message=tail)
    env.assertContains("Internal error in RDB reading", log, message=tail)
    env.assertContains("Unexpected EOF reading RDB file", log, message=tail)
    env.assertNotContains(FAILED, log, message=tail)


class testRdbLoadTruncatedGraphKey():
    def __init__(self):
        # the loading server exits under test; sanitizers report on exit
        if VALGRIND or SANITIZER:
            Environment.skip(None)
        self.env, self.db = Env()
        self.conn = self.env.getConnection()

    def test_truncated_graph_key(self):
        _build_graph(self.conn)
        _assert_truncated_load(self.env, self.conn, "graphdata")


class testRdbLoadTruncatedVirtualKey():
    def __init__(self):
        if VALGRIND or SANITIZER:
            Environment.skip(None)
        # small virtual keys, so the graph is saved across several keys
        self.env, self.db = Env(moduleArgs="VKEY_MAX_ENTITY_COUNT 10")
        self.conn = self.env.getConnection()

    def test_truncated_virtual_key(self):
        self.conn.execute_command("GRAPH.QUERY", GRAPH_ID,
            "UNWIND range(1, 100) AS i CREATE (:N {v: i})")
        _assert_truncated_load(self.env, self.conn, "graphmeta")


class testRdbLoadMalformedGraphKey():
    def __init__(self):
        if VALGRIND or SANITIZER:
            Environment.skip(None)
        self.env, self.db = Env()
        self.conn = self.env.getConnection()

    def test_malformed_graph_key(self):
        _build_graph(self.conn)
        rdb = _save_graph_rdb(self.conn)
        values = [v for v in rdb_module_values(rdb) if v[1] == "graphdata"]
        self.env.assertEqual(len(values), 1)
        _, _, start, _ = values[0]

        # corrupt the index language, then re-stamp the file's checksum so the
        # load fails on the graph data, not on the checksum
        gv = GraphValue(rdb, start)
        f = gv.fields[f"{_schema(gv, 'N')}.index.language"]
        self.env.assertEqual(f.value, b"english\0")
        gv.set_bytes(f.off + 6, b"X")
        bad = bytes(gv.buf[:-8])
        bad += crc64(bad).to_bytes(8, "little")

        # after a failed load Redis runs redis-check-rdb and exits with its
        # verdict; the checker validates a module value's framing only, so for
        # well-framed graph data the module rejected it exits 0. what matters:
        # the server exited by itself (no signal, no hang)
        exit_code, log = _start_on_rdb(self.env, bad)
        tail = f"exit code {exit_code}: " + log[-2000:]
        self.env.assertIsNotNone(exit_code, message=tail)
        self.env.assertGreaterEqual(-1 if exit_code is None else exit_code, 0,
                                    message=tail)
        ours = log.find(f"{FAILED} '{GRAPH_ID}': unsupported index language")
        redis_line = log.find("Internal error in RDB reading")
        self.env.assertGreater(ours, -1, message=tail)
        self.env.assertGreater(redis_line, ours, message=tail)
        self.env.assertContains("not able to load", log[redis_line:],
                                message=tail)
