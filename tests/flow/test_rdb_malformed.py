from common import *
from index_utils import create_node_range_index, create_node_vector_index
from constraint_utils import (create_unique_node_constraint,
                              create_mandatory_node_constraint)
from rdb_payload import (GraphValue, reframe, crc64, rdb_module_values,
                         T_ARRAY, T_INT64, T_NULL, T_STRING, T_INTERN_STRING,
                         T_VECTOR_F32)

import os
import shutil
import tempfile
import subprocess

GRAPH_ID = "g"
FAILED = "Failed loading graph key"


# a small graph touching every part of the encoding the tests below corrupt:
# indexes (range, vector, CCH), constraints, properties of several types, a
# multi-edge (tensor), and a deleted node and edge
def _build_graph(db):
    g = db.select_graph(GRAPH_ID)
    create_node_range_index(g, "N", "v", sync=True)
    create_node_vector_index(g, "V", "e", dim=4, sync=True)
    g.query("CREATE (a:N {v: 1, s: 'one', a: [1, 2]}), (b:N {v: 2}), "
            "(c:N {v: 3}), (:V {e: vecf32([1, 2, 3, 4])}), "
            "(a)-[:R {w: 1}]->(b), (a)-[:R {w: 2}]->(b), (b)-[:R {w: 3}]->(c)")
    g.query("CREATE (:N {v: 4})-[:R {w: 4}]->(:N {v: 5})")
    g.query("MATCH (n:N {v: 4}) DETACH DELETE n")
    g.query("CREATE CCH INDEX FOR ()-[e:R]->() ON (e.w)")

    # only active constraints are encoded
    create_unique_node_constraint(g, "N", "v", sync=True)
    create_mandatory_node_constraint(g, "N", "v", sync=True)


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
             if ".property[" in n and ".element[" not in n]
    assert names, t
    return names[0]


# the id of the attribute named 'name'
def _attribute(gv, name):
    names = gv.find("", name.encode() + b"\0")
    names = [n for n in names if n.startswith("attribute[")]
    assert len(names) == 1, names
    return int(names[0][len("attribute["):-1])


# the fields prefix ("label_matrix[k].M.p") of the first matrix vector of
# component 'part' (p, h, i, x, b) holding data of a multi-byte integer type,
# with that type's size
_INT_TYPES = {b"GrB_UINT64\0": 8, b"GrB_INT64\0": 8,
              b"GrB_UINT32\0": 4, b"GrB_INT32\0": 4}

def _int_vector(gv, part):
    for n, f in gv.fields.items():
        if n.endswith(f".{part}.type") and f.value in _INT_TYPES:
            v = n[:-len(".type")]
            if gv.fields[f"{v}.bytes"].value > 0:
                return v, _INT_TYPES[f.value]
    raise AssertionError(f"no integer matrix vector .{part}")


#-------------------------------------------------------------------------------
# a payload with one field corrupted must be rejected, leave no key behind and
# log exactly one line naming the key and what was wrong
#-------------------------------------------------------------------------------

class testRdbMalformed():
    def __init__(self):
        self.env, self.db = Env()
        self.conn = self.env.getConnection()

        self.conn.flushall()
        _build_graph(self.db)

        # uncompressed, so the walker can read the serializer buffers
        self.conn.config_set("rdbcompression", "no")
        try:
            payload = self.conn.dump(GRAPH_ID)
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

    def test_unknown_nested_value_type(self):
        # an array element of a type no value has, its value removed
        gv = self._value()
        p = _property_of_type(gv, T_ARRAY)
        e = f"{p}.element[0]"
        self.env.assertEqual(gv.fields[f"{e}.type"].value, T_INT64)
        gv.set_uint(f"{e}.type", 1 << 24)
        gv.remove_value(f"{e}.value")
        self._assert_rejected(gv, f"unknown value type {1 << 24}")

    def test_unterminated_string_property(self):
        gv = self._value()
        names = [n[:-len(".value")] for n in gv.find(".value", b"one\0")
                 if ".property[" in n]
        self.env.assertEqual(len(names), 1)
        self.env.assertContains(gv.fields[f"{names[0]}.type"].value,
                                (T_STRING, T_INTERN_STRING))
        f = gv.fields[f"{names[0]}.value"]
        gv.set_bytes(f.off + f.size - 1, b"X")
        self._assert_rejected(gv, "string of 4 bytes is not NUL-terminated")

    # entity ids run below the header's live + deleted count; the datablocks
    # are allocated with room past it, so the bound is the count, not the
    # allocation
    def _id_limit(self, gv, kind):
        f = gv.fields
        return f[f"{kind}_count"].value + f[f"deleted_{kind}_count"].value

    def test_node_id_out_of_range(self):
        gv = self._value()
        limit = self._id_limit(gv, "node")
        gv.set_uint("node[0].id", limit)
        self._assert_rejected(gv, f"node id {limit} out of range")

    def test_edge_id_out_of_range(self):
        gv = self._value()
        limit = self._id_limit(gv, "edge")
        gv.set_uint("edge[0].id", limit)
        self._assert_rejected(gv, f"edge id {limit} out of range")

    def test_deleted_nodes_buffer_size(self):
        gv = self._value()
        size = gv.fields["deleted_nodes"].size
        self.env.assertEqual(size, 8)
        gv.set_length("deleted_nodes", size - 1)
        self._assert_rejected(gv, "deleted nodes buffer size mismatch")

    def test_deleted_node_id_out_of_range(self):
        gv = self._value()
        limit = self._id_limit(gv, "node")
        self.env.assertEqual(gv.fields["deleted_nodes"].size, 8)
        gv.set_bytes(gv.fields["deleted_nodes"].off,
                     limit.to_bytes(8, "little"))
        self._assert_rejected(gv, f"deleted node id {limit} out of range")

    def test_deleted_edge_id_out_of_range(self):
        gv = self._value()
        limit = self._id_limit(gv, "edge")
        self.env.assertEqual(gv.fields["deleted_edges"].size, 8)
        gv.set_bytes(gv.fields["deleted_edges"].off,
                     limit.to_bytes(8, "little"))
        self._assert_rejected(gv, f"deleted edge id {limit} out of range")

    #---------------------------------------------------------------------------
    # schema
    #---------------------------------------------------------------------------

    def test_duplicate_attribute_name(self):
        # the second attribute renamed after the first (same length)
        gv = self._value()
        a0, a1 = gv.fields["attribute[0]"], gv.fields["attribute[1]"]
        self.env.assertEqual(a0.size, a1.size)
        self.env.assertNotEqual(a0.value, a1.value)
        gv.set_bytes(a1.off, a0.value)
        self._assert_rejected(gv, "duplicate attribute name")

    def test_unterminated_attribute_name(self):
        gv = self._value()
        f = gv.fields["attribute[0]"]
        gv.set_bytes(f.off + f.size - 1, b"X")
        self._assert_rejected(gv,
            f"string of {f.size} bytes is not NUL-terminated")

    def test_duplicate_label(self):
        # label V renamed N
        gv = self._value()
        gv.set_bytes(gv.fields[f"{_schema(gv, 'V')}.name"].off, b"N")
        self._assert_rejected(gv, "duplicate or out of order schema")

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

    def test_constraint_without_attributes(self):
        gv = self._value()
        c = f"{_schema(gv, 'N')}.constraint[0]"
        self.env.assertEqual(gv.fields[f"{c}.attribute_count"].value, 1)
        gv.set_uint(f"{c}.attribute_count", 0)
        gv.remove_value(f"{c}.attribute[0]")
        self._assert_rejected(gv, "constraint with 0 attributes")

    # counts and ids are read as 64 bits and checked before they are narrowed:
    # each corruption below wraps to the healthy value once narrowed

    def test_constraint_attribute_count_wraps(self):
        # 257 narrows to 1 (uint8_t), the count the payload holds
        gv = self._value()
        c = f"{_schema(gv, 'N')}.constraint[0]"
        self.env.assertEqual(gv.fields[f"{c}.attribute_count"].value, 1)
        gv.set_uint(f"{c}.attribute_count", 257)
        self._assert_rejected(gv, "constraint with 257 attributes")

    def test_constraint_attribute_wraps(self):
        # narrows to the same attribute (AttributeID is 16 bits)
        gv = self._value()
        name = f"{_schema(gv, 'N')}.constraint[0].attribute[0]"
        gv.set_uint(name, (1 << 16) + gv.fields[name].value)
        self._assert_rejected(gv, "constraint on an unknown attribute")

    def test_unique_constraint_without_index(self):
        # the unique constraint moved to an attribute no index covers
        gv = self._value()
        s = _schema(gv, "N")
        unique = [n[:-len(".type")] for n in gv.find(".type", 0)   # CT_UNIQUE
                  if n.startswith(f"{s}.constraint[")]
        self.env.assertEqual(len(unique), 1)
        gv.set_uint(f"{unique[0]}.attribute[0]", _attribute(gv, "s"))
        self._assert_rejected(gv, "constraint can't be created",
                              "missing supporting exact-match index")

    def test_cch_without_relationship_types(self):
        gv = self._value()
        self.env.assertEqual(gv.fields["cch[0].relation_count"].value, 1)
        gv.set_uint("cch[0].relation_count", 0)
        gv.remove_value("cch[0].relation[0]")
        self._assert_rejected(gv, "CCH index without relationship types")

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
        self._assert_rejected(gv, "label matrix for an unknown label: 99")

    def test_label_matrix_negative_label(self):
        # read into a signed 32-bit LabelID, 0xFFFFFFFF passed for -1
        gv = self._value()
        gv.set_uint("label_matrix[0].label", 0xFFFFFFFF)
        self._assert_rejected(gv,
            f"label matrix for an unknown label: {0xFFFFFFFF}")

    def test_label_matrix_missing(self):
        # the last label matrix dropped, count adjusted: well formed, one short
        gv = self._value()
        n = gv.fields["label_matrix_count"].value
        self.env.assertEqual(n, gv.fields["label_count"].value)
        gv.set_uint("label_matrix_count", n - 1)
        gv.remove_values(f"label_matrix[{n - 1}].label",
                         f"label_matrix[{n - 1}].DM.b.handling")
        self._assert_rejected(gv, f"{n - 1} label matrices for {n} labels")

    def test_label_matrix_twice(self):
        gv = self._value()
        self.env.assertEqual(gv.fields["label_matrix[0].label"].value, 0)
        gv.set_uint("label_matrix[1].label", 0)
        self._assert_rejected(gv, "label matrix decoded twice")

    def test_relation_matrix_out_of_order(self):
        gv = self._value()
        self.env.assertEqual(gv.fields["relation_matrix[0].relation"].value, 0)
        gv.set_uint("relation_matrix[0].relation", 5)
        self._assert_rejected(gv, "relation matrix out of order")

    def test_relation_matrix_id_wraps(self):
        # narrows to 0 (RelationID is 32 bits)
        gv = self._value()
        self.env.assertEqual(gv.fields["relation_matrix[0].relation"].value, 0)
        gv.set_uint("relation_matrix[0].relation", 1 << 32)
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

    def test_matrix_vector_entries_overflow(self):
        # entries * type size wraps to 0, which GraphBLAS' own size check
        # accepts; nothing later checks the length of an index (i) vector
        gv = self._value()
        v, size = _int_vector(gv, "i")
        entries = (1 << 64) // size
        gv.set_uint(f"{v}.entries", entries)
        self._assert_rejected(gv, f"matrix vector declares {entries} entries")

    def test_unknown_matrix_type(self):
        # GxB_Type_from_name reports an unknown name as success, NULL type
        gv = self._value()
        v, _ = _int_vector(gv, "p")
        f = gv.fields[f"{v}.type"]
        gv.set_bytes(f.off + 4, b"X")
        self._assert_rejected(gv, "unknown matrix value type 'GrB_X")

    #---------------------------------------------------------------------------
    # framing
    #---------------------------------------------------------------------------

    def test_unexpected_value_type(self):
        gv = self._value()
        gv.set_tag("node_count", 0x7F)
        self._assert_rejected(gv, "unexpected value type")

    def test_buffer_length_overrun(self):
        gv = self._value()
        gv.set_length("graph_name", 1 << 32)
        self._assert_rejected(gv, "buffer length overruns its buffer")

    def test_unknown_payload_type(self):
        gv = self._value()
        gv.set_uint("payload[0].type", 99)
        self._assert_rejected(gv, "unknown payload type")


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
        _build_graph(self.db)
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
        _build_graph(self.db)
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
