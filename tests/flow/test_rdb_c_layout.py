import os

from common import *
from constraint_utils import *

GRAPH_ID = "rdb_c_layout"

# BufferedWriter frames every unsigned word as a 1-byte type tag plus eight
# little-endian bytes.
TYPE_UNSIGNED = 4
WORD_BYTES = 9

# Constraint type discriminators, as written to the RDB.
CT_MANDATORY = 1

LABEL = "Person"
CONSTRAINT_FIELDS = ["p1", "p2", "p3", "p4", "p5"]


class testRdbCLayout():
    """Guard the RDB schema layout against C's v19 encoder.

    These assert the bytes of a real saved RDB, which is unusual for a flow test
    and is the whole point: **a Rust-to-Rust reload cannot fail on either defect
    they cover.** Our decoder strips an optional `range:` field-name prefix with
    `unwrap_or`, so it accepts either spelling; and the constraint status word it
    used to read was one it also wrote, so reader and writer agreed with each
    other while both disagreed with C. Every persistency test in this suite passed
    with both bugs present.

    What they protect is cross-engine loading, verified by hand against
    `falkordb/falkordb-server:edge-c` in both directions. Neither failure was
    graceful on the C side -- a two-constraint RDB written by C was refused
    outright, and C segfaulted in `Constraint_SetStatus` loading ours -- so there
    is no error message to assert on instead of the layout.

    Reference: `_RdbSaveConstraint` and `_RdbSaveIndexField` in
    `src/serializers/encoder/v19/encode_schema.c` on the `master` branch.
    """

    def __init__(self):
        self.env, self.db = Env()
        # These read the RDB off disk, which the shared services container does not
        # expose to the test process.
        if os.getenv("FALKORDB_USE_SERVICE"):
            Environment.skip(None)
        self.con = self.env.getConnection()
        # LZF would defeat every assertion here: a compressed payload contains
        # none of the literals or word framing being looked for.
        self.con.execute_command("CONFIG", "SET", "rdbcompression", "no")

    def _reset(self):
        """Start from an empty keyspace.

        Not just tidiness: a dump.rdb these tests wrote is loaded by the next run
        of the suite, so without this a test inherits graphs from the previous run
        -- including, during a bisect, ones written by a differently-behaving
        build.
        """
        self.con.execute_command("FLUSHALL")

    def _dump_bytes(self, label=None):
        """SAVE and return the RDB as raw bytes.

        The dump covers the whole keyspace, so any other graph carrying the same
        label would land in it and be indistinguishable from the one under test.
        These tests also leave a dump.rdb behind that the *next* run loads at
        startup, so isolation has to be asserted, not assumed -- an earlier run's
        stale graph is what made the anchored lookups below find two schema
        entries and fail.
        """
        if label is not None:
            # The engine keeps its own `telemetry{...}` graph beside each user
            # graph; it carries none of these labels, so it is not a confounder.
            keys = [k for k in self.con.execute_command("KEYS", "*")
                    if not str(k).startswith("telemetry")]
            self.env.assertEquals(len(keys), 1,
                message=f"exactly one user graph should exist while dumping; found "
                        f"{keys}. Another graph with the same label would be "
                        f"anchored on too")
        self.con.execute_command("SAVE")
        directory = self.con.execute_command("CONFIG", "GET", "dir")[1]
        filename = self.con.execute_command("CONFIG", "GET", "dbfilename")[1]
        with open(os.path.join(directory, filename), "rb") as f:
            return f.read()

    def _schema_words(self, blob, label, count=12):
        """The words following each occurrence of `label` in the RDB.

        A schema entry is the label name, then the index block, then the
        constraint block, so anchoring on the name gives a known offset into the
        part under test. Every occurrence is returned rather than the first,
        because the name is not guaranteed to appear only once -- searching the
        whole RDB for a word pattern instead is what this replaced, and it produced
        false failures from `[1][0][5]` occurring by chance elsewhere.
        """
        out = []
        needle = label.encode() + b"\x00"
        i = blob.find(needle)
        while i >= 0:
            tail = blob[i + len(needle):]
            ws, j = [], 0
            while len(ws) < count and j + WORD_BYTES <= len(tail) \
                    and tail[j] == TYPE_UNSIGNED:
                ws.append(int.from_bytes(tail[j + 1:j + WORD_BYTES], "little"))
                j += WORD_BYTES
            out.append(ws)
            i = blob.find(needle, i + 1)
        return out

    def test01_index_field_names_are_bare_attribute_names(self):
        # C writes an index field's bare attribute name and puts the field type in
        # a separate word. We hold the RediSearch field name internally, which
        # carries a type prefix, and used to write that instead.
        #
        # The consequence on C is not a cosmetically odd index name: loading such
        # an RDB, C looks for the exact-match index supporting a UNIQUE constraint
        # on `name`, finds ours filed under `range:name`, and dereferences the NULL
        # that `Constraint_New` hands back.
        self._reset()
        graph = self.db.select_graph(GRAPH_ID + "_idx")
        graph.query("CREATE (:Person {name:'a', age:1, embedding:vecf32([1,2,3,4])})")
        graph.query("CREATE INDEX FOR (n:Person) ON (n.age)")
        graph.query("CREATE VECTOR INDEX FOR (n:Person) ON (n.embedding) "
                    "OPTIONS {dimension: 4, similarityFunction: 'euclidean'}")

        blob = self._dump_bytes()

        self.env.assertTrue(b"age\x00" in blob,
            message="the bare attribute name must be written")
        self.env.assertFalse(b"range:age\x00" in blob,
            message="a `range:`-prefixed field name reached the RDB; C reads that "
                    "name literally and can then no longer match a constraint to "
                    "its supporting index")
        self.env.assertFalse(b"vector:embedding\x00" in blob,
            message="a `vector:`-prefixed field name reached the RDB")
        graph.delete()

    def test02_constraint_block_matches_c_layout(self):
        # The status word is invisible from inside this engine -- we stopped writing
        # it and stopped reading it in the same change -- so the only way to notice
        # its return is to look at the wire.
        #
        # No index is created here, which keeps the index block a single `false`
        # word and puts the constraint block at a known offset from the label name.
        self._reset()
        graph = self.db.select_graph(GRAPH_ID + "_cons")
        props = ", ".join(f"{p}:{i}" for i, p in enumerate(CONSTRAINT_FIELDS))
        graph.query(f"CREATE (:{LABEL} {{{props}}})")
        create_mandatory_node_constraint(graph, LABEL, *CONSTRAINT_FIELDS, sync=True)

        n = len(CONSTRAINT_FIELDS)
        # has_index=false, one constraint, MANDATORY, n fields -- then n attribute
        # ids. C's layout is `type, field_count, ids...` with no status word; the
        # status word would sit where field_count is and push everything along.
        expected = [0, 1, CT_MANDATORY, n]
        found = self._schema_words(self._dump_bytes(LABEL), LABEL)

        matches = [ws for ws in found if ws[:len(expected)] == expected]
        self.env.assertEquals(len(matches), 1,
            message=f"expected exactly one schema entry reading "
                    f"{expected} (has_index, constraint count, type, field count); "
                    f"got {found}. A zero word between the type and the field count "
                    f"is the status field, which C never writes -- it leaves C's "
                    f"reader one field out of step for the rest of the block, which "
                    f"is why C refused a two-constraint RDB outright while a "
                    f"one-constraint RDB parsed by luck")
        if matches:
            ids = matches[0][len(expected):len(expected) + n]
            self.env.assertEquals(sorted(ids), list(range(n)),
                message=f"the {n} words after the field count must be the "
                        f"constraint's attribute ids; got {ids}")
        graph.delete()

    def test03_only_active_constraints_are_encoded(self):
        # C encodes only CT_ACTIVE constraints and writes the *filtered* count. That
        # is what makes the status word unnecessary rather than merely absent: an RDB
        # cannot describe an unfinished constraint, so there is nothing for a status
        # field to say. Encoding a FAILED one would hand C a constraint it then marks
        # active.
        #
        # The second node lacks `p1`, so the constraint cannot be satisfied.
        # MANDATORY is deliberate: UNIQUE pulls in a supporting exact-match index,
        # which would fill the index block and move the constraint block.
        self._reset()
        graph = self.db.select_graph(GRAPH_ID + "_failed")
        props = ", ".join(f"{p}:{i}" for i, p in enumerate(CONSTRAINT_FIELDS))
        graph.query(f"CREATE (:{LABEL} {{{props}}}), (:{LABEL} {{other:1}})")
        create_mandatory_node_constraint(graph, LABEL, *CONSTRAINT_FIELDS, sync=True)

        statuses = [r[0] for r in graph.query(
            "CALL db.constraints() YIELD status RETURN status").result_set]
        self.env.assertEquals(statuses, ["FAILED"],
            message=f"this test needs a FAILED constraint to be meaningful; "
                    f"got {statuses}")

        found = self._schema_words(self._dump_bytes(LABEL), LABEL)
        # has_index=false, then a constraint count of zero.
        matches = [ws for ws in found if ws[:2] == [0, 0]]
        self.env.assertEquals(len(matches), 1,
            message=f"expected the schema to encode zero constraints, reading "
                    f"[0, 0] for (has_index, constraint count); got {found}. A "
                    f"count of 1 means the FAILED constraint was encoded, and C "
                    f"would load it as active because its format has no field to "
                    f"say otherwise")
        graph.delete()

    def test04_udf_strings_are_nul_terminated(self):
        # UDF libraries are not in the graph key at all -- they are a module-level
        # AUX field written with raw RedisModule_Save* calls rather than the tagged
        # writer everything above uses, so nothing else here covers them.
        #
        # C writes `strlen(s) + 1`, including the terminator, and reads the buffers
        # back as C strings. The Rust helper writes `s.len()`. A library saved
        # without its terminator loaded on C as `XLibte` -- the name ran into
        # whatever followed it in the RDB -- and none of its functions could be
        # found. `GRAPH.UDF LIST` still reported a library, so it failed quietly.
        #
        # Only the *writing* half needs asserting here. The reading half cannot
        # regress unnoticed: since we now write the terminator, a reader that stops
        # stripping it cannot load its own output, and the whole of test_udf fails
        # to run. Ablating each half separately is what established that.
        self._reset()
        lib = "NulTermLib"
        self.db.udf_load(lib, "function double (x) { return x * 2; }\n"
                              f"falkor.register ('double', double);", True)
        # A graph must exist for the keyspace to be worth saving.
        self.db.select_graph(GRAPH_ID + "_udf").query("RETURN 1")

        blob = self._dump_bytes()
        self.env.assertTrue(lib.encode() + b"\x00" in blob,
            message=f"the UDF library name must be written NUL-terminated, as C's "
                    f"`AUXSaveUDF_latest` does; found {lib!r} without a terminator, "
                    f"which C reads as that name plus whatever bytes follow it")
