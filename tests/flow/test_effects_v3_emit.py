from common import *
from index_utils import wait_for_indices_to_sync
from constraint_utils import *
import os

GRAPH_ID = "effects_v3_emit"

# The EMIT side of effects v3, end to end over real replication.
#
# test_effects_v3.py covers DECODE by feeding hand-built payloads to a reader,
# and says so: "nothing here depends on the write path". This is the other
# half - a master that ENCODES v3 and a replica that has to agree, with no
# hand-built bytes anywhere.
#
# DDL is what this is really for. Every record here reaches the replica only
# if the effect was routed into the v3 accumulator, encoded, and applied: a
# record with no producing path emits a well-formed EMPTY payload instead of
# failing, so the master stays correct and the replica silently does not. That
# shape passes every check that looks only at the master.
#
# The multi-field cases are the ones that cannot be replaced by single-field
# ones. C emits one effect per FIELD while a v3 CREATE_INDEX is one record per
# STATEMENT, so a single-field index exercises none of the accumulation and
# passes whether or not it works.


class testEffectsV3Emit(FlowTestsBase):

    def __init__(self):
        if VALGRIND or SANITIZER:
            Environment.skip(None)

        # EFFECTS_THRESHOLD 0 forces every write onto the effects path rather
        # than letting a small one replicate verbatim, which would hide an
        # unencodable effect behind a correct replica
        self.env, self.db = Env(env='oss', useSlaves=True,
                moduleArgs='EFFECTS_THRESHOLD 0 EFFECTS_VERSION 3')

        self.src_con = self.env.getConnection()
        self.replica_con = self.env.getSlaveConnection()
        self.replica_con.config_set("slave-read-only", "no")

        self.src = Graph(self.src_con, GRAPH_ID)
        self.replica = Graph(self.replica_con, GRAPH_ID)

    def _sync(self):
        wait_for_indices_to_sync(self.src)
        self.src_con.execute_command("WAIT", "1", "0")

    def _both(self, q):
        """run a read on each side and return the two result sets"""
        return (self.src.ro_query(q).result_set,
                self.replica.ro_query(q).result_set)

    def _assert_same(self, q, msg):
        a, b = self._both(q)
        self.env.assertEqual(a, b, message=f"{msg}: {a} != {b}")

    def _fields(self, con, label):
        """(name -> textWeight) for one label's index, read off db.indexes"""
        g = Graph(con, GRAPH_ID)
        res = g.ro_query("CALL db.indexes() YIELD label, info "
                         "RETURN label, info").result_set
        for row in res:
            if row[0] != label:
                continue
            return {f['name']: f['textWeight'] for f in row[1]['fields']
                    if f['name'] != 'NONE_INDEXABLE_FIELDS'}
        return {}

    def test01_data_replicates(self):
        self.src.query("UNWIND range(1,5) AS i CREATE (:A {x:i})")
        self.src.query("MATCH (a:A {x:1}), (b:A {x:2}) CREATE (a)-[:R {w:1}]->(b)")
        self.src.query("MATCH (n:A) SET n.y = 'v'")
        self.src.query("MATCH (n:A {x:1}) SET n:B")
        self.src.query("MATCH (n:A {x:5}) DELETE n")
        self._sync()

        self._assert_same("MATCH (n) RETURN count(n)", "node count")
        self._assert_same("MATCH ()-[e]->() RETURN count(e)", "edge count")
        self._assert_same("MATCH (n) RETURN id(n), labels(n), properties(n) "
                          "ORDER BY id(n)", "nodes")
        self._assert_same("MATCH ()-[e]->() RETURN id(e), properties(e) "
                          "ORDER BY id(e)", "edges")

    def test02_multi_field_index_is_one_statement(self):
        # TWO FIELDS IN ONE STATEMENT. C emits one effect per field, so this
        # is the case statement-level accumulation exists for - a single-field
        # index passes with or without it.
        self.src.query("CREATE INDEX FOR (n:A) ON (n.x, n.y)")
        self._sync()

        self._assert_same(
            "CALL db.indexes() YIELD label, properties, types "
            "RETURN label, properties, types ORDER BY label, properties",
            "index definitions")

        # and the replica's index actually answers a query through it
        plan = self.replica.explain("MATCH (n:A {x: 2}) RETURN n")
        self.env.assertIn("Index Scan", str(plan))

    def test03_edge_index(self):
        self.src.query("CREATE INDEX FOR ()-[e:R]->() ON (e.w)")
        self._sync()
        self._assert_same(
            "CALL db.indexes() YIELD label, properties, entitytype "
            "RETURN label, properties, entitytype ORDER BY label, properties",
            "edge index")

    def test04_fulltext_per_field_weights(self):
        # PER-FIELD OPTIONS THAT DIFFER. A v3 CREATE_INDEX carries ONE options
        # block and apply hands it to every field in the record, so two fields
        # that disagree cannot share one - if they did, both would land on the
        # replica with whichever weight was written.
        #
        # Reachable only through this procedure: the CREATE INDEX ... OPTIONS
        # syntax passes one map to every field.
        self.src.query(
            "CALL db.idx.fulltext.createNodeIndex('Doc', "
            "{field:'a', weight:2.0}, {field:'b', weight:5.0})")
        self._sync()

        master = self._fields(self.src_con, 'Doc')
        replica = self._fields(self.replica_con, 'Doc')

        self.env.assertEqual(master, replica,
                message=f"per-field weights diverged: {master} != {replica}")
        # pinned explicitly too - equality alone would pass if BOTH sides
        # collapsed onto one weight
        self.env.assertEqual(replica.get('a'), 2.0)
        self.env.assertEqual(replica.get('b'), 5.0)

    def test05_fulltext_stopwords_with_differing_weights(self):
        # THE CASE THAT DISTINGUISHES. Index_SetStopwords refuses when
        # stopwords are already set, while Index_SetLanguage objects only when
        # the language DIFFERS - so a statement that splits into two records
        # must state the index-level options once. A language-only version of
        # this test is green either way, and a uniform-weight version never
        # splits at all.
        self.src.query(
            "CALL db.idx.fulltext.createNodeIndex("
            "{label:'Note', stopwords:['the','a'], language:'english'}, "
            "{field:'t', weight:2.0}, {field:'u', weight:7.0})")
        self._sync()

        master = self._fields(self.src_con, 'Note')
        replica = self._fields(self.replica_con, 'Note')
        self.env.assertEqual(master, replica,
                message=f"split-record index diverged: {master} != {replica}")
        self.env.assertEqual(replica.get('t'), 2.0)
        self.env.assertEqual(replica.get('u'), 7.0)

        # the stopwords themselves have to have crossed
        self._assert_same(
            "CALL db.indexes() YIELD label, stopwords "
            "RETURN label, stopwords ORDER BY label", "stopwords")

    def test05b_unstated_options_do_not_leak_between_fields(self):
        # The fulltext procedure builds ONE options map and reuses it for every
        # field, so an option this field did not state must be REMOVED from the
        # map and not merely left unset - otherwise the previous field's value
        # is still there and this field inherits it.
        #
        # It matters beyond the leak: v3's presence byte means "the statement
        # said this", and the procedure used to add all three options for every
        # field whether stated or not. An unstated phonetic then travelled as
        # C's default string "no", which Rust reads as phonetic ENABLED.
        self.src.query(
            "CALL db.idx.fulltext.createNodeIndex('Mix', "
            "{field:'p', weight:9.0, nostem:true}, {field:'q'}, "
            "{field:'r', weight:3.0})")
        self._sync()

        master = self._fields(self.src_con, 'Mix')
        replica = self._fields(self.replica_con, 'Mix')
        self.env.assertEqual(master, replica,
                message=f"mixed statedness diverged: {master} != {replica}")

        # q states nothing, so it takes the default - NOT p's 9.0
        self.env.assertEqual(replica.get('p'), 9.0)
        self.env.assertEqual(replica.get('q'), 1.0)
        self.env.assertEqual(replica.get('r'), 3.0)

    def test05c_stated_phonetic_travels(self):
        # A field that states phonetic and NOTHING ELSE. Unreachable until the
        # procedure stopped adding a weight to every field: _validateOptions
        # checks weight first and continues, so this branch was never taken,
        # and it rejected exactly the phonetics it should accept.
        self.src.query(
            "CALL db.idx.fulltext.createNodeIndex('Ph', "
            "{field:'v', phonetic:'dm:en'})")
        self._sync()

        self._assert_same(
            "CALL db.indexes() YIELD label, properties, types "
            "RETURN label, properties, types ORDER BY label, properties",
            "phonetic index")

        master = self._fields(self.src_con, 'Ph')
        replica = self._fields(self.replica_con, 'Ph')
        self.env.assertEqual(master, replica,
                message=f"phonetic field diverged: {master} != {replica}")
        self.env.assertEqual(list(replica.keys()), ['v'])

    def test06_vector_index_options(self):
        self.src.query(
            "CREATE VECTOR INDEX FOR (n:Vec) ON (n.v) "
            "OPTIONS {dimension:4, similarityFunction:'euclidean', M:24}")
        self._sync()
        self._assert_same(
            "CALL db.indexes() YIELD label, properties, types "
            "RETURN label, properties, types ORDER BY label, properties",
            "vector index")

        # and it accepts a vector, which a wrong dimension would refuse
        self.src.query("CREATE (:Vec {v: vecf32([1.0,2.0,3.0,4.0])})")
        self._sync()
        self._assert_same("MATCH (n:Vec) RETURN count(n)", "vector node")

    def test07_drop_index(self):
        # dropped one field of a two-field index, so the record is a real drop
        # rather than the whole index going away
        self.src.query("DROP INDEX FOR (n:A) ON (n.y)")
        self._sync()
        self._assert_same(
            "CALL db.indexes() YIELD label, properties, types "
            "RETURN label, properties, types ORDER BY label, properties",
            "after dropping one field")

    def test08_constraints_carry_their_status(self):
        # v3 states a ConstraintStatus, which v2 has no field for. A replica
        # never validates, so the announcement is the only thing that can tell
        # it an enforcing constraint from one still building.
        create_unique_constraint(self.src, 'NODE', 'A', 'x', sync=True)
        create_mandatory_constraint(self.src, 'NODE', 'A', 'y', sync=True)
        self._sync()

        self._assert_same(
            "CALL db.constraints() YIELD type, label, properties, status "
            "RETURN type, label, properties, status "
            "ORDER BY type, label, properties", "constraints")

        # OPERATIONAL specifically - a replica that defaulted to PENDING would
        # still compare equal if the master were somehow pending too
        res = self.replica.ro_query(
            "CALL db.constraints() YIELD status RETURN DISTINCT status"
        ).result_set
        self.env.assertEqual(res, [['OPERATIONAL']])

    def test09_drop_constraint(self):
        drop_unique_constraint(self.src, 'NODE', 'A', 'x')
        self._sync()
        self._assert_same(
            "CALL db.constraints() YIELD type, label, properties "
            "RETURN type, label, properties ORDER BY type, label, properties",
            "after dropping a constraint")

    def test10_no_verbatim_fallback_was_needed(self):
        # THE CHECK THE STATE COMPARISONS CANNOT MAKE. When v3 cannot encode an
        # effect the master replicates the QUERY TEXT instead; the replica then
        # converges anyway and every assertion above passes while v3 encoded
        # nothing. Only the master's log separates "the new path works" from
        # "the old path was still running".
        # globbed rather than asked for: RLTest has no accessor for it, and an
        # earlier version of this test called one that does not exist and
        # returned early - passing while asserting nothing. Finding no log is
        # a FAILURE here for that reason.
        import glob
        logs = glob.glob(os.path.join('logs', '*master*.log'))
        self.env.assertTrue(len(logs) > 0,
                message="found no master log to check for the fallback")

        for path in logs:
            with open(path, 'r', errors='replace') as fh:
                body = fh.read()
            self.env.assertNotContains("v3 cannot encode this effect", body)

            # and the guard behind it: an effect written as v2 into a v3 buffer
            self.env.assertNotContains("v3 buffer received a v2 record write",
                    body)
