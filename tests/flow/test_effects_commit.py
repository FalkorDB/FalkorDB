"""Effects -- what one commit carries, and what it must not leak.

Three properties of a commit rather than of an opcode: that a non-deterministic
write ships its *outcome* rather than its query, that a statement which fails
leaves no trace of the schema it began to introduce, and that an entity updated
and then deleted in the same statement reports the update while shipping only
the delete.

See `effects_common.py` for the shared fixture.
"""

from common import *

from effects_common import _EffectsBase


class testEffects_01_NonDeterministic(_EffectsBase):
    """`rand()`, `timestamp()` and friends evaluated once, on the primary.

    It used to be that these had to *force* effects, because a cheap write
    would otherwise replay the query and the two sides would evaluate `rand()`
    separately. Effects are now the only mechanism, so the hazard is gone --
    but a regression that reintroduced query replay would show up here first.
    """

    GRAPH_ID = "effects_nondeterministic"

    def __init__(self):
        self._setup()
        self.start_monitor('GRAPH.EFFECT')
        # Discard anything setup put on the wire so the first test's window
        # starts empty and `assert_effect_emitted` counts only what it wrote.
        self.monitor_mark()

    def test01_create_node_with_random_and_timestamp(self):
        q = "CREATE ({r:rand(), t:timestamp()})"
        res = self.query_and_sync(q)
        self.env.assertEqual(res.nodes_created, 1)
        self.env.assertEqual(res.properties_set, 2)

        self.assert_effect_emitted()
        self.assert_graph_eq()

    def test02_non_deterministic_queries(self):
        """A non deterministic query still replicates correctly.

        It used to be the case that these had to *force* effects, because a
        cheap write would otherwise replay the query and the two sides would
        evaluate `rand()` or `date()` separately. Effects are now the only
        mechanism, so the hazard is gone — but the queries are still worth
        replicating and comparing.
        """

        self.env.flush()        # clean slate

        self.master_graph  = Graph(self.master, self.GRAPH_ID)
        self.replica_graph = Graph(self.replica, self.GRAPH_ID)

        # each of the following queries contains a non deterministic element
        queries = [
            "WITH date()                  AS x CREATE ()",
            "WITH rand()                  AS x CREATE ()",
            "WITH timestamp()             AS x CREATE ()",
            "WITH localtime()             AS x CREATE ()",
            "WITH randomuuid()            AS x CREATE ()",
            "WITH localdatetime()         AS x CREATE ()",
            "WITH date.transaction()      AS x CREATE ()",
            "WITH localtime.transaction() AS x CREATE ()",

            "CREATE ({v:date()})",
            "CREATE ({v:rand()})",
            "CREATE ({v:timestamp()})",
            "CREATE ({v:localtime()})",
            "CREATE ({v:randomuuid()})",
            "CREATE ({v:localdatetime()})",
            "CREATE ({v:date.transaction()})",
            "CREATE ({v:localtime.transaction()})",

            # duplicated query for DB internal execution-plan cache utilization
            "CREATE ({v:date()})",
            "CREATE ({v:rand()})",
            "CREATE ({v:timestamp()})",
            "CREATE ({v:localtime()})",
            "CREATE ({v:randomuuid()})",
            "CREATE ({v:localdatetime()})",
            "CREATE ({v:date.transaction()})",
            "CREATE ({v:localtime.transaction()})",
            ]

        for q in queries:
            self.master_graph.query(q)

            # although effects are disabled
            # we're still expecting replication to use effect
            self.assert_effect_emitted()

        # make sure graphs are the same!
        self.wait_for_replica_offset()
        self.assert_graph_eq()


class testEffects_02_SchemaRollback(_EffectsBase):
    """A statement that introduces a schema and then fails leaves none behind.

    The schema is created before the clause that raises, so this is about what
    the rollback undoes locally *and* what it declines to put on the wire.
    """

    GRAPH_ID = "effects_schema_rollback"

    def __init__(self):
        self._setup()

    def test01_schema_replication(self):
        """
        Make sure a query which introduces a new schema
        but fails doesn't replicate the schema creation
        and removes the schema
        """

        # clean slate
        self.env.flush()

        # replicate via effects

        self.master_graph  = Graph(self.master, self.GRAPH_ID)
        self.replica_graph = Graph(self.replica, self.GRAPH_ID)

        # create a new node schame 'A' mapped to schema id 0
        q = "CREATE (a:A) RETURN a / 0"
        try:
            self.master_graph.query (q)
            # we shouldn't be here
            self.env.assertTrue(False)
        except Exception:
            # as expected
            pass

        # graph should remain empty
        q = "CALL db.labels()"
        res = self.master_graph.ro_query(q).result_set
        self.env.assertEqual(len(res), 0)

        # try to create a second label
        q = "CREATE (b:B)"
        #res = self.master_graph.query (q)
        res = self.query_and_sync(q)
        self.env.assertEqual(res.labels_added, 1)

        q = "CALL db.meta.stats()"
        master_stats  = self.master_graph.ro_query  (q).result_set
        replica_stats = self.replica_graph.ro_query (q).result_set

        self.env.assertEqual(master_stats, replica_stats)


class testEffects_03_UpdateThenDelete(_EffectsBase):
    """An entity updated and deleted by the same statement."""

    GRAPH_ID = "effects_update_then_delete"

    def __init__(self):
        self._setup()
        self.start_monitor('GRAPH.EFFECT')
        # Discard anything setup put on the wire so the first test's window
        # starts empty and `assert_effect_emitted` counts only what it wrote.
        self.monitor_mark()

    def test01_update_then_delete_in_one_transaction(self):
        # An entity updated and deleted by the same query. The update is not
        # part of the payload — the replica has no entity to apply it to, and
        # the DELETE record puts it in the same place anyway.
        #
        # Two things are asserted here that unit tests cannot both reach. The
        # cascade form (`DELETE n`, taking its edges with it) only removes the
        # edge inside `commit`, via `delete_implicit_edges`, so it needs a real
        # graph. And `Properties set` must stay 1: the SET did happen, and
        # suppressing the update at the source rather than in the payload would
        # silently change what the query reports — the node form has always
        # reported 1, and the edge forms have to agree with it.
        #
        # Without the payload filter, `digest_updates` reads the type of an
        # edge `commit` has already cleared, and the ordinary accessor panics:
        # the master goes down on an ordinary query.
        self.monitor_mark()

        # A label per case: an earlier case leaves its endpoint behind, and a
        # shared label would make the node case match that leftover too.
        cases = [
            ("UDa", "MATCH (:UDa)-[e:UDR]->() SET e.x = 1 DELETE e",         "explicit edge"),
            ("UDb", "MATCH (n:UDb)-[e:UDR]->() SET e.x = 1 DELETE n",        "cascade"),
            ("UDc", "MATCH (n:UDc)-[e:UDR]->() SET e.x = 1 DETACH DELETE n", "detach"),
            ("UDd", "MATCH (n:UDd) SET n.x = 1 DELETE n",                    "node"),
        ]

        for label, q, name in cases:
            self.query_and_sync(f"CREATE (:{label})-[:UDR]->(:UDB)")
            res = self.query_and_sync(q)

            # The SET ran. What the payload carries is a separate question.
            self.env.assertEqual(res.properties_set, 1, message=name)

            self.assert_effect_emitted()
            self.assert_graph_eq()

            # And the master is still here — this is the regression.
            self.env.assertEqual(self.master.ping(), True, message=name)

        # Neither engine kept a trace of the deleted entities.
        for g in (self.master_graph, self.replica_graph):
            self.env.assertEqual(
                g.ro_query("MATCH ()-[e:UDR]->() RETURN count(e)").result_set[0][0], 0)
            for label in ("UDb", "UDc", "UDd"):
                self.env.assertEqual(
                    g.ro_query(f"MATCH (n:{label}) RETURN count(n)").result_set[0][0], 0)
