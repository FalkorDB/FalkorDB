"""Invariants of the query set that used to be comments or an import-time assert.

The `write N`-last ordering in particular was enforced only by a comment, and
getting it wrong is expensive but silent: it inflates node capacity / matrix
dimension for every query measured afterwards (algo.pageRank went 150x when
those queries ran first).
"""

from falkorbench import callgrind
from falkorbench import queries as qs


class TestQuerySet:
    def test_names_are_unique(self):
        """Names are the CSV key; a duplicate would silently overwrite a row."""
        names = [q.name for q in qs.QUERIES]
        assert len(names) == len(set(names))

    def test_command_follows_the_write_flag(self):
        for q in qs.QUERIES:
            assert q.command == ("GRAPH.QUERY" if q.write else "GRAPH.RO_QUERY")

    def test_every_query_has_text(self):
        for q in qs.QUERIES:
            assert q.cypher.strip(), q.name

    def test_sized_write_queries_stay_last(self):
        """They inflate node capacity / matrix dimension to max(N), which would
        slow every full-graph query measured after them."""
        sized_prefixes = ("write ", "create ", "delete ")
        sized_idx = [
            i
            for i, q in enumerate(qs.QUERIES)
            if any(q.name.startswith(p) for p in sized_prefixes) and q.name.split()[-1][0].isdigit()
        ]
        assert sized_idx, "expected to find the sized write queries"
        first_sized = min(sized_idx)
        # Everything from the first sized write query onward must also be sized.
        assert sized_idx == list(range(first_sized, len(qs.QUERIES)))

    def test_sized_write_queries_ascend_in_magnitude(self):
        """No sized row may be preceded by one an order of magnitude larger.

        A write costs more while the engine still carries a large deletion:
        creating one node measured 0.05 ms after a 10k delete and 0.69 ms after
        a 1M one. With the create/delete pairs grouped after the mixed rows,
        `create 100` ran directly after `write 1m` and measured recovery from it
        rather than the cost of a create -- 4.4x C, where a fresh graph shows
        1.28x. Ascending order keeps each row's context comparable to its own
        size, and only a test keeps it that way; the previous ordering rule was
        a comment and drifted.
        """
        sized = [
            q.name
            for q in qs.QUERIES
            if any(q.name.startswith(p) for p in ("write ", "create ", "delete "))
            and q.name.split()[-1][0].isdigit()
        ]
        suffix = {"": 1, "k": 1_000, "m": 1_000_000}

        def magnitude(name: str) -> int:
            n = name.split()[-1]
            return int(n.rstrip("km")) * suffix[n[-1] if n[-1] in "km" else ""]

        sizes = [magnitude(n) for n in sized]
        assert sizes == sorted(sizes), f"sized rows must ascend in magnitude, got {sized}"

    def test_reps_are_positive_when_set(self):
        for q in qs.QUERIES:
            assert q.reps is None or q.reps > 0, q.name


class TestCallgrindSubset:
    def test_the_control_query_is_in_the_subset(self):
        """`RETURN 1` is the fixed floor the whole callgrind table is read
        against — if it is absent there is nothing to sanity-check a run with."""
        cg_names = {q.name for q in qs.QUERIES if q.cg}
        assert "RETURN 1" in cg_names

    def test_subset_is_a_strict_subset(self):
        cg = [q for q in qs.QUERIES if q.cg]
        assert 0 < len(cg) < len(qs.QUERIES)

    def test_cg_queries_only_need_labels_cg_setup_provides(self):
        """CG_SETUP builds :Person (id/name/age/score + index), a :KNOWS ring and
        :Tmp. A cg-flagged query needing anything else — a fulltext or vector
        index, a constraint, a UDF, LOAD CSV — fails inside CI at ~14s per query,
        which is a slow way to learn about a typo.
        """
        provided = {"Person", "KNOWS", "Tmp"}
        # Labels the reduced graph does not have. :Doc and friends only exist in
        # the full SETUP.
        forbidden = {
            "Doc",
            "SIMILAR",
            "MEnd",
            "MULTI",
            "CIdx",
            "Geo",
            "ZIdx",
            "IDoc",
            "SIdx",
            "Place",
            "UREL",
            "REF",
        }
        for q in (q for q in qs.QUERIES if q.cg):
            for label in forbidden:
                assert f":{label}" not in q.cypher, f"{q.name} needs :{label}, not in CG_SETUP"
        assert provided  # documents what is available

    def test_cg_queries_avoid_udfs_and_csv(self):
        for q in (q for q in qs.QUERIES if q.cg):
            lowered = q.cypher.lower()
            assert "load csv" not in lowered, q.name
            assert "udf." not in lowered, q.name

    def test_tmp_pool_outlasts_the_widest_span(self):
        """A destructive cg query runs up to MAX_SPAN + n1 times. Running dry
        does not fail loudly — it quietly measures a no-op and halves the
        reported cost — so the pool must be bigger than the widest span."""
        pool_stmt = next(s for s in callgrind.CG_SETUP if ":Tmp" in s)
        # "UNWIND range(0, 4999) AS i CREATE (:Tmp {x: i})"
        upper = int(pool_stmt.split("range(0, ")[1].split(")")[0])
        assert upper + 1 > callgrind.MAX_SPAN


class TestSetup:
    def test_setup_is_non_empty_and_ordered(self):
        assert qs.SETUP
        # The Person index must be created before the ring build so the build is
        # index-driven rather than a 10000x10000 nested scan.
        idx = next(i for i, s in enumerate(qs.SETUP) if "CREATE INDEX FOR (p:Person)" in s)
        ring = next(i for i, s in enumerate(qs.SETUP) if "KNOWS" in s and "CREATE (a)-" in s)
        assert idx < ring

    def test_error_queries_are_triples_with_a_known_command(self):
        for name, command, cypher in qs.ERROR_QUERIES:
            assert name and cypher
            assert command in ("GRAPH.QUERY", "GRAPH.RO_QUERY")

    def test_csv_fixtures_are_named_and_non_empty(self):
        for name, body in qs.CSV_FILES.items():
            assert name.endswith(".csv")
            assert body.strip()
