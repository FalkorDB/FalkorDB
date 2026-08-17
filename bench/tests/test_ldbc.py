"""The LDBC query set, its parameters, and the rewrites applied to it.

The value of this benchmark rests entirely on the vendored query texts still
being the LDBC queries. These tests pin the three ways that can quietly stop
being true: a query text that no longer parses out of its file, a rewrite that
is applied but not declared, and a parameter set that does not match what the
queries actually take.

Nothing here needs a server. The behaviours that do — that the rewrites are
accepted by the engine and the originals are not — are covered by
`bench ldbc run`, which fails when a query returns no rows on every parameter
row.
"""

import csv
import re

import pytest

from falkorbench.ldbc import dataset
from falkorbench.ldbc import loader as loader_mod
from falkorbench.ldbc import params as params_mod
from falkorbench.ldbc import queries as query_mod
from falkorbench.ldbc import runner
from falkorbench.ldbc import schema

# --- query texts -------------------------------------------------------------


def test_all_fourteen_complex_reads_are_present():
    loaded = query_mod.load_queries()
    assert [q.number for q in loaded] == list(range(1, 15))
    assert [q.name for q in loaded] == [f"IC{n}" for n in range(1, 15)]


def test_param_header_comment_is_stripped():
    """The `:param` block is a Neo4j Browser directive, not Cypher.

    It is stripped so what is sent equals what is measured; if the regex ever
    stops matching, the queries still run (FalkorDB reads it as a comment) and
    nothing would otherwise notice.
    """
    for query in query_mod.load_queries():
        assert ":param" not in query.cypher, f"{query.name} kept its header"
        assert query.cypher.lstrip().startswith(("MATCH", "//")), query.name


def test_every_query_references_exactly_its_declared_parameters():
    """A parameter the text does not use, or uses without declaring, is a bug.

    An undeclared parameter is the dangerous direction: it evaluates to null,
    its predicate fails, and the query returns zero rows very fast — which reads
    as a fast query rather than a broken one.
    """
    for query in query_mod.load_queries():
        used = set(re.findall(r"\$(\w+)", query.cypher))
        assert used == set(query_mod.PARAM_NAMES[query.number]), query.name


def test_rewritten_queries_are_declared_and_annotated():
    """A rewrite must be both listed in REWRITES and commented in the file.

    Otherwise a number gets reported as an LDBC result while the query it came
    from has silently drifted from the reference text.
    """
    for query in query_mod.load_queries():
        annotated = "FalkorDB rewrite:" in query.cypher
        assert annotated == query.rewritten, (
            f"{query.name}: declared={query.rewritten} but annotated={annotated}"
        )


def test_the_known_dialect_gaps_are_the_rewritten_ones():
    """Pins which queries depart from upstream, and why.

    Each was verified against a running engine: shortestPath() is rejected
    inside MATCH, allShortestPaths() needs pre-bound endpoints, and there is no
    datetime(). If a future engine version accepts the original, this test is
    the prompt to drop the rewrite rather than carry it forever.
    """
    assert set(query_mod.REWRITES) == {1, 10, 13, 14}


def _code(cypher: str) -> str:
    """`cypher` with `//` comment lines removed.

    The rewrite annotations name the constructs they replaced, so a check for
    "this query no longer uses X" must look at the code, not the commentary.
    """
    return "\n".join(line for line in cypher.splitlines() if not line.strip().startswith("//"))


def test_no_query_still_calls_datetime():
    """FalkorDB has no temporal type; `datetime()` is `Unknown function`."""
    for query in query_mod.load_queries():
        assert "datetime(" not in _code(query.cypher), query.name


def test_shortest_path_is_never_bound_inside_match():
    """`MATCH path = shortestPath(...)` is rejected by FalkorDB.

    Checked structurally rather than by remembering which queries do it, so a
    future upstream bump reintroducing the pattern fails here.
    """
    for query in query_mod.load_queries():
        assert "= shortestPath(" not in _code(query.cypher), query.name


def test_select_resolves_names_and_rejects_unknown_ones():
    assert [q.name for q in query_mod.select(("IC3", "ic7"))] == ["IC3", "IC7"]
    assert len(query_mod.select(())) == 14
    with pytest.raises(ValueError, match="IC99"):
        query_mod.select(("IC99",))


def test_validate_rejects_missing_and_extra_parameters():
    query_mod.validate(13, {"person1Id": 1, "person2Id": 2})
    with pytest.raises(ValueError, match="missing"):
        query_mod.validate(13, {"person1Id": 1})
    with pytest.raises(ValueError, match="unexpected"):
        query_mod.validate(13, {"person1Id": 1, "person2Id": 2, "nope": 3})


# --- schema ------------------------------------------------------------------


def test_self_referencing_edges_have_distinct_endpoint_columns():
    """`Person.id|Person.id` collapses to one key once parsed into a map.

    Both endpoints would then resolve to the same node and every such edge would
    become a self-loop — a graph that loads without error and answers every
    query wrongly.
    """
    for edge in schema.EDGE_FILES:
        if edge.from_label == edge.to_label:
            assert edge.from_id != edge.to_id, edge.file


def test_message_superlabel_is_on_both_post_and_comment():
    """IC7 and IC8 match `(:Message)` and would otherwise find nothing."""
    by_label = {n.label: n for n in schema.NODE_FILES}
    assert "Message" in by_label["Post"].extra_labels
    assert "Message" in by_label["Comment"].extra_labels


def test_every_unique_constraint_has_a_supporting_index():
    """FalkorDB rejects a unique constraint with no exact-match index.

    The failure is `missing supporting exact-match index` at constraint-creation
    time, so this is a hard requirement rather than a tuning choice.
    """
    indexed = {(label, prop) for label, prop in schema.INDICES}
    for label in schema.UNIQUE_CONSTRAINTS:
        assert (label, "id") in indexed, label


def test_load_uses_fieldterminator_not_delimiter():
    """FalkorDB has no `DELIMITER` keyword; the LDBC CSVs are pipe-separated.

    `LOAD CSV WITH HEADERS DELIMITER '|' FROM ...` is rejected outright with
    `Invalid input 'DELIMITER': expected From`, and the standard-Cypher
    spelling is `FIELDTERMINATOR`, placed *after* `AS <var>`. Getting this
    wrong fails every single load, so pin the exact shape.
    """
    prefix = loader_mod._LOAD_PREFIX
    assert "DELIMITER" not in prefix
    assert re.search(r"AS\s+row\s+FIELDTERMINATOR\s+'\|'", prefix), prefix


def test_every_edge_endpoint_label_is_loadable_and_indexed():
    """An endpoint label the loader never creates cannot ever match.

    `LOAD CSV` into a `MATCH` that finds nothing drops the row silently, so this
    would surface as an edge file that loads zero edges.
    """
    creatable = set()
    for node in schema.NODE_FILES:
        if node.type_column is None:
            creatable.add(node.label)
        else:
            mapping = schema.PLACE_LABELS if node.label == "Place" else schema.ORGANISATION_LABELS
            creatable.update({node.label, *mapping.values()})
        creatable.update(node.extra_labels)

    indexed = {label for label, prop in schema.INDICES if prop == "id"}
    for edge in schema.EDGE_FILES:
        for label in (edge.from_label, edge.to_label):
            assert label in creatable, f"{edge.file}: no node file creates :{label}"
            assert label in indexed, f"{edge.file}: :{label}(id) is not indexed"


def test_person_properties_cover_what_ic1_and_ic10_return():
    """IC1 returns email/speaks as lists; IC10 filters on birthday parts."""
    person = next(n for n in schema.NODE_FILES if n.label == "Person")
    for prop in ("email", "speaks", "birthdayMonth", "birthdayDay"):
        assert prop in person.properties


# --- parameters --------------------------------------------------------------


def test_duration_days_is_converted_to_an_end_date():
    """IC3 and IC4 ship `durationDays`; the queries take `endDate`.

    Upstream drivers derive it. Doing the same here is what lets the query texts
    stay equal to upstream.
    """
    row = params_mod._coerce(4, {"personId": "42", "startDate": "1000", "durationDays": "3"})
    assert row == {"personId": 42, "startDate": 1000, "endDate": 1000 + 3 * 86_400_000}
    query_mod.validate(4, row)


def test_coerce_keeps_strings_as_strings_and_numbers_as_numbers():
    row = params_mod._coerce(1, {"personId": "933", "firstName": "Jose"})
    assert row == {"personId": 933, "firstName": "Jose"}


def test_official_parameter_directory_round_trips(tmp_path):
    """A parameter file in LDBC's own pipe-separated format is parsed."""
    for n in query_mod.COMPLEX_READS:
        names = [p for p in query_mod.PARAM_NAMES[n] if p != "endDate"]
        if n in (3, 4):
            names = [*names, "durationDays"]
        values = ["1000" if _numeric(p) else "Jose" for p in names]
        path = tmp_path / f"interactive_{n}_param.txt"
        with path.open("w", newline="") as fh:
            writer = csv.writer(fh, delimiter="|", lineterminator="\n")
            writer.writerow(names)
            writer.writerow(values)

    param_set = params_mod.from_directory(tmp_path, echo=lambda _: None)
    assert param_set.official is True
    assert "NOT comparable" not in param_set.caveat()
    for n in query_mod.COMPLEX_READS:
        assert len(param_set.for_query(n)) == 1


def _numeric(name: str) -> bool:
    return name.endswith(("Id", "Date", "Year")) or name in ("month", "durationDays")


def test_sampled_parameters_are_labelled_as_not_comparable():
    """The caveat is the point: sampled numbers are not LDBC results.

    Constructed directly rather than by sampling a graph — this asserts the
    reporting contract, which is what a reader of the CSV depends on.
    """
    sampled = params_mod.ParamSet(rows={}, official=False, source="seed=1, n=5")
    assert "NOT comparable to published LDBC results" in sampled.caveat()


def test_missing_parameter_file_is_an_error_not_an_empty_set(tmp_path):
    with pytest.raises(params_mod.ParamError, match=r"missing interactive_1_param\.txt"):
        params_mod.from_directory(tmp_path, echo=lambda _: None)


# --- results -----------------------------------------------------------------


def test_percentiles_are_values_that_actually_occurred():
    """Nearest-rank, so a reported p99 is a latency that was measured."""
    result = runner.QueryResult(name="IC1", rewritten=False)
    result.latencies_ms = [float(n) for n in range(1, 101)]
    assert result.pct(50) == 50.0
    assert result.pct(99) == 99.0
    assert result.pct(100) == 100.0


def test_a_query_that_always_returned_nothing_is_a_problem():
    """The failure mode this benchmark is most likely to produce silently.

    It does not error, and its latency looks excellent, but it measured nothing.
    """
    empty = runner.QueryResult(name="IC7", rewritten=False)
    empty.latencies_ms = [1.0, 2.0]
    empty.empty_runs = 2
    assert runner.problems([empty]) == ["IC7: returned zero rows on all 2 runs"]

    fine = runner.QueryResult(name="IC7", rewritten=False)
    fine.latencies_ms = [1.0, 2.0]
    fine.empty_runs = 1
    assert runner.problems([fine]) == []


def test_a_query_with_no_measurement_is_a_problem():
    failed = runner.QueryResult(name="IC14", rewritten=True)
    failed.failures = ["timeout"]
    assert runner.problems([failed]) == ["IC14: produced no measurement (timeout)"]


def test_result_csv_is_written_with_the_declared_columns(tmp_path):
    result = runner.QueryResult(name="IC1", rewritten=True)
    result.latencies_ms = [1.5, 2.5]
    result.rows_total = 40

    out = tmp_path / "ldbc.csv"
    runner.write_csv(out, [result])
    rows = list(csv.DictReader(out.open()))
    assert list(rows[0]) == list(runner.CSV_FIELDS)
    assert rows[0]["query"] == "IC1"
    assert rows[0]["rewritten"] == "yes"
    assert rows[0]["runs"] == "2"


# --- dataset preparation -----------------------------------------------------


def test_ambiguous_edge_header_is_rewritten_without_touching_the_rows(tmp_path):
    """Only the header line changes; the body is copied bytes-for-bytes.

    At SF1 these files are hundreds of MB, so re-encoding them would be the
    slowest part of a load for no benefit.
    """
    path = tmp_path / "person_knows_person_0_0.csv"
    path.write_text("Person.id|Person.id|creationDate\n1|2|100\n3|4|200\n")

    dataset._rewrite_header(path, ["FromPerson.id", "ToPerson.id"])
    lines = path.read_text().splitlines()
    assert lines[0] == "FromPerson.id|ToPerson.id|creationDate"
    assert lines[1:] == ["1|2|100", "3|4|200"]


def test_rewriting_a_header_twice_is_a_no_op(tmp_path):
    path = tmp_path / "edge.csv"
    path.write_text("Person.id|Person.id\n1|2\n")
    dataset._rewrite_header(path, ["FromPerson.id", "ToPerson.id"])
    first = path.read_text()
    dataset._rewrite_header(path, ["FromPerson.id", "ToPerson.id"])
    assert path.read_text() == first


def test_birthday_parts_are_derived_from_epoch_millis(tmp_path):
    """IC10 needs the calendar month and day, and FalkorDB has no datetime()."""
    path = tmp_path / "person_0_0.csv"
    path.write_text("id|firstName|birthday\n933|Mahinda|628646400000\n")

    dataset._derive_birthday_parts(path)
    rows = list(csv.DictReader(path.open(), delimiter="|"))
    assert rows[0]["birthdayMonth"] == "12"
    assert rows[0]["birthdayDay"] == "3"


def test_deriving_birthday_parts_twice_does_not_duplicate_columns(tmp_path):
    """A second run must not append the columns again.

    `prepare` guards this with a marker file, but the guard is only as good as
    the operation being idempotent on its own.
    """
    path = tmp_path / "person_0_0.csv"
    path.write_text("id|birthday\n933|628646400000\n")
    dataset._derive_birthday_parts(path)
    once = path.read_text()
    dataset._derive_birthday_parts(path)
    assert path.read_text() == once


def test_a_person_file_without_a_birthday_column_is_an_error(tmp_path):
    path = tmp_path / "person_0_0.csv"
    path.write_text("id|firstName\n1|Jose\n")
    with pytest.raises(dataset.DatasetError, match="no `birthday` column"):
        dataset._derive_birthday_parts(path)


def test_an_unknown_scale_factor_is_refused_before_any_download(tmp_path):
    with pytest.raises(dataset.DatasetError, match="unknown scale factor"):
        dataset.fetch(tmp_path, "1000", echo=lambda _: None)
