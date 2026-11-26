"""The LDBC query set, the dataset preparation, and `bench ldbc check`.

Nothing here needs a server or docker: the check's analysis is pure, its
measurement loop runs against scripted sides, and kit's docker calls are
replaced. Every result a test makes is held to falkorbench.contract.
"""

import csv
import io
import json
import os
import re
import shutil
import subprocess
import tarfile
from pathlib import Path
from types import SimpleNamespace

import pytest
from click.testing import CliRunner

from falkorbench import cli as cli_mod
from falkorbench import contract
from falkorbench import kit
from falkorbench.ldbc import check as ck
from falkorbench.ldbc import dataset
from falkorbench.ldbc import loader as loader_mod
from falkorbench.ldbc import params as params_mod
from falkorbench.ldbc import queries as query_mod
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
    inside MATCH, allShortestPaths() needs pre-bound endpoints, there is no
    datetime(), a traversal pattern is not a boolean in a projection, and a
    pattern comprehension's WHERE cannot see an outer list-comprehension
    variable. IC6 and IC9 used to be rewritten around two Rust wrong-result
    bugs (FalkorDB/FalkorDB#2556, #2557); both are fixed and the two queries
    run the upstream text again. If a future engine version accepts another
    original, this test is the prompt to drop that rewrite rather than carry
    it forever.
    """
    assert set(query_mod.REWRITES) == {1, 7, 10, 13, 14}


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


class _StubGraph:
    """Enough of a graph for sample() to run: canned columns and a date span."""

    LO, HI = 1_262_808_638_903, 1_347_528_281_121

    def ro_query(self, cypher, params=None):
        if "min(m.creationDate)" in cypher:
            rows = [[self.LO, self.HI]]
        else:
            n = (params or {}).get("n", 2)
            rows = [
                [f"v{i}"] if "Person)" not in cypher or "firstName" in cypher else [i]
                for i in range(n)
            ]
            if "MATCH (p:Person)-[:KNOWS]" in cypher or "p.id" in cypher:
                rows = [[1000 + i] for i in range(n)]
        return SimpleNamespace(result_set=rows)


def test_ic3_gets_a_far_wider_date_window_than_the_other_queries():
    """A 30-day window returns nothing for IC3 at SF0.1 on both engines, so a
    sampled run would report it as instant while measuring nothing."""
    param_set = params_mod.sample(
        SimpleNamespace(graph=_StubGraph()), count=3, seed=1, echo=lambda *a: None
    )
    span = _StubGraph.HI - _StubGraph.LO
    for ic3, ic4 in zip(param_set.for_query(3), param_set.for_query(4), strict=True):
        ic3_width = ic3["endDate"] - ic3["startDate"]
        assert ic3_width > 365 * 86_400_000, "IC3 needs at least a year to match anything"
        assert ic3_width == span - span // 4
        assert ic3_width > (ic4["endDate"] - ic4["startDate"]) * 10


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
    rows = list(csv.DictReader(path.read_text().splitlines(), delimiter="|"))
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


def _tar_zst(tmp_path, members):
    """A .tar.zst holding `members` ({name: bytes}), via the zstd binary."""
    if not shutil.which("zstd"):
        pytest.skip("zstd not on PATH")
    raw = tmp_path / "x.tar"
    with tarfile.open(raw, "w") as tar:
        for name, data in members.items():
            if data is None:  # a directory entry, as the LDBC archive has
                info = tarfile.TarInfo(name)
                # 0700 on one of them: the data filter ignores a directory's
                # archived mode, so the fallback must too.
                info.type, info.mode = tarfile.DIRTYPE, 0o700 if name == "ds" else 0o755
                tar.addfile(info)
                continue
            info = tarfile.TarInfo(name)
            info.size, info.mode = len(data), 0o4775  # setuid + group write
            tar.addfile(info, io.BytesIO(data))
    subprocess.run(["zstd", "-q", "-f", str(raw), "-o", str(tmp_path / "x.tar.zst")], check=True)
    return tmp_path / "x.tar.zst"


def test_extraction_applies_the_data_filter(tmp_path):
    archive = _tar_zst(tmp_path, {"ds": None, "ds/static": None, "ds/static/a.csv": b"id|x\n1|2\n"})
    dest = tmp_path / "cache"
    dest.mkdir()
    dataset._extract_zst(archive, dest)
    assert (dest / "ds/static/a.csv").read_bytes() == b"id|x\n1|2\n"
    # archived 0o4775: the data filter drops setuid and group write
    assert (dest / "ds/static/a.csv").stat().st_mode & 0o7777 == 0o755


def test_extraction_stops_on_a_python_without_the_data_filter(tmp_path, monkeypatch):
    """Python 3.11.2 (Debian 12) has no tarfile data filter: the dataset is
    not extracted unsafely, the fetch stops with a clear error."""
    archive = _tar_zst(tmp_path, {"ds": None, "ds/a.csv": b"id\n1\n"})
    monkeypatch.delattr(tarfile, "data_filter")
    dest = tmp_path / "cache"
    dest.mkdir()
    with pytest.raises(dataset.DatasetError, match=r"needs Python 3\.11\.4 or newer"):
        dataset._extract_zst(archive, dest)
    assert not (dest / "ds").exists()


def test_an_interrupted_extraction_does_not_pass_for_a_dataset(tmp_path, monkeypatch):
    """`fetch` treats an existing dataset directory as extracted, so a run
    stopped mid-extraction must not leave one behind."""
    sf = "0.1"
    tarball = tmp_path / schema.DATASET_TARBALL.format(sf=sf)
    tarball.write_bytes(b"x" * schema.SCALE_FACTORS[sf])
    target = dataset.dataset_dir(tmp_path, sf)

    def half(archive, dest):
        (dest / target.name / "static").mkdir(parents=True)
        (dest / target.name / "static/place_0_0.csv").write_text("id|")
        raise dataset.DatasetError("zstd exited 1")

    monkeypatch.setattr(dataset, "_extract_zst", half)
    with pytest.raises(dataset.DatasetError):
        dataset.fetch(tmp_path, sf, echo=lambda _: None)
    assert not target.exists()

    def whole(archive, dest):
        (dest / target.name / "static").mkdir(parents=True)

    monkeypatch.setattr(dataset, "_extract_zst", whole)
    assert dataset.fetch(tmp_path, sf, echo=lambda _: None) == target
    assert target.is_dir() and not list(tmp_path.glob(".extracting-*"))


# --- the nightly check (`bench ldbc check`) ----------------------------------
#
# Everything below runs without docker: the analysis is pure, the measurement
# loop is driven through scripted sides, and kit's docker calls are replaced.
# Each result is held to the perf contract (falkorbench.contract).

MAIN = "ghcr.io/falkordb/falkordb-server@sha256:" + "a" * 64
OTHER = "falkordb/falkordb-server@sha256:" + "b" * 64


def assert_valid(result) -> None:
    """Hold a result (or the path of a result.json) to the contract."""
    if isinstance(result, Path):
        _kind, problems = contract.validate_file(result, "result")
    else:
        _kind, problems = contract.validate_document(result, "result")
    assert problems == [], problems


def args(tmp_path, *extra, candidate=MAIN, baseline=OTHER, mode="nightly"):
    """The check's flags, as `bench ldbc check` parses them."""
    flags = ["--candidate", candidate, "--baseline", baseline, "--out", str(tmp_path / "out")]
    return ck.parse_args([*flags, "--mode", mode, "--cache", str(tmp_path), *extra])


def S(side, param, outcome="ok", ms=10.0, rows=3, phase="main", rnd=0, q="IC2"):
    """A scripted sample."""
    latency = ms if outcome == "ok" else None
    return ck.Sample(
        phase,
        rnd,
        q,
        param,
        side,
        outcome,
        latency,
        rows if outcome == "ok" else None,
        "" if outcome == "ok" else f"{outcome} here",
    )


def pair(param, *, base=None, cand=None, rounds=2):
    """Both sides on one parameter row, `rounds` times."""
    base, cand = base or {}, cand or {}
    out = []
    for r in range(rounds):
        out.append(S(ck.BASELINE, param, rnd=r, **base))
        out.append(S(ck.CANDIDATE, param, rnd=r, **cand))
    return out


def test_percentile_is_nearest_rank():
    assert ck.percentile([], 50) is None
    assert ck.percentile([5.0], 95) == 5.0
    values = [float(v) for v in range(1, 51)]
    assert ck.percentile(values, 50) == 25.0
    assert ck.percentile(values, 95) == 48.0


def test_params_header_uses_plain_names_and_quotes_strings():
    header = ck.params_header({"personId": 7, "tagName": 'a "b"'})
    assert header == 'CYPHER personId=7 tagName="a \\"b\\"" '
    assert ck.params_header({}) == ""
    with pytest.raises(ValueError, match="plain identifier"):
        ck.params_header({"a b": 1})


@pytest.mark.parametrize(
    ("metric", "cand", "base", "expected"),
    [
        ("p50_ms", 1.0, 0.5, None),  # exactly 2.0x and exactly the floor: strict >
        ("p50_ms", 2.0, 1.0, None),  # exactly 2.0x, 1 ms over the floor
        ("p50_ms", 0.25, 0.75, None),  # 3x faster, but exactly the floor
        ("p50_ms", 0.5, 0.0, None),  # baseline 0: exactly the floor
        ("p50_ms", 0.6, 0.0, "regression"),  # baseline 0: over the floor
        ("p50_ms", 1.25, 0.6, "regression"),  # IC13's size: 2.08x and 0.65 ms
        ("p50_ms", 2.0, 0.9, "regression"),  # 2.2x and 1.1 ms
        ("p50_ms", 0.8, 0.35, None),  # 2.3x but 0.45 ms, under the floor
        ("p50_ms", 30.0, 20.0, None),  # 10 ms slower but only 1.5x
        ("p95_ms", 3.0, 1.0, None),  # 3x but exactly the 2 ms p95 floor
        ("p95_ms", 3.1, 1.0, "regression"),
        ("p95_ms", 2.5, 1.0, None),  # 2.5x but 1.5 ms, under the 2 ms p95 floor
        ("p50_ms", 5.0, 10.0, None),  # exactly 0.5x
        ("p50_ms", 4.0, 10.0, "improvement"),
        ("p50_ms", 0.2, 0.6, None),  # 3x faster but only 0.4 ms
    ],
)
def test_a_latency_crosses_only_outside_both_the_ratio_and_the_floor(metric, cand, base, expected):
    assert ck.LatencyCompare(metric, cand, base).crossing() == expected


def test_baseline_that_cannot_run_a_query_leaves_it_out_as_info():
    """C's dialect gaps: the baseline errors on every row."""
    samples = []
    for p in range(3):
        samples += pair(p, base={"outcome": "error"})
    v = ck.analyse("IC1", samples)
    assert v.left_out and v.measured and not v.crossed
    assert [(f["name"], f["status"], f["candidate"], f["baseline"]) for f in v.findings] == [
        ("IC1 completed", "info", 1, 0)
    ]


def test_a_query_neither_side_can_run_is_left_out_as_info():
    """A dialect gap on the baseline leaves the query out, compared on neither side: the
    candidate's failure is not judged (null), so #4065's rule (0 is a regression) does not apply."""
    samples = []
    for p in range(2):
        samples += pair(p, base={"outcome": "error"}, cand={"outcome": "error"})
    v = ck.analyse("IC13", samples)
    assert v.left_out
    assert [(f["status"], f["candidate"], f["baseline"]) for f in v.findings] == [("info", None, 0)]


B, C = ck.BASELINE, ck.CANDIDATE


@pytest.mark.parametrize(
    ("samples", "expected"),
    [
        pytest.param(
            [*pair(0), S(B, 1), S(C, 1, outcome="timeout")],
            ("regression", 0, 1),
            id="candidate-times-out-where-the-baseline-finished",
        ),
        pytest.param(
            [S(B, 0, outcome="timeout"), S(C, 0, outcome="timeout")],
            ("regression", 0, 0),
            id="both-time-out-on-the-same-row",
        ),
        pytest.param(
            [*pair(0), S(B, 1, outcome="timeout"), S(C, 1, outcome="timeout")],
            ("regression", 0, 0),
            id="both-fail-one-row-another-finished",
        ),
        pytest.param(
            [*pair(0), S(B, 1, outcome="error"), S(C, 1)],
            ("improvement", 1, 0),
            id="one-baseline-error-is-not-a-dialect-gap",
        ),
        pytest.param(  # review: a baseline error on another row hid this
            [*pair(0), S(B, 1), S(C, 1, outcome="timeout"), S(B, 2, outcome="error"), S(C, 2)],
            ("regression", 0, 1),
            id="not-hidden-by-another-rows-baseline-error",
        ),
        pytest.param(  # review: per parameter row, a failure in another round hid this
            [
                S(B, 0, outcome="error", rnd=0),
                S(C, 0, rnd=0),
                S(B, 0, rnd=1),
                S(C, 0, outcome="timeout", rnd=1),
            ],
            ("regression", 0, 1),
            id="judged-per-request-not-per-row",
        ),
    ],
)
def test_completion_is_judged_request_by_request(samples, expected):
    """A candidate failure is a regression at once, on both sides too (#4065); the
    release gate counts it only on a request the baseline finished. The baseline
    failing where the candidate finished is an improvement."""
    v = ck.analyse("IC2", samples)
    assert not v.left_out
    (f,) = [f for f in v.findings if f["metric"] == "completed"]
    assert (f["status"], f["candidate"], f["baseline"]) == expected
    if f["status"] == "regression":
        gate = f.get(kit.INTRODUCED, kit.introduced(f["candidate"], f["baseline"]))
        assert gate is (expected[2] == 1)


def test_a_candidate_failure_on_a_request_the_baseline_never_ran_is_not_new_for_the_gate():
    v = ck.analyse("IC2", [*pair(0), S(C, 1, outcome="timeout")])
    (f,) = [f for f in v.findings if f["metric"] == "completed"]
    assert (f["status"], f["candidate"], f["baseline"]) == ("regression", 0, 1)
    assert f[kit.INTRODUCED] is False


def test_a_row_count_difference_is_a_regression_with_the_totals():
    samples = pair(0, cand={"rows": 2}) + pair(1)
    v = ck.analyse("IC9", samples)
    (f,) = [f for f in v.findings if f["metric"] == "rows"]
    assert (f["name"], f["candidate"], f["baseline"], f["status"]) == (
        "IC9 rows",
        5,
        6,
        "regression",
    )
    assert f["direction"] == "mismatch" and f["ratio"] is None


def test_row_differences_that_cancel_out_name_the_parameter_row():
    samples = pair(0, cand={"rows": 2}) + pair(1, cand={"rows": 4})
    v = ck.analyse("IC9", samples)
    (f,) = [f for f in v.findings if f["metric"] == "rows"]
    assert (f["name"], f["candidate"], f["baseline"]) == ("IC9 rows (param 0)", 2, 3)


def test_zero_rows_on_every_run_on_both_sides_is_reported_as_info():
    samples = pair(0, base={"rows": 0}, cand={"rows": 0}) + pair(
        1, base={"rows": 0}, cand={"rows": 0}
    )
    v = ck.analyse("IC6", samples)
    assert [(f["name"], f["metric"], f["status"]) for f in v.findings] == [
        ("IC6 returns rows", "rows", "info")  # a note (0 vs 0 rows), not a broken check
    ]


def test_zero_rows_on_the_candidate_only_is_a_row_mismatch():
    samples = pair(0, cand={"rows": 0}) + pair(1, cand={"rows": 0})
    v = ck.analyse("IC6", samples)
    assert [f["status"] for f in v.findings if f["metric"] == "rows"] == ["regression"]


def test_a_latency_crossing_waits_for_the_re_run():
    samples = []
    for p in range(5):
        samples += pair(p, base={"ms": 10.0}, cand={"ms": 40.0})
    v = ck.analyse("IC3", samples)
    assert v.crossed == ["p50_ms", "p95_ms"]
    assert [f for f in v.findings if f["metric"].endswith("_ms")] == []

    repeated = []
    for p in range(5):
        repeated += pair(
            p, base={"ms": 10.0, "phase": "rerun"}, cand={"ms": 35.0, "phase": "rerun"}
        )
    confirmed = ck.confirm("IC3", v.crossed, repeated).findings
    assert [(f["name"], f["status"], f["ratio"]) for f in confirmed] == [
        ("IC3 p50", "regression", 3.5),
        ("IC3 p95", "regression", 3.5),
    ]

    noise = []
    for p in range(5):
        noise += pair(p, base={"ms": 10.0, "phase": "rerun"}, cand={"ms": 11.0, "phase": "rerun"})
    settled = ck.confirm("IC3", v.crossed, noise)
    assert settled.findings == [] and not settled.inconclusive
    assert {r["status"] for r in settled.rows} == {"not repeated"}  # raw/rows.csv only


def test_a_candidate_that_stops_finishing_in_the_re_run_is_a_regression():
    rerun = [S(ck.BASELINE, 0, phase="rerun"), S(ck.CANDIDATE, 0, outcome="timeout", phase="rerun")]
    found = ck.confirm("IC3", ["p50_ms"], rerun).findings
    assert [(f["name"], f["status"]) for f in found] == [("IC3 completed", "regression")]


def test_an_improvement_needs_no_re_run():
    samples = []
    for p in range(4):
        samples += pair(p, base={"ms": 50.0}, cand={"ms": 10.0})
    v = ck.analyse("IC10", samples)
    assert v.crossed == []
    assert {f["status"] for f in v.findings} == {"improvement"}


def test_warmup_latencies_are_not_measured():
    samples = [
        S(ck.BASELINE, 0, ms=900.0, phase="warmup"),
        S(ck.CANDIDATE, 0, ms=1.0, phase="warmup"),
    ]
    samples += pair(0)
    assert ck.analyse("IC2", samples).crossed == []


def test_nothing_measured_is_not_a_pass():
    skipped = [S(ck.BASELINE, 0, outcome="skipped"), S(ck.CANDIDATE, 0, outcome="skipped")]
    v = ck.analyse("IC2", skipped)
    assert not v.measured


def test_findings_put_every_crossing_first_and_cap_the_informative_rows():
    def f(name, status, ratio):
        return kit.finding(name, "p50_ms", ratio, 1.0, status)

    crossings = [f("a", "regression", 3.0), f("b", "regression", 9.0), f("c", "improvement", 0.2)]
    info = [f(f"i{n}", "info", 1.0) for n in range(12)]
    chosen = kit.select(ck.worst_first(crossings + info))
    assert [x["name"] for x in chosen[:3]] == ["b", "a", "c"]
    assert [x["status"] for x in chosen[3:]] == ["info"] * 10


# --- the measurement loop, with scripted sides --------------------------------


class ScriptedSide(ck.Side):
    """A side whose replies come from a function of (query, params, call no.)."""

    def __init__(self, side, script):
        super().__init__(side, MAIN)
        self.script = script
        self.calls = []

    def run(self, cypher, params, timeout_ms):
        self.calls.append((cypher, dict(params)))
        return self.script(cypher, params, len(self.calls))


def _ok(ms=5.0, rows=2):
    return lambda *_: ck.Sample("", 0, "", 0, "", ck.OK, ms, rows)


def scripted_check(tmp_path, base_script, cand_script, *, names=("IC2", "IC7"), params=3):
    labels = ["--candidate-label", "main@4c1088949", "--baseline-label", "6.2.0"]
    commit = ["--candidate-commit", "4c1088949a5852e3949c05bfb3d92a603c4cd4ca"]
    check = ck.Check(args(tmp_path, *labels, *commit, *names), echo=lambda _: None)
    check.queries = query_mod.select(names)
    check.param_set = params_mod.ParamSet(
        rows={q.number: [{"personId": i} for i in range(params)] for q in check.queries},
        source="test",
    )
    base = ScriptedSide(ck.BASELINE, base_script)
    cand = ScriptedSide(ck.CANDIDATE, cand_script)
    check.sides = {ck.BASELINE: base, ck.CANDIDATE: cand}
    for side in (base, cand):
        original = side.script

        def stamped(cypher, p, n, _orig=original, _side=side.side):
            sample = _orig(cypher, p, n)
            sample.side = _side
            return sample

        side.script = stamped
    return check


def test_the_rounds_interleave_the_sides_and_swap_their_order(tmp_path):
    order = []

    def recorder(name):
        def script(cypher, params, n):
            order.append((name, params["personId"]))
            return ck.Sample("", 0, "", 0, "", ck.OK, 5.0, 2)

        return script

    check = scripted_check(tmp_path, recorder("b"), recorder("c"), names=("IC2",), params=2)
    check.measure()
    # warm-up b,c on row 0; round 0 baseline first; round 1 candidate first.
    assert order == [("b", 0), ("c", 0), ("b", 0), ("c", 0), ("b", 1), ("c", 1),
                     ("c", 0), ("b", 0), ("c", 1), ("b", 1)]  # fmt: skip
    assert check.problems == []
    assert check.compared_queries == 1


def test_a_timed_out_row_is_skipped_on_both_sides_and_reported_at_once(tmp_path):
    def cand(cypher, params, n):
        if params["personId"] == 1:
            return ck.Sample("", 0, "", 0, "", ck.TIMEOUT, error="Query timed out")
        return ck.Sample("", 0, "", 0, "", ck.OK, 5.0, 2)

    check = scripted_check(tmp_path, _ok(), cand, names=("IC2",), params=3)
    check.measure()
    # warm-up row 0; round 0 rows 0, 1 (timed out), 2; round 1 rows 0 and 2 only.
    assert [p["personId"] for _, p in check.sides[ck.CANDIDATE].calls] == [0, 0, 1, 2, 0, 2]
    assert [p["personId"] for _, p in check.sides[ck.BASELINE].calls] == [0, 0, 1, 2, 0, 2]
    assert [(f["name"], f["status"]) for f in check.findings] == [("IC2 completed", "regression")]
    skipped = [s for s in check.samples if s.outcome == ck.SKIPPED]
    assert len(skipped) == 2 and all("timed out on this parameter row" in s.error for s in skipped)


def test_a_side_stops_a_query_after_its_second_timeout(tmp_path):
    def hangs(cypher, params, n):
        return ck.Sample("", 0, "", 0, "", ck.TIMEOUT, error="Query timed out")

    def ic3_hangs(cypher, params, n):
        if "countryXName" in cypher:
            return hangs(cypher, params, n)
        return ck.Sample("", 0, "", 0, "", ck.OK, 5.0, 2)

    check = scripted_check(tmp_path, ic3_hangs, ic3_hangs, names=("IC2", "IC3"), params=25)
    check.measure()
    # IC3: warm-up row 0 and round 0 row 1 time out; nothing else is sent.
    sent = [c for c, _ in check.sides[ck.CANDIDATE].calls if "countryXName" in c]
    assert len(sent) == ck.MAX_TIMEOUTS == 2
    result = check.result()
    assert result["verdict"] == "regression"
    assert result["summary"].startswith("1 of 2 LDBC complex reads compared")
    assert "IC3 did not finish (baseline too)" in result["summary"]
    assert result["broken"] == [
        {"name": "IC3 completed", "baseline_too": True, "issue": 4001, "introduced": False}
    ]
    assert_valid(result)


def test_a_slow_candidate_is_re_run_and_confirmed(tmp_path):
    check = scripted_check(tmp_path, _ok(ms=5.0), _ok(ms=50.0), names=("IC2",), params=4)
    check.measure()
    phases = {s.phase for s in check.samples}
    assert phases == {"warmup", "main", "rerun"}
    names = [(f["name"], f["status"]) for f in check.findings]
    assert ("IC2 p50", "regression") in names and ("IC2 p95", "regression") in names


def test_the_deadline_leaves_the_rest_unmeasured_and_says_error(tmp_path):
    check = scripted_check(tmp_path, _ok(), _ok(), names=("IC2", "IC7"))
    check.t0 -= check.mode.deadline_s + 1
    check.measure()
    assert any("not measured" in p for p in check.problems)
    result = check.result()
    assert result["verdict"] == "error"
    assert_valid(result)


def test_an_a_a_run_passes_and_its_result_is_valid(tmp_path):
    check = scripted_check(tmp_path, _ok(), _ok())
    check.findings.append(
        kit.finding("load", "throughput", 1000.0, 1010.0, "info", direction="lower_is_worse")
    )
    check.measure()
    result = check.build(None)
    assert result["verdict"] == "pass", result
    assert result["confirmed"] is False and result["error"] is None
    assert result["candidate"]["commit"] == "4c1088949a5852e3949c05bfb3d92a603c4cd4ca"
    assert result["baseline"] == {"image": OTHER, "commit": None, "label": "6.2.0"}
    assert "2 of 2 LDBC complex reads compared at SF0.1" in result["summary"]
    assert "raw/samples.csv" in result["artifacts"]
    # Findings carry regression / improvement / info only; the rows that
    # crossed nothing are in raw/rows.csv (contract v1.12).
    assert {f["status"] for f in result["findings"]} <= {"regression", "improvement", "info"}
    assert "raw/rows.csv" in result["artifacts"]
    rows = list(csv.DictReader((check.out / "raw/rows.csv").read_text().splitlines()))
    assert {r["status"] for r in rows} == {"ok"} and len(rows) == 4  # 2 queries x p50, p95
    assert_valid(result)


def test_a_confirmed_regression_result_is_valid(tmp_path):
    check = scripted_check(tmp_path, _ok(ms=3.0), _ok(ms=10.0))
    check.measure()
    result = check.result()
    assert result["verdict"] == "regression" and result["confirmed"] is True
    assert result["summary"].count("\n") == 0
    assert "regression: IC2 p50 3.333x" in result["summary"]  # ratios to 3 places
    assert_valid(result)


def test_the_c_gaps_are_left_out_and_the_run_still_passes(tmp_path):
    def c_engine(cypher, params, n):
        if "shortestPath" in cypher:
            return ck.Sample("", 0, "", 0, "", ck.ERROR, error="unsupported")
        return ck.Sample("", 0, "", 0, "", ck.OK, 5.0, 2)

    check = scripted_check(tmp_path, c_engine, c_engine, names=("IC1", "IC2", "IC13"))
    check.measure()
    result = check.result()
    assert result["verdict"] == "pass", result
    assert "IC1, IC13 left out (baseline cannot run)" in result["summary"]
    assert {(f["name"], f["status"]) for f in result["findings"] if f["metric"] == "completed"} == {
        ("IC1 completed", "info"),
        ("IC13 completed", "info"),
    }
    assert_valid(result)


def test_an_image_not_pinned_by_digest_is_an_error_result_not_a_crash(tmp_path):
    out = tmp_path / "out"
    res = CliRunner().invoke(
        cli_mod.cli,
        ["ldbc", "check", "--candidate", "falkordb/falkordb-server:edge", "--baseline", OTHER,
         "--out", str(out), "--mode", "nightly"],
    )  # fmt: skip
    assert res.exit_code == 0, res.output
    result = json.loads((out / "result.json").read_text())
    assert result["verdict"] == "error"
    assert "--candidate 'falkordb/falkordb-server:edge' is not" in result["error"]


@pytest.fixture
def docker(monkeypatch):
    """kit's docker calls, replaced: every container runs, and what was
    started and removed is recorded."""
    seen = {"started": [], "removed": []}

    def start(image, name, **kwargs):
        seen["started"].append((name, kwargs["volumes"], kwargs["env"]))
        return kit.Container(name, 1)

    monkeypatch.setattr(kit, "pull", lambda image: None)
    monkeypatch.setattr(kit, "start", start)
    monkeypatch.setattr(kit, "logs", lambda name: "log\n")
    monkeypatch.setattr(kit, "state", lambda name: "running")
    monkeypatch.setattr(kit, "remove", lambda *names: seen["removed"].extend(names))
    monkeypatch.setattr(ck.Check, "fetch", lambda self: self.args.cache)
    return seen


def test_a_baseline_that_cannot_load_is_an_error(tmp_path, monkeypatch, docker):
    monkeypatch.setattr(ck.Check, "connect", lambda self, s: SimpleNamespace())

    def boom(client, root, *, import_root, echo):
        raise loader_mod.LoadError("person_0_0.csv: created 1 of 2 rows")

    monkeypatch.setattr(ck.loader_mod, "load", boom)
    result = ck.run_check(args(tmp_path, "IC2", baseline=MAIN), echo=lambda _: None)
    assert result["verdict"] == "error"
    assert "baseline could not load the dataset" in result["error"]
    assert (tmp_path / "out/raw/baseline-server.log").exists()
    # The image's own server, with the dataset mounted where IMPORT_FOLDER points.
    assert docker["started"] == [
        (f"perf-ldbc-baseline-{os.getpid()}", {tmp_path: "/import:ro"}, ck.SERVER_ENV)
    ]
    assert docker["removed"] == [f"perf-ldbc-baseline-{os.getpid()}"]
    assert_valid(tmp_path / "out/result.json")


def test_a_candidate_that_cannot_load_is_a_regression(tmp_path, monkeypatch, docker):
    monkeypatch.setattr(ck.Check, "connect", lambda self, s: s.side)

    def load(client, root, *, import_root, echo):
        if client == ck.CANDIDATE:
            raise loader_mod.LoadError("comment_0_0.csv: created 0 of 151043 rows")
        return loader_mod.LoadStats(nodes=10, edges=20, seconds=1.0)

    monkeypatch.setattr(ck.loader_mod, "load", load)
    result = ck.run_check(args(tmp_path, "IC2"), echo=lambda _: None)
    assert result["verdict"] == "regression"
    assert [(f["name"], f["status"]) for f in result["findings"]] == [
        ("load completed", "regression")
    ]
    assert len(docker["removed"]) == 2
    assert_valid(tmp_path / "out/result.json")


def test_a_stop_during_startup_is_an_error_result_and_removes_both_servers(
    tmp_path, monkeypatch, docker
):
    """kit.run_check() turns the stop into the error; cleanup() removes the
    container named before kit.start() returned."""
    import signal

    def connect(self, s):
        if s.side == ck.CANDIDATE:
            os.kill(os.getpid(), signal.SIGTERM)
            for _ in range(10_000):  # a pending handler runs between bytecodes
                pass
        return s.side

    monkeypatch.setattr(ck.Check, "connect", connect)
    monkeypatch.setattr(
        ck.loader_mod, "load", lambda *a, **k: loader_mod.LoadStats(nodes=1, edges=1, seconds=1.0)
    )
    result = ck.run_check(args(tmp_path, "IC2"), echo=lambda _: None)
    assert result["verdict"] == "error" and "stopped by signal 15" in result["error"]
    assert sorted(docker["removed"]) == sorted(
        f"perf-ldbc-{side}-{os.getpid()}" for side in ck.SIDES
    )
    assert_valid(tmp_path / "out/result.json")


def test_a_baseline_that_runs_nothing_is_an_error_not_a_pass(tmp_path):
    def broken(cypher, params, n):
        return ck.Sample("", 0, "", 0, "", ck.ERROR, error="unknown argument TIMEOUT")

    check = scripted_check(tmp_path, broken, _ok(), names=("IC2", "IC7"))
    check.measure()
    result = check.result()
    assert result["verdict"] == "error"
    assert "no query could be compared" in result["error"]
    assert_valid(result)


def test_a_deadline_that_cuts_the_rounds_is_an_error(tmp_path):
    check = scripted_check(tmp_path, _ok(), _ok(), names=("IC2",), params=2)
    real = check.past_deadline
    calls = {"n": 0}

    def after_round_zero():
        calls["n"] += 1
        return calls["n"] > 3 or real()  # warm-up and round 0 run, round 1 does not

    check.past_deadline = after_round_zero
    check.measure()
    assert any(s.error == "past the check's deadline" for s in check.samples)
    result = check.result()
    assert result["verdict"] == "error"
    assert "deadline passed" in result["error"]
    assert_valid(result)


def test_a_candidate_server_that_died_after_its_last_reply_is_a_regression(tmp_path):
    check = scripted_check(tmp_path, _ok(), _ok(), names=("IC2",))
    check.measure()
    check.sides[ck.CANDIDATE].down = "container exited 139"
    check.server_findings(set(ck.SIDES))
    result = check.result()
    assert result["verdict"] == "regression"
    assert ("server up", "regression") in [(f["name"], f["status"]) for f in result["findings"]]
    assert_valid(result)


def test_a_baseline_server_that_died_is_an_error(tmp_path):
    check = scripted_check(tmp_path, _ok(), _ok(), names=("IC2",))
    check.measure()
    check.sides[ck.BASELINE].down = "container exited 137 (OOM-killed)"
    check.server_findings(set(ck.SIDES))
    result = check.result()
    assert result["verdict"] == "error"
    assert "baseline server went down" in result["error"]
    assert_valid(result)


def test_a_bad_commit_is_reported_and_dropped_so_the_result_stays_valid(tmp_path):
    out = tmp_path / "out"
    res = CliRunner().invoke(
        cli_mod.cli,
        ["ldbc", "check", "--candidate", MAIN, "--baseline", OTHER, "--out", str(out),
         "--mode", "nightly", "--candidate-commit", "4c10889"],
    )  # fmt: skip
    assert res.exit_code == 0, res.output
    result = json.loads((out / "result.json").read_text())
    assert result["verdict"] == "error" and "--candidate-commit '4c10889'" in result["error"]
    assert result["candidate"]["commit"] is None
    assert_valid(out / "result.json")


def test_no_latency_finding_for_a_query_whose_rows_differ():
    """6.0.4's IC9 returned nothing (#2557), in 2 ms instead of 210 ms."""
    samples = []
    for p in range(4):
        samples += pair(p, base={"ms": 210.0, "rows": 20}, cand={"ms": 2.0, "rows": 0})
    v = ck.analyse("IC9", samples)
    assert v.rows_differ and v.crossed == []
    assert [(f["name"], f["status"]) for f in v.findings] == [("IC9 rows", "regression")]


def test_a_warm_up_alone_is_not_a_comparison(tmp_path):
    """The warm-up finishes on both sides, every timed request times out on both."""

    def warm_then_hang(cypher, params, n):
        if n == 1:
            return ck.Sample("", 0, "", 0, "", ck.OK, 5.0, 2)
        return ck.Sample("", 0, "", 0, "", ck.TIMEOUT, error="Query timed out")

    check = scripted_check(tmp_path, warm_then_hang, warm_then_hang, names=("IC2",), params=3)
    check.measure()
    result = check.result()
    assert check.compared_queries == 0
    # the candidate's timeouts are a regression (#4065), which outranks the error
    assert (
        result["verdict"] == "regression" and "no query could be compared" in result["incomplete"]
    )
    assert_valid(result)


def test_a_candidate_that_times_out_in_the_warm_up_is_a_regression(tmp_path):
    """The warm-up is judged with the rounds: the baseline finished it."""

    def cand(cypher, params, n):
        if n == 1:
            return ck.Sample("", 0, "", 0, "", ck.TIMEOUT, error="Query timed out")
        return ck.Sample("", 0, "", 0, "", ck.OK, 5.0, 2)

    check = scripted_check(tmp_path, _ok(), cand, names=("IC2",), params=3)
    check.measure()
    result = check.result()
    assert result["verdict"] == "regression"
    assert [(f["name"], f["status"]) for f in result["findings"]] == [
        ("IC2 completed", "regression")
    ]
    assert_valid(result)


def test_a_query_that_finished_on_neither_side_alone_is_not_a_comparison(tmp_path):
    def hangs(cypher, params, n):
        return ck.Sample("", 0, "", 0, "", ck.TIMEOUT, error="Query timed out")

    check = scripted_check(tmp_path, hangs, hangs, names=("IC3",), params=5)
    check.measure()
    result = check.result()
    assert result["verdict"] == "regression"  # #4065: the candidate did not finish IC3
    assert "no query could be compared" in result["incomplete"]
    assert_valid(result)


@pytest.mark.parametrize(
    ("state", "verdict"), [("gone", "pass"), ("exited 139", "regression"), ("running", "pass")]
)
def test_only_an_exited_container_counts_as_a_server_that_went_down(
    tmp_path, monkeypatch, state, verdict
):
    """`gone` (no such container) says nothing about how the server ran."""
    states = {"b": "running", "c": state}
    monkeypatch.setattr(kit, "state", states.__getitem__)
    monkeypatch.setattr(kit, "logs", lambda name: "")
    check = scripted_check(tmp_path, _ok(), _ok(), names=("IC2",))
    check.measure()
    check.sides[ck.BASELINE].container = kit.Container("b", 1)
    check.sides[ck.CANDIDATE].container = kit.Container("c", 1)
    check.collect()
    result = check.result()
    assert result["verdict"] == verdict, result
    assert_valid(result)


def test_row_counts_are_compared_per_request_not_from_the_warm_up():
    """Both sides return one row in the warm-up; the candidate then returns none."""
    samples = [
        S(ck.BASELINE, 0, rows=1, phase="warmup"),
        S(ck.CANDIDATE, 0, rows=1, phase="warmup"),
    ]
    samples += pair(0, base={"rows": 1}, cand={"rows": 0})
    v = ck.analyse("IC2", samples)
    assert [(f["metric"], f["status"]) for f in v.findings] == [("rows", "regression")]


def test_latencies_are_paired_per_request():
    """A candidate time whose baseline counterpart failed is not compared."""
    samples = [S(ck.BASELINE, 0, outcome="error", rnd=0), S(ck.CANDIDATE, 0, ms=900.0, rnd=0)]
    samples += [S(ck.BASELINE, 0, ms=10.0, rnd=1), S(ck.CANDIDATE, 0, ms=10.0, rnd=1)]
    base = ck.side_stats([s for s in samples if s.side == ck.BASELINE])
    cand = ck.side_stats([s for s in samples if s.side == ck.CANDIDATE])
    assert [(c.candidate, c.baseline) for c in ck.compare_latency(cand, base)] == [
        (10.0, 10.0),
        (10.0, 10.0),
    ]


def test_a_query_that_returns_nothing_on_both_sides_gets_no_latency_finding():
    samples = []
    for p in range(4):
        samples += pair(p, base={"rows": 0, "ms": 1.0}, cand={"rows": 0, "ms": 5.0})
    v = ck.analyse("IC6", samples)
    assert v.no_rows and v.crossed == []
    assert [f["name"] for f in v.findings] == ["IC6 returns rows"]


def test_a_re_run_that_returns_different_rows_is_a_rows_regression_not_a_timing():
    rerun = []
    for p in range(3):
        rerun += pair(
            p, base={"rows": 4, "phase": "rerun"}, cand={"rows": 0, "ms": 1.0, "phase": "rerun"}
        )
    found = ck.confirm("IC2", ["p50_ms"], rerun).findings
    assert [(f["name"], f["metric"], f["status"]) for f in found] == [
        ("IC2 rows", "rows", "regression")
    ]


def test_a_budget_cut_is_an_error_not_a_pass(tmp_path):
    check = scripted_check(tmp_path, _ok(), _ok(), names=("IC2",), params=3)
    clock = iter(range(0, 10_000, 200))  # every reading 200 s after the last
    real = ck.time.monotonic
    ck.time.monotonic = lambda: float(next(clock))
    try:
        check.t0 = 0.0
        check.mode = ck.Mode("nightly", "0.1", 3, 2, 2, 60_000, 300.0, 10**9, 600.0)
        check.measure()
    finally:
        ck.time.monotonic = real
    assert check.budget_skips > 0
    result = check.result()
    assert result["verdict"] == "error" and "per-round budget cut" in result["error"]
    assert_valid(result)


@pytest.mark.parametrize(("answers", "down"), [(True, ""), (False, "no answer to PING for 60s")])
def test_a_client_side_timeout_gives_the_server_the_ping_grace(monkeypatch, answers, down):
    """After a client-side timeout the side is down only if it does not answer
    PING within PING_GRACE_S; a busy server that answers keeps its row timeout."""
    waits = []

    def wait_ping(port, timeout_s, container):
        waits.append((port, timeout_s, container.name))
        return answers

    def hangs(*command):
        raise ck.RedisTimeoutError("Timeout reading from socket")

    monkeypatch.setattr(kit, "wait_ping", wait_ping)
    side = ck.Side(ck.CANDIDATE, MAIN, container=kit.Container("c", 7))
    side.conn = SimpleNamespace(
        execute_command=hangs, connection_pool=SimpleNamespace(disconnect=lambda: None)
    )
    sample = side.run("RETURN 1", {}, 1000)
    assert sample.outcome == ck.TIMEOUT
    assert waits == [(7, ck.PING_GRACE_S, "c")]
    assert side.down.startswith(down) and bool(side.down) is not answers


def test_the_state_is_read_even_when_the_logs_fail(tmp_path, monkeypatch):
    def logs(name):
        raise kit.DockerError("`docker logs` did not return within 60 s")

    monkeypatch.setattr(kit, "logs", logs)
    monkeypatch.setattr(kit, "state", {"b": "running", "c": "exited 139"}.__getitem__)
    check = scripted_check(tmp_path, _ok(), _ok(), names=("IC2",))
    check.measure()
    check.sides[ck.BASELINE].container = kit.Container("b", 1)
    check.sides[ck.CANDIDATE].container = kit.Container("c", 1)
    check.collect()
    assert ("server up", "regression") in [(f["name"], f["status"]) for f in check.findings]


def test_cleanup_removes_a_container_that_was_named_but_never_handed_over(tmp_path, monkeypatch):
    removed = []
    monkeypatch.setattr(kit, "remove", lambda *names: removed.extend(names))
    check = scripted_check(tmp_path, _ok(), _ok(), names=("IC2",))
    check.sides[ck.CANDIDATE].container_name = "perf-ldbc-candidate-7"
    check.cleanup()
    assert removed == ["perf-ldbc-candidate-7"]


def test_a_dataset_extracted_under_umask_077_is_readable_by_the_server(tmp_path, monkeypatch):
    sf = "0.1"
    tarball = tmp_path / schema.DATASET_TARBALL.format(sf=sf)
    tarball.write_bytes(b"x" * schema.SCALE_FACTORS[sf])
    target = dataset.dataset_dir(tmp_path, sf)

    def extract(archive, dest):
        (dest / target.name / "static").mkdir(parents=True, mode=0o700)
        (dest / target.name / "static/a.csv").write_text("id\n")
        os.chmod(dest / target.name / "static/a.csv", 0o600)
        os.chmod(dest / target.name, 0o700)

    monkeypatch.setattr(dataset, "_extract_zst", extract)
    dataset.fetch(tmp_path, sf, echo=lambda _: None)
    for d in (target, target / "static"):
        assert d.stat().st_mode & 0o777 == 0o755
    assert (target / "static/a.csv").stat().st_mode & 0o777 == 0o644


def test_a_prepared_file_keeps_its_mode_under_umask_077(tmp_path):
    path = tmp_path / "person_knows_person_0_0.csv"
    path.write_text("Person.id|Person.id|creationDate\n1|2|100\n")
    os.chmod(path, 0o644)
    old = os.umask(0o077)
    try:
        dataset._rewrite_header(path, ["FromPerson.id", "ToPerson.id"])
    finally:
        os.umask(old)
    assert path.stat().st_mode & 0o777 == 0o644


def test_a_filesystem_error_while_preparing_is_a_dataset_error(tmp_path, monkeypatch):
    def full(path):
        raise OSError(28, "No space left on device")

    monkeypatch.setattr(dataset, "_rewrite_header", lambda path, names: None)
    monkeypatch.setattr(dataset, "_derive_birthday_parts", full)
    with pytest.raises(dataset.DatasetError, match="No space left"):
        dataset.prepare(tmp_path, echo=lambda _: None)


def test_a_download_that_stalls_times_out(tmp_path, monkeypatch):
    seen = {}

    def urlopen(request, timeout=None):
        seen["timeout"] = timeout
        raise TimeoutError("timed out")

    monkeypatch.setattr(dataset.urllib.request, "urlopen", urlopen)
    with pytest.raises(OSError):
        dataset._download("https://example.invalid/x", tmp_path / "x.part")
    assert seen["timeout"] == dataset._DOWNLOAD_STALL_S


def test_a_duplicated_endpoint_id_fails_the_load():
    assert loader_mod.UNCONSTRAINED_ENDPOINTS == ("Place", "TagClass")

    def client(dupes):
        return SimpleNamespace(
            graph=SimpleNamespace(ro_query=lambda q: SimpleNamespace(result_set=[[dupes]]))
        )

    loader_mod._assert_unique_endpoint_ids(client(0))
    with pytest.raises(loader_mod.LoadError, match="Place holds 2 duplicated id"):
        loader_mod._assert_unique_endpoint_ids(client(2))


def test_a_warm_up_with_rows_does_not_make_empty_rounds_worth_timing():
    samples = [
        S(ck.BASELINE, 0, rows=1, phase="warmup"),
        S(ck.CANDIDATE, 0, rows=1, phase="warmup"),
    ]
    for p in range(3):
        samples += pair(p, base={"rows": 0, "ms": 1.0}, cand={"rows": 0, "ms": 9.0})
    v = ck.analyse("IC2", samples)
    assert v.no_rows and v.crossed == []


def test_a_re_run_with_no_rows_on_either_side_is_inconclusive():
    rerun = []
    for p in range(3):
        rerun += pair(
            p, base={"rows": 0, "phase": "rerun"}, cand={"rows": 0, "ms": 50.0, "phase": "rerun"}
        )
    result = ck.confirm("IC2", ["p50_ms"], rerun, {0, 1, 2})
    assert result.findings == [] and "no rows" in result.inconclusive


def test_a_re_run_that_did_not_retime_a_row_does_not_clear_the_crossing():
    """Row 2 was the slow one; both sides fail it in the re-run: the candidate's failure is
    the finding (#4065), so the crossing is not cleared either."""
    rerun = pair(0, base={"phase": "rerun"}, cand={"phase": "rerun"})
    rerun += pair(1, base={"phase": "rerun"}, cand={"phase": "rerun"})
    rerun += pair(
        2,
        base={"outcome": "timeout", "phase": "rerun"},
        cand={"outcome": "timeout", "phase": "rerun"},
    )
    result = ck.confirm("IC2", ["p50_ms"], rerun, {0, 1, 2})
    # the shared failure is the candidate's break (#4065), and the crossing stays unsettled
    assert [(f["name"], f["status"], f["baseline"]) for f in result.findings] == [
        ("IC2 completed", "regression", 0)
    ]
    assert "[2]" in result.inconclusive


def test_a_shared_failure_in_the_re_run_does_not_hide_a_repeated_crossing():
    """Row 2 fails on both sides in the re-run; rows 0 and 1 are 3x slower again."""
    rerun = []
    for p in (0, 1):
        rerun += pair(p, base={"ms": 10.0, "phase": "rerun"}, cand={"ms": 30.0, "phase": "rerun"})
    rerun += pair(
        2,
        base={"outcome": "timeout", "phase": "rerun"},
        cand={"outcome": "timeout", "phase": "rerun"},
    )
    result = ck.confirm("IC2", ["p50_ms"], rerun, {0, 1})
    assert [(f["name"], f["status"]) for f in result.findings] == [
        ("IC2 completed", "regression"),
        ("IC2 p50", "regression"),
    ]
    assert result.findings[0][kit.INTRODUCED] is False and not result.inconclusive


def test_a_re_run_the_candidate_alone_fails_is_the_finding():
    rerun = pair(0, base={"phase": "rerun"}, cand={"phase": "rerun", "ms": 30.0})
    rerun += pair(1, base={"phase": "rerun"}, cand={"outcome": "timeout", "phase": "rerun"})
    result = ck.confirm("IC2", ["p50_ms"], rerun, {0, 1})
    assert [(f["name"], f["status"], f["baseline"]) for f in result.findings] == [
        ("IC2 completed", "regression", 1)
    ]


def test_an_inconclusive_re_run_is_an_error_in_the_result(tmp_path):
    def fast_then_empty(cypher, params, n):
        return ck.Sample("", 0, "", 0, "", ck.OK, 50.0 if n <= 7 else 5.0, 2 if n <= 7 else 0)

    check = scripted_check(tmp_path, _ok(ms=5.0), fast_then_empty, names=("IC2",), params=3)
    check.measure()
    result = check.result()
    assert result["verdict"] in ("error", "regression")
    assert_valid(result)


def test_rows_excluded_by_the_timeout_rules_are_not_budget_skips(tmp_path, monkeypatch):
    """Two shared timeouts use up both the timeout cap and the round budget:
    the rows after them are excluded by the cap, not cut by the budget."""
    now = {"t": 0.0}
    monkeypatch.setattr(ck.time, "monotonic", lambda: now["t"])

    def hangs(cypher, params, n):
        now["t"] += 200.0  # each request takes 200 s of the 300 s budget
        return ck.Sample("", 0, "", 0, "", ck.TIMEOUT, error="Query timed out")

    check = scripted_check(tmp_path, hangs, hangs, names=("IC3",), params=6)
    check.t0 = 0.0
    check.mode = ck.Mode("nightly", "0.1", 6, 2, 2, 60_000, 300.0, 1e9, 600.0)
    check.measure()
    assert check.budget_skips == 0
    assert not any("budget" in p for p in check.problems)


def test_the_summary_keeps_warm_ups_out_of_the_main_rows(tmp_path):
    check = scripted_check(tmp_path, _ok(rows=3), _ok(rows=3), names=("IC2",), params=1)
    check.measure()
    by = {(r["phase"], r["side"]): r for r in check.summary_rows}
    assert by[("warmup", ck.BASELINE)]["attempted"] == 1
    assert by[("main", ck.BASELINE)]["attempted"] == 2
    assert by[("main", ck.BASELINE)]["rows_total"] == 6


def test_the_cache_root_is_searchable_by_the_server(tmp_path, monkeypatch):
    cache = tmp_path / "cache"
    old = os.umask(0o077)
    try:
        # an already extracted dataset, so fetch() returns before any download
        monkeypatch.setattr(dataset, "dataset_dir", lambda cache, sf: tmp_path)
        dataset.fetch(cache, "0.1", echo=lambda _: None)
    finally:
        os.umask(old)
    assert cache.stat().st_mode & 0o055 == 0o055


def test_a_regression_that_was_also_cut_short_says_so_in_the_summary(tmp_path):
    """6.0.4 vs 6.2.0 on GCE: regressions, plus a deadline that cut IC12's re-run."""
    check = scripted_check(tmp_path, _ok(ms=5.0), _ok(ms=50.0), names=("IC2",))
    check.measure()
    check.problem(
        "the 35 min deadline passed before every query had its rounds (22 parameter-row runs skipped)"
    )
    result = check.result()
    assert result["verdict"] == "regression" and result["error"] is None
    assert "also not fully measured: the 35 min deadline passed" in result["summary"]
    assert_valid(result)
