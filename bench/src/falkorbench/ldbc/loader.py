"""Loading the LDBC SNB dataset into FalkorDB with `LOAD CSV`.

Ordering is not incidental. Indices come first because the edge phase looks up
both endpoints by `id` on every row — without them each of the ~17M SF1 edges
drives two label scans. Unique constraints come last, because FalkorDB validates
a constraint against existing data asynchronously and a constraint created up
front would be validated once per inserted row.
"""

from __future__ import annotations

import time
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path

from redis.exceptions import ResponseError

from falkorbench.client import BenchClient
from falkorbench.ldbc import schema

Echo = Callable[[str], None]


class LoadError(RuntimeError):
    """The dataset could not be loaded."""


@dataclass
class LoadStats:
    nodes: int = 0
    edges: int = 0
    seconds: float = 0.0

    def line(self) -> str:
        return f"{self.nodes:,} nodes, {self.edges:,} edges in {self.seconds:.1f}s"


def load(
    client: BenchClient,
    dataset: Path,
    *,
    import_root: Path,
    echo: Echo = print,
) -> LoadStats:
    """Load `dataset` into `client`'s graph.

    `dataset` must live under `import_root`, which is the server's
    IMPORT_FOLDER — `LOAD CSV` resolves `file://` against it, so a path outside
    that tree is not reachable by the server no matter what this process can
    see.
    """
    try:
        rel = dataset.resolve().relative_to(import_root.resolve())
    except ValueError as e:
        raise LoadError(
            f"dataset {dataset} is not under the server's import folder {import_root}"
        ) from e

    stats = LoadStats()
    started = time.monotonic()

    create_indices(client, echo=echo)
    for node in schema.NODE_FILES:
        stats.nodes += _load_nodes(client, rel, node, echo=echo)
    for edge in schema.EDGE_FILES:
        stats.edges += _load_edges(client, rel, edge, echo=echo)
    create_constraints(client, echo=echo)

    stats.seconds = time.monotonic() - started
    echo(f"loaded {stats.line()}")
    return stats


def create_indices(client: BenchClient, *, echo: Echo = print) -> None:
    """Create the exact-match indices, tolerating ones that already exist."""
    for label, prop in schema.INDICES:
        try:
            client.graph.query(f"CREATE INDEX FOR (n:{label}) ON (n.{prop})")
        except ResponseError as e:
            if "already indexed" not in str(e).lower():
                raise LoadError(f"index :{label}({prop}) failed: {e}") from e
    echo(f"created {len(schema.INDICES)} indices")


def create_constraints(
    client: BenchClient,
    *,
    timeout: float = 600.0,
    echo: Echo = print,
) -> None:
    """Create the unique constraints and wait for them to become OPERATIONAL.

    FalkorDB rejects `CREATE CONSTRAINT ... ASSERT`, so these go through
    GRAPH.CONSTRAINT, which replies PENDING and validates in the background —
    reporting itself as `UNDER CONSTRUCTION` in `db.constraints()` until it
    settles. A constraint still under construction is not yet enforcing
    anything, and one that ends FAILED means the data violated it; both would
    otherwise pass unnoticed, since the create call itself succeeded.
    """
    for label in schema.UNIQUE_CONSTRAINTS:
        try:
            client.command(
                "GRAPH.CONSTRAINT",
                "CREATE",
                client.graph_name,
                "UNIQUE",
                "NODE",
                label,
                "PROPERTIES",
                "1",
                "id",
            )
        except ResponseError as e:
            if "already exists" in str(e).lower():
                continue
            raise LoadError(f"constraint on :{label}(id) failed: {e}") from e

    deadline = time.monotonic() + timeout
    while True:
        pending, failed = _constraint_states(client)
        if failed:
            raise LoadError(f"unique constraint(s) failed validation: {', '.join(failed)}")
        if not pending:
            break
        if time.monotonic() > deadline:
            raise LoadError(f"constraints still pending after {timeout:.0f}s: {', '.join(pending)}")
        time.sleep(0.5)
    echo(f"created {len(schema.UNIQUE_CONSTRAINTS)} unique constraints")


def _constraint_states(client: BenchClient) -> tuple[list[str], list[str]]:
    """(pending, failed) constraint labels, from `db.constraints()`.

    The engine's three statuses are `UNDER CONSTRUCTION`, `OPERATIONAL` and
    `FAILED` (`ConstraintStatus`'s Display impl). It is *not* `PENDING` — that
    is only what the `GRAPH.CONSTRAINT CREATE` call itself replies. An
    in-progress status can also carry a progress prefix, as in
    `[Indexing] 3/9: UNDER CONSTRUCTION`, so match on substring rather than
    equality or the poll would call a healthy constraint failed.
    """
    res = client.graph.ro_query("CALL db.constraints()")
    headers = [h[1] if isinstance(h, (list, tuple)) else h for h in (res.header or [])]
    try:
        label_i, status_i = headers.index("label"), headers.index("status")
    except ValueError as e:
        raise LoadError(f"unexpected db.constraints() shape: {headers}") from e

    pending: list[str] = []
    failed: list[str] = []
    for row in res.result_set:
        status = str(row[status_i]).upper()
        if "UNDER CONSTRUCTION" in status:
            pending.append(str(row[label_i]))
        elif "OPERATIONAL" not in status:
            failed.append(f"{row[label_i]} ({status})")
    return pending, failed


def _load_nodes(client: BenchClient, rel: Path, node: schema.NodeFile, *, echo: Echo) -> int:
    """Load one node file, returning the number of nodes created."""
    props = ", ".join(f"{k}: {v}" for k, v in node.properties.items())
    labels = ":".join((node.label or "", *node.extra_labels)).strip(":")

    if node.type_column is None:
        created = _run_load(client, node.file, rel, f"CREATE (:{labels} {{{props}}})")
    else:
        # The polymorphic static files carry the subtype in a column, and a
        # label cannot come from an expression. One pass per subtype, filtered —
        # these are the two smallest files in the dataset (a few thousand rows
        # even at SF1), so the repeated scan costs nothing worth optimising.
        mapping = schema.PLACE_LABELS if node.label == "Place" else schema.ORGANISATION_LABELS
        _assert_types_covered(client, node, rel, set(mapping))
        created = 0
        for value, sub in mapping.items():
            all_labels = ":".join((node.label, sub, *node.extra_labels))
            created += _run_load(
                client,
                node.file,
                rel,
                f"CREATE (:{all_labels} {{{props}}})",
                where=f"row.{node.type_column} = '{value}'",
            )

    echo(f"  {node.file:<48} {created:>9,} nodes")
    return created


#: The pipe-separated dataset read with a header row. `FIELDTERMINATOR` is
#: standard Cypher and goes *after* `AS row`; FalkorDB has no `DELIMITER`
#: keyword (`Invalid input 'DELIMITER': expected From`).
_LOAD_PREFIX = "LOAD CSV WITH HEADERS FROM $file AS row FIELDTERMINATOR '|' "


def _assert_types_covered(
    client: BenchClient,
    node: schema.NodeFile,
    rel: Path,
    known: set[str],
) -> None:
    """Fail if the file holds a subtype the label mapping does not name.

    Without this an unrecognised `type` value would simply not be loaded, and
    the first symptom would be a query quietly returning fewer rows.
    """
    try:
        res = client.graph.query(
            f"{_LOAD_PREFIX}WITH row.{node.type_column} AS t RETURN collect(DISTINCT t)",
            {"file": _url(rel, node.file)},
        )
    except ResponseError as e:
        raise LoadError(f"{node.file}: {e}") from e
    found = {str(v) for v in (res.result_set[0][0] or [])}
    if unknown := found - known:
        raise LoadError(f"{node.file}: unmapped {node.type_column} value(s): {sorted(unknown)}")


def _load_edges(client: BenchClient, rel: Path, edge: schema.EdgeFile, *, echo: Echo) -> int:
    """Load one edge file, returning the number of relationships created."""
    props = ", ".join(f"{k}: {v}" for k, v in edge.properties.items())
    props = f" {{{props}}}" if props else ""
    cypher = (
        f"MATCH (f:{edge.from_label} {{id: toInteger(row.`{edge.from_id}`)}}) "
        f"WITH f, row "
        f"MATCH (t:{edge.to_label} {{id: toInteger(row.`{edge.to_id}`)}}) "
        f"CREATE (f)-[:{edge.type}{props}]->(t)"
    )
    created = _run_load(client, edge.file, rel, cypher)
    echo(f"  {edge.file:<48} {created:>9,} edges")
    return created


def _run_load(
    client: BenchClient,
    name: str,
    rel: Path,
    tail: str,
    *,
    where: str | None = None,
) -> int:
    """Run one `LOAD CSV ... <tail>` and return entities created.

    The row count is checked against what was created: `LOAD CSV` piped into a
    `MATCH` silently drops rows whose endpoints are missing, so an edge file
    loaded before its nodes would report success having created nothing. The
    same `where` filter is applied to both, so a filtered subtype load is
    compared against the rows it was actually supposed to create.
    """
    url = _url(rel, name)
    prefix = _LOAD_PREFIX
    if where:
        prefix += f"WITH row WHERE {where} "
    try:
        expected = client.graph.query(prefix + "RETURN count(row)", {"file": url}).result_set[0][0]
        res = client.graph.query(prefix + tail, {"file": url})
    except ResponseError as e:
        raise LoadError(f"{name}: {e}") from e

    # The client reports these as floats; keep them integral so a count reads
    # as "1,460" rather than "1,460.0" and compares exactly against count(row).
    created = int(res.nodes_created) + int(res.relationships_created)
    if created != expected:
        raise LoadError(
            f"{name}: created {created} of {expected} rows — "
            f"endpoints missing, or a duplicate id in the source"
        )
    return created


def _url(rel: Path, name: str) -> str:
    return f"file://{rel.as_posix()}/{name}"
