"""The 14 Interactive v1 complex reads, and the parameters they are run with.

The query texts are vendored verbatim from
`ldbc/ldbc_snb_interactive_v1_impls` (`cypher/queries/`), with four rewrites
that FalkorDB's dialect requires. Each rewrite is commented in the `.cypher`
file next to the line it replaces, and `REWRITES` below records them in one
place so a report can state exactly how far the run departs from upstream.

Keeping the texts as files rather than string literals means a future upstream
bump is a readable diff.
"""

from __future__ import annotations

import re
from collections.abc import Callable
from functools import cache
from pathlib import Path
from typing import Any
from typing import NamedTuple

QUERY_DIR = Path(__file__).parent / "queries"

COMPLEX_READS = tuple(range(1, 15))

#: Departures from the upstream query text, by query number. Reported alongside
#: results: a number produced by a rewritten query is not measuring quite the
#: same thing as one produced by the original, and that should be visible
#: without reading the diff.
REWRITES: dict[int, str] = {
    1: "shortestPath() moved from MATCH into WITH (unsupported in MATCH)",
    10: "datetime({epochMillis: ...}).month/.day replaced by loader-derived "
    "birthdayMonth/birthdayDay properties (no temporal type)",
    13: "shortestPath() moved from MATCH into WITH (unsupported in MATCH)",
    14: "allShortestPaths() endpoints pre-bound in a preceding MATCH "
    "(endpoints must already be resolved)",
}

#: Parameter names each query takes, in the order the upstream header declares
#: them. Used to validate a parameter file rather than discovering a missing
#: parameter as a null comparison at run time.
PARAM_NAMES: dict[int, tuple[str, ...]] = {
    1: ("personId", "firstName"),
    2: ("personId", "maxDate"),
    3: ("personId", "countryXName", "countryYName", "startDate", "endDate"),
    4: ("personId", "startDate", "endDate"),
    5: ("personId", "minDate"),
    6: ("personId", "tagName"),
    7: ("personId",),
    8: ("personId",),
    9: ("personId", "maxDate"),
    10: ("personId", "month"),
    11: ("personId", "countryName", "workFromYear"),
    12: ("personId", "tagClassName"),
    13: ("person1Id", "person2Id"),
    14: ("person1Id", "person2Id"),
}

_HEADER = re.compile(r"^\s*(//.*?\n)?\s*/\*.*?\*/", re.S)


class ComplexRead(NamedTuple):
    number: int
    cypher: str

    @property
    def name(self) -> str:
        return f"IC{self.number}"

    @property
    def rewritten(self) -> bool:
        return self.number in REWRITES


@cache
def load_queries() -> tuple[ComplexRead, ...]:
    """Read the vendored query texts, stripping the `:param` header comment.

    The header is a Neo4j Browser directive, not Cypher. FalkorDB parses it as a
    comment and ignores it, but removing it keeps what is sent equal to what is
    measured.
    """
    out = []
    for n in COMPLEX_READS:
        path = QUERY_DIR / f"interactive-complex-{n}.cypher"
        text = path.read_text(encoding="utf-8")
        body = _HEADER.sub("", text, count=1).strip()
        if not body:
            raise ValueError(f"{path.name}: no query body after the header comment")
        out.append(ComplexRead(number=n, cypher=body))
    return tuple(out)


def select(names: tuple[str, ...]) -> list[ComplexRead]:
    """Resolve `IC3`-style names to queries, erroring on an unknown name."""
    queries = load_queries()
    if not names:
        return list(queries)
    wanted = {n.upper() for n in names}
    chosen = [q for q in queries if q.name in wanted]
    if missing := wanted - {q.name for q in chosen}:
        raise ValueError(f"unknown queries: {sorted(missing)}")
    return chosen


def validate(number: int, params: dict[str, Any]) -> None:
    """Raise if `params` does not supply exactly what query `number` needs.

    An absent parameter is not an error in Cypher — it evaluates to null, the
    predicate that uses it silently fails, and the query returns zero rows very
    quickly. That reads as a fast query rather than a broken one, which is
    exactly the failure this benchmark must not report.
    """
    expected = set(PARAM_NAMES[number])
    got = set(params)
    if missing := expected - got:
        raise ValueError(f"IC{number}: missing parameter(s) {sorted(missing)}")
    if extra := got - expected:
        raise ValueError(f"IC{number}: unexpected parameter(s) {sorted(extra)}")


def rewrite_note(echo: Callable[[str], None]) -> None:
    """Print the rewrite list, so a run never reports numbers without it."""
    echo(f"{len(REWRITES)} of {len(COMPLEX_READS)} queries differ from upstream:")
    for n in sorted(REWRITES):
        echo(f"  IC{n}: {REWRITES[n]}")
