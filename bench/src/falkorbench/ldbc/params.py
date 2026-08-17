"""Substitution parameters for the complex reads.

LDBC ships *substitution parameters* generated alongside each dataset, chosen so
that the queries hit a representative spread of the data rather than whichever
person happens to be first. Running with hand-picked ids produces numbers that
are not comparable to anything, because the cost of these queries varies by
orders of magnitude with how well-connected the chosen person is.

Two sources, in order:

1. `--params DIR` — an official `interactive_N_param.txt` set. This is the only
   source whose numbers are comparable with published LDBC results.
2. Sampling the loaded graph, seeded. Deterministic, so two runs of the same
   scale factor compare against each other, but **not** comparable to published
   results: the sampler picks arbitrary well-connected entities rather than
   LDBC's calibrated ones.

The distinction is carried on `ParamSet.official` so a report can state which
was used, rather than leaving a reader to assume the stronger one.
"""

from __future__ import annotations

import csv
import random
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from falkorbench.client import BenchClient
from falkorbench.ldbc import queries as query_mod

Echo = Callable[[str], None]

#: A day in milliseconds. IC3 and IC4 ship `durationDays`, but the queries take
#: an `endDate`; upstream drivers derive it. Doing that here keeps the query
#: texts equal to upstream.
_DAY_MS = 86_400_000

#: Parameter file column -> query parameter name, where they differ.
_ALIASES = {
    "personId": "personId",
    "person1Id": "person1Id",
    "person2Id": "person2Id",
}


class ParamError(RuntimeError):
    """Parameters could not be sourced."""


@dataclass
class ParamSet:
    """Parameter rows for each query number.

    official  True only when these came from an LDBC-generated parameter set.
              Reported with the results; sampled parameters do not support a
              comparison against published numbers.
    """

    rows: dict[int, list[dict[str, Any]]]
    official: bool
    source: str

    def for_query(self, number: int) -> list[dict[str, Any]]:
        return self.rows.get(number, [])

    def caveat(self) -> str:
        if self.official:
            return f"official LDBC substitution parameters ({self.source})"
        return (
            f"parameters sampled from the loaded graph ({self.source}); "
            f"deterministic and self-comparable, but NOT comparable to "
            f"published LDBC results"
        )


def load(
    client: BenchClient,
    *,
    directory: Path | None,
    count: int,
    seed: int,
    echo: Echo = print,
) -> ParamSet:
    """Official parameters from `directory` if given, else sampled ones."""
    if directory is not None:
        return from_directory(directory, echo=echo)
    return sample(client, count=count, seed=seed, echo=echo)


def from_directory(directory: Path, *, echo: Echo = print) -> ParamSet:
    """Parse an LDBC `interactive_N_param.txt` set.

    The files are pipe-separated with a header naming each column; dates are
    epoch milliseconds. IC3 and IC4 supply `durationDays` rather than an
    `endDate`, which is derived here.
    """
    if not directory.is_dir():
        raise ParamError(f"parameter directory not found: {directory}")

    rows: dict[int, list[dict[str, Any]]] = {}
    for n in query_mod.COMPLEX_READS:
        path = directory / f"interactive_{n}_param.txt"
        if not path.exists():
            raise ParamError(f"missing {path.name} in {directory}")
        with path.open(encoding="utf-8", newline="") as fh:
            parsed = [_coerce(n, row) for row in csv.DictReader(fh, delimiter="|")]
        if not parsed:
            raise ParamError(f"{path.name} has no rows")
        for row in parsed:
            query_mod.validate(n, row)
        rows[n] = parsed
        echo(f"  IC{n:<3} {len(parsed):>6,} parameter rows")
    return ParamSet(rows=rows, official=True, source=str(directory))


def _coerce(number: int, row: dict[str, str]) -> dict[str, Any]:
    """Convert one raw parameter row to the query's parameter map."""
    out: dict[str, Any] = {}
    duration_days: int | None = None
    for key, raw in row.items():
        if key is None or raw is None:
            continue
        name = _ALIASES.get(key, key)
        if name == "durationDays":
            duration_days = int(raw)
            continue
        out[name] = int(raw) if _is_int(raw) else raw

    if duration_days is not None:
        start = out.get("startDate")
        if not isinstance(start, int):
            raise ParamError(f"IC{number}: durationDays given without an integer startDate")
        out["endDate"] = start + duration_days * _DAY_MS
    return out


def _is_int(text: str) -> bool:
    return bool(text) and (text[1:] if text[0] in "+-" else text).isdigit()


def sample(
    client: BenchClient,
    *,
    count: int,
    seed: int,
    echo: Echo = print,
) -> ParamSet:
    """Derive parameters from the loaded graph, deterministically.

    People are drawn from the most-connected end of the `KNOWS` degree
    distribution rather than uniformly. A uniformly drawn person is very often
    isolated, and a query anchored on an isolated person returns nothing in
    microseconds — a set of those would report the benchmark as uniformly fast
    while measuring almost none of the work it exists to measure.
    """
    rng = random.Random(seed)

    people = _column(
        client,
        "MATCH (p:Person)-[:KNOWS]-() WITH p, count(*) AS deg "
        "RETURN p.id ORDER BY deg DESC, p.id ASC LIMIT $n",
        {"n": count * 4},
    )
    if not people:
        raise ParamError("no :Person with a :KNOWS edge — is the graph loaded?")

    names = _column(
        client,
        "MATCH (p:Person) WITH p.firstName AS n, count(*) AS c "
        "RETURN n ORDER BY c DESC, n ASC LIMIT $n",
        {"n": count},
    )
    countries = _column(
        client,
        "MATCH (c:Country)<-[:IS_PART_OF]-()<-[:IS_LOCATED_IN]-(p:Person) "
        "WITH c.name AS n, count(p) AS c2 RETURN n ORDER BY c2 DESC, n ASC LIMIT $n",
        {"n": max(count, 2)},
    )
    tags = _column(
        client,
        "MATCH (t:Tag)<-[:HAS_TAG]-() WITH t.name AS n, count(*) AS c "
        "RETURN n ORDER BY c DESC, n ASC LIMIT $n",
        {"n": count},
    )
    tag_classes = _column(
        client,
        "MATCH (tc:TagClass)<-[:HAS_TYPE]-(t:Tag)<-[:HAS_TAG]-() "
        "WITH tc.name AS n, count(*) AS c RETURN n ORDER BY c DESC, n ASC LIMIT $n",
        {"n": count},
    )
    lo, hi = _date_range(client)

    missing = [
        label
        for label, values in (
            ("Person.firstName", names),
            ("Country.name", countries),
            ("Tag.name", tags),
            ("TagClass.name", tag_classes),
        )
        if not values
    ]
    if missing:
        raise ParamError(f"cannot sample parameters, nothing found for: {', '.join(missing)}")

    def person() -> int:
        return int(rng.choice(people))

    def window() -> tuple[int, int]:
        """A start date and a 30-day end date, drawn from the dense end of the corpus.

        Messages accumulate over the simulated period, so a window drawn
        uniformly across the whole span usually lands in the sparse early
        history and matches nothing. That is the same failure mode as an
        isolated person: it returns instantly and measures none of the work the
        query exists to measure. Draw from the last quarter, where the data is.
        """
        floor = hi - (hi - lo) // 4
        start = rng.randrange(floor, max(floor + 1, hi - 30 * _DAY_MS))
        return start, start + 30 * _DAY_MS

    def max_date() -> int:
        """A cut-off with history behind it.

        IC2 and IC9 select the most recent messages *before* maxDate, so a
        cut-off early in the corpus leaves nothing to find.
        """
        return rng.randrange(hi - (hi - lo) // 4, hi + 1)

    rows: dict[int, list[dict[str, Any]]] = {n: [] for n in query_mod.COMPLEX_READS}
    for _ in range(count):
        start, end = window()
        pair = rng.sample(people, 2) if len(people) >= 2 else [people[0], people[0]]
        two_countries = (
            rng.sample(countries, 2) if len(countries) >= 2 else [countries[0], countries[0]]
        )
        rows[1].append({"personId": person(), "firstName": rng.choice(names)})
        rows[2].append({"personId": person(), "maxDate": max_date()})
        rows[3].append(
            {
                "personId": person(),
                "countryXName": two_countries[0],
                "countryYName": two_countries[1],
                "startDate": start,
                "endDate": end,
            }
        )
        rows[4].append({"personId": person(), "startDate": start, "endDate": end})
        rows[5].append({"personId": person(), "minDate": start})
        rows[6].append({"personId": person(), "tagName": rng.choice(tags)})
        rows[7].append({"personId": person()})
        rows[8].append({"personId": person()})
        rows[9].append({"personId": person(), "maxDate": max_date()})
        rows[10].append({"personId": person(), "month": rng.randint(1, 12)})
        rows[11].append(
            {
                "personId": person(),
                "countryName": rng.choice(countries),
                "workFromYear": rng.randint(2000, 2013),
            }
        )
        rows[12].append({"personId": person(), "tagClassName": rng.choice(tag_classes)})
        rows[13].append({"person1Id": int(pair[0]), "person2Id": int(pair[1])})
        rows[14].append({"person1Id": int(pair[0]), "person2Id": int(pair[1])})

    for n, param_rows in rows.items():
        for row in param_rows:
            query_mod.validate(n, row)

    echo(f"sampled {count} parameter rows per query (seed {seed})")
    return ParamSet(rows=rows, official=False, source=f"seed={seed}, n={count}")


def _column(client: BenchClient, cypher: str, params: dict[str, Any]) -> list[Any]:
    return [row[0] for row in client.graph.ro_query(cypher, params).result_set]


def _date_range(client: BenchClient) -> tuple[int, int]:
    """(min, max) message creationDate, the corpus' time span."""
    res = client.graph.ro_query("MATCH (m:Message) RETURN min(m.creationDate), max(m.creationDate)")
    lo, hi = res.result_set[0]
    if lo is None or hi is None:
        raise ParamError("no :Message creationDate — is the graph loaded?")
    return int(lo), int(hi)
