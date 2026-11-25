"""Substitution parameters for the complex reads, sampled from the loaded graph.

Seeded and deterministic, so two runs compare against each other; not
comparable to published LDBC results. bench/README.md says how the sampler
picks people and dates.
"""

from __future__ import annotations

import random
from collections.abc import Callable
from dataclasses import dataclass
from typing import Any

from falkorbench.client import BenchClient
from falkorbench.ldbc import queries as query_mod

Echo = Callable[[str], None]

#: A day in milliseconds; the sampled windows are whole days.
_DAY_MS = 86_400_000


class ParamError(RuntimeError):
    """Parameters could not be sourced."""


@dataclass
class ParamSet:
    """Parameter rows for each query number, and where they came from."""

    rows: dict[int, list[dict[str, Any]]]
    source: str

    def for_query(self, number: int) -> list[dict[str, Any]]:
        return self.rows.get(number, [])


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

    def wide_window() -> tuple[int, int]:
        """IC3's window, which has to be far wider than the other queries'.

        IC3 counts friends who are *not* resident in either country yet posted
        from both of them inside the window. That conjunction is rare: at SF0.1,
        anchored on the most-connected person and the two densest countries, a
        30-day window returns 0 rows, 90 days returns 0, 365 returns 1 and 730
        returns 2 — measured on the C engine, which agrees with Rust here, so
        this is a property of the dataset rather than of either engine.
        `durationDays` is an explicit LDBC parameter, so widening it stays
        within the query's own contract; LDBC's curated parameter sets pick
        values that guarantee a non-empty result, and sampling cannot.
        """
        start = lo + (hi - lo) // 4
        return start, hi

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
        ic3_start, ic3_end = wide_window()
        rows[3].append(
            {
                "personId": person(),
                "countryXName": two_countries[0],
                "countryYName": two_countries[1],
                "startDate": ic3_start,
                "endDate": ic3_end,
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
    return ParamSet(rows=rows, source=f"seed={seed}, n={count}")


def _column(client: BenchClient, cypher: str, params: dict[str, Any]) -> list[Any]:
    return [row[0] for row in client.graph.ro_query(cypher, params).result_set]


def _date_range(client: BenchClient) -> tuple[int, int]:
    """(min, max) message creationDate, the corpus' time span."""
    res = client.graph.ro_query("MATCH (m:Message) RETURN min(m.creationDate), max(m.creationDate)")
    lo, hi = res.result_set[0]
    if lo is None or hi is None:
        raise ParamError("no :Message creationDate — is the graph loaded?")
    return int(lo), int(hi)
