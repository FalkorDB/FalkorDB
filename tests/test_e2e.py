from decimal import Decimal
import os
import subprocess
import sys
from time import sleep
from typing import Counter
import common
from falkordb import Node, Edge, Path
from hypothesis import given, settings, strategies as st
import itertools
import math
import pytest
from redis import ResponseError

text_st = st.text().filter(lambda s: all(0x00 < ord(c) < 0x80 for c in s))
at_least_1_text_st = st.text("abcdefghijklmnopqrstuvwxyz", min_size=1)
is_extra = False


def setup_module(module):
    global is_extra
    from conftest import pytest_config

    is_extra = "extra" in pytest_config.getoption("-m")
    common.start_redis(moduleEnvs=["IMPORT_FOLDER", "data/"])


def teardown_module(module):
    common.shutdown_redis()


def setup_function(function):
    if common.g.name in common.client.list_graphs():
        common.g.delete()


def memory_usage():
    common.g.execute_command("MEMORY PURGE")

    memory_info = common.g.execute_command("INFO", "memory")
    _used_memory = memory_info["used_memory"]

    return _used_memory


def query(
    query: str,
    params=None,
    write: bool = False,
    compare_results: bool = True,
    steps=None,
):
    global is_extra
    if not is_extra:
        try:
            if not write:
                record_query = common.g._build_params_header(params) + query
                res = common.g.execute_command(
                    "GRAPH.RECORD", common.g.name, record_query
                )
                if steps is not None:
                    assert len(res[0]) == steps
            assert True
        except:
            assert False
    if write:
        try:
            common.g.query("RETURN 1")
            read_res = common.g.ro_query(query, params)
            assert False
        except ResponseError as e:
            assert "graph.RO_QUERY is to be executed only on read-only queries" == str(
                e
            )
        return common.g.query(query, params)
    else:
        write_res = common.g.query(query, params)
        read_res = common.g.ro_query(query, params)
        if compare_results:
            assert len(write_res.result_set) == len(read_res.result_set)
            for i in range(len(write_res.result_set)):
                assert len(write_res.result_set[i]) == len(read_res.result_set[i])
                for j in range(len(write_res.result_set[i])):
                    assert write_res.result_set[i][j] == read_res.result_set[i][j] or (
                        math.isnan(write_res.result_set[i][j])
                        and math.isnan(read_res.result_set[i][j])
                    )
        return write_res


def query_exception(query: str, message: str, params=None):
    try:
        common.g.query(query, params)
        assert False, "Expected an error"
    except ResponseError as e:
        assert message in str(e)


def assert_result_set_equal_no_order(res, expected):
    assert len(res.result_set) == len(expected)
    for record in expected:
        assert record in res.result_set


def assert_nrm_rows(result_set, expected_vs):
    """Assert n-r-m rows have correct labels, relation, and property values (ignoring IDs)."""
    vs = sorted([row[0].properties["v"] for row in result_set])
    assert vs == expected_vs
    for row in result_set:
        n, r, m = row
        v = n.properties["v"]
        assert n.labels == ["N"]
        assert n.properties == {"v": v}
        assert r.relation == "R"
        assert r.properties == {"v": v}
        assert m.labels == ["M"]
        assert m.properties == {"v": v}


def assert_float_equal(f1, f2):
    assert abs(f1 - f2) < 1e-10, f"Expected {f1} to be close to {f2}"


def test_return_values():
    res = query("RETURN null")
    assert res.result_set == [[None]]

    for b in [True, False]:
        res = query(f"RETURN {b}")
        assert res.result_set == [[1 if b else 0]]

    for i in range(0, 100):
        for sign in ["", "-", "- ", "+", "+ "]:
            res = query(f"RETURN {sign}{i}")
            assert res.result_set == [[eval(f"{sign}{i}")]]

            res = query(f"RETURN {sign}{i / 10.0}")
            assert res.result_set == [[eval(f"{sign}{i / 10.0}")]]

            # test number in hex format 0x...
            n = hex(i)
            res = query(f"RETURN {sign}{n}")
            assert res.result_set == [[eval(f"{sign}{n}")]]

            # Test engineering notation
            eng_notation = f"{sign}{i / 10.0:e}"
            res = query(f"RETURN {eng_notation} AS literal")
            assert res.result_set == [[eval(f"{eng_notation}")]]

    # Test specific cases
    res = query("RETURN .5 AS literal")
    assert res.result_set == [[0.5]]

    res = query("RETURN -.5 AS literal")
    assert res.result_set == [[-0.5]]

    res = query("RETURN 1e-3 AS literal")
    assert res.result_set == [[0.001]]

    res = query("RETURN -1e3 AS literal")
    assert res.result_set == [[-1000.0]]

    res = query("RETURN 'Avi'")
    assert res.result_set == [["Avi"]]

    res = query("RETURN []")
    assert res.result_set == [[[]]]

    res = query("RETURN ['Avi', [1, 2]]")
    assert res.result_set == [[["Avi", [1, 2]]]]

    res = query("RETURN {}")
    assert res.result_set == [[{}]]

    res = query("RETURN {a: 'Avi', b: 42}")
    assert res.result_set == [[{"a": "Avi", "b": 42}]]

    res = query("WITH 1 AS a, 'Avi' AS b RETURN b, a")
    assert res.result_set == [["Avi", 1]]

    query_exception("RETURN 0/0 AS a", "Division by zero")

    query_exception("RETURN 1/0 AS a", "Division by zero")

    res = query("RETURN 0.0/0.0 AS a")
    assert math.isnan(res.result_set[0][0])

    res = query("RETURN 0.1/0.0 AS a")
    assert res.result_set == [[float("inf")]]

    res = query("RETURN 0.0/0 AS a")
    assert math.isnan(res.result_set[0][0])

    res = query("RETURN 0.1/0 AS a")
    assert res.result_set == [[float("inf")]]

    res = query("RETURN 0/0.0 AS a")
    assert math.isnan(res.result_set[0][0])

    res = query("RETURN 1/0.0 AS a")
    assert res.result_set == [[float("inf")]]


@pytest.mark.extra
def test_numerical_bases():
    for i in range(0, 100):
        for sign in ["", "-", "- ", "+", "+ "]:
            n = oct(i)
            res = query(f"RETURN {sign}{n}")
            assert res.result_set == [[eval(f"{sign}{n}")]]

            n = bin(i)
            res = query(f"RETURN {sign}{n}")
            assert res.result_set == [[eval(f"{sign}{n}")]]


def test_parameters():
    for value in [None, True, False, 1, -1, 0.1, "Avi", [1], {"a": 2}, {}]:
        res = query("RETURN $p", params={"p": value})
        assert res.result_set == [[value]]


class CustomNumber:
    def __init__(self, value):
        self.value = value

    def __add__(self, other):
        return CustomNumber(self.value + other.value)

    def __sub__(self, other):
        return CustomNumber(self.value - other.value)

    def __mul__(self, other):
        return CustomNumber(self.value * other.value)

    def __truediv__(self, other):
        if isinstance(self.value, int) and isinstance(other.value, int):
            return CustomNumber(self.value // other.value)
        return CustomNumber(self.value / other.value)

    def __mod__(self, other):
        return CustomNumber(self.value % other.value)


def test_operators():
    for op in ["and", "or"]:
        for a in [True, False]:
            for b in [True, False]:
                res = query(f"RETURN {a} {op} {b}")
                assert res.result_set == [[1 if eval(f"{a} {op} {b}") else 0]]

    for op1 in ["and", "or"]:
        for op2 in ["and", "or"]:
            for a in [True, False]:
                for b in [True, False]:
                    for c in [True, False]:
                        res = query(f"RETURN {a} {op1} {b} {op2} {c}")
                        assert res.result_set == [
                            [1 if eval(f"{a} {op1} {b} {op2} {c}") else 0]
                        ]

    for a in [True, False]:
        for b in [True, False]:
            res = query(f"RETURN {a} = {b}")
            assert res.result_set == [[1 if a == b else 0]]

    for a in range(-10, 10):
        for b in range(-10, 10):
            res = query(f"RETURN {a} = {b}")
            assert res.result_set == [[1 if a == b else 0]]

    for a in range(-10, 10):
        for b in range(-10, 10):
            res = query(f"RETURN {a} + {b}")
            assert res.result_set == [[a + b]]

            res = query(f"RETURN {a} * {b}")
            assert res.result_set == [[a * b]]

            if a != 0:
                res = query(f"RETURN {a} ^ {b}")
                assert res.result_set == [[float("{:.15g}".format(pow(a, b)))]]

            if a >= 0 and b > 0:
                res = query(f"RETURN {a} % {b}")
                assert res.result_set == [[a % b]]

            res = query(f"RETURN {a} + {b} * ({a} + {b})")
            assert res.result_set == [[a + b * (a + b)]]

    for op1 in ["+", "-", "*", "/", "%"]:
        for op2 in ["+", "-", "*", "/", "%"]:
            for op3 in ["+", "-", "*", "/", "%"]:
                for op4 in ["+", "-", "*", "/", "%"]:
                    for n1 in [2, 2.0]:
                        for n2 in [4, 4.0]:
                            for n3 in [8, 8.0]:
                                for n4 in [16, 16.0]:
                                    for n5 in [32, 32.0]:
                                        res = query(
                                            f"RETURN {n1} {op1} {n2} {op2} {n3} {op3} {n4} {op4} {n5}"
                                        )
                                        assert res.result_set == [
                                            [
                                                eval(
                                                    f"CustomNumber({n1}) {op1} CustomNumber({n2}) {op2} CustomNumber({n3}) {op3} CustomNumber({n4}) {op4} CustomNumber({n5})"
                                                ).value
                                            ]
                                        ]

    for i, a in enumerate([True, 1, "Avi", [1]]):
        res = query(f"RETURN {{a0: true, a1: 1, a2: 'Avi', a3: [1]}}.a{i}")
        assert res.result_set == [[a]]

        res = query(f"RETURN {{a: {{a0: true, a1: 1, a2: 'Avi', a3: [1]}}}}.a.a{i}")
        assert res.result_set == [[a]]

    for a in range(5):
        res = query(f"RETURN [][{a}]")
        assert res.result_set == [[None]]

        res = query(f"RETURN [0, 1, 2, 3, 4][{a}]")
        assert res.result_set == [[[0, 1, 2, 3, 4][a]]]

        res = query(f"RETURN [[0, 1, 2, 3, 4]][0][{a}]")
        assert res.result_set == [[[0, 1, 2, 3, 4][a]]]

    res = query(f"UNWIND [NULL, true, false, 1, 'Avi', [], {{}}] AS x RETURN x IS NULL")
    assert res.result_set == [
        [True],
        [False],
        [False],
        [False],
        [False],
        [False],
        [False],
    ]


@given(st.integers(-100, 100), st.integers(-100, 100))
def test_unwind(f, t):
    res = query(f"UNWIND range({f}, {t}) AS x RETURN x")
    assert res.result_set == [[i] for i in range(f, t + 1)]

    res = query(f"UNWIND {list(range(f, t + 1))} AS x RETURN x")
    assert res.result_set == [[i] for i in range(f, t + 1)]


@given(st.integers(-100, 100), st.integers(-100, 100), st.integers(-100, 100))
def test_unwind_range_step(f, t, s):
    if s == 0:
        query_exception(
            f"UNWIND range({f}, {t}, {s}) AS x RETURN x",
            "ArgumentError: step argument to range() can't be 0",
        )
        return
    res = query(f"UNWIND range({f}, {t}, {s}) AS x RETURN x")
    if s > 0:
        if f == t:
            assert res.result_set == [[f]]
        else:
            assert res.result_set == [[i] for i in range(f, t + 1, s)]
    else:
        assert res.result_set == [[i] for i in range(f, t - 1, s)]


@given(
    st.integers(-10, 10),
    st.integers(-10, 10),
    st.integers(-10, 10),
    st.integers(-10, 10),
)
def test_nested_unwind_range(f1, t1, f2, t2):
    res = query(
        f"UNWIND range({f1}, {t1}) AS x UNWIND range({f2}, {t2}) AS y RETURN x, y"
    )
    assert res.result_set == [
        [i, j] for i in range(f1, t1 + 1) for j in range(f2, t2 + 1)
    ]


def test_graph_crud():
    res = query("CREATE ()", write=True)
    assert res.result_set == []
    assert res.nodes_created == 1

    res = query("MATCH (n) RETURN n")
    assert res.result_set == [[Node(0)]]

    res = query("MATCH p=() RETURN p")
    assert res.result_set == [[Path([Node(0)], [])]]

    res = query("MATCH (n) DELETE n", write=True)
    assert res.nodes_deleted == 1

    res = query("MATCH (n) RETURN n")
    assert res.result_set == []

    res = query("UNWIND range(1, 3) AS x CREATE (n:N) RETURN n", write=True)
    assert len(res.result_set) == 3
    assert all(row[0].labels == ["N"] for row in res.result_set)
    assert res.nodes_created == 3

    res = query("MATCH (n:N), (m:N) RETURN n, m")
    assert len(res.result_set) == 9
    assert all(row[0].labels == ["N"] and row[1].labels == ["N"] for row in res.result_set)

    common.g.delete()

    res = query(
        "UNWIND range(0, 2) AS x CREATE (n:N {v: x})-[r:R {v: x}]->(m:M {v: x}) RETURN n, r, m",
        write=True,
    )
    assert res.nodes_created == 6
    assert res.relationships_created == 3
    assert len(res.result_set) == 3
    assert_nrm_rows(res.result_set, [0, 1, 2])

    res = query("MATCH (n)-[r:R]->(m) RETURN n, r, m")
    assert len(res.result_set) == 3
    assert_nrm_rows(res.result_set, [0, 1, 2])

    res = query("MATCH (m)<-[r:R]-(n) RETURN n, r, m")
    assert len(res.result_set) == 3
    assert_nrm_rows(res.result_set, [0, 1, 2])

    res = query("MATCH p=()-[:R]->() RETURN p")
    assert len(res.result_set) == 3
    vs = sorted([row[0].nodes()[0].properties["v"] for row in res.result_set])
    assert vs == [0, 1, 2]
    for row in res.result_set:
        p = row[0]
        assert len(p.nodes()) == 2
        assert len(p.edges()) == 1
        v = p.nodes()[0].properties["v"]
        assert p.nodes()[0].labels == ["N"]
        assert p.nodes()[0].properties == {"v": v}
        assert p.nodes()[1].labels == ["M"]
        assert p.nodes()[1].properties == {"v": v}
        assert p.edges()[0].relation == "R"
        assert p.edges()[0].properties == {"v": v}

    res = query("MATCH (n:N) RETURN n.v")
    assert_result_set_equal_no_order(res, [[0], [1], [2]])

    res = query("MATCH (n:N) DELETE n", write=True)
    assert res.nodes_deleted == 3
    assert res.relationships_deleted == 3

def test_match_node_by_id():
    query("UNWIND range(0, 1000) AS x CREATE (n:N {v: x})", write=True)

    res1 = query("MATCH (n) WHERE id(n) = 1000 RETURN n.v")
    res2 = query("MATCH (n) WHERE n.v = 1000 RETURN n.v")

    assert res1.result_set == res2.result_set == [[1000]]
    assert res1.run_time_ms < res2.run_time_ms

def test_node_labels():
    res = query("CREATE ()", write=True)
    assert res.result_set == []
    assert res.nodes_created == 1

    res = query("MATCH (n) RETURN labels(n)")
    assert res.result_set == [[[]]]

    res = query("MATCH (n) DELETE n", write=True)
    assert res.nodes_deleted == 1

    res = query("CREATE (:N:M)", write=True)
    assert res.result_set == []
    assert res.nodes_created == 1

    res = query("MATCH (n) RETURN labels(n)")
    assert res.result_set == [[["N", "M"]]]


def test_large_graph():
    query(
        "UNWIND range(0, 100000) AS x CREATE (n:N {v: x})-[r:R {v: x}]->(m:M {v: x})",
        write=True,
    )


def test_toInteger():
    for v in [None, ""]:
        res = query("RETURN toInteger($p)", params={"p": v})
        assert res.result_set == [[None]]

    for v in [True, False]:
        res = query("RETURN toInteger($p)", params={"p": v})
        assert res.result_set == [[int(float(v))]]


@given(st.integers(-100, 100)) # | st.decimals(-100, 100, places=13))
def test_prop_toInteger(x):
    x = float(x) if isinstance(x, Decimal) else x
    res = query(f"RETURN toInteger({x}), toInteger('{x}')")
    if isinstance(x, float):
        assert res.result_set == [[int(math.floor(x)), int(math.floor(x))]]
    else:
        assert res.result_set == [[int(x), int(x)]]


def test_list_range():
    res = query("RETURN [1, 2, 3][null..1] AS r")
    assert res.result_set == [[None]]
    res = query("RETURN [1, 2, 3][1..null] AS r")
    assert res.result_set == [[None]]
    res = query("RETURN [1, 2, 3][..] AS r")
    assert res.result_set == [[[1, 2, 3]]]


@given(st.integers(-10, 10), st.integers(-10, 10))
def test_prop_list_range(a, b):
    res = query(f"RETURN [1, 2, 3, 4, 5][{a}..{b}] AS r")
    assert res.result_set == [[[1, 2, 3, 4, 5][a:b]]]
    res = query("RETURN [1, 2, 3, 4, 5][$from..$to] AS r", params={"from": a, "to": b})
    assert res.result_set == [[[1, 2, 3, 4, 5][a:b]]]

    res = query(f"RETURN [1, 2, 3, 4, 5][{a}..] AS r")
    assert res.result_set == [[[1, 2, 3, 4, 5][a:]]]
    res = query(f"RETURN [1, 2, 3, 4, 5][..{a}] AS r")
    assert res.result_set == [[[1, 2, 3, 4, 5][:a]]]


@given(
    st.lists(st.booleans() | st.integers(-10, 10) | text_st),
    st.lists(st.booleans() | st.integers(-10, 10) | text_st),
)
def test_list_concat(a, b):
    res = query(f"RETURN $a + $b", params={"a": a, "b": b})
    assert res.result_set == [[a + b]]


@given(
    st.lists(st.booleans() | st.integers(-10, 10) | text_st),
    st.booleans() | st.integers(-10, 10) | text_st,
)
def test_list_append(a, b):
    res = query(f"RETURN $a + $b", params={"a": a, "b": b})
    assert res.result_set == [[a + [b]]]


def test_in_list():
    # test for simple values
    for value in [True, False, [1]]:
        res = query("RETURN $p IN [$p]", params={"p": value})
        assert res.result_set == [[True]]


@given(st.lists(st.integers(-10, 10) | text_st), st.integers(-10, 10) | text_st)
def test_prop_in_list(a, b):
    res = query("RETURN $b IN $a", params={"a": a, "b": b})
    assert res.result_set == [[b in a]]


@given(
    st.none()
    | st.booleans()
    | st.integers(-10, 10)
    | text_st
    | st.lists(st.none() | st.booleans() | st.integers(-10, 10) | text_st)
    | st.dictionaries(
        at_least_1_text_st, st.none() | st.booleans() | st.integers(-10, 10) | text_st
    )
)
def test_equal_null(a):
    res = query("RETURN $a = null", params={"a": a})
    assert res.result_set == [[None]]

    res = query("RETURN null = $a", params={"a": a})
    assert res.result_set == [[None]]


@given(
    st.booleans()
    | st.integers(-10, 10)
    | text_st
    | st.lists(st.booleans() | st.integers(-10, 10) | text_st)
    | st.dictionaries(
        at_least_1_text_st, st.booleans() | st.integers(-10, 10) | text_st
    )
)
def test_prop_equal(a):
    res = query("RETURN $a = $a", params={"a": a})
    assert res.result_set == [[True]]


@pytest.mark.extra
@given(
    st.booleans()
    | st.integers(-10, 10)
    | text_st
    | st.lists(st.booleans() | st.integers(-10, 10) | text_st)
    | st.dictionaries(
        at_least_1_text_st, st.booleans() | st.integers(-10, 10) | text_st
    )
)
def test_prop_equal_extra(a):
    res = query("RETURN $a = $a = $a AS res", params={"a": a})
    assert res.result_set == [[True]]

    res = query("RETURN $a = $a = $a = $b AS res", params={"a": a, "b": "foo"})
    assert res.result_set == [[False]]


@given(
    st.booleans()
    | st.integers(-10, 10)
    | text_st
    | st.lists(st.booleans() | st.integers(-10, 10) | text_st)
    | st.dictionaries(
        at_least_1_text_st, st.booleans() | st.integers(-10, 10) | text_st
    ),
    st.booleans()
    | st.integers(-10, 10)
    | text_st
    | st.lists(st.booleans() | st.integers(-10, 10) | text_st)
    | st.dictionaries(
        at_least_1_text_st, st.booleans() | st.integers(-10, 10) | text_st
    ),
)
def test_prop_equal2(a, b):
    res = query("RETURN $a = $b", params={"a": a, "b": b})
    if isinstance(a, list) and isinstance(b, list):
        assert res.result_set == [
            [a == b and all(type(x) == type(y) for x, y in zip(a, b))]
        ]
    elif isinstance(a, dict) and isinstance(b, dict):
        assert res.result_set == [
            [a == b and all(type(a[k]) == type(b[k]) for k in a.keys())]
        ]
    else:
        assert res.result_set == [[a == b and type(a) == type(b)]]


def test_is_equal():
    res = query("RETURN $a = $a AS res", params={"a": None})
    assert res.result_set == [[None]]
    res = query("RETURN [null] = [null] AS res")
    assert res.result_set == [[None]]

    res = query("RETURN [1] = [1, null] AS res")
    assert res.result_set == [[False]]

    res = query("RETURN [1, 2] = [null, 'foo'] AS res")
    assert res.result_set == [[False]]

    res = query("RETURN [1, 2] = [null, 2] AS res")
    assert res.result_set == [[None]]

    res = query("RETURN [[1]] = [[1], [null]] AS res")
    assert res.result_set == [[False]]

    res = query("RETURN [[1, 2], [1, 3]] = [[1, 2], [null, 'foo']] AS res")
    assert res.result_set == [[False]]

    res = query("RETURN [[1, 2], ['foo', 'bar']] = [[1, 2], [null, 'bar']] AS res")
    assert res.result_set == [[None]]


@given(
    st.none()
    | text_st
    | st.lists(
        st.none()
        | st.booleans()
        | st.integers(-10, 10)
        | text_st
        | st.lists(st.none() | st.booleans() | st.integers(-10, 10) | text_st)
    )
)
def test_list_size(a):
    res = query("RETURN size($a)", params={"a": a})
    assert res.result_set == [[len(a) if a is not None else None]]


@given(
    st.none()
    | st.lists(
        st.none()
        | st.booleans()
        | st.integers(-10, 10)
        | text_st
        | st.lists(st.none() | st.booleans() | st.integers(-10, 10) | text_st)
    )
)
def test_list_head(a):
    res = query("RETURN head($a)", params={"a": a})
    assert res.result_set == [[a[0] if a else None]]


@given(
    st.none()
    | st.lists(
        st.none()
        | st.booleans()
        | st.integers(-10, 10)
        | text_st
        | st.lists(st.none() | st.booleans() | st.integers(-10, 10) | text_st)
    )
)
def test_list_last(a):
    res = query("RETURN last($a)", params={"a": a})
    assert res.result_set == [[a[-1] if a else None]]


@given(
    st.none()
    | st.lists(
        st.none()
        | st.booleans()
        | st.integers(-10, 10)
        | text_st
        | st.lists(st.none() | st.booleans() | st.integers(-10, 10) | text_st)
    )
)
def test_list_tail(a):
    res = query("RETURN tail($a)", params={"a": a})
    if a is None:
        assert res.result_set == [[None]]
    elif len(a) == 0:
        assert res.result_set == [[[]]]
    else:
        assert res.result_set == [[a[1:]]]


@given(
    st.none()
    | st.lists(
        st.none()
        | st.booleans()
        | st.integers(-10, 10)
        | text_st
        | st.lists(st.none() | st.booleans() | st.integers(-10, 10) | text_st)
    )
)
def test_list_reverse(a):
    res = query("RETURN reverse($a)", params={"a": a})
    assert res.result_set == [[a[::-1] if a is not None else None]]


def cypher_xor(a, b, c):
    """
    This function simulates the XOR operation for three boolean values.
    It returns True if an odd number of inputs are True, otherwise it returns False.
    """
    if a == "null" or b == "null" or c == "null":
        return None
    else:
        return a ^ b ^ c


def test_xor():
    # Define the possible values
    values = [True, False, "null"]

    # Generate all possible triples
    triples = list(itertools.product(values, repeat=3))

    for a, b, c in triples:
        res = query(f"RETURN {a} XOR {b} XOR {c} AS r")
        expected = cypher_xor(a, b, c)
        assert res.result_set == [[expected]]


def test_literals():
    for i in range(-100, 101):
        hex_representation = hex(i)
        res = query(f"RETURN {hex_representation} AS literal")
        assert res.result_set == [[i]]

        # octal representation with leading zero, old format
        res = query("RETURN 02613152366 AS literal")
        assert res.result_set == [[372036854]]

        res = query("RETURN .2 AS literal")
        assert res.result_set == [[0.2]]

        res = query("RETURN -.2 AS literal")
        assert res.result_set == [[-0.2]]


@given(st.none() | text_st, st.none() | text_st)
def test_split(a, b):
    res = query("RETURN split($a, $b)", params={"a": a, "b": b})
    if a is None or b is None:
        assert res.result_set == [[None]]
    elif b == "":
        if a == "":
            assert res.result_set == [[[""]]]
        else:
            assert res.result_set == [[list(a)]]
    else:
        if a == "":
            assert res.result_set == [[[""]]]
        else:
            assert res.result_set == [[a.split(b) if a else []]]


@given(st.none() | text_st)
def test_letter_casing(a):
    res = query("RETURN toUpper($a)", params={"a": a})
    assert res.result_set == [[a.upper() if a is not None else None]]


def test_add():
    res = query("RETURN null + 1 AS name")
    assert res.result_set == [[None]]

    res = query("RETURN 1 + null AS name")
    assert res.result_set == [[None]]

    res = query("RETURN 9223372036854775807 + 2 AS name")
    assert res.result_set == [[-9223372036854775807]]

    res = query("RETURN 1 + 1 AS name")
    assert res.result_set == [[2]]

    res = query("RETURN 1.0 + 1.0 AS name")
    assert res.result_set == [[2.0]]

    res = query("RETURN 1.1 + 1 AS name")
    assert res.result_set == [[2.1]]

    res = query("RETURN 1 + 1.1 AS name")
    assert res.result_set == [[2.1]]

    res = query("RETURN [1] + [1] AS name")
    assert res.result_set == [[[1, 1]]]

    res = query("RETURN [1] + 1 AS name")
    assert res.result_set == [[[1, 1]]]

    res = query("RETURN [] + 1 AS name")
    assert res.result_set == [[[1]]]

    res = query("RETURN 'a' + [1, 2 ,3] AS name")
    assert res.result_set == [[["a", 1, 2, 3]]]

    res = query("RETURN 'a' + 'b' + 'c' AS name")
    assert res.result_set == [["abc"]]

    res = query("RETURN 'a' + 'b' + 1 AS name")
    assert res.result_set == [["ab1"]]

    res = query("RETURN 'a' + 'b' + 0.100000 AS name")
    assert res.result_set == [["ab0.100000"]] or res.result_set == [["ab0.1"]]

    res = query("RETURN 'a' + True AS name")
    assert res.result_set == [["atrue"]]

    query_exception("RETURN {} + 1 AS name", "")


@given(st.none() | text_st, st.none() | text_st)
def test_starts_with(a, b):
    res = query("RETURN $a STARTS WITH $b", params={"a": a, "b": b})
    assert res.result_set == [
        [a.startswith(b) if a is not None and b is not None else None]
    ]


@given(st.none() | text_st, st.none() | text_st)
def test_ends_with(a, b):
    res = query("RETURN $a ENDS WITH $b", params={"a": a, "b": b})
    assert res.result_set == [
        [a.endswith(b) if a is not None and b is not None else None]
    ]


@given(st.none() | text_st, st.none() | text_st)
def test_contains(a, b):
    res = query("RETURN $a CONTAINS $b", params={"a": a, "b": b})
    assert res.result_set == [[b in a if a is not None and b is not None else None]]


@given(st.none() | text_st, st.none() | text_st, st.none() | text_st)
def test_replace(a, b, c):
    res = query("RETURN replace($a, $b, $c)", params={"a": a, "b": b, "c": c})
    assert res.result_set == [
        [a.replace(b, c) if a is not None and b is not None and c is not None else None]
    ]


@pytest.mark.extra
def test_regex_matches():
    res = query("RETURN 'abc' =~ 'a.*' AS result")
    assert res.result_set == [[True]]

    res = query("RETURN 'abc' =~ 'd.*' AS result")
    assert res.result_set == [[False]]

    res = query("RETURN 'abc' =~ 'a.*c' AS result")
    assert res.result_set == [[True]]

    res = query("RETURN 'abc' =~ 'a.*d' AS result")
    assert res.result_set == [[False]]

    res = query("RETURN 'abc' =~ '^a.*c$' AS result")
    assert res.result_set == [[True]]

    res = query("RETURN 'abc' =~ '^d.*c$' AS result")
    assert res.result_set == [[False]]

    # Null handling
    res = query("RETURN null =~ 'a.*' AS result")
    assert res.result_set == [[None]]

    res = query("RETURN 'abc' =~ null AS result")
    assert res.result_set == [[None]]


@given(st.none() | text_st, st.none() | st.integers(-10, 10))
def test_left(a, b):
    if a is None:
        res = query("RETURN left($a, $b)", params={"a": a, "b": b})
        assert res.result_set == [[None]]
    elif b is None or b < 0:
        query_exception(
            "RETURN left($a, $b)",
            "length must be a non-negative integer",
            params={"a": a, "b": b},
        )
    else:
        res = query("RETURN left($a, $b)", params={"a": a, "b": b})
        assert res.result_set == [[a[:b]]]


@given(st.none() | text_st)
def test_ltrim(a):
    res = query("RETURN ltrim($a)", params={"a": a})
    assert res.result_set == [[a.lstrip(" ") if a is not None else None]]


@given(st.none() | text_st, st.none() | st.integers(-10, 10))
def test_right(a, b):
    if a is None:
        res = query("RETURN right($a, $b)", params={"a": a, "b": b})
        assert res.result_set == [[None]]
    elif b is None or b < 0:
        query_exception(
            "RETURN right($a, $b)",
            "length must be a non-negative integer",
            params={"a": a, "b": b},
        )
    else:
        res = query("RETURN right($a, $b)", params={"a": a, "b": b})
        assert res.result_set == [[a[-b:] if b > 0 else ""]]


@given(st.none() | text_st, st.integers(-10, 10), st.none() | st.integers(-10, 10))
def test_substring(a, b, c):
    q = "RETURN substring($a, $b)" if c is None else "RETURN substring($a, $b, $c)"
    if a is None:
        res = query(q, params={"a": a, "b": b, "c": c})
        assert res.result_set == [[None]]
    elif b < 0:
        query_exception(
            q, "start must be a non-negative integer", params={"a": a, "b": b, "c": c}
        )
    elif b >= len(a):
        res = query(q, params={"a": a, "b": b, "c": c})
        assert res.result_set == [[a[b : (b + c if c is not None else None)]]]
    elif c is not None and c < 0:
        query_exception(
            q, "length must be a non-negative integer", params={"a": a, "b": b, "c": c}
        )
    else:
        res = query(q, params={"a": a, "b": b, "c": c})
        assert res.result_set == [[a[b : (b + c if c is not None else None)]]]


@settings(deadline=None)
@given(st.lists(at_least_1_text_st, unique=True))
def test_graph_list(a):
    for i in a:
        common.client.select_graph(i).query("RETURN 1")
        common.client.connection.set(f"ng:{i}", "ng")
    graphs = common.client.list_graphs()

    assert len(graphs) == len(a)
    for i in a:
        assert i in graphs
        common.client.select_graph(i).delete()


@given(st.lists(text_st), text_st)
def test_string_join(a, b):
    q = "RETURN string.join($a, $b)" if b else "RETURN string.join($a)"
    if a is None:
        res = query(q, params={"a": a, "b": b})
        assert res.result_set == [[None]]
    else:
        res = query(q, params={"a": a, "b": b})
        assert res.result_set == [[(b if b else "").join(a)]]


@pytest.mark.extra
def test_string_match_regex():
    res = query("RETURN string.matchRegEx(null, null) AS name")
    assert res.result_set == [[[]]]

    res = query("RETURN string.matchRegEx('foo bar', null) AS name")
    assert res.result_set == [[[]]]

    res = query("RETURN string.matchRegEx(null, '.*') AS name")
    assert res.result_set == [[[]]]

    res = query("RETURN string.matchRegEx('foo bar', '.*') AS name")
    assert res.result_set == [[[["foo bar"]]]]

    res = query("RETURN string.matchRegEx('foo bar', '[a-z]+\\s+[a-z]+') AS name")
    assert res.result_set == [[[["foo bar"]]]]

    ## multiple groups
    res = query("RETURN string.matchRegEx('foo bar', '([a-z]+)\\s+([a-z]+)') AS name")
    assert res.result_set == [[[["foo bar", "foo", "bar"]]]]

@pytest.mark.extra
def test_string_replace_regex():
    res = query(
        "RETURN string.replaceRegEx('foo-bar baz-qux', '(?<first>[a-z]+)-(?<last>[a-z]+)', '$first $last') AS name"
    )
    assert res.result_set == [["foo bar baz qux"]]

    res = query(
        "RETURN string.replaceRegEx('foo-bar baz-qux', '([a-z]+)-([a-z]+)', '$1 $2') AS name"
    )
    assert res.result_set == [["foo bar baz qux"]]

    res = query(
        "RETURN string.replaceRegEx('foo-bar baz-qux', '([a-z]+)-([a-z]+)', '${1}_${2}') AS name"
    )
    assert res.result_set == [["foo_bar baz_qux"]]

    res = query(
        "RETURN string.replaceRegEx('foo-bar baz-qux', '(\\w+)-(\\w+)', '${1}_${2}') AS name"
    )
    assert res.result_set == [["foo_bar baz_qux"]]

    res = query(
        "RETURN string.replaceRegEx('123', '(\\w+)-(\\w+)', '${1}_${2}') AS name"
    )
    assert res.result_set == [["123"]]

    ## broken regex
    query_exception(
        "RETURN string.replaceRegEx('foo bar', '**', 'a') AS name", "Invalid regex"
    )


@given(st.none() | st.integers(-100, 100) | st.decimals(-100, 100, places=13))
def test_abs(a):
    a = float(a) if isinstance(a, Decimal) else a
    res = query("RETURN abs($a)", params={"a": a})
    assert res.result_set == [[abs(a) if a is not None else None]]


@given(st.none() | st.integers(-100, 100) | st.decimals(-100, 100, places=13))
def test_ceil(a):
    a = float(a) if isinstance(a, Decimal) else a
    res = query("RETURN ceil($a)", params={"a": float(a) if a is not None else None})
    assert res.result_set == [[math.ceil(a) if a is not None else None]]


def test_e():
    res = query("RETURN e()")
    assert res.result_set == [[2.71828182845905e0]]


def test_exp():
    res = query("RETURN exp(1) AS name")
    assert res.result_set == [[2.71828182845905]]

    res = query("RETURN exp(0) AS name")
    assert res.result_set == [[1]]

    res = query("RETURN exp(-1) AS name")
    assert res.result_set == [[0.367879441171442]]

    res = query("RETURN exp(-1.0) AS name")
    assert res.result_set == [[0.367879441171442]]

    res = query("RETURN exp(null) AS name")
    assert res.result_set == [[None]]


@given(st.none() | st.integers(-100, 100) | st.floats(-100, 100, allow_subnormal=False))
def test_floor(a):
    res = query("RETURN floor($a)", params={"a": a})
    assert res.result_set == [[math.floor(a) if a is not None else None]]


def test_log():
    res = query("RETURN log(1) AS name")
    assert res.result_set == [[0]]

    res = query("RETURN log(1.0) AS name")
    assert res.result_set == [[0]]

    res = query("RETURN log(0) AS name")
    assert res.result_set == [[float("-inf")]]

    res = query("RETURN log(-1) AS name")
    assert math.isnan(res.result_set[0][0])

    res = query("RETURN log(null) AS name")
    assert res.result_set == [[None]]


def test_log10():
    res = query("RETURN log10(1) AS name")
    assert res.result_set == [[0]]

    res = query("RETURN log10(1.0) AS name")
    assert res.result_set == [[0]]

    res = query("RETURN log10(0) AS name")
    assert res.result_set == [[float("-inf")]]

    res = query("RETURN log10(-1) AS name")
    assert math.isnan(res.result_set[0][0])

    res = query("RETURN log10(null) AS name")
    assert res.result_set == [[None]]


def test_pow():
    res = query("RETURN pow(2, 3) AS name")
    assert res.result_set == [[8]]

    res = query("RETURN pow(2.0, 3) AS name")
    assert res.result_set == [[8.0]]

    res = query("RETURN pow(2.0, 3.0) AS name")
    assert res.result_set == [[8.0]]

    res = query("RETURN pow(2, 3.0) AS name")
    assert res.result_set == [[8.0]]

    res = query("RETURN pow(2, -3) AS name")
    assert res.result_set == [[0.125]]

    res = query("RETURN pow(2, 0) AS name")
    assert res.result_set == [[1]]

    res = query("RETURN pow(-2, 3) AS name")
    assert res.result_set == [[-8]]

    res = query("RETURN pow(-2, -3) AS name")
    assert res.result_set == [[-0.125]]

    res = query("RETURN pow(-2, 0) AS name")
    assert res.result_set == [[1]]

    res = query("RETURN pow(null, 3) AS name")
    assert res.result_set == [[None]]

    res = query("RETURN pow(3, null) AS name")
    assert res.result_set == [[None]]


def shannon_entropy(data):
    n = len(data)
    counts = Counter(data)
    probabilities = [count / n for count in counts.values()]

    return -sum(p * math.log2(p) for p in probabilities if p > 0)


def test_rand():
    data = []
    for _ in range(1000):
        res = query("RETURN rand()", compare_results=False)
        data.append(res.result_set[0][0])
        assert res.result_set[0][0] >= 0.0
        assert res.result_set[0][0] < 1.0
    assert shannon_entropy(data) > 0.9  # Check for randomness


def round_away_from_zero(num):
    if num > 0:
        return math.floor(num + 0.5)
    else:
        return math.ceil(num - 0.5)


@given(st.none() | st.integers(-100, 100) | st.floats(-100, 100, allow_subnormal=False))
def test_round(a):
    res = query("RETURN round($a)", params={"a": a})
    assert res.result_set == [[round_away_from_zero(a) if a is not None else None]]


def signum(x):
    return (x > 0) - (x < 0)


@given(st.none() | st.integers(-100, 100) | st.floats(-100, 100, allow_subnormal=False))
def test_sign(a):
    res = query("RETURN sign($a)", params={"a": a})
    assert res.result_set == [[signum(a) if a is not None else None]]


def test_sqrt():
    res = query("RETURN sqrt(4) AS result")
    assert res.result_set == [[2]]

    res = query("RETURN sqrt(4.0) AS result")
    assert res.result_set == [[2.0]]

    res = query("RETURN sqrt(0) AS result")
    assert res.result_set == [[0]]

    res = query("RETURN sqrt(-1) AS result")
    assert math.isnan(res.result_set[0][0])

    res = query("RETURN sqrt(-1.0) AS result")
    assert math.isnan(res.result_set[0][0])

    res = query("RETURN sqrt(null) AS result")
    assert res.result_set == [[None]]


@given(st.integers(-100, 100), st.integers(-100, 100))
def test_range(a, b):
    res = query("RETURN range($a, $b)", params={"a": a, "b": b})
    assert res.result_set == [[list(range(a, b + 1))]]


@given(st.integers(-10, 10), st.integers(-100, 10), st.integers(-10, 10))
def test_range_step(a, b, c):
    if c == 0:
        query_exception(
            "RETURN range($a, $b, $c)",
            "ArgumentError: step argument to range() can't be 0",
            params={"a": a, "b": b, "c": c},
        )
        return
    res = query("RETURN range($a, $b, $c)", params={"a": a, "b": b, "c": c})
    if c > 0:
        if a == b:
            assert res.result_set == [[[a]]]
        else:
            assert res.result_set == [[[i for i in range(a, b + 1, c)]]]
    else:
        assert res.result_set == [[[i for i in range(a, b - 1, c)]]]


@given(
    st.lists(
        st.none()
        | st.booleans()
        | st.integers(-10, 10)
        | text_st
        | st.lists(st.none() | st.booleans() | st.integers(-10, 10) | text_st)
    )
)
def test_collect(a):
    res = query("UNWIND $a AS x RETURN collect(x)", params={"a": a})
    assert_result_set_equal_no_order(res, [[[x for x in a if x is not None]]])

    res = query(
        "UNWIND $a AS x WITH collect(x) AS xs UNWIND xs AS y RETURN collect(y)",
        params={"a": a},
    )
    assert_result_set_equal_no_order(res, [[[x for x in a if x is not None]]])


@given(
    st.lists(
        st.none()
        | st.booleans()
        | st.integers(-10, 10)
        | text_st
        | st.lists(st.none() | st.booleans() | st.integers(-10, 10) | text_st)
    )
)
def test_count(a):
    res = query("UNWIND $a AS x RETURN count(x)", params={"a": a})
    assert_result_set_equal_no_order(res, [[len([x for x in a if x is not None])]])


@given(st.lists(st.none() | st.integers(-10, 10)))
def test_sum(a):
    res = query("UNWIND $a AS x RETURN sum(x)", params={"a": a})
    assert_result_set_equal_no_order(res, [[sum(x for x in a if x is not None)]])

    res = query("UNWIND $a AS x RETURN sum(distinct x)", params={"a": a})
    assert_result_set_equal_no_order(res, [[sum(set(x for x in a if x is not None))]])


@given(st.lists(st.none() | st.integers(-10, 10)))
def test_min(a):
    res = query("UNWIND $a AS x RETURN min(x)", params={"a": a})
    if not a or all(x is None for x in a):
        assert res.result_set == [[None]]
    else:
        assert res.result_set == [[min(x for x in a if x is not None)]]


@given(st.lists(st.none() | st.integers(-10, 10)))
def test_max(a):
    res = query("UNWIND $a AS x RETURN max(x)", params={"a": a})
    if not a or all(x is None for x in a):
        assert res.result_set == [[None]]
    else:
        assert res.result_set == [[max(x for x in a if x is not None)]]


@given(st.lists(st.none() | st.integers(-10, 10)))
def test_stdev(a):
    res = query("UNWIND $a AS x RETURN stdev(x)", params={"a": a})
    valid_values = [x for x in a if x is not None]
    if len(valid_values) < 2:
        assert res.result_set == [[0.0]]
    else:
        mean = sum(valid_values) / len(valid_values)
        variance = sum((x - mean) ** 2 for x in valid_values) / (len(valid_values) - 1)
        assert_float_equal(res.result_set[0][0], math.sqrt(variance))


@given(st.lists(st.none() | st.integers(-10, 10)))
def test_stdevp(a):
    res = query("UNWIND $a AS x RETURN stdevp(x)", params={"a": a})
    valid_values = [x for x in a if x is not None]
    if len(valid_values) < 2:
        assert res.result_set == [[0.0]]
    else:
        mean = sum(valid_values) / len(valid_values)
        variance = sum((x - mean) ** 2 for x in valid_values) / len(valid_values)
        assert_float_equal(res.result_set[0][0], math.sqrt(variance))


def test_aggregation():
    res = query("UNWIND range(1, 10) AS x RETURN sum(x / 10.0)")
    assert_result_set_equal_no_order(res, [[5.5]])

    res = query(
        "UNWIND range(1, 11) AS x RETURN x % 2, count(x)", compare_results=False
    )
    assert_result_set_equal_no_order(res, [[1, 6], [0, 5]])

    res = query("UNWIND range(1, 100) AS x RETURN min(x), max(x)")
    assert_result_set_equal_no_order(res, [[1, 100]])

    res = query("UNWIND range(1, 100) AS x RETURN {min: min(x), max: max(x)}")
    assert_result_set_equal_no_order(res, [[{"min": 1, "max": 100}]])

    res = query("UNWIND range(0,-1) AS a RETURN count(a), 1 + sum(a)")
    assert res.result_set == [[0, 1]]

    res = query(
        "UNWIND [1, 2, 3, 1, 2, 3] AS x RETURN x % 2 = 0, sum(x), sum(distinct x)",
        compare_results=False,
    )
    assert_result_set_equal_no_order(res, [[False, 8, 4], [True, 4, 2]])

    res = query(
        "UNWIND [1, 2, 3, 1, 2, 3] AS x RETURN sum(x), sum(distinct x)",
        compare_results=False,
    )
    assert_result_set_equal_no_order(res, [[12, 6]])

    res = query(
        "UNWIND [[1, 1], [1, 2], [1, 3], [1, 1], [2, 1], [2, 2], [2, 3], [2, 1]] AS x RETURN x[0], sum(x[1]), sum(distinct x[1])",
        compare_results=False,
    )
    assert_result_set_equal_no_order(res, [[1, 7.0, 6.0], [2, 7.0, 6.0]])


@given(st.lists(st.integers(-10, 10)))
def test_avg_integers(a):
    res = query("UNWIND $a AS x RETURN avg(x)", params={"a": a})
    if not a:
        assert res.result_set == [[None]]
    else:
        expected_avg = sum(a) / len(a)
        assert_float_equal(res.result_set[0][0], expected_avg)

    res = query("UNWIND $a AS x RETURN avg(distinct x)", params={"a": a})
    if not a:
        assert res.result_set == [[None]]
    else:
        a = list(set(a))
        expected_avg = sum(a) / len(a)
        assert_float_equal(res.result_set[0][0], expected_avg)


@given(
    st.lists(
        st.floats(
            min_value=-10.0,
            max_value=10.0,
            allow_nan=False,
            allow_infinity=False,
            allow_subnormal=False,
        )
    )
)
def test_avg_floats(a):
    res = query("UNWIND $a AS x RETURN avg(x)", params={"a": a})
    if not a:
        assert res.result_set == [[None]]
    else:
        expected_avg = sum(a) / len(a)
        assert_float_equal(res.result_set[0][0], expected_avg)

    res = query("UNWIND $a AS x RETURN avg(distinct x)", params={"a": a})
    if not a:
        assert res.result_set == [[None]]
    else:
        a = list(set(a))
        expected_avg = sum(a) / len(a)
        assert_float_equal(res.result_set[0][0], expected_avg)


@given(
    st.lists(
        st.none()
        | st.integers(-10, 10)
        | st.floats(
            min_value=-10.0,
            max_value=10.0,
            allow_nan=False,
            allow_infinity=False,
            allow_subnormal=False,
        )
    )
)
def test_avg_mixed(a):
    res = query("UNWIND $a AS x RETURN avg(x)", params={"a": a})
    if not any(True for x in a if x is not None):
        assert res.result_set == [[None]]
    else:
        valid_values = [x for x in a if x is not None]
        expected_avg = sum(valid_values) / len(valid_values)
        assert_float_equal(res.result_set[0][0], expected_avg)

    res = query("UNWIND $a AS x RETURN avg(distinct x)", params={"a": a})
    if not any(True for x in a if x is not None):
        assert res.result_set == [[None]]
    else:
        valid_values = list(set(x for x in a if x is not None))
        expected_avg = sum(valid_values) / len(valid_values)
        assert_float_equal(res.result_set[0][0], expected_avg)


def test_avg_overflow():
    # Test with very large values that might cause overflow
    large_values = [1e308, 1e308, 1e308]
    res = query("UNWIND $a AS x RETURN avg(x)", params={"a": large_values})
    assert res.result_set[0][0] == 1e308

    # Test with very large values with different signs
    mixed_large_values = [1e308, -1e308, 1e308]
    res = query("UNWIND $a AS x RETURN avg(x)", params={"a": mixed_large_values})
    expected = 1e308 / 3
    assert abs(res.result_set[0][0] - expected) < 1e-10 * expected

    # Test with values close to max float
    max_float = sys.float_info.max / 10  # Using a smaller value to avoid overflow
    near_max_values = [max_float, max_float / 2, max_float / 4]
    res = query("UNWIND $a AS x RETURN avg(x)", params={"a": near_max_values})
    expected = sum(near_max_values) / len(near_max_values)
    assert abs(res.result_set[0][0] - expected) < expected * 1e-10


def test_avg_nan():
    # Test with NaN values
    res = query("UNWIND [0.0/0.0, 1, 2] AS x RETURN avg(x)")
    assert math.isnan(res.result_set[0][0])


def test_avg_inf():
    # Test with positive infinity
    res = query("UNWIND [1, 2, 1.0/0] AS x RETURN avg(x)")
    assert res.result_set[0][0] == float("inf")

    # Test with negative infinity
    res = query("UNWIND [-1, -2, -1/0.0] AS x RETURN avg(x)")
    assert res.result_set[0][0] == float("-inf")

    # Test with mixed infinities
    res = query("UNWIND [1, 2, -1/0.0, 1/0.0] AS x RETURN avg(x)")
    assert math.isnan(res.result_set[0][0])


@pytest.mark.extra
@given(
    st.lists(
        st.none() | st.integers(-100, 100) | st.floats(-100, 100, allow_subnormal=False)
    ),
    st.integers(0, 1) | st.floats(min_value=0.0, max_value=1.0, allow_subnormal=False),
)
def test_percentile_disc(values, percentile):
    sorted_values = sorted([x for x in values if x is not None])

    index = math.ceil(percentile * len(sorted_values)) - 1
    index = max(0, min(index, len(sorted_values) - 1))  # Ensure index is in bounds
    expected = sorted_values[index] if len(sorted_values) > 0 else None

    res = query(
        "UNWIND $values AS x RETURN percentileDisc(x, $p) AS result",
        params={"values": values, "p": percentile},
    )
    if expected is None:
        assert res.result_set == [[None]]
    else:
        assert_float_equal(res.result_set[0][0], float(expected))


@pytest.mark.extra
def test_percentile_extra():
    # Test with NULL percentile
    q = "UNWIND [1, 2, 3, 4, 5] AS x RETURN percentileDisc(x, null) AS result"
    query_exception(q, "Type mismatch: expected Integer or Float but was Null")


def test_percentile_disc_edge_cases():
    # Test with percentile < 0
    q = "UNWIND [1, 2, 3, 4, 5] AS x RETURN percentileDisc(x, -0.5) AS result"
    query_exception(
        q, "is not a valid argument, must be a number in the range 0.0 to 1.0"
    )

    # Test with percentile > 1
    q = "UNWIND [1, 2, 3, 4, 5] AS x RETURN percentileDisc(x, 1.2) AS result"
    query_exception(
        q, "is not a valid argument, must be a number in the range 0.0 to 1.0"
    )


@pytest.mark.extra
@given(
    st.lists(
        st.none() | st.integers(-100, 100) | st.floats(-100, 100, allow_subnormal=False)
    ),
    st.integers(0, 1) | st.floats(min_value=0.0, max_value=1.0, allow_subnormal=False),
)
def test_percentile_cont(values, percentile):
    sorted_values = sorted([x for x in values if x is not None])

    if len(sorted_values) == 0:
        expected = None
    elif percentile == 1.0 or len(sorted_values) == 1:
        expected = float(sorted_values[-1])
    else:
        float_idx = (len(sorted_values) - 1) * percentile
        int_idx = int(float_idx)
        fraction = float_idx - int_idx

        if fraction == 0.0:
            expected = float(sorted_values[int_idx])
        else:
            lhs = sorted_values[int_idx] * (1.0 - fraction)
            rhs = sorted_values[int_idx + 1] * fraction
            expected = lhs + rhs

    res = query(
        "UNWIND $values AS x RETURN percentileCont(x, $p) AS result",
        params={"values": values, "p": percentile},
    )
    if expected is None:
        assert res.result_set == [[None]]
    else:
        assert_float_equal(res.result_set[0][0], expected)


def test_percentile_cont_edge_cases():
    # Test with percentile < 0
    q = "UNWIND [1, 2, 3, 4, 5] AS x RETURN percentileCont(x, -0.5) AS result"
    query_exception(
        q, "is not a valid argument, must be a number in the range 0.0 to 1.0"
    )

    # Test with percentile > 1
    q = "UNWIND [1, 2, 3, 4, 5] AS x RETURN percentileCont(x, 1.2) AS result"
    query_exception(
        q, "is not a valid argument, must be a number in the range 0.0 to 1.0"
    )


def test_case():
    res = query("RETURN CASE 1 + 2 WHEN 'a' THEN 1 END")
    assert res.result_set == [[None]]
    res = query("RETURN CASE WHEN 1 = 2 THEN 1 END")
    assert res.result_set == [[None]]
    res = query("RETURN CASE WHEN '1 = 2' THEN 1 END")
    assert res.result_set == [[1]]
    res = query("RETURN CASE 1 + 2 WHEN 'a' THEN 1 ELSE 2 END")
    assert res.result_set == [[2]]
    res = query("RETURN CASE WHEN 1 = 3 THEN 1 ELSE 2 END")
    assert res.result_set == [[2]]
    res = query("RETURN CASE 1 + 2 WHEN 3 THEN 1 + 2 WHEN 2 THEN 2 ELSE 2 END")
    assert res.result_set == [[3]]
    res = query(
        "RETURN CASE WHEN False THEN 1 WHEN 1 = 1 THEN 1 + 1 WHEN 3 = 3 THEN 3 ELSE 2 END"
    )
    assert res.result_set == [[2]]


def test_quantifier():
    # Test non-boolean expressions
    q = "RETURN all(x IN [1, 2, 3] WHERE x + 1) AS res"
    query_exception(q, "Type mismatch: expected Boolean but was Integer")

    res = query("RETURN any(x IN [1, 2, 3] WHERE null) AS res")
    assert res.result_set == [[None]]

    res = query("RETURN none(x IN [1, 2, 3] WHERE null) AS res")
    assert res.result_set == [[None]]

    res = query("RETURN single(x IN [1, 2, 3] WHERE null) AS res")
    assert res.result_set == [[None]]

    # Test mixed boolean and null values
    res = query("RETURN all(x IN [true, null] WHERE x) AS res")
    assert res.result_set == [[None]]

    res = query("RETURN any(x IN [false, null] WHERE x) AS res")
    assert res.result_set == [[None]]

    res = query("RETURN none(x IN [false, null] WHERE x) AS res")
    assert res.result_set == [[None]]

    res = query("RETURN single(x IN [true, null] WHERE x) AS res")
    assert res.result_set == [[None]]


@given(st.lists(st.integers(-10, 10)))
def test_prop_quantifier(a):
    res = query("RETURN all(x IN $a WHERE x > 0)", params={"a": a})
    assert res.result_set == [[all(x > 0 for x in a if x is not None)]]

    res = query("RETURN any(x IN $a WHERE x > 0)", params={"a": a})
    assert res.result_set == [[any(x > 0 for x in a if x is not None)]]

    res = query("RETURN none(x IN $a WHERE x > 0)", params={"a": a})
    assert res.result_set == [[not any(x > 0 for x in a if x is not None)]]

    res = query("RETURN single(x IN $a WHERE x > 0)", params={"a": a})
    assert res.result_set == [[len([x for x in a if x is not None and x > 0]) == 1]]


def test_list_comprehension():
    ## without where and without expr
    res = query("RETURN [x IN range(1, 10)] AS result")
    assert res.result_set == [[[1, 2, 3, 4, 5, 6, 7, 8, 9, 10]]]

    ## with where and without expr
    res = query("RETURN [x IN range(1, 10) WHERE x % 2 = 0] AS result")
    assert res.result_set == [[[2, 4, 6, 8, 10]]]

    ## with where and with expr
    res = query("RETURN [x IN range(1, 10) WHERE x % 2 = 0 | x + 1] AS result")
    assert res.result_set == [[[3, 5, 7, 9, 11]]]

    ## error in where
    q = "RETURN [x IN range(1, 10) WHERE x % 'a' = 2] AS result"
    query_exception(q, "Type mismatch: expected Integer, Float, or Null but was")

    ## error in expr
    q = "RETURN [x IN range(1, 10) WHERE x % 2 = 0 | x / 'a'] AS result"
    query_exception(q, "Type mismatch: expected Integer, Float, or Null but was")

    ## embedded
    res = query("RETURN [y IN [x IN range(1, 10)]] AS result")
    assert res.result_set == [[[1, 2, 3, 4, 5, 6, 7, 8, 9, 10]]]

    res = query("RETURN [x IN range(1, 10) | range(1, x)] AS result")
    expected = [[[list(range(1, i + 1)) for i in range(1, 11)]]]
    assert res.result_set == expected

    res = query("RETURN [x IN range(1, 10) WHERE x > 5] AS result")
    assert res.result_set == [[[6, 7, 8, 9, 10]]]

    res = query("RETURN [x IN range(1, 10) WHERE x < 5] AS result")
    assert res.result_set == [[[1, 2, 3, 4]]]

    res = query("RETURN [x IN range(1, 10) WHERE x = 5] AS result")
    assert res.result_set == [[[5]]]

    res = query("RETURN [x IN range(1, 10) WHERE x < 0] AS result")
    assert res.result_set == [[[]]]

    res = query("RETURN [x IN range(1, 10) WHERE x > 0] AS result")
    assert res.result_set == [[[1, 2, 3, 4, 5, 6, 7, 8, 9, 10]]]

    res = query("RETURN [x IN range(1, 10) WHERE x < -5] AS result")
    assert res.result_set == [[[]]]


@pytest.mark.extra
def test_parentheses():
    lparen = "(" * 10000
    rparen = ")" * 10000
    res = query(f"RETURN {lparen}1{rparen}")
    assert res.result_set == [[1]]


@pytest.mark.extra
def test_nested_list():
    lparen = "[" * 100
    rparen = "]" * 100
    res = query(f"RETURN {lparen}1{rparen}")
    expected = [1]
    for _ in range(100):
        expected = [expected]
    assert res.result_set == [expected]


def test_index():
    res = query(
        "UNWIND range(1, 100000) AS x CREATE (n:Node {vi: x, vs: tostring(x)})",
        write=True,
    )
    assert res.nodes_created == 100000

    memory_usage_before = memory_usage()

    res = query("MATCH (n:Node {vi: 5}) RETURN n")
    assert res.result_set == [
        [Node(4, labels=["Node"], properties={"vi": 5, "vs": "5"})]
    ]
    runtime_ms = res.run_time_ms

    query("CREATE INDEX FOR (n:Node) ON (n.vi, n.vs)", write=True)

    common.wait_for_indices_to_sync(common.g)

    res = query("MATCH (n:Node {vi: 5}) RETURN n", steps=2)
    assert res.result_set == [
        [Node(4, labels=["Node"], properties={"vi": 5, "vs": "5"})]
    ]
    assert res.run_time_ms < runtime_ms / 100

    res = query("MATCH (n:Node {vs: '5'}) RETURN n", steps=2)
    assert res.result_set == [
        [Node(4, labels=["Node"], properties={"vi": 5, "vs": "5"})]
    ]

    res = query("MATCH (n:Node {vi: 5}) SET n.vi = 0", write=True)
    assert res.properties_set == 1

    res = query("MATCH (n:Node {vi: 5}) RETURN n")
    assert res.result_set == []

    res = query("MATCH (n:Node {vi: 0}) RETURN n")
    assert res.result_set == [
        [Node(4, labels=["Node"], properties={"vi": 0, "vs": "5"})]
    ]

    res = query("MATCH (n:Node {vi: 0}) REMOVE n:Node", write=True)

    res = query("MATCH (n:Node {vi: 0}) RETURN n")
    assert res.result_set == []

    res = query("MATCH (n {vi: 0}) SET n:Node", write=True)

    res = query("MATCH (n:Node {vi: 0}) RETURN n")
    assert res.result_set == [
        [Node(4, labels=["Node"], properties={"vi": 0, "vs": "5"})]
    ]

    res = query("MATCH (n:Node {vi: 0}) DELETE n", write=True)
    assert res.nodes_deleted == 1

    res = query("MATCH (n:Node {vi: 0}) RETURN n")
    assert res.result_set == []

    res = query("CREATE (n:Node {vi: 5, vs: '5'})", write=True)
    assert res.nodes_created == 1

    res = query("MATCH (n:Node {vi: 5}) RETURN n")
    assert res.result_set == [
        [Node(4, labels=["Node"], properties={"vi": 5, "vs": "5"})]
    ]

    query("DROP INDEX FOR (n:Node) ON (n.vi, n.vs)", write=True)

    # global is_extra
    # if not is_extra:
    #     # wait for index drop to complete
    #     sleep(5)

    #     memory_usage_after = memory_usage()
    #     assert abs(memory_usage_after - memory_usage_before) < 1024 * 1024 / 2


@pytest.mark.extra
@pytest.mark.skipif(
    os.environ.get("EXISTING_ENV", "").lower() == "1",
    reason="LOAD CSV reads from the redis process's filesystem; under EXISTING_ENV "
    "redis runs in a sibling container that can't see test-written files.",
)
def test_load_csv():
    subprocess.run(["mkdir", "-p", "data"], check=True)
    with open("data/test.csv", "w") as f:
        f.write("name,age\nAlice,30\nBob,25\nCharlie,35\n")

    res = query("LOAD CSV WITH HEADERS FROM 'file://test.csv' AS row RETURN row")
    expected = [
        [{"name": "Alice", "age": "30"}],
        [{"name": "Bob", "age": "25"}],
        [{"name": "Charlie", "age": "35"}],
    ]
    assert res.result_set == expected

    res = query("LOAD CSV FROM 'file://test.csv' AS row RETURN row")
    expected = [
        [["name", "age"]],
        [["Alice", "30"]],
        [["Bob", "25"]],
        [["Charlie", "35"]],
    ]
    assert res.result_set == expected


def test_optional_match_null_merge():
    """Test that explicitly-bound Null values from OPTIONAL MATCH survive
    correctly through env merge operations.

    When OPTIONAL MATCH produces no results, the unmatched variables are
    explicitly bound to Null.  The env merge function must propagate these
    bound-Null values (instead of skipping them because the value is Null),
    so that downstream operators see the correct Null rather than a stale
    value from a prior operation.
    """
    # Create two :A nodes; only the first has an outgoing :R edge.
    query(
        "CREATE (:A {v: 1})-[:R]->(:B {v: 10}), (:A {v: 2})",
        write=True,
    )

    # OPTIONAL MATCH binds b to a Node for v=1 and to Null for v=2.
    # The MERGE clause triggers env.merge() internally when joining the
    # match result env back into the input env.
    res = query(
        """
        MATCH (a:A)
        OPTIONAL MATCH (a)-[:R]->(b)
        MERGE (c:C {v: a.v})
        RETURN a.v AS av, b.v AS bv, c.v AS cv
        ORDER BY av
        """,
        write=True,
        compare_results=False,
    )

    assert res.result_set == [
        [1, 10, 1],
        [2, None, 2],
    ]

    # Also test aggregation with an optional-match variable as a grouping key.
    # The aggregate operator calls acc.merge(&key) during finalization; if
    # bound-Null is not propagated correctly the Null group may show a stale
    # value instead of Null.
    res = query(
        """
        MATCH (a:A)
        OPTIONAL MATCH (a)-[:R]->(b)
        RETURN b.v AS bv, collect(a.v) AS vs
        ORDER BY vs
        """,
    )

    assert res.result_set == [
        [10, [1]],
        [None, [2]],
    ]


def test_value_hash_join_batch_boundary():
    """Regression: a probe row whose matches exactly fill an output batch must
    not cause the following probe row to be skipped.

    The Value Hash Join emits output in BATCH_SIZE (1024) chunks. When a single
    probe (left) row has exactly 1024 matches, draining them fills the builder
    to the batch boundary and advances to the next probe row before returning.
    A prior bug then re-advanced the probe cursor on the next `next()` call,
    skipping the following probe row entirely.

    Setup: probe side `:L` has two rows (k=1, k=2); build side `:R` has 1024
    rows with k=1 (so L{k:1} gets exactly 1024 matches) plus one row with k=2.
    Without the fix the k=2 group is dropped from the result.
    """
    query("CREATE (:L {k: 1}), (:L {k: 2})", write=True)
    query("UNWIND range(1, 1024) AS i CREATE (:R {k: 1})", write=True)
    query("CREATE (:R {k: 2})", write=True)

    res = query(
        """
        MATCH (a:L), (b:R)
        WHERE a.k = b.k
        RETURN a.k AS ak, count(b) AS cnt
        ORDER BY ak
        """
    )

    assert res.result_set == [
        [1, 1024],
        [2, 1],
    ]
