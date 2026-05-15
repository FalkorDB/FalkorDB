"""Regression tests for integer-overflow runtime crashes.

Each of the following queries used to crash the server (SIGSEGV / Rust panic
abort) before being fixed:

  * abs(i64::MIN)             -> n.abs() overflow panic
  * unary minus on i64::MIN   -> -i overflow panic
  * range(0, i64::MAX)        -> length overflow / huge allocation

The fixes replace the offending arithmetic with `checked_neg` / `checked_abs`
and saturating/checked length math in `range`. These tests assert that the
server (a) does NOT crash and (b) returns a clear error to the client.
"""

from redis import ResponseError
import pytest

import common


def setup_module(module):
    common.start_redis()


def teardown_module(module):
    common.shutdown_redis()


def setup_function(function):
    if common.g.name in common.client.list_graphs():
        common.g.delete()


def _ensure_alive():
    # If the server crashed, this raises ConnectionError instead of returning.
    assert common.client.connection.ping()


def test_abs_i64_min_does_not_crash():
    with pytest.raises(ResponseError, match=r"(?i)overflow"):
        common.g.query("RETURN abs(-9223372036854775808)")
    _ensure_alive()


def test_unary_minus_i64_min_does_not_crash():
    with pytest.raises(ResponseError, match=r"(?i)overflow"):
        common.g.query("WITH -9223372036854775808 AS x RETURN -x")
    _ensure_alive()


def test_unary_minus_i64_min_param_does_not_crash():
    # Same bug, exercised through the parameter-evaluation path.
    with pytest.raises(ResponseError, match=r"(?i)overflow"):
        common.g.query("CYPHER x=-9223372036854775808 RETURN -$x")
    _ensure_alive()


def test_range_to_i64_max_does_not_crash():
    with pytest.raises(ResponseError, match=r"(?i)range too large"):
        common.g.query("RETURN range(0, 9223372036854775807)")
    _ensure_alive()


def test_range_full_int_span_does_not_crash():
    # Spans the entire i64 domain; old code overflowed `end - start`.
    with pytest.raises(ResponseError, match=r"(?i)range too large"):
        common.g.query(
            "RETURN range(-9223372036854775808, 9223372036854775807)"
        )
    _ensure_alive()


def test_abs_normal_values_still_work():
    res = common.g.query("RETURN abs(-3), abs(3), abs(-1.5), abs(NULL)")
    assert res.result_set == [[3, 3, 1.5, None]]


def test_unary_minus_normal_values_still_work():
    res = common.g.query("RETURN -5, -(-7), -1.25, -NULL")
    assert res.result_set == [[-5, 7, -1.25, None]]


def test_range_normal_values_still_work():
    res = common.g.query("RETURN range(1, 5)")
    assert res.result_set == [[[1, 2, 3, 4, 5]]]
    res = common.g.query("RETURN range(10, 2, -2)")
    assert res.result_set == [[[10, 8, 6, 4, 2]]]
