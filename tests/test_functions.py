import datetime
from itertools import product
from random import choice
import string
from typing import Optional

from redis import ResponseError
import common
from enum import IntFlag
import pytest

def setup_module(module):
    common.start_redis()


def teardown_module(module):
    common.shutdown_redis()

def setup_function(function):
    if common.g.name in common.client.list_graphs():
        common.g.delete()


class Type(IntFlag):
    NULL = 1
    BOOL = 2
    INT = 4
    FLOAT = 8
    STRING = 16
    LIST = 32
    MAP = 64
    NODE = 128
    RELATIONSHIP = 256
    PATH = 512

all_types = [
    Type.NULL,
    Type.BOOL,
    Type.INT,
    Type.FLOAT,
    Type.STRING,
    Type.LIST
]

def random_value(type):
    if type == Type.NULL:
        return "null"
    elif type == Type.BOOL:
        return choice(["true", "false"])
    elif type == Type.INT:
        return str(choice(range(-1000, 1000)))
    elif type == Type.FLOAT:
        return str(choice(list(map(lambda x: x / 10, range(-1000, 1000)))))
    elif type == Type.STRING:
        return f"\"{''.join(choice(string.ascii_letters + string.digits) for _ in range(choice(range(0, 100))))}\""
    elif type == Type.LIST:
        return str([random_value(choice(all_types)) for _ in range(choice(range(0, 10)))])
    elif isinstance(type, Optional):
        return choice([random_value(type.__args__[0]), None])
    return None

def is_valid_type(actual_type, expected_type):
    if isinstance(expected_type, Type):
        return actual_type & expected_type
    return actual_type & expected_type.__args__[0]

def validate_function(func, expected_args_type):
    if len(expected_args_type) == 0:
        query = f"RETURN {func}()"
        try:
            common.g.query(query)
            assert True
        except Exception as e:
            assert False, f"Function {func} should accept no arguments but raised an error: {e}"
        
        query = f"RETURN {func}(null)"
        try:
            common.g.query(query)
            assert False
        except Exception as e:
            assert "Received 1 arguments to function" in str(e), f"Function {func} should not accept null as an argument but raised an error: {e}"

    args_type = []
    for _ in range(len(expected_args_type)):
        args_type.append(all_types)
    for args_type in product(*args_type):
        args = [x for x in [random_value(arg_type) for arg_type in args_type] if x is not None]
        is_unexpected = False
        for (arg_type, expected_type) in zip(args_type, expected_args_type):
            if not is_valid_type(arg_type, expected_type):
                is_unexpected = True
                break
        if not is_unexpected:
            args.append("null")
            query = f"RETURN {func}({', '.join(args)})"
            try:
                common.g.query(query)
                assert False
            except Exception as e:
                most = len(expected_args_type)
                assert f"Received {len(args)} arguments to function '{func}', expected at most {most}" in str(e)
            least = len([x for x in expected_args_type if isinstance(x, Type)])
            while len(args) > least:
                args.pop()
            while len(args) > 0:
                args.pop()
                query = f"RETURN {func}({', '.join(args)})"
                try:
                    common.g.query(query)
                    assert False
                except Exception as e:
                    least = len([x for x in expected_args_type if isinstance(x, Type)])
                    assert f"Received {len(args)} arguments to function '{func}', expected at least {least}" in str(e)
        else:
            query = f"RETURN {func}({', '.join(args)})"
            try:
                common.g.query(query)
                assert False
            except Exception as e:
                assert True

@pytest.mark.parametrize("name, types", [
    ("tointeger", [Type.STRING | Type.BOOL | Type.INT | Type.FLOAT | Type.NULL]),
    ("tofloat", [Type.STRING | Type.INT | Type.FLOAT | Type.NULL]),
    ("tostring", [Type.STRING | Type.INT | Type.BOOL | Type.NULL]),
    ("size", [Type.LIST | Type.STRING | Type.NULL]),
    ("head", [Type.LIST | Type.NULL]),
    ("last", [Type.LIST | Type.NULL]),
    ("tail", [Type.LIST | Type.NULL]),
    ("reverse", [Type.LIST | Type.STRING | Type.NULL]),
    ("substring", [Type.STRING | Type.NULL, Type.INT, Optional[Type.INT]]),
    ("split", [Type.STRING | Type.NULL, Type.STRING | Type.NULL]),
    ("tolower", [Type.STRING | Type.NULL]),
    ("toupper", [Type.STRING | Type.NULL]),
    ("replace", [Type.STRING | Type.NULL, Type.STRING | Type.NULL, Type.STRING | Type.NULL]),
    ("left", [Type.STRING | Type.NULL, Type.INT | Type.NULL]),
    ("ltrim", [Type.STRING | Type.NULL]),
    ("right", [Type.STRING | Type.NULL, Type.INT | Type.NULL]),
    ("string.join", [Type.LIST | Type.NULL, Optional[Type.STRING]]),
    ("abs", [Type.INT | Type.FLOAT | Type.NULL]),
    ("ceil", [Type.INT | Type.FLOAT | Type.NULL]),
    ("e", []),
    ("exp", [Type.INT | Type.FLOAT | Type.NULL]),
    ("floor", [Type.INT | Type.FLOAT | Type.NULL]),
    ("log", [Type.INT | Type.FLOAT | Type.NULL]),
    ("log10", [Type.INT | Type.FLOAT | Type.NULL]),
    ("pow", [Type.INT | Type.FLOAT | Type.NULL, Type.INT | Type.FLOAT | Type.NULL]),
    ("rand", []),
    ("round", [Type.INT | Type.FLOAT | Type.NULL]),
    ("sign", [Type.INT | Type.FLOAT | Type.NULL]),
    ("sqrt", [Type.INT | Type.FLOAT | Type.NULL]),
    ("range", [Type.INT, Type.INT, Optional[Type.INT]]),
    ("keys", [Type.MAP | Type.NODE | Type.RELATIONSHIP | Type.NULL]),
    ("toBoolean", [Type.BOOL | Type.STRING | Type.INT | Type.NULL])
])
def test_functions(name, types):
    validate_function(name, types)

@pytest.mark.extra
def test_extra_functions():
    validate_function("string.matchRegEx", [Type.STRING | Type.NULL, Type.STRING | Type.NULL])
    validate_function("string.replaceRegEx", [Type.STRING | Type.NULL, Type.STRING | Type.NULL, Optional[Type.STRING | Type.NULL]])

def query_exception(query: str, message: str, params=None):
    try:
        common.g.query(query, params)
        assert False, "Expected an error"
    except ResponseError as e:
        assert message in str(e)

def test_struct_constructor_extra_args_rejected():
    """A temporal constructor called with more arguments than it accepts is
    refused by arity validation, with the message the C implementation gives.

    These constructors were declared `var_arg` so the zero-argument "now" form
    would validate, which also let *any* arity through. That mattered because
    `rewrite_struct_constructor` only rewrites a call with one Map argument and
    emits exactly one child per slot, while the evaluator inferred "this was
    rewritten" from `num_children > 1` — so a genuine two-argument call reached
    a `struct_fn` expecting `slots.len()` values and indexed past the end of a
    2-slice. A debug assertion in test builds; in release, a read past the end
    that killed the server. Found by the fuzzer.

    Capping the arity is what makes the bad call unreachable, and matches C,
    which rejects every one of these. The `== struct_slots.len()` guard in
    eval.rs is the second line of defence behind it.
    """
    for query, message in [
        # the fuzzer's input, reduced
        ("""WITH localdatetime({year: 1980, month: 12, day: 11, hour: 12,
                               minute: 31, second: 14}) AS x,
                 localdatetime({year: 1984, month: 10, day: 11, hour: 12},
                               {list: [6], fd: 645876123}) AS d
            RETURN x = d""",
         "Received 2 arguments to function 'localdatetime', expected at most 1"),
        ("RETURN localdatetime({year: 1984}, {x: 1})",
         "Received 2 arguments to function 'localdatetime', expected at most 1"),
        ("RETURN localdatetime({year: 1984}, {x: 1}, {y: 2})",
         "Received 3 arguments to function 'localdatetime', expected at most 1"),
        # a non-Map second argument takes the same path
        ("RETURN localdatetime({year: 1984}, 7)",
         "Received 2 arguments to function 'localdatetime', expected at most 1"),
        ("RETURN localtime({hour: 1}, {x: 1})",
         "Received 2 arguments to function 'localtime', expected at most 1"),
        ("RETURN date({year: 1984}, {x: 1})",
         "Received 2 arguments to function 'date', expected at most 1"),
        # duration and point already capped their arity; they must stay capped
        ("RETURN duration({days: 1}, {x: 1})",
         "Received 2 arguments to function 'duration', expected at most 1"),
        ("RETURN point({x: 1, y: 2}, {x: 1})",
         "Received 2 arguments to function 'point', expected at most 1"),
    ]:
        query_exception(query, message)
        assert common.g.query("RETURN 1").result_set == [[1]], \
            f"server did not survive: {query}"

def test_struct_constructor_accepted_arities():
    """Capping the arity must leave the two forms that are meant to work: the
    zero-argument "now" form the loose signature existed for, and the single-Map
    constructor the binder rewrites into positional slots."""
    for query in [
        "RETURN localdatetime() IS NOT NULL",
        "RETURN localtime() IS NOT NULL",
        "RETURN date() IS NOT NULL",
    ]:
        assert common.g.query(query).result_set == [[True]], query

    for query, expected in [
        ("RETURN localdatetime({year: 1984, month: 10, day: 11})",
         datetime.datetime(1984, 10, 11, tzinfo=datetime.timezone.utc)),
        ("RETURN date({year: 1984, month: 10, day: 11})",
         datetime.date(1984, 10, 11)),
        ("RETURN localtime({hour: 12, minute: 31})",
         datetime.time(12, 31)),
    ]:
        assert common.g.query(query).result_set[0][0] == expected, query

    # duration decodes to a relativedelta, which compares cleanly to its parts
    duration = common.g.query("RETURN duration({days: 1, hours: 2})").result_set[0][0]
    assert (duration.days, duration.hours) == (1, 2)
