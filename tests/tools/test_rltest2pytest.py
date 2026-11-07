"""Unit tests for the RLTest -> pytest codemod.

These matter more than usual: the tool rewrites ~5,600 assertions across 126
files, so a wrong translation would be both silent and pervasive. The cases
below pin the semantics that were read out of RLTest/env.py — especially
argument order and operator precedence, the two places a plausible-looking
rewrite can change meaning.
"""

import os
import sys

import pytest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# The codemod is a one-shot dev tool, so libcst is deliberately not in
# tests/requirements.txt — it would bloat every CI test image for something
# CI never runs. Skip rather than error when it isn't installed.
pytest.importorskip("libcst")

from tools.rltest2pytest import rewrite_source  # noqa: E402


def rw(src):
    out, _ = rewrite_source(src)
    return out.strip()


def refusals(src):
    _, r = rewrite_source(src)
    return r.refusals


class TestComparisons:
    @pytest.mark.parametrize("call,expected", [
        ("self.env.assertEqual(a, b)", "assert a == b"),
        ("self.env.assertNotEqual(a, b)", "assert a != b"),
        ("self.env.assertGreater(a, b)", "assert a > b"),
        ("self.env.assertGreaterEqual(a, b)", "assert a >= b"),
        ("self.env.assertLess(a, b)", "assert a < b"),
        ("self.env.assertLessEqual(a, b)", "assert a <= b"),
        ("self.env.assertIsNone(a)", "assert a is None"),
        ("self.env.assertIsNotNone(a)", "assert a is not None"),
        ("self.env.assertTrue(a)", "assert a"),
        ("self.env.assertFalse(a)", "assert not a"),
    ])
    def test_operator_mapping(self, call, expected):
        assert rw(call) == expected

    def test_contains_keeps_value_first(self):
        # RLTest: assertContains(value, holder) -> `value in holder`.
        # Getting this backwards would still run, and would still pass for
        # symmetric cases, which is exactly why it is pinned here.
        assert rw("self.env.assertContains(needle, haystack)") == "assert needle in haystack"
        assert rw("self.env.assertNotContains(needle, haystack)") == "assert needle not in haystack"

    def test_receiver_forms(self):
        assert rw("env.assertEqual(a, b)") == "assert a == b"
        assert rw("cls.env.assertEqual(a, b)") == "assert a == b"

    def test_ignores_non_env_receiver(self):
        src = "other.assertEqual(a, b)"
        assert rw(src) == src


class TestPrecedence:
    def test_boolean_operand_is_parenthesised(self):
        # `assert a or b == c` would parse as `a or (b == c)` — a bug that
        # silently passes whenever `a` is truthy.
        assert rw("self.env.assertEqual(a or b, c)") == "assert (a or b) == c"

    def test_conditional_operand_is_parenthesised(self):
        assert rw("self.env.assertEqual(a if p else b, c)") == "assert (a if p else b) == c"

    def test_chained_comparison_operand_is_parenthesised(self):
        assert rw("self.env.assertEqual(a < b, c)") == "assert (a < b) == c"

    def test_bare_tuple_is_parenthesised(self):
        assert rw("self.env.assertEqual(a, b, )").startswith("assert a == b")

    def test_arithmetic_needs_no_parens(self):
        # Binds tighter than comparison already; parenthesising would only
        # add noise across thousands of call sites.
        assert rw("self.env.assertEqual(a + b, c)") == "assert a + b == c"

    def test_assert_false_of_comparison(self):
        # `not` binds looser than `==`, so this is already correct unparenthesised.
        assert rw("self.env.assertFalse(a == b)") == "assert not a == b"

    def test_assert_false_of_boolean(self):
        assert rw("self.env.assertFalse(a or b)") == "assert not (a or b)"


class TestApprox:
    def test_almost_equal_uses_abs_delta(self):
        # RLTest: abs(v1 - v2) <= delta, which is pytest.approx(v2, abs=delta).
        out = rw("self.env.assertAlmostEqual(a, b, 0.0001)")
        assert out.endswith("assert a == pytest.approx(b, abs=0.0001)")

    def test_import_added_once(self):
        out, _ = rewrite_source(
            "x = 1\nself.env.assertAlmostEqual(a, b, 1)\n"
            "self.env.assertAlmostEqual(c, d, 2)\n")
        assert out.count("import pytest") == 1
        assert out.startswith("import pytest")

    def test_import_not_duplicated(self):
        out, _ = rewrite_source("import pytest\nself.env.assertAlmostEqual(a, b, 1)\n")
        assert out.count("import pytest") == 1

    def test_no_import_when_unused(self):
        out, _ = rewrite_source("self.env.assertEqual(a, b)\n")
        assert "import pytest" not in out


class TestMessageAndDepth:
    def test_message_becomes_assert_message(self):
        assert rw("self.env.assertEqual(a, b, message=msg)") == "assert a == b, msg"

    def test_depth_is_dropped(self):
        # depth is RLTest's own stack bookkeeping; it has no pytest meaning.
        assert rw("self.env.assertEqual(a, b, depth=1, message=msg)") == "assert a == b, msg"


class TestSetupClass:
    def test_init_becomes_setup_class(self):
        out = rw(
            "class testFoo():\n"
            "    def __init__(self):\n"
            "        self.env, self.db = Env()\n"
            "        self.graph = self.db.select_graph('g')\n")
        assert "@classmethod" in out
        assert "def setup_class(cls):" in out
        assert "cls.env, cls.db = Env()" in out
        assert "cls.graph = cls.db.select_graph('g')" in out
        assert "self." not in out

    def test_other_methods_untouched(self):
        out = rw(
            "class testFoo():\n"
            "    def test01(self):\n"
            "        self.env.assertEqual(a, b)\n")
        assert "def test01(self):" in out
        assert "assert a == b" in out

    def test_init_with_extra_args_is_refused(self):
        src = ("class testFoo():\n"
               "    def __init__(self, extra):\n"
               "        self.x = extra\n")
        assert rw(src) == src.strip()
        assert len(refusals(src)) == 1

    def test_helper_class_init_is_left_alone(self):
        # QueryInfo / Constraint / MemoryUsage live alongside test classes but
        # are ordinary objects; converting their constructors would break them.
        src = ("class QueryInfo():\n"
               "    def __init__(self):\n"
               "        self.x = 1\n")
        out = rw(src)
        assert "def __init__(self):" in out
        assert "setup_class" not in out
        assert refusals(src) == []

    def test_helpers_reached_from_init_are_promoted(self):
        # `cls.populate_graph()` on an instance method is an unbound call and
        # fails with "missing 1 required positional argument: 'self'".
        out = rw(
            "class testFoo():\n"
            "    def __init__(self):\n"
            "        self.env, self.db = Env()\n"
            "        self.populate_graph()\n"
            "    def populate_graph(self):\n"
            "        self.graph = self.db.select_graph('g')\n")
        assert "def populate_graph(cls):" in out
        assert out.count("@classmethod") == 2
        assert "self." not in out

    def test_promotion_is_transitive(self):
        out = rw(
            "class testFoo():\n"
            "    def __init__(self):\n"
            "        self.a()\n"
            "    def a(self):\n"
            "        self.b()\n"
            "    def b(self):\n"
            "        pass\n")
        assert "def a(cls):" in out and "def b(cls):" in out
        assert out.count("@classmethod") == 3

    def test_helpers_keep_their_extra_parameters(self):
        out = rw(
            "class testFoo():\n"
            "    def __init__(self):\n"
            "        self.populate(10)\n"
            "    def populate(self, n):\n"
            "        self.n = n\n")
        assert "def populate(cls, n):" in out

    def test_test_methods_are_never_promoted(self):
        # Promoting a test method would change how pytest calls it.
        out = rw(
            "class testFoo():\n"
            "    def __init__(self):\n"
            "        self.test01()\n"
            "    def test01(self):\n"
            "        pass\n")
        assert "def test01(self):" in out

    def test_helper_only_used_by_tests_is_not_promoted(self):
        out = rw(
            "class testFoo():\n"
            "    def __init__(self):\n"
            "        self.env = Env()\n"
            "    def helper(self):\n"
            "        pass\n"
            "    def test01(self):\n"
            "        self.helper()\n")
        assert "def helper(self):" in out
        assert out.count("@classmethod") == 1

    def test_already_a_classmethod_is_not_double_decorated(self):
        out = rw(
            "class testFoo():\n"
            "    def __init__(self):\n"
            "        self.helper()\n"
            "    @classmethod\n"
            "    def helper(cls):\n"
            "        pass\n")
        assert out.count("@classmethod") == 2


class TestXunitHooks:
    """pytest only honours setUp/tearDown on unittest.TestCase subclasses."""

    def test_setup_and_teardown_are_renamed(self):
        out = rw(
            "class testFoo():\n"
            "    def setUp(self):\n"
            "        self.conn.delete(KEY)\n"
            "    def tearDown(self):\n"
            "        self.conn.flushall()\n")
        assert "def setup_method(self):" in out
        assert "def teardown_method(self):" in out
        assert "setUp" not in out and "tearDown" not in out

    def test_hooks_renamed_without_an_init_present(self):
        out = rw("class testFoo():\n"
                 "    def setUp(self):\n"
                 "        pass\n")
        assert "def setup_method(self):" in out

    def test_helper_class_hooks_are_left_alone(self):
        src = ("class QueryInfo():\n"
               "    def setUp(self):\n"
               "        pass\n")
        assert "def setUp(self):" in rw(src)


class TestLatentBugs:
    def test_assert_true_positional_message(self):
        # RLTest binds the string to `depth` and dies on `depth + 1`; the
        # intended reading is an assertion message.
        assert rw('self.env.assertTrue(False, "boom")') == 'assert False, "boom"'

    def test_non_literal_second_positional_is_refused(self):
        # Could be a genuine depth argument, so refuse rather than guess.
        src = "self.env.assertTrue(False, n)"
        assert rw(src) == src
        assert len(refusals(src)) == 1

    def test_almost_equal_delta_keyword(self):
        out = rw("self.env.assertAlmostEqual(a, b, delta=0.1)")
        assert out.endswith("assert a == pytest.approx(b, abs=0.1)")


class TestMultilineExpressions:
    """Splicing out the call's parentheses must not break line continuation."""

    def test_multiline_boolean_gets_parens(self):
        out = rw("self.env.assertTrue(a == b or\n"
                 "                    c == d)")
        assert out.startswith("assert (")
        compile(out, "<t>", "exec")

    def test_multiline_operand_in_comparison(self):
        out = rw("self.env.assertEqual(a +\n"
                 "                     b, c)")
        compile(out, "<t>", "exec")

    def test_single_line_stays_unparenthesised(self):
        # The multi-line guard must not add noise to the common case.
        assert rw("self.env.assertTrue(a == b)") == "assert a == b"

    def test_multiline_call_needs_no_extra_parens(self):
        # Newlines already sit inside the call's own brackets.
        out = rw("self.env.assertTrue(foo(a,\n"
                 "                        b))")
        assert out == "assert foo(a,\n                        b)"
        compile(out, "<t>", "exec")

    def test_multiline_message_gets_parens(self):
        # Implicit string concatenation split over lines has no brackets
        # of its own once the call's parens are removed.
        out = rw('self.env.assertTrue(False, "part one "\n'
                 '                           "part two")')
        assert out.endswith('"part two")')
        compile(out, "<t>", "exec")


class TestOutputAlwaysCompiles:
    def test_invalid_output_is_refused_not_written(self, monkeypatch):
        import tools.rltest2pytest as mod

        # Force a broken rewrite to prove the guard returns the original.
        monkeypatch.setattr(mod, "_paren_if_multiline", lambda n: n)
        src = "self.env.assertTrue(a == b or\n                    c == d)\n"
        out, rw_ = mod.rewrite_source(src, "x.py")
        assert out == src
        assert any("invalid syntax" in r for r in rw_.refusals)


class TestSharedClassState:
    """RLTest reused one instance per class; pytest builds one per test."""

    def test_write_to_class_state_is_redirected(self):
        out = rw(
            "class testFoo():\n"
            "    def __init__(self):\n"
            "        self.env, self.db = Env()\n"
            "    def test01(self):\n"
            "        self.env, self.db = Env(moduleArgs='X 1')\n")
        assert "type(self).env, type(self).db = Env(moduleArgs='X 1')" in out

    def test_unrelated_instance_attrs_are_left_alone(self):
        out = rw(
            "class testFoo():\n"
            "    def __init__(self):\n"
            "        self.env = Env()\n"
            "    def test01(self):\n"
            "        self.scratch = 1\n")
        assert "self.scratch = 1" in out
        assert "type(self)" not in out

    def test_reads_are_left_alone(self):
        out = rw(
            "class testFoo():\n"
            "    def __init__(self):\n"
            "        self.graph = g\n"
            "    def test01(self):\n"
            "        self.graph.query('X')\n")
        assert "self.graph.query('X')" in out

    def test_helpers_promoted_to_classmethod_are_not_touched(self):
        out = rw(
            "class testFoo():\n"
            "    def __init__(self):\n"
            "        self.build()\n"
            "    def build(self):\n"
            "        self.graph = g\n")
        assert "cls.graph = g" in out
        assert "type(self)" not in out

    def test_teardown_rebuilt_handles_reach_the_next_test(self):
        # RLTest ran tearDown on the same shared instance, so a handle
        # rebuilt there was visible to the following test.
        out = rw(
            "class testFoo():\n"
            "    def __init__(self):\n"
            "        self.graph = g\n"
            "    def tearDown(self):\n"
            "        self.graph = fresh()\n")
        assert "def teardown_method(self):" in out
        assert "type(self).graph = fresh()" in out


class TestBareSelfAsserts:
    """`self.assertX(...)` with the `.env` dropped.

    Flow test classes are not `unittest.TestCase`, so these raise
    AttributeError rather than asserting — and only on the failure path,
    where the real reason then gets masked. Four sites in
    test_replication_states.py and one in test_constraint.py.
    """

    def test_bare_self_assert_true_is_rewritten(self):
        assert rw("self.assertTrue(False)") == "assert False"

    def test_bare_self_assert_false_is_rewritten(self):
        assert rw("self.assertFalse(1)") == "assert not 1"

    def test_bare_cls_assert_is_rewritten(self):
        assert rw("cls.assertEqual(a, b)") == "assert a == b"

    def test_locally_defined_helper_is_untouched(self):
        # assertLoggedQuery is defined by test_graph_info.py itself. Only
        # RLTest's own method names are rewritten.
        src = "self.assertLoggedQuery(logged[0], q, cached)"
        assert rw(src) == src

    def test_unrelated_receiver_is_untouched(self):
        src = "other.assertEqual(a, b)"
        assert rw(src) == src


class TestReproducibility:
    def test_rewrite_is_idempotent(self):
        src = ("class testFoo():\n"
               "    def __init__(self):\n"
               "        self.env, self.db = Env()\n"
               "    def test01(self):\n"
               "        self.env.assertEqual(a, b)\n"
               "        self.env.assertAlmostEqual(c, d, 0.1)\n")
        once, _ = rewrite_source(src)
        twice, _ = rewrite_source(once)
        assert once == twice

    def test_formatting_is_preserved(self):
        # libcst keeps comments and blank lines; a regex pass would not.
        src = ("# leading comment\n"
               "\n"
               "def f():\n"
               "    # inner comment\n"
               "    self.env.assertEqual(a, b)  # trailing\n")
        out, _ = rewrite_source(src)
        assert "# leading comment" in out
        assert "# inner comment" in out
        assert "# trailing" in out
