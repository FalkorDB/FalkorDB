#!/usr/bin/env python3
"""Rewrite RLTest-style flow tests into plain pytest.

Two transformations, both mechanical:

  1. `self.env.assertEqual(a, b)` and friends  ->  `assert a == b`
  2. a test class's `__init__`                 ->  `@classmethod setup_class(cls)`

The point of doing this with libcst rather than regex is that the result is
*reproducible*: re-running the tool over the migrated tree is a no-op, so the
large mechanical commit can be regenerated and diffed rather than trusted.

Anything the tool cannot prove safe is left alone and reported, so the
residue is a short list for manual conversion instead of a silent
mistranslation.

    python tests/tools/rltest2pytest.py tests/flow          # rewrite
    python tests/tools/rltest2pytest.py --check tests/flow  # verify no-op

Requires libcst (`pip install libcst`), which is deliberately kept out of
tests/requirements.txt: this is a one-shot dev tool, not a test dependency.
"""

import argparse
import os
import sys
import warnings

import libcst as cst
import libcst.matchers as m
from libcst.metadata import MetadataWrapper, PositionProvider

# name -> (arity, operator). Semantics verified against RLTest/env.py:
#   assertEqual(first, second)          first == second
#   assertAlmostEqual(v1, v2, delta)    abs(v1 - v2) <= delta
#   assertContains(value, holder)       value in holder    <- value is FIRST
#   assertTrue(val)                     bool(val) == True
COMPARISONS = {
    "assertEqual": (2, "=="),
    "assertNotEqual": (2, "!="),
    "assertGreater": (2, ">"),
    "assertGreaterEqual": (2, ">="),
    "assertLess": (2, "<"),
    "assertLessEqual": (2, "<="),
    "assertIn": (2, "in"),
    "assertContains": (2, "in"),
    "assertNotContains": (2, "not in"),
    "assertIsNone": (1, "is None"),
    "assertIsNotNone": (1, "is not None"),
}
TRUTHY = {"assertTrue": False, "assertFalse": True}
APPROX = "assertAlmostEqual"

ALL_ASSERTS = set(COMPARISONS) | set(TRUTHY) | {APPROX}

_OP = {
    "==": cst.Equal, "!=": cst.NotEqual,
    ">": cst.GreaterThan, ">=": cst.GreaterThanEqual,
    "<": cst.LessThan, "<=": cst.LessThanEqual,
    "in": cst.In, "not in": cst.NotIn,
}

# Operand types that bind more loosely than a comparison, so they must be
# parenthesised when spliced into one. Everything else (calls, subscripts,
# literals, arithmetic) already binds tighter and is left bare for readability.
_LOOSER_THAN_COMPARISON = (
    cst.BooleanOperation,
    cst.Lambda,
    cst.IfExp,
    cst.NamedExpr,
    cst.Yield,
)
_NEEDS_PARENS = _LOOSER_THAN_COMPARISON + (cst.Comparison,)

# Expressions whose line breaks are *not* protected by brackets of their own.
# Inside a call the newlines were held by the call's parentheses; once the call
# is gone they need brackets or the result won't parse. Implicit string
# concatenation ("a" "b" split over lines) has the same problem.
_SPLITTABLE = (
    cst.BooleanOperation,
    cst.BinaryOperation,
    cst.Comparison,
    cst.UnaryOperation,
    cst.IfExp,
    cst.Lambda,
    cst.NamedExpr,
    cst.ConcatenatedString,
)

# Rendering helper for the multi-line check; carries no state.
_MODULE = cst.Module(body=[])

_STRING_NODES = (cst.SimpleString, cst.ConcatenatedString, cst.FormattedString)

# unittest hook names RLTest honoured, mapped to pytest's xunit equivalents.
_XUNIT_HOOKS = {
    "setUp": "setup_method",
    "tearDown": "teardown_method",
    "setUpClass": "setup_class",
    "tearDownClass": "teardown_class",
}

# Classes pytest will collect, per `python_classes = test* Test*` in
# pytest.ini. Only these get the __init__ -> setup_class treatment: helper
# classes in the same files (QueryInfo, Constraint, MemoryUsage, LoggedQuery)
# are ordinary objects whose constructors must stay constructors.
def _is_test_class(name):
    return name.startswith("test") or name.startswith("Test")


class Refusal(Exception):
    """Raised when a construct can't be translated with confidence."""


def _paren(node):
    """Parenthesise `node` if splicing it into an assert would change meaning.

    Two independent reasons to add parentheses:

    1. Precedence — an operand that binds looser than a comparison would
       re-associate, e.g. `assertEqual(a or b, c)` must not become
       `assert a or b == c`.
    2. Line continuation — the operand's newlines used to sit inside the
       call's parentheses. Removing the call makes a multi-line operator
       expression a syntax error, so it needs brackets of its own.
    """
    if isinstance(node, _NEEDS_PARENS):
        return node.with_changes(lpar=[cst.LeftParen()], rpar=[cst.RightParen()])
    if isinstance(node, (cst.Tuple, cst.GeneratorExp)) and not node.lpar:
        return node.with_changes(lpar=[cst.LeftParen()], rpar=[cst.RightParen()])
    return _paren_if_multiline(node)


def _paren_if_multiline(node):
    """Bracket an unbracketed operator expression that spans several lines."""
    if not isinstance(node, _SPLITTABLE) or node.lpar:
        return node
    if "\n" not in _MODULE.code_for_node(node):
        return node
    return node.with_changes(lpar=[cst.LeftParen()], rpar=[cst.RightParen()])


def _is_env_receiver(node):
    """True for `env.<assert>` / `self.env.<assert>` / `cls.env.<assert>`,
    and for a bare `self.<assert>` / `cls.<assert>`.

    Requiring an `env` receiver keeps the rewrite from touching an unrelated
    object that happens to expose a method of the same name. `self`/`cls` is
    included because a handful of call sites drop the `.env` — those are
    latent AttributeErrors today (flow test classes are not
    `unittest.TestCase`, so they have no `assertTrue`), reachable only on the
    failure path, which is precisely where they do the most damage. Locally
    defined helpers are unaffected: only RLTest's own method names in
    ALL_ASSERTS are rewritten, and no flow class defines one.
    """
    return (m.matches(node, m.Name("env"))
            or m.matches(node, m.Attribute(attr=m.Name("env")))
            or m.matches(node, m.Name("self"))
            or m.matches(node, m.Name("cls")))


def _split_args(call, allow_positional_message=False):
    """Positional args plus an optional assertion message.

    `depth=` is RLTest's stack-unwinding hint for its own reporting and has no
    pytest equivalent, so it is dropped. `message=` becomes the assert message.

    `allow_positional_message` handles a latent bug in the existing tests.
    RLTest's signature is `assertTrue(val, depth=0, message=None)`, so
    `assertTrue(False, "some message")` binds the string to `depth`, and
    RLTest then evaluates `depth + 1` and dies with

        TypeError: can only concatenate str (not "int") to str

    rather than reporting the message. The tests still fail (so nothing was
    passing that shouldn't), but with a confusing error. Treating a trailing
    string literal as the message translates these to `assert False, "..."`,
    which is what the author meant. Only string *literals* qualify — a
    non-literal second positional could be a genuine depth and is refused.
    """
    positional, message = [], None
    for arg in call.args:
        if arg.keyword is None:
            if arg.star:
                raise Refusal("*args / **kwargs at call site")
            positional.append(arg.value)
        elif arg.keyword.value in ("message", "delta"):
            # `delta` is simply the named form of assertAlmostEqual's third
            # positional parameter.
            if arg.keyword.value == "delta":
                positional.append(arg.value)
            else:
                message = arg.value
        elif arg.keyword.value == "depth":
            continue
        else:
            raise Refusal(f"unexpected keyword {arg.keyword.value!r}")

    if (allow_positional_message and message is None and len(positional) == 2
            and isinstance(positional[1], _STRING_NODES)):
        message = positional.pop()
    if message is not None:
        message = _paren_if_multiline(message)
    return positional, message


class _InstanceWriteToClass(cst.CSTTransformer):
    """`self.x = v` -> `type(self).x = v` for attributes that are class state.

    RLTest built one instance per class, so a write in one test was visible
    in the next. pytest builds a fresh instance per test, which would drop
    the write on the floor and leave later tests reading the stale value.
    """

    def __init__(self, names):
        self.names = names
        self.rewrites = 0

    def _fix(self, node):
        if isinstance(node, (cst.Tuple, cst.List)):
            return node.with_changes(elements=[
                e.with_changes(value=self._fix(e.value)) for e in node.elements])
        if (m.matches(node, m.Attribute(value=m.Name("self")))
                and node.attr.value in self.names):
            self.rewrites += 1
            return node.with_changes(value=cst.Call(
                func=cst.Name("type"), args=[cst.Arg(value=cst.Name("self"))]))
        return node

    def leave_Assign(self, original, updated):
        return updated.with_changes(targets=[
            t.with_changes(target=self._fix(t.target)) for t in updated.targets])

    def leave_AugAssign(self, original, updated):
        return updated.with_changes(target=self._fix(updated.target))

    def leave_AnnAssign(self, original, updated):
        return updated.with_changes(target=self._fix(updated.target))


def _class_state_names(body):
    """Attributes assigned through `cls` by any classmethod of the class."""
    names = set()
    for stmt in body:
        if not isinstance(stmt, cst.FunctionDef):
            continue
        if not any(m.matches(d.decorator, m.Name("classmethod"))
                   for d in stmt.decorators):
            continue
        for assign in m.findall(stmt, m.Assign()):
            for target in assign.targets:
                for node in m.findall(target, m.Attribute(value=m.Name("cls"))):
                    names.add(node.attr.value)
    return names


class _SelfToCls(cst.CSTTransformer):
    """Rename `self` to `cls` inside a converted setup_class body."""

    def leave_Name(self, original, updated):
        return updated.with_changes(value="cls") if updated.value == "self" else updated


def _is_instance_method(fn):
    """A plain `def m(self, ...)` — not already a class/static method."""
    for d in fn.decorator_list if hasattr(fn, "decorator_list") else fn.decorators:
        if m.matches(d.decorator, m.Name("classmethod")) or \
           m.matches(d.decorator, m.Name("staticmethod")):
            return False
    ps = fn.params.params
    return bool(ps) and ps[0].name.value == "self"


def _reachable_helpers(init, methods):
    """Helpers that `__init__` reaches through `self.x`, transitively.

    `__init__` becomes a classmethod, so every helper it touches has to
    become one too — otherwise `cls.populate_graph()` is an unbound
    function and complains about a missing `self`.
    """
    seen, queue = set(), [init]
    while queue:
        fn = queue.pop()
        for node in m.findall(fn, m.Attribute(value=m.Name("self"))):
            name = node.attr.value
            target = methods.get(name)
            if (target is None or name in seen or name.startswith("test")
                    or not _is_instance_method(target)):
                continue
            seen.add(name)
            queue.append(target)
    return seen


def _to_classmethod(fn, name):
    params = fn.params
    return fn.with_changes(
        name=cst.Name(name),
        params=params.with_changes(
            params=[params.params[0].with_changes(name=cst.Name("cls")),
                    *params.params[1:]]),
        body=fn.body.visit(_SelfToCls()),
        decorators=[*fn.decorators, cst.Decorator(decorator=cst.Name("classmethod"))],
    )


class Rewriter(cst.CSTTransformer):
    METADATA_DEPENDENCIES = (PositionProvider,)

    def __init__(self, path):
        self.path = path
        self.refusals = []
        self.rewrites = 0
        self.needs_pytest = False
        self._has_pytest_import = False
        self._class_stack = []
        self.converted_inits = 0
        self.renamed_hooks = 0
        self.shared_writes = 0

    def visit_ClassDef(self, node):
        self._class_stack.append(node.name.value)

    def leave_ClassDef(self, original, updated):
        name = self._class_stack.pop()
        # Only collected test classes get a setup_class; helper classes in the
        # same file keep their constructors.
        if not _is_test_class(name):
            return updated

        updated = self._rename_xunit_hooks(updated)

        methods = {s.name.value: s for s in updated.body.body
                   if isinstance(s, cst.FunctionDef)}
        init = methods.get("__init__")
        if init is None:
            return updated
        if "setup_class" in methods:
            self.refusals.append(
                f"{self.path}:{self._line(original)}: {name} has both __init__ "
                "and setUpClass/setup_class; merge them by hand")
            return updated
        if m.findall(init, m.Call(func=m.Attribute(
                value=m.Call(func=m.Name("super")), attr=m.Name("__init__")))):
            # `super().__init__()` needs an instance; as a classmethod it
            # raises "descriptor '__init__' of 'object' object needs an
            # argument". Never silently drop it.
            self.refusals.append(
                f"{self.path}:{self._line(original)}: {name}.__init__ calls "
                "super().__init__(); convert by hand")
            return updated

        params = init.params
        if (len(params.params) != 1 or params.kwonly_params or params.posonly_params
                or params.star_arg is not cst.MaybeSentinel.DEFAULT):
            self.refusals.append(
                f"{self.path}:{self._line(original)}: __init__ takes arguments "
                "beyond self; convert by hand")
            return updated
        if params.params[0].name.value != "self":
            return updated

        promote = _reachable_helpers(init, methods)
        body = []
        for stmt in updated.body.body:
            if isinstance(stmt, cst.FunctionDef) and stmt.name.value == "__init__":
                body.append(_to_classmethod(stmt, "setup_class"))
            elif isinstance(stmt, cst.FunctionDef) and stmt.name.value in promote:
                body.append(_to_classmethod(stmt, stmt.name.value))
            else:
                body.append(stmt)
        self.converted_inits += 1
        state = _class_state_names(body)
        body = [self._share_class_state(s, state) for s in body]
        return updated.with_changes(body=updated.body.with_changes(body=body))

    def _share_class_state(self, stmt, state):
        """Route class-state writes in instance methods back to the class.

        This includes setUp/tearDown: RLTest ran them against the same
        single instance, so a handle rebuilt in tearDown was visible to the
        next test.
        """
        if not isinstance(stmt, cst.FunctionDef) or not state:
            return stmt
        if not _is_instance_method(stmt):
            return stmt
        fixer = _InstanceWriteToClass(state)
        stmt = stmt.with_changes(body=stmt.body.visit(fixer))
        self.shared_writes += fixer.rewrites
        return stmt

    def _rename_xunit_hooks(self, cls_node):
        """`setUp`/`tearDown` -> `setup_method`/`teardown_method`.

        These are unittest names. RLTest ran them; pytest only honours them
        on `unittest.TestCase` subclasses, so on a plain class they would
        silently stop running and leave state leaking between tests.
        """
        body, changed = [], False
        for stmt in cls_node.body.body:
            new = _XUNIT_HOOKS.get(
                stmt.name.value) if isinstance(stmt, cst.FunctionDef) else None
            if new is None:
                body.append(stmt)
                continue
            body.append(stmt.with_changes(name=cst.Name(new)))
            self.renamed_hooks += 1
            changed = True
        if not changed:
            return cls_node
        return cls_node.with_changes(body=cls_node.body.with_changes(body=body))
        return updated

    # ── assertions ──────────────────────────────────────────────────────
    def leave_SimpleStatementLine(self, original, updated):
        if len(updated.body) != 1 or not isinstance(updated.body[0], cst.Expr):
            return updated
        call = updated.body[0].value
        if not isinstance(call, cst.Call) or not isinstance(call.func, cst.Attribute):
            return updated
        name = call.func.attr.value
        if name not in ALL_ASSERTS or not _is_env_receiver(call.func.value):
            return updated
        try:
            assert_stmt = self._build_assert(name, call)
        except Refusal as e:
            self.refusals.append(f"{self.path}:{self._line(original)}: {name}: {e}")
            return updated
        self.rewrites += 1
        return updated.with_changes(body=[assert_stmt])

    def _line(self, node):
        try:
            return self.get_metadata(PositionProvider, node).start.line
        except Exception:
            return "?"

    def _build_assert(self, name, call):
        args, message = _split_args(
            call, allow_positional_message=name in TRUTHY)

        if name in TRUTHY:
            if len(args) != 1:
                raise Refusal(f"expected 1 argument, got {len(args)}")
            expr = args[0]
            if TRUTHY[name]:  # assertFalse
                # `not` binds looser than comparison, so `not a == b` already
                # means `not (a == b)`; only boolean/conditional operands
                # actually need parens for precedence.
                expr = (_paren(expr) if isinstance(expr, _LOOSER_THAN_COMPARISON)
                        else _paren_if_multiline(expr))
                expr = cst.UnaryOperation(operator=cst.Not(), expression=expr)
            else:
                expr = _paren_if_multiline(expr)
            return cst.Assert(test=expr, msg=message)

        if name == APPROX:
            if len(args) != 3:
                raise Refusal(f"expected 3 arguments (v1, v2, delta), got {len(args)}")
            self.needs_pytest = True
            approx = cst.Call(
                func=cst.Attribute(value=cst.Name("pytest"), attr=cst.Name("approx")),
                args=[
                    cst.Arg(value=args[1]),
                    cst.Arg(value=args[2], keyword=cst.Name("abs"),
                            equal=cst.AssignEqual(
                                whitespace_before=cst.SimpleWhitespace(""),
                                whitespace_after=cst.SimpleWhitespace(""))),
                ],
            )
            return cst.Assert(
                test=cst.Comparison(
                    left=_paren(args[0]),
                    comparisons=[cst.ComparisonTarget(
                        operator=cst.Equal(), comparator=approx)]),
                msg=message)

        arity, op = COMPARISONS[name]
        if len(args) != arity:
            raise Refusal(f"expected {arity} argument(s), got {len(args)}")

        if op in ("is None", "is not None"):
            operator = cst.Is() if op == "is None" else cst.IsNot()
            return cst.Assert(
                test=cst.Comparison(
                    left=_paren(args[0]),
                    comparisons=[cst.ComparisonTarget(
                        operator=operator, comparator=cst.Name("None"))]),
                msg=message)

        # assertContains(value, holder) reads `value in holder`, so operands
        # stay in source order for `in` just as they do for `==`.
        return cst.Assert(
            test=cst.Comparison(
                left=_paren(args[0]),
                comparisons=[cst.ComparisonTarget(
                    operator=_OP[op](), comparator=_paren(args[1]))]),
            msg=message)

    # ── imports ─────────────────────────────────────────────────────────
    def visit_Import(self, node):
        for alias in node.names:
            if m.matches(alias.name, m.Name("pytest")):
                self._has_pytest_import = True

    def leave_Module(self, original, updated):
        if not self.needs_pytest or self._has_pytest_import:
            return updated
        imp = cst.SimpleStatementLine(
            body=[cst.Import(names=[cst.ImportAlias(name=cst.Name("pytest"))])])
        return updated.with_changes(body=[imp, *updated.body])


def rewrite_source(src, path="<memory>"):
    """Rewrite one module's source. Returns (new_source, rewriter).

    The output is re-parsed before being handed back: a rewrite that doesn't
    compile is a tool bug, and it should surface as a refusal on that one
    file rather than as a broken tree on disk.
    """
    wrapper = MetadataWrapper(cst.parse_module(src), unsafe_skip_copy=True)
    rewriter = Rewriter(path)
    new = wrapper.visit(rewriter)
    code = new.code
    try:
        # Suppress SyntaxWarning: several test files contain regex literals
        # with invalid escapes ('Project\s+Filter'). Those warnings predate
        # this tool and are not what the guard is checking for.
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", SyntaxWarning)
            compile(code, path, "exec")
    except SyntaxError as e:
        rewriter.refusals.append(
            f"{path}:{e.lineno}: rewrite produced invalid syntax ({e.msg}); "
            "file left unchanged")
        return src, rewriter
    return code, rewriter


def _collect(paths):
    files = []
    for p in paths:
        if os.path.isdir(p):
            for root, _, names in os.walk(p):
                files += [os.path.join(root, n) for n in sorted(names)
                          if n.endswith(".py")]
        else:
            files.append(p)
    return files


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("paths", nargs="+", help="files or directories to rewrite")
    ap.add_argument("--check", action="store_true",
                    help="don't write; exit non-zero if anything would change")
    args = ap.parse_args(argv)

    changed, refusals, total = [], [], 0
    for path in _collect(args.paths):
        with open(path) as f:
            src = f.read()
        try:
            out, rw = rewrite_source(src, path)
        except cst.ParserSyntaxError as e:
            refusals.append(f"{path}: parse error: {e}")
            continue
        refusals += rw.refusals
        total += rw.rewrites
        if out != src:
            changed.append(path)
            if not args.check:
                with open(path, "w") as f:
                    f.write(out)

    verb = "would change" if args.check else "rewrote"
    print(f"{verb} {len(changed)} file(s); {total} assertion(s) converted")
    if refusals:
        print(f"\n{len(refusals)} construct(s) need manual conversion:")
        for r in refusals:
            print(f"  {r}")
    if args.check and changed:
        for c in changed:
            print(f"  would change: {c}")
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
