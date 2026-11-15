#!/usr/bin/env python3
"""Reconcile a flow-test run against the native-index xfail ledger.

Under ``--features index-falkordb`` the native index is the only index, so predicates it cannot
serve fail loudly. The ledger (``nextgen_index_xfail.txt``) records which tests that affects.

The check runs in BOTH directions and either one fails the build:

* an **unexpected failure** — a test failed that the ledger does not list, or a listed test failed
  in a way its group does not name. Either a regression, or a gap nobody wrote down. Both need a
  human.
* an **unexpected pass** — a ledgered test now passes. The fix must delete its line, in the same
  PR that fixed it. Without this direction the ledger silently rots into a list of things that
  quietly started working, and it stops meaning anything.

Usage::

    check_xfail_ledger.py --ledger nextgen_index_xfail.txt --results results.txt
    ... | check_xfail_ledger.py --ledger nextgen_index_xfail.txt   # results on stdin

``--results`` is RLTest console output; the parser reads its "Failed Tests Summary" block and the
per-test result lines, so it works on the combined output of several suites.
"""

from __future__ import annotations

import argparse
import re
import sys
from typing import NamedTuple

# `file:Class.test`, the identifier RLTest prints. Tolerates surrounding whitespace and the ANSI
# colour codes RLTest emits when it thinks it is on a terminal.
TEST_ID = re.compile(r"([A-Za-z0-9_]+):([A-Za-z0-9_]+)\.([A-Za-z0-9_]+)")
# A summary entry for a whole class, not one test: RLTest records a class whose `__init__` raised
# as `file:Class [__init__]`, and a Redis that did not exit cleanly under the class's name. Neither
# names a test, so neither can be ledgered, and neither may be read as "nothing failed". Anchored
# on the `test_` file and `test` class prefixes, so a `key:value` line a test prints is not one.
# RLTest takes any class whose name starts with `test` in any case, so the class prefix does too.
CLASS_ENTRY = re.compile(r"(test_[A-Za-z0-9_]+):((?i:test)[A-Za-z0-9_]+)( \[__init__\])?")
ANSI = re.compile(r"\x1b\[[0-9;]*m")
# What the native index says when it cannot serve a predicate; group 1 is the predicate's shape,
# the part a ledger group names.
NATIVE_ERROR = re.compile(
    r"native index cannot serve this predicate on (?:node|relationship) :\S* — (.*?)\. "
    r"The `index-falkordb` build"
)
# An assertion RLTest reports, `(FAIL):\t<message>\t<file>:<line>`, printed above the verdict of
# the test that made it. Group 1 is the message.
ASSERTION = re.compile(r"\(FAIL\):\s*(.*?)\s+\S+\.py:\d+$")


def shape_pattern(shape: str) -> re.Pattern[str]:
    """Compile a ledger `error:` or `assertion:` pattern. A `*` stands for one name or value
    without spaces, so `` `*` `` matches any attribute in backticks; everything else is literal."""
    return re.compile(re.escape(shape).replace(r"\*", r"[^`\s]*"))


class Group(NamedTuple):
    """What one ledger group's tests fail with: the shapes the native index's error names
    (``error:``), and the messages of assertions that fail without such an error
    (``assertion:``)."""

    errors: list[re.Pattern[str]]
    assertions: list[re.Pattern[str]]

    def accounts_for(self, failure: str) -> bool:
        """Whether `failure`, an exception or an assertion message, is one this group names."""
        if shape := NATIVE_ERROR.search(failure):
            return any(p.fullmatch(shape.group(1)) for p in self.errors)
        return any(p.fullmatch(failure) for p in self.assertions)


def read_ledger(path: str) -> tuple[dict[str, Group], set[str]]:
    """Return (expected_failures, ignored), skipping comments and blank lines.

    Each expected failure maps to its group: one or more ``error: <shape>`` and
    ``assertion: <message>`` lines followed by its entries. ``<shape>`` is what the native index's
    error says it cannot serve; ``<message>`` is an assertion that fails without such an error, as
    RLTest prints it; ``*`` stands for a name. A blank line after the entries ends the group, as
    does the next such line. An entry outside any group cannot say what it fails with, so it is an
    error.

    A line prefixed with ``!`` is *ignored in both directions* — neither required to fail nor
    required to pass. That exists for tests whose failure is not about the flag at all and is not
    reproducible everywhere (a harness bug on one platform, say): the normal check would be red on
    one environment whichever direction it took. It is deliberately not a general suppression
    mechanism — a bidirectional ledger only means something if entries have to justify themselves,
    so the file requires evidence that the flag is uninvolved before a line gets the prefix.
    """
    expected: dict[str, Group] = {}
    ignored: set[str] = set()
    group = Group([], [])
    group_has_entries = False
    with open(path, encoding="utf-8") as fh:
        for number, raw in enumerate(fh, 1):
            if group_has_entries and not raw.strip():
                group, group_has_entries = Group([], []), False
            line = raw.split("#", 1)[0].strip()
            if not line:
                continue
            if line.startswith(("error:", "assertion:")):
                if group_has_entries:
                    group, group_has_entries = Group([], []), False
                kind, _, pattern = line.partition(":")
                target = group.errors if kind == "error" else group.assertions
                target.append(shape_pattern(pattern.strip()))
            elif line.startswith("!"):
                ignored.add(line[1:].strip())
            elif not group.errors and not group.assertions:
                sys.exit(f"{path}:{number}: `{line}` has no `error:` or `assertion:` line above it")
            else:
                expected[line] = group
                group_has_entries = True
    return expected, ignored


def parse_results(text: str) -> tuple[set[str], set[str], set[str], dict[str, list[str]]]:
    """Return (failed, seen, class_failures, failures) from RLTest output.

    RLTest prints a test id twice, and the two forms are distinguishable without tracking any
    state:

    * ``\ttest_file:Class.test:`` — trailing colon; RLTest's ``printPass`` / ``printFail`` /
      ``printError`` / ``printSkip`` all emit ``'%s:\\r\\n\\t%s'``, so this line is printed when the
      test *finishes*, immediately above its verdict. It means "this test produced a result".
    * ``\ttest_file:Class.test``  — no trailing colon; an entry in a "Failed Tests Summary"
      block, followed by an indented reason.

    Keying off the trailing colon rather than the summary header matters when several suites are
    concatenated into one log: the header appears once per *suite*, so a parser that latches on it
    treats every later suite's output as failures. Matching on shape is immune to that. It also
    catches ``[ERROR]`` failures (an unhandled exception), which never print ``[FAIL]`` at all.

    ``fullmatch`` is deliberate and safe. The only decoration RLTest puts on these lines is the
    ANSI colouring (stripped above) and, for class methods, a leading tab from
    ``_runTest(prefix='\\t')`` — ``.strip()`` removes it. ``msgPrefix`` never reaches stdout; it is
    only a fallback key for recording the failure. Checked against a full 78 KB six-suite run:
    zero lines carrying a test id are rejected.

    A ``[FAIL]`` or ``[ERROR]`` verdict counts as a failure on its own. The summary block is
    printed only when RLTest finishes, so a runner that dies after a test failed never lists it
    there.

    ``failures`` maps a test to what it failed with: each assertion RLTest printed above its
    verdict (``(FAIL): <message>``), and the exception it prints on the line after an ``[ERROR]``
    (``Unhandled exception: ...``). The summary only says "Exception raised during test
    execution", so that line is the one place the error is named.
    """
    text = ANSI.sub("", text).replace("\r", "")
    failed: set[str] = set()
    seen: set[str] = set()
    skipped: set[str] = set()
    class_failures: set[str] = set()
    failures: dict[str, list[str]] = {}
    current: str | None = None
    # The test whose `[ERROR]` was the last verdict, until its exception line or the next test.
    raised: str | None = None
    # Assertions printed since the last test line: they belong to the next test to finish.
    asserted: list[str] = []

    for line in text.splitlines():
        stripped = line.strip()
        match = TEST_ID.fullmatch(stripped.rstrip(":"))
        if match:
            raised = None
            test_id = f"{match.group(1)}:{match.group(2)}.{match.group(3)}"
            if stripped.endswith(":"):
                seen.add(test_id)   # the test produced a result
                current = test_id
                if asserted:
                    failures.setdefault(test_id, []).extend(asserted)
            else:
                failed.add(test_id)  # a summary entry
                seen.add(test_id)
                current = None
            asserted = []
            continue
        # The summary repeats each failure's assertions; none of them belongs to a later test.
        if stripped.startswith(("Total Tests Run:", "SUITE_EXIT")):
            asserted = []
        if CLASS_ENTRY.fullmatch(stripped):
            class_failures.add(stripped)
            continue
        if assertion := ASSERTION.search(stripped):
            asserted.append(assertion.group(1))
            continue
        if raised is not None and stripped.startswith("Unhandled exception:"):
            failures.setdefault(raised, []).append(
                stripped.removeprefix("Unhandled exception:").strip()
            )
            raised = None
            continue
        if current is None:
            continue
        # A skipped test prints the same `name:` header as a pass. Counting it as a pass would
        # report every ledgered-but-skipped test as "now green" and send someone to delete a
        # line for a test that never executed.
        if "[SKIP]" in stripped:
            skipped.add(current)
            current = None
        elif "[FAIL]" in stripped or "[ERROR]" in stripped:
            failed.add(current)
            if "[ERROR]" in stripped:
                raised = current
            current = None
        elif "[PASS]" in stripped:
            current = None
    return failed, seen - skipped, class_failures, failures


def split_suites(text: str) -> list[tuple[str, int, str]]:
    """Split the runner's log into (suite, exit status, output), one per ``SUITE_EXIT`` line.

    Output after the last ``SUITE_EXIT`` line belongs to no suite and is not returned.
    """
    suites: list[tuple[str, int, str]] = []
    chunk: list[str] = []
    for line in text.splitlines():
        parts = line.split()
        if len(parts) == 3 and parts[0] == "SUITE_EXIT" and parts[2].isdigit():
            suites.append((parts[1], int(parts[2]), "\n".join(chunk)))
            chunk = []
        else:
            chunk.append(line)
    return suites


def suite_problem(status: int, output: str) -> str | None:
    """Say why one suite's output cannot be reconciled, or return None if it can.

    Every suite needs RLTest's closing summary, proof that the runner finished: a suite that stops
    early, whatever its exit status, leaves its later tests unrun and unreported. A suite that
    exits 1 has failing tests, and the ledger can only account for failures it can read, so it
    also needs at least one failure the parser recognises. An import error exits 1 with neither.
    """
    if status > 1:
        return f"exit {status}: the harness died rather than reporting test failures"
    failed, seen, class_failures, _ = parse_results(output)
    if not seen and not class_failures:
        return "reported no test results"
    if "Total Tests Run:" not in output:
        return f"exited {status} before RLTest printed its summary"
    if status == 0:
        return None
    if not failed and not class_failures:
        return "exited 1 but reported no failure this check can read"
    return None


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--ledger", required=True)
    ap.add_argument("--results", help="RLTest output file; omit to read stdin")
    args = ap.parse_args()

    ledger, ignored = read_ledger(args.ledger)
    if args.results:
        with open(args.results, encoding="utf-8") as fh:
            text = fh.read()
    else:
        text = sys.stdin.read()
    failed, seen, class_failures, failures = parse_results(text)

    if not seen and not class_failures:
        print("error: no test results parsed — the run produced nothing to check", file=sys.stderr)
        return 2

    # `!`-prefixed entries drop out before either direction is computed.
    failed -= ignored
    seen -= ignored

    listed = set(ledger)
    # A class-level failure names no test, so no ledger line can account for it.
    unexpected_failures = sorted(failed - listed) + sorted(class_failures)
    # A listed test is accounted for only by what its group names: each of its failures has to be
    # one. Any other — another gap or a regression reached first, or a different assertion — is
    # not the failure the ledger recorded.
    wrong_errors = []
    for test in sorted(failed & listed):
        texts = failures.get(test, [])
        unaccounted = [t for t in texts if not ledger[test].accounts_for(t)]
        if not texts:
            wrong_errors.append((test, "no failure message this check can read"))
        elif unaccounted:
            wrong_errors.append((test, unaccounted[0]))
    # Only count a ledgered test as passing if the run actually executed it.
    unexpected_passes = sorted((listed & seen) - failed)
    # A ledgered test that never ran is neither a pass nor a failure, so without this it
    # reconciles silently — and so does every other test in the suite that died with it.
    never_ran = sorted(listed - seen)
    # The runner records each suite's exit status. Checking each suite on its own matters: a
    # suite with no ledger entries that dies on import leaves nothing behind, and the other
    # suites' results would otherwise reconcile without it.
    dead_suites = [
        f"{suite}: {problem}"
        for suite, status, output in split_suites(text)
        if (problem := suite_problem(status, output)) is not None
    ]

    if unexpected_failures:
        print("\nUNEXPECTED FAILURES — not in the ledger:")
        for t in unexpected_failures:
            print(f"  {t}")
        print(
            "\n  Either this is a regression, or it is a real gap that was never recorded.\n"
            "  Fix it, or add it to the ledger WITH the reason it cannot be served."
        )

    if wrong_errors:
        print("\nLEDGERED, BUT FAILING IN A WAY ITS GROUP DOES NOT NAME:")
        for test, error in wrong_errors:
            print(f"  {test}\n    {error}")
        print(
            "\n  The ledger accounts for a test only by the failure it recorded. Fix whatever this\n"
            "  test now hits first, or move the line to the group that names it."
        )

    if unexpected_passes:
        print("\nUNEXPECTED PASSES — ledgered but now green:")
        for t in unexpected_passes:
            print(f"  {t}")
        print(
            "\n  Delete these lines from the ledger, in the PR that made them pass.\n"
            "  A ledger that keeps entries for working tests stops describing anything."
        )

    if never_ran:
        print("\nLEDGERED BUT NEVER RAN:")
        for t in never_ran:
            print(f"  {t}")
        print(
            "\n  These cannot be reconciled — the test did not report a result. Usually the suite\n"
            "  died early, which also hides every other test in it. Fix the run, do not delete\n"
            "  the lines."
        )

    if dead_suites:
        print("\nSUITES THAT CANNOT BE RECONCILED — their results are not usable:")
        for suite in dead_suites:
            print(f"  {suite}")

    if unexpected_failures or wrong_errors or unexpected_passes or never_ran or dead_suites:
        return 1

    skipped = f", {len(ignored)} ignored" if ignored else ""
    print(
        f"ledger reconciled: {len(seen)} tests run, "
        f"{len(failed)} failed, all accounted for by {len(ledger)} ledger entries{skipped}"
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
