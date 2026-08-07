#!/usr/bin/env python3
"""Reconcile a flow-test run against the native-index xfail ledger.

Under ``--features index-falkordb`` the native index is the only index, so predicates it cannot
serve fail loudly. The ledger (``nextgen_index_xfail.txt``) records which tests that affects.

The check runs in BOTH directions and either one fails the build:

* an **unexpected failure** — a test failed that the ledger does not list. Either a regression, or
  a gap nobody wrote down. Both need a human.
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

# `file:Class.test`, the identifier RLTest prints. Tolerates surrounding whitespace and the ANSI
# colour codes RLTest emits when it thinks it is on a terminal.
TEST_ID = re.compile(r"([A-Za-z0-9_]+):([A-Za-z0-9_]+)\.([A-Za-z0-9_]+)")
ANSI = re.compile(r"\x1b\[[0-9;]*m")


def read_ledger(path: str) -> tuple[set[str], set[str]]:
    """Return (expected_failures, ignored), skipping comments and blank lines.

    A line prefixed with ``!`` is *ignored in both directions* — neither required to fail nor
    required to pass. That exists for tests whose failure is not about the flag at all and is not
    reproducible everywhere (a harness bug on one platform, say): the normal check would be red on
    one environment whichever direction it took. It is deliberately not a general suppression
    mechanism — a bidirectional ledger only means something if entries have to justify themselves,
    so the file requires evidence that the flag is uninvolved before a line gets the prefix.
    """
    expected: set[str] = set()
    ignored: set[str] = set()
    with open(path, encoding="utf-8") as fh:
        for raw in fh:
            line = raw.split("#", 1)[0].strip()
            if not line:
                continue
            if line.startswith("!"):
                ignored.add(line[1:].strip())
            else:
                expected.add(line)
    return expected, ignored


def parse_results(text: str) -> tuple[set[str], set[str]]:
    """Return (failed, seen) test ids from RLTest output.

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
    """
    text = ANSI.sub("", text).replace("\r", "")
    failed: set[str] = set()
    seen: set[str] = set()
    skipped: set[str] = set()
    current: str | None = None

    for line in text.splitlines():
        stripped = line.strip()
        match = TEST_ID.fullmatch(stripped.rstrip(":"))
        if match:
            test_id = f"{match.group(1)}:{match.group(2)}.{match.group(3)}"
            if stripped.endswith(":"):
                seen.add(test_id)   # the test produced a result
                current = test_id
            else:
                failed.add(test_id)  # a summary entry
                seen.add(test_id)
                current = None
            continue
        # A skipped test prints the same `name:` header as a pass. Counting it as a pass would
        # report every ledgered-but-skipped test as "now green" and send someone to delete a
        # line for a test that never executed.
        if current is not None and "[SKIP]" in stripped:
            skipped.add(current)
            current = None
    return failed, seen - skipped


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--ledger", required=True)
    ap.add_argument("--results", help="RLTest output file; omit to read stdin")
    args = ap.parse_args()

    ledger, ignored = read_ledger(args.ledger)
    text = open(args.results, encoding="utf-8").read() if args.results else sys.stdin.read()
    failed, seen = parse_results(text)

    if not seen:
        print("error: no test results parsed — the run produced nothing to check", file=sys.stderr)
        return 2

    # `!`-prefixed entries drop out before either direction is computed.
    failed -= ignored
    seen -= ignored

    unexpected_failures = sorted(failed - ledger)
    # Only count a ledgered test as passing if the run actually executed it.
    unexpected_passes = sorted((ledger & seen) - failed)
    # A ledgered test that never ran is neither a pass nor a failure, so without this it
    # reconciles silently — and so does every other test in the suite that died with it.
    never_ran = sorted(ledger - seen)
    # The runner records each suite's exit status. >1 means the harness itself died rather
    # than merely reporting test failures, so its results are not trustworthy input.
    dead_suites = [
        line.split()[1]
        for line in text.splitlines()
        if line.startswith("SUITE_EXIT") and len(line.split()) == 3 and int(line.split()[2]) > 1
    ]

    if unexpected_failures:
        print("\nUNEXPECTED FAILURES — not in the ledger:")
        for t in unexpected_failures:
            print(f"  {t}")
        print(
            "\n  Either this is a regression, or it is a real gap that was never recorded.\n"
            "  Fix it, or add it to the ledger WITH the reason it cannot be served."
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
        print("\nSUITES THAT DIED (exit > 1) — their results are not usable:")
        for suite in dead_suites:
            print(f"  {suite}")

    if unexpected_failures or unexpected_passes or never_ran or dead_suites:
        return 1

    skipped = f", {len(ignored)} ignored" if ignored else ""
    print(
        f"ledger reconciled: {len(seen)} tests run, "
        f"{len(failed)} failed, all accounted for by {len(ledger)} ledger entries{skipped}"
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
