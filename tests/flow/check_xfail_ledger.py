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


def read_ledger(path: str) -> set[str]:
    """Ledger entries, ignoring comments and blank lines."""
    entries = set()
    with open(path, encoding="utf-8") as fh:
        for raw in fh:
            line = raw.split("#", 1)[0].strip()
            if line:
                entries.add(line)
    return entries


def parse_results(text: str) -> tuple[set[str], set[str]]:
    """Return (failed, seen) test ids from RLTest output.

    RLTest prints a test id twice, and the two forms are distinguishable without tracking any
    state:

    * ``\ttest_file:Class.test:``  — trailing colon; the test is *starting*. Followed by
      ``[PASS]``, ``[FAIL]`` or ``[ERROR]``.
    * ``\ttest_file:Class.test``   — no trailing colon; an entry in a "Failed Tests Summary"
      block, followed by an indented reason.

    Keying off the trailing colon rather than the summary header matters when several suites are
    concatenated into one log: the header appears once per *suite*, so a parser that latches on it
    treats every later suite's output as failures. Matching on shape is immune to that. It also
    catches ``[ERROR]`` failures (an unhandled exception), which never print ``[FAIL]`` at all.
    """
    text = ANSI.sub("", text).replace("\r", "")
    failed: set[str] = set()
    seen: set[str] = set()

    for line in text.splitlines():
        stripped = line.strip()
        match = TEST_ID.fullmatch(stripped.rstrip(":"))
        if not match:
            continue
        test_id = f"{match.group(1)}:{match.group(2)}.{match.group(3)}"
        if stripped.endswith(":"):
            seen.add(test_id)       # the test ran
        else:
            failed.add(test_id)     # a summary entry
            seen.add(test_id)
    return failed, seen


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--ledger", required=True)
    ap.add_argument("--results", help="RLTest output file; omit to read stdin")
    args = ap.parse_args()

    ledger = read_ledger(args.ledger)
    text = open(args.results, encoding="utf-8").read() if args.results else sys.stdin.read()
    failed, seen = parse_results(text)

    if not seen:
        print("error: no test results parsed — the run produced nothing to check", file=sys.stderr)
        return 2

    unexpected_failures = sorted(failed - ledger)
    # Only count a ledgered test as passing if the run actually executed it.
    unexpected_passes = sorted((ledger & seen) - failed)

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

    if unexpected_failures or unexpected_passes:
        return 1

    print(
        f"ledger reconciled: {len(seen)} tests run, "
        f"{len(failed)} failed, all accounted for by {len(ledger)} ledger entries"
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
