#!/usr/bin/env python3
"""Count passed, failed and skipped tests per server, for the CI run summary.

Each CI cell that runs a suite records what it ran:

    count_results.py record --server valkey --run "release flow (image, amd64)" \\
        --rltest flow.log --logs tests/flow/logs --status failure --out counts/x.json
    count_results.py record --server redis --run "image" \\
        --junit e2e.xml tck.xml --logs tests/flow/logs --out counts/y.json

and one job adds them up:

    count_results.py summary counts/ >> "$GITHUB_STEP_SUMMARY"

A record is a JSON list of rows: server, the version its start-up banner names,
the run (how the cell ran), the suite, and the counts. RLTest output gives the
counts from its per-test verdict lines (`[PASS]`, `[FAIL]`, `[ERROR]`,
`[SKIP]`); pytest gives them through `--junitxml`, split by test file into the
e2e, function, MVCC and concurrency suites, and TCK.

A cell whose test step failed but recorded no failed test (the harness died, a
class's `__init__` raised, the server crashed between tests) records one failed
test named for the cell, so a broken cell never reads as a clean one.
"""

import argparse
import json
import os
import re
import sys
import xml.etree.ElementTree as ET
from collections import defaultdict

ANSI = re.compile(r"\x1b\[[0-9;]*m")
# The banner every Redis or Valkey start logs, e.g.
# `Valkey version=9.1.2, bits=64, commit=00000000, modified=0, pid=9, just started`.
BANNER = re.compile(r"\b(Valkey|Redis) version=([0-9][0-9A-Za-z.\-]*)")
VERDICTS = {"[PASS]": "passed", "[FAIL]": "failed", "[ERROR]": "failed", "[SKIP]": "skipped"}
# pytest file name -> the suite name the issue tracker and the summary use.
PYTEST_SUITES = {
    "test_e2e": "e2e",
    "test_functions": "function",
    "test_mvcc": "MVCC",
    "test_concurrency": "concurrency",
    "test_tck": "TCK",
}


def empty():
    return {"passed": 0, "failed": 0, "skipped": 0}


def count_rltest(text):
    """Counts from RLTest output. RLTest prints each test's verdict on a line of
    its own, right under the test's name, so counting those lines counts tests;
    the failure summary at the end repeats names, not verdicts."""
    counts = empty()
    for line in ANSI.sub("", text).splitlines():
        key = VERDICTS.get(line.strip())
        if key:
            counts[key] += 1
    return counts


def suite_of(case):
    """The suite a pytest testcase belongs to, from its file or class name."""
    where = case.get("file") or case.get("classname") or ""
    for part in re.split(r"[./\\]", where):
        if part in PYTEST_SUITES:
            return PYTEST_SUITES[part]
    return "pytest"


def count_junit(paths):
    """Counts per suite from pytest --junitxml files."""
    suites = defaultdict(empty)
    for path in paths:
        if not os.path.exists(path):
            print(f"count_results: {path} does not exist, skipped", file=sys.stderr)
            continue
        try:
            root = ET.parse(path).getroot()
        except ET.ParseError as e:
            # A truncated or empty file (pytest killed mid-run): report it and
            # count nothing from it, so the caller's --status failure fallback
            # counts the cell instead of the record crashing.
            print(f"count_results: {path} is not valid JUnit XML ({e}), skipped", file=sys.stderr)
            continue
        for case in root.iter("testcase"):
            counts = suites[suite_of(case)]
            if case.find("skipped") is not None:
                counts["skipped"] += 1
            elif case.find("failure") is not None or case.find("error") is not None:
                counts["failed"] += 1
            else:
                counts["passed"] += 1
    return dict(suites)


def banner_versions(paths):
    """`{server: {versions}}` from the start-up banners in the given files and
    directories, the same banners tests/assert_server_log.sh checks."""
    found = defaultdict(set)
    files = []
    for path in paths:
        if os.path.isdir(path):
            for root, _, names in os.walk(path):
                files.extend(os.path.join(root, n) for n in names)
        elif os.path.isfile(path):
            files.append(path)
    for name in files:
        try:
            with open(name, errors="replace") as f:
                for m in BANNER.finditer(f.read()):
                    found[m.group(1).lower()].add(m.group(2))
        except OSError:
            continue
    return found


def record(args):
    if args.rltest:
        text = ""
        if os.path.exists(args.rltest):
            with open(args.rltest, errors="replace") as f:
                text = f.read()
        else:
            print(f"count_results: {args.rltest} does not exist", file=sys.stderr)
        suites = {args.suite or "flow": count_rltest(text)}
    else:
        suites = count_junit(args.junit)

    if args.status == "failure" and not any(c["failed"] for c in suites.values()):
        # The step failed, but no test did: count the cell itself as a failure,
        # under the suite the cell ran (RLTest), or a row of its own (pytest,
        # where it can't be told which file the harness died in).
        name = (args.suite or "flow") if args.rltest else "harness (step failed, no test did)"
        suites.setdefault(name, empty())["failed"] += 1

    versions = banner_versions(args.logs)
    version = ", ".join(sorted(versions.get(args.server, ()))) or "not logged"
    rows = [
        {"server": args.server, "version": version, "run": args.run, "suite": suite, **counts}
        for suite, counts in sorted(suites.items())
    ]
    os.makedirs(os.path.dirname(os.path.abspath(args.out)), exist_ok=True)
    with open(args.out, "w") as f:
        json.dump(rows, f, indent=1)
    for row in rows:
        print(f"{row['server']} {row['version']} | {row['run']} | {row['suite']}: "
              f"{row['passed']} passed, {row['failed']} failed, {row['skipped']} skipped")
    return 0


def summary(args):
    rows = []
    for root, _, names in os.walk(args.dir):
        for name in sorted(names):
            if name.endswith(".json"):
                with open(os.path.join(root, name)) as f:
                    rows.extend(json.load(f))

    out = ["## Test results per server", ""]
    if not rows:
        out.append("No suite recorded any results in this run.")
        print("\n".join(out))
        return 0

    # (server, run, suite) -> counts, plus the versions each server's banners named.
    table = defaultdict(empty)
    versions = defaultdict(set)
    cells = defaultdict(int)
    for row in rows:
        key = (row["server"], row["run"], row["suite"])
        for k in ("passed", "failed", "skipped"):
            table[key][k] += row[k]
        cells[key] += 1
        if row.get("version") and row["version"] != "not logged":
            versions[row["server"]].update(v.strip() for v in row["version"].split(","))

    for server in sorted({k[0] for k in table}):
        named = ", ".join(sorted(versions[server])) or "version not logged"
        out += [f"### {server.capitalize()} ({named})", "",
                "| Run | Suite | Cells | Passed | Failed | Skipped |",
                "|---|---|---:|---:|---:|---:|"]
        total = empty()
        for (srv, run, suite), counts in sorted(table.items()):
            if srv != server:
                continue
            out.append(f"| {run} | {suite} | {cells[(srv, run, suite)]} | {counts['passed']} "
                       f"| {counts['failed']} | {counts['skipped']} |")
            for k in total:
                total[k] += counts[k]
        out += [f"| **all runs** | | | **{total['passed']}** | **{total['failed']}** "
                f"| **{total['skipped']}** |", ""]
    print("\n".join(out))
    return 0


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    sub = parser.add_subparsers(dest="command", required=True)

    rec = sub.add_parser("record", help="record one cell's counts")
    rec.add_argument("--server", required=True, choices=["redis", "valkey"])
    rec.add_argument("--run", required=True, help="how the cell ran, e.g. 'release flow (image, amd64)'")
    rec.add_argument("--suite", help="suite name for RLTest output (default: flow)")
    src = rec.add_mutually_exclusive_group(required=True)
    src.add_argument("--rltest", help="RLTest (flow.sh) output")
    src.add_argument("--junit", nargs="+", help="pytest --junitxml files")
    rec.add_argument("--logs", nargs="*", default=[], help="server log files or directories")
    rec.add_argument("--status", default="success", help="the test step's outcome")
    rec.add_argument("--out", required=True, help="JSON file to write")
    rec.set_defaults(func=record)

    summ = sub.add_parser("summary", help="add up recorded counts per server, as Markdown")
    summ.add_argument("dir", help="directory of recorded JSON files")
    summ.set_defaults(func=summary)

    args = parser.parse_args(argv)
    return args.func(args)


if __name__ == "__main__":
    sys.exit(main())
