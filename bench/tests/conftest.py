"""Every repository file these tests read must be in perf-contract.yml's `paths:`
filter: otherwise a pull request that changes only that file (a workflow a test
parses, say) skips the tests that read it.

An audit hook records each file the run opens inside the repository. At the end
of the run, a file the filter does not match fails the run, and the summary
names it.
"""

from __future__ import annotations

import os
import re
import sys
from pathlib import Path

import pytest
import yaml

REPO = os.path.realpath(Path(__file__).parents[2])
FILTER_WORKFLOW = Path(REPO, ".github", "workflows", "perf-contract.yml")
_opened: set[str] = set()
_unfiltered: list[str] = []


def _record_open(event: str, args: tuple) -> None:
    if event != "open" or not isinstance(args[0], (str, bytes, os.PathLike)):
        return  # os.open on a file descriptor, or another event
    path = os.path.realpath(os.fsdecode(args[0]))
    if not path.startswith(REPO + os.sep):
        return
    relative = os.path.relpath(path, REPO).replace(os.sep, "/")
    if not {".venv", "__pycache__"} & set(relative.split("/")):  # the environment, not inputs
        _opened.add(relative)


sys.addaudithook(_record_open)


def _glob(pattern: str) -> re.Pattern:
    """A GitHub path filter pattern: `**` crosses directories, `*` and `?` do not."""
    if pattern.startswith("!"):
        raise ValueError(f"negated path filter {pattern!r}: teach conftest.py about it")
    regex = []
    for part in re.split(r"(\*\*/|\*\*|\*|\?)", pattern):
        regex.append(
            {"**/": "(?:.*/)?", "**": ".*", "*": "[^/]*", "?": "[^/]"}.get(part, re.escape(part))
        )
    return re.compile("".join(regex))


def paths_filter() -> list[re.Pattern]:
    workflow = yaml.safe_load(FILTER_WORKFLOW.read_text(encoding="utf-8"))
    on = workflow.get("on", workflow.get(True))  # PyYAML reads the key `on` as True
    return [_glob(pattern) for pattern in on["pull_request"]["paths"]]


def pytest_sessionfinish(session):
    patterns = paths_filter()
    _unfiltered[:] = sorted(p for p in _opened if not any(r.fullmatch(p) for r in patterns))
    if _unfiltered:
        session.exitstatus = pytest.ExitCode.TESTS_FAILED


def pytest_terminal_summary(terminalreporter):
    if _unfiltered:
        terminalreporter.section("perf-contract.yml paths filter", red=True)
        terminalreporter.write_line(
            "these tests read files that .github/workflows/perf-contract.yml's "
            "`paths:` filter does not match; add them there:"
        )
        for path in _unfiltered:
            terminalreporter.write_line(f"  {path}")
