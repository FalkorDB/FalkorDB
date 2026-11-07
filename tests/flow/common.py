"""Compatibility surface for the flow test suite.

Every flow test does `from common import *`, so this module is the one
import they all share. The orchestration itself lives in `tests/infra`
(shared with the e2e/TCK/MVCC suites); this file only re-exports it
alongside the falkordb/redis names the tests expect.
"""

import os
import sys
import time
from functools import wraps

import redis
from redis import ResponseError
from falkordb import FalkorDB, Graph, Node, Edge, Path, ExecutionPlan

sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(
    os.path.abspath(__file__)))))

import pytest

from base import FlowTestsBase
from infra import (NO_RETRY, SOCKET_TIMEOUT, Env, FalkorEnv, mountable_mkdtemp,
                   mountable_mkstemp, teardown_all)

SANITIZER     = os.getenv('SANITIZER', '')      != ''
CODE_COVERAGE = os.getenv('CODE_COVERAGE', '0') == '1'

# Normalized OS name for cross-platform test guards (matches the C FalkorDB
# convention used by tests/memcheck): "macos", "linux", "windows", or the
# raw sys.platform value if unrecognized.
OS = {'darwin': 'macos', 'linux': 'linux', 'win32': 'windows'}.get(sys.platform, sys.platform)


def skip(reason="unconditionally skipped"):
    """Decorator form of an unconditional skip, kept for existing call sites.

    Deliberately narrower than RLTest's `skip`, which also took `cluster=`,
    `macos=`, `asan=` and friends: a conditional skip belongs in
    `@pytest.mark.skipif`. Passing one of those keywords raises TypeError
    rather than silently skipping under the wrong condition.
    """
    def decorate(f):
        @wraps(f)
        def wrapper(*args, **kwargs):
            pytest.skip(reason)
        return wrapper
    return decorate


__all__ = [
    "Env", "FalkorEnv", "teardown_all", "FlowTestsBase",
    "FalkorDB", "Graph", "Node", "Edge", "Path", "ExecutionPlan",
    "redis", "ResponseError", "pytest",
    "SANITIZER", "CODE_COVERAGE", "OS", "SOCKET_TIMEOUT", "NO_RETRY",
    "mountable_mkdtemp", "mountable_mkstemp", "skip",
    "os", "sys", "time",
]
