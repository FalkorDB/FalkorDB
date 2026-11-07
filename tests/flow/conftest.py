"""Pytest wiring for the flow suite.

Two things happen here that RLTest used to do for us: reclaiming the servers
a test class started, and making `from common import *` resolve.
"""

import os
import sys

import pytest

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from infra import teardown_all  # noqa: E402  (needs the sys.path above)

LOG_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "logs")


@pytest.fixture(scope="class", autouse=True)
def _reclaim_environments():
    """Stop every server a test class started, however it was created.

    Class-scoped and autouse so no test file has to remember it. It drains
    the registry in tests/infra rather than looking at `cls.env`, which is
    what makes it correct for the classes that swap their environment
    part-way through (test_timeout, test_cache, test_config, test_graph_copy):
    the superseded server is registered too, so it gets reclaimed as well.
    """
    yield
    teardown_all()


def pytest_sessionstart(session):
    os.makedirs(LOG_DIR, exist_ok=True)
