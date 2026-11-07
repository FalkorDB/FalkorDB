"""Harness for the root suites (e2e, functions, TCK, MVCC, concurrency).

Redis lifecycle is delegated to `tests.infra`, the same layer the flow
suite uses, so all suites resolve the module the same way and behave the
same against a local build, a spawned container, or an existing service.

The module-level `client` / `g` globals are the interface the suites
actually use (`common.g.query(...)`), so they are kept as-is.
"""

import os
import sys
import time
import warnings

from falkordb import FalkorDB

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from infra import Env, partition, teardown_all, uses_existing_service

env = None
client = None
g = None
shutdown = False
_saved_addr = None

GRAPH_ID = "test"


def start_redis(release=None, moduleEnvs=()):
    """Bring up a server and bind the `client` / `g` globals to it.

    `moduleEnvs` is the flat `[KEY, VALUE, ...]` argv form the suites
    already pass; `release` overrides the RELEASE environment variable for
    module lookup.
    """
    global env, client, g, shutdown, _saved_addr
    if release is not None:
        os.environ["RELEASE"] = "1" if release else "0"
    module_args = " ".join(moduleEnvs)
    if module_args and uses_existing_service():
        # The server is already up and its load-time args are fixed, so
        # load-time-only keys can't be applied — GRAPH.CONFIG SET rejects
        # them outright. Drop them rather than failing the whole module:
        # the suites that pass them already skip the tests that depend on
        # them in this mode (see the EXISTING_ENV skip in tests/test_e2e.py).
        mutable, immutable = partition(module_args)
        if immutable:
            warnings.warn(
                "ignoring load-time-only module args "
                + ", ".join(f"{k}={v}" for k, v in immutable)
                + " — attaching to an existing server, which was loaded with"
                " whatever values it was started with.",
                stacklevel=2)
        module_args = " ".join(f"{k} {v}" for k, v in mutable)
    env, client = Env(moduleArgs=module_args)
    # Several suites fan work out over multiprocessing pools. Those children
    # re-import this module from scratch (macOS spawns rather than forks), so
    # the address has to travel in the environment for falkordb() to find it.
    # CI sets these itself to point at the service container, so remember
    # what was there and put it back on shutdown.
    _saved_addr = (os.environ.get("FALKORDB_HOST"),
                   os.environ.get("FALKORDB_PORT"))
    os.environ["FALKORDB_HOST"] = str(env.host)
    os.environ["FALKORDB_PORT"] = str(env.port)
    g = client.select_graph(GRAPH_ID)
    shutdown = True


def falkordb():
    """A second client pointed at the running server.

    Bare `FalkorDB()` defaults to localhost:6379, which silently bypasses
    the docker-services CI mode (where redis runs as a sibling container).
    """
    if env is not None:
        return FalkorDB(host=env.host, port=env.port)
    return FalkorDB(
        host=os.environ.get("FALKORDB_HOST", "localhost"),
        port=int(os.environ.get("FALKORDB_PORT", os.environ.get("PORT", "6379"))),
    )


def shutdown_redis():
    global env, client, g, shutdown, _saved_addr
    teardown_all()
    env = client = g = None
    shutdown = False
    # Leaving our address set would point a later falkordb() at a dead
    # server; blanket-deleting would drop CI's service address.
    for name, prev in zip(("FALKORDB_HOST", "FALKORDB_PORT"),
                          _saved_addr or (None, None)):
        if prev is None:
            os.environ.pop(name, None)
        else:
            os.environ[name] = prev
    _saved_addr = None


def wait_for_indices_to_sync(graph):
    q = "CALL db.indexes() YIELD status WHERE status <> 'OPERATIONAL' RETURN count(1)"
    while True:
        result = graph.ro_query(q)
        if result.result_set[0][0] == 0:
            break
        time.sleep(0.5) # sleep 500ms
