import os
import common
from falkordb import FalkorDB
from multiprocessing import Pool
from redis import Redis


def setup_module(module):
    global is_extra
    from conftest import pytest_config

    is_extra = "extra" in pytest_config.getoption("-m")
    common.start_redis()


def teardown_module(module):
    common.shutdown_redis()


def setup_function(function):
    if common.g.name in common.client.list_graphs():
        common.g.delete()

def run_write(id):
    db = common.falkordb()
    g = db.select_graph("test")
    res = g.query("CREATE (n:Node {id: $id})", params={"id": id})
    version = int(res._raw_stats[-1][15:])
    return (id, version)

def test_concurrent_writes():
    with Pool(processes=8) as pool:
        write_results = pool.map(run_write, range(0, 1000))

    versions = [version for _, version in write_results]
    assert len(set(versions)) == 1000

    res = common.g.query("MATCH (n:Node) RETURN count(DISTINCT n.id)")
    assert res.result_set[0][0] == 1000

def run_write_burst(id):
    db = common.falkordb()
    g = db.select_graph("test")
    for i in range(100):
        g.query("CREATE (n:Node {id: $id})", params={"id": id * 100 + i})
    return id

def test_record_concurrent_with_writes():
    # GRAPH.RECORD used to execute synchronously on the main thread: it held
    # the GIL while waiting for the ThreadedGraph read lock, while a
    # committing write query held the write lock and waited for the GIL —
    # deadlocking the whole server. The socket timeout makes the old code
    # fail fast instead of hanging the test run forever.
    common.g.query("CREATE (:Node {id: -1})")
    r = Redis(
        host=os.environ.get("FALKORDB_HOST", "localhost"),
        port=int(os.environ.get("FALKORDB_PORT", os.environ.get("PORT", "6379"))),
        socket_timeout=30,
    )
    with Pool(processes=8) as pool:
        writers = pool.map_async(run_write_burst, range(8))
        while not writers.ready():
            r.execute_command("GRAPH.RECORD", "test", "MATCH (n:Node) RETURN n.id LIMIT 1")
        assert sorted(writers.get()) == list(range(8))

    res = common.g.query("MATCH (n:Node) RETURN count(n)")
    assert res.result_set[0][0] == 801