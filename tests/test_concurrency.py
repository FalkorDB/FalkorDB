import common
from falkordb import FalkorDB
from multiprocessing import Pool


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