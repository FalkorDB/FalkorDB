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


def _hammer_under_write_load(command):
    """Repeatedly invoke `command(r)` on the main-thread client while 8 writer
    processes burst writes into graph "test". `command` must return (not hang)
    each call; a #726 lock-order deadlock (write loop holding L1 while
    acquiring the GIL vs. a main-thread command holding the GIL and waiting for
    L1) would hang the server, which the 30s socket timeout turns into a fast
    failure. Returns after all writers complete.
    """
    common.g.query("CREATE (:Node {id: -1})")
    r = Redis(
        host=os.environ.get("FALKORDB_HOST", "localhost"),
        port=int(os.environ.get("FALKORDB_PORT", os.environ.get("PORT", "6379"))),
        socket_timeout=30,
    )
    with Pool(processes=8) as pool:
        writers = pool.map_async(run_write_burst, range(8))
        while not writers.ready():
            command(r)
        assert sorted(writers.get()) == list(range(8))
    # >= 801 (800 burst writes + the initial -1 node); a hammering command that
    # itself writes (MULTI) may add more. The point of this guard is no hang.
    res = common.g.query("MATCH (n:Node) RETURN count(n)")
    assert res.result_set[0][0] >= 801


def test_slowlog_concurrent_with_writes():
    # GRAPH.SLOWLOG runs inline on the main thread (it is not moved to a
    # worker), so it directly exercises the two-phase write-loop fix: while it
    # holds the GIL and takes the per-graph read lock, the write loop must not
    # be holding the write lock while waiting for the GIL.
    _hammer_under_write_load(lambda r: r.execute_command("GRAPH.SLOWLOG", "test"))


def test_memory_concurrent_with_writes():
    # GRAPH.MEMORY is dispatched to a worker thread; it must not deadlock
    # against committing writes.
    _hammer_under_write_load(lambda r: r.execute_command("GRAPH.MEMORY", "USAGE", "test"))


def test_explain_concurrent_with_writes():
    # GRAPH.EXPLAIN is dispatched to a worker thread (this PR supersedes the
    # side-branch #735 fix). It must not deadlock against committing writes.
    _hammer_under_write_load(
        lambda r: r.execute_command(
            "GRAPH.EXPLAIN", "test", "MATCH (n:Node) RETURN n.id LIMIT 1"
        )
    )


def _create_index(r):
    # First call creates the index; later calls error ("already indexed") but
    # still plan as DDL and run through the GIL->L1-write DDL branch (eager
    # create_rs_index / GilGuard). We only care that the command returns rather
    # than hanging, so swallow the expected error.
    try:
        r.execute_command("GRAPH.QUERY", "test", "CREATE INDEX FOR (n:Node) ON (n.id)")
    except Exception:
        pass


def test_create_index_concurrent_with_writes():
    # DDL runs under GIL->L1-write with the GilGuard made a no-op; must not
    # deadlock or self-deadlock against concurrent committing writes.
    _hammer_under_write_load(_create_index)


def _multi_write(r):
    # A MULTI-wrapped write runs synchronously on the main thread (query_sync).
    # Before the #726 fix this DEADLOCKED against the pool write loop (query_sync
    # holds the GIL and waits for L1; the write loop holds L1 and waits for the
    # GIL). Now it either succeeds or, when it races the write loop for the MVCC
    # slot in the brief commit gap, returns a retryable "write lock unavailable"
    # (the client is expected to retry). Either way the server does not hang —
    # which is what this test guards. Swallow only that expected transient error.
    try:
        p = r.pipeline(transaction=True)
        p.execute_command("GRAPH.QUERY", "test", "CREATE (:Node {id: -2})")
        p.execute()
    except Exception as e:  # noqa: BLE001 - narrow check below
        if "write lock unavailable" not in str(e):
            raise


def test_multi_write_concurrent_with_writes():
    _hammer_under_write_load(_multi_write)


def _profile_write(r):
    # GRAPH.PROFILE of a write now routes through the SAME write queue as
    # GRAPH.QUERY (two-phase GIL->L1 commit), instead of the old bespoke path
    # that held L1-write across execute+commit. Under the two-phase change that
    # old path could also panic on a busy MVCC slot (.expect). It must return,
    # not hang or crash, under concurrent write load.
    r.execute_command("GRAPH.PROFILE", "test", "CREATE (:Node {id: -3})")


def test_profile_write_concurrent_with_writes():
    _hammer_under_write_load(_profile_write)


def _profile_ddl(r):
    # GRAPH.PROFILE of DDL (CREATE INDEX) is the case that previously took the
    # GIL under L1-write on the background profile path — an uncovered #726
    # inversion (the profile path installed no L1HeldScope, so the assertion
    # could not even catch it). It now runs through the write loop's
    # GIL->L1-write DDL branch. Swallow the expected "already indexed" after the
    # first success; we only assert the command returns rather than hanging.
    try:
        r.execute_command("GRAPH.PROFILE", "test", "CREATE INDEX FOR (n:Node) ON (n.v)")
    except Exception:
        pass


def test_profile_ddl_concurrent_with_writes():
    _hammer_under_write_load(_profile_ddl)