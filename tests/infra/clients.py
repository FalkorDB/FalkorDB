"""Redis/FalkorDB client construction policy shared by every backend.

Every client handed to a test is built here, so the deadline and retry policy
are stated in exactly one place.
"""

import redis
from falkordb import FalkorDB
from redis.backoff import NoBackoff
from redis.retry import Retry

# Per-socket read timeout (seconds) applied to redis clients constructed by
# the test infrastructure. Tests that issue heavy synthetic workloads
# (test_index_create.test08_async_index_creation — 1 M-node CREATE,
# test_slowlog.populate_slowlog — 19 concurrent UNWIND(250k),
# test_pending_queries_limit — 16 concurrent UNWIND(1M)) run well past any
# short deadline, and under ASAN/coverage instrumentation the
# GraphBLAS+falkordb binary is another 3-5x slower (see PR #539).
#
# An *unset* value is not "no deadline": redis-py >= 6 defaults
# socket_timeout to 5s. Combined with its default retrying Retry policy that
# is worse than a plain timeout — the command is silently re-issued on a new
# connection. A re-issued WAIT is the dangerous one: WAIT compares against
# the calling connection's own write offset, which is 0 on a fresh
# connection, so it returns "acked" while the replica is still behind (this
# is what let test_effects.test17_random_ops compare a stale replica). A
# re-issued write can also apply twice. So state the deadline explicitly and
# pair it with NO_RETRY below.
SOCKET_TIMEOUT = 600

# Retry policy for every client handed to a test: none. A tripped deadline
# must surface as a TimeoutError naming the slow command, not be quietly
# re-run. Connect-time retries are separately disabled where we probe for
# liveness (see backends' readiness waits).
NO_RETRY = Retry(NoBackoff(), 0)


def make_client(host, port, decode_responses=True):
    """A test-facing redis client carrying the deadline and no-retry policy."""
    return redis.Redis(host=host, port=port, decode_responses=decode_responses,
                       socket_timeout=SOCKET_TIMEOUT, retry=NO_RETRY)


def probe_client(host, port):
    """A short-deadline client for liveness probing.

    Separate from make_client because probes want to fail fast. redis-py 7.4
    retries ConnectionError with effectively no upper bound by default, so a
    plain client against a dead server hangs regardless of
    socket_connect_timeout; the explicit no-retry policy is what makes a
    per-attempt budget real.
    """
    return redis.Redis(host=host, port=port, socket_connect_timeout=1,
                       retry=Retry(NoBackoff(), 0))


def no_retry_client(c):
    """Pin SOCKET_TIMEOUT and NO_RETRY onto an already-built client.

    Clients built by make_client get both as kwargs; this is for the ones
    falkordb-py builds itself, which pass neither. set_retry() updates the
    pool's kwargs *and* every connection already in it, so it also covers a
    client that has already been used.
    """
    try:
        c.connection_pool.connection_kwargs['socket_timeout'] = SOCKET_TIMEOUT
        c.set_retry(NO_RETRY)
    except Exception:
        pass
    return c


def db_handle(host, port):
    """FalkorDB handle whose underlying redis client carries our deadline and
    NO_RETRY. falkordb-py builds that client itself, so normalize after."""
    db = FalkorDB(host, port)
    no_retry_client(db.connection)
    return db
