from common import *
import redis
import time

GRAPH_ID = "v3_always"

# a threshold no small write can exceed, which is the ONLY configuration in
# which the two replication paths differ. The DEFAULT is 0 (config.c:962),
# and _should_replicate_effects returns true unconditionally at 0 - so a test
# run at the default proves nothing about either version: both emit effects
# there and the verbatim path is unreachable.
HIGH_THRESHOLD = 1000000000


def _settle_replication(src_con, timeout=30):
    """block until the replica is online and caught up

    Until it is, a write on the primary is not propagated as a COMMAND at all -
    it is folded into the full-sync RDB, and the replica ends up holding the
    data with an empty command stream. That is indistinguishable from "nothing
    was ever replicated", and it is what an unsettled test measures.
    """
    deadline = time.time() + timeout
    while time.time() < deadline:
        info = src_con.execute_command("INFO", "replication")
        slave0 = info.get('slave0') if isinstance(info, dict) else None
        online = (slave0.get('state') == 'online'
                  if isinstance(slave0, dict)
                  else (slave0 is not None and 'state=online' in str(slave0)))
        if online:
            # and caught up: WAIT returns once the replica acks this offset
            src_con.execute_command("WAIT", "1", "0")
            return
        time.sleep(0.1)
    raise AssertionError("replica never came online")


def _replica_stream(src_con, replica_con, write, settle=1.0):
    """the commands the replica received from its master while 'write' ran

    MONITOR, not INFO commandstats: a replica does NOT count master-link
    commands in commandstats. Measured - a plain SET on the primary lands on
    the replica (GET returns it) while cmdstat_set stays absent - so the
    counters report 0 for "replicated" and 0 for "never sent" alike. MONITOR
    is fed from call() and does show the replicated stream.
    """
    kw = replica_con.connection_pool.connection_kwargs
    mon_con = redis.Redis(
            host=kw.get('host'), port=kw.get('port'),
            unix_socket_path=kw.get('unix_socket_path'),
            socket_timeout=settle,
            # a GRAPH.EFFECT argument is binary and is NOT utf-8; strict
            # decoding raises inside the monitor reader and loses the line
            # we came to see
            encoding_errors='replace')
    seen = []
    try:
        m = mon_con.monitor()
        m.__enter__()
        try:
            write()
            # the replica has acknowledged the stream up to this write, so
            # anything it was going to run for it has been dispatched
            src_con.execute_command("WAIT", "1", "0")
            while True:
                try:
                    seen.append(m.next_command()['command'])
                except redis.exceptions.TimeoutError:
                    break
        finally:
            m.__exit__(None, None, None)
    finally:
        mon_con.close()
    return seen


def _kinds(stream):
    """(effects, verbatim) command counts in a replica stream"""
    eff = sum(1 for c in stream if c.upper().startswith("GRAPH.EFFECT"))
    qry = sum(1 for c in stream if c.upper().startswith("GRAPH.QUERY"))
    return eff, qry


# v3 replicates as effects at ANY threshold; v2 keeps the cost heuristic.
#
# The threshold sends a cheap write to the replica as QUERY TEXT instead, which
# is only correct if the replica re-executing that text produces the same result
# and the same ids. That is the assumption effects exist to replace, and it does
# not hold across engines - so a C primary with a non-zero threshold hands a Rust
# replica a query to re-run, and nothing guarantees they agree. Rust dropped the
# mechanism outright (EFFECTS_THRESHOLD_DEPRECATED).
#
# ASSERTED ON THE REPLICATED COMMAND, NOT ON GRAPH STATE. Both paths leave the
# replica's data identical when they work, so nothing in the graph distinguishes
# "re-executed the text" from "applied the effect"; the stream does.
class testEffectsV3AlwaysUsesEffects(FlowTestsBase):
    def __init__(self):
        if VALGRIND or SANITIZER:
            Environment.skip(None)
        self.env, self.db = Env(env='oss', useSlaves=True,
                                moduleArgs=f'EFFECTS_VERSION 3 '
                                           f'EFFECTS_THRESHOLD {HIGH_THRESHOLD}')
        self.src_con = self.env.getConnection()
        self.replica_con = self.env.getSlaveConnection()
        self.src = Graph(self.src_con, GRAPH_ID)

    def test01_small_write_at_high_threshold_uses_effects(self):
        # one node: as cheap as a write gets, so the heuristic would have sent
        # this verbatim
        _settle_replication(self.src_con)
        stream = _replica_stream(self.src_con, self.replica_con,
                lambda: self.src.query("CREATE (:Tiny {x:1})"))
        eff, qry = _kinds(stream)

        self.env.assertTrue(eff > 0,
                message=f"v3 must replicate as effects even at a threshold "
                        f"that sends this same write verbatim under v2; "
                        f"replica received {stream}")
        self.env.assertEqual(qry, 0,
                message=f"v3 must not replicate verbatim; replica received "
                        f"{stream}")


# and the mirror: v2 must be untouched by the change above
class testEffectsV2StillUsesThreshold(FlowTestsBase):
    def __init__(self):
        if VALGRIND or SANITIZER:
            Environment.skip(None)
        self.env, self.db = Env(env='oss', useSlaves=True,
                                moduleArgs=f'EFFECTS_VERSION 2 '
                                           f'EFFECTS_THRESHOLD {HIGH_THRESHOLD}')
        self.src_con = self.env.getConnection()
        self.replica_con = self.env.getSlaveConnection()
        self.src = Graph(self.src_con, GRAPH_ID)

    def test01_small_write_at_high_threshold_still_goes_verbatim(self):
        _settle_replication(self.src_con)
        stream = _replica_stream(self.src_con, self.replica_con,
                lambda: self.src.query("CREATE (:Tiny {x:1})"))
        eff, qry = _kinds(stream)

        # this is the half that proves v2 did not move, and the half that
        # proves the v3 assertion above is testing something: the very same
        # write, at the very same threshold, goes verbatim here
        self.env.assertTrue(qry > 0,
                message=f"v2 must keep the cost heuristic - a cheap write "
                        f"should still replicate verbatim; replica received "
                        f"{stream}")
        self.env.assertEqual(eff, 0,
                message=f"v2 must not have been pulled onto the effects path; "
                        f"replica received {stream}")
