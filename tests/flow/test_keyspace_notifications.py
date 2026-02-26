import time
from common import Env

GRAPH_ID = "keyspace_notifications_test"


def _notifications_enabled(conn):
    """Check if keyspace notifications are configured for module events.

    Module events require 'K' or 'E' (keyspace/keyevent) AND 'd' or 'A'
    (module events / all events) to be set in notify-keyspace-events.
    """
    config = conn.execute_command("CONFIG", "GET", "notify-keyspace-events")
    # config is a list ['notify-keyspace-events', '<value>'] with decode_responses=True
    if isinstance(config, dict):
        val = config.get("notify-keyspace-events", "")
    elif isinstance(config, list) and len(config) >= 2:
        val = config[1]
    else:
        val = ""
    if not val:
        return False
    has_scope = ('K' in val or 'E' in val)
    has_module = ('d' in val or 'A' in val)
    return has_scope and has_module


def _drain_pubsub(pubsub, timeout=1.0):
    """Drain all pending pubsub messages, including subscription confirmations."""
    deadline = time.time() + timeout
    while time.time() < deadline:
        msg = pubsub.get_message(timeout=0.05)
        if msg is None:
            break


def _wait_for_message(pubsub, timeout=3.0):
    """Wait for the next pubsub message of type 'pmessage', with retries."""
    deadline = time.time() + timeout
    while time.time() < deadline:
        msg = pubsub.get_message(timeout=0.1)
        if msg is not None and msg.get('type') == 'pmessage':
            return msg
    return None


class testKeyspaceNotifications():
    def __init__(self):
        self.env, self.db = Env()
        self.conn = self.env.getConnection()
        self.graph = self.db.select_graph(GRAPH_ID)

        # Detect if keyspace notifications are already enabled.
        # Do NOT call CONFIG SET at runtime: Redis 8.x's
        # updateClientMemUsageAndBucket() asserts pthread_equal(self, main_thread_id),
        # so calling RedisModule_NotifyKeyspaceEvent from a FalkorDB thread-pool
        # worker (non-main thread) causes an immediate server crash.
        # The graph.deleted notification in cmd_delete.c is safe because that
        # command handler runs on the Redis main thread.
        self.notifications_enabled = _notifications_enabled(self.conn)

    def test01_graph_deleted_notification(self):
        """Test that graph.deleted notification is sent when a graph is deleted.

        This notification is emitted from cmd_delete.c which runs on the
        Redis main thread, so it is safe to call RedisModule_NotifyKeyspaceEvent
        there.
        """
        if not self.notifications_enabled:
            self.env.skip()

        # Create some data in the graph first
        self.graph.query("CREATE (n:Person {name: 'Alice'})")

        pubsub = self.conn.pubsub()
        pubsub.psubscribe("__keyevent@0__:graph.deleted")

        # Drain the initial subscribe confirmation frame
        _drain_pubsub(pubsub)

        # Delete the graph
        self.graph.delete()

        # Wait for the notification
        message = _wait_for_message(pubsub, timeout=3.0)

        # Verify the notification was received for the correct graph key
        self.env.assertIsNotNone(message)
        if message:
            self.env.assertEquals(message['type'], 'pmessage')
            self.env.assertEquals(message['data'], GRAPH_ID)

        pubsub.punsubscribe()
        pubsub.close()
