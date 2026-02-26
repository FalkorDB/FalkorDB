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
        # Do NOT call CONFIG SET at runtime as that can destabilize the server
        # under sanitizer/ASAN builds.
        self.notifications_enabled = _notifications_enabled(self.conn)

    def test01_graph_modified_notification(self):
        """Test that graph.modified notification is sent on write operations"""
        if not self.notifications_enabled:
            self.env.skip()

        # Subscribe before the write so we don't miss the message
        pubsub = self.conn.pubsub()
        pubsub.psubscribe("__keyevent@0__:graph.modified")

        # Drain initial subscribe confirmation message
        _drain_pubsub(pubsub)

        # Perform a write operation
        result = self.graph.query("CREATE (n:Person {name: 'Alice'})")
        self.env.assertEquals(result.nodes_created, 1)

        # Wait for the notification
        message = _wait_for_message(pubsub, timeout=3.0)

        # Verify the notification was received for the correct graph key
        self.env.assertIsNotNone(message)
        if message:
            self.env.assertEquals(message['type'], 'pmessage')
            self.env.assertEquals(message['data'], GRAPH_ID)

        pubsub.punsubscribe()
        pubsub.close()

    def test02_graph_modified_on_different_operations(self):
        """Test that graph.modified is sent for various write operations"""
        if not self.notifications_enabled:
            self.env.skip()

        pubsub = self.conn.pubsub()
        pubsub.psubscribe("__keyevent@0__:graph.modified")
        _drain_pubsub(pubsub)

        # Test CREATE operation
        result = self.graph.query("CREATE (n:Person {name: 'Bob'})")
        self.env.assertEquals(result.nodes_created, 1)

        message = _wait_for_message(pubsub, timeout=3.0)
        self.env.assertIsNotNone(message)
        if message:
            self.env.assertEquals(message['data'], GRAPH_ID)

        # Test SET operation (property update)
        result = self.graph.query("MATCH (n:Person) SET n.age = 30")
        self.env.assertGreaterEqual(result.properties_set, 1)

        message = _wait_for_message(pubsub, timeout=3.0)
        self.env.assertIsNotNone(message)
        if message:
            self.env.assertEquals(message['data'], GRAPH_ID)

        # Test DELETE operation (node deletion)
        result = self.graph.query("MATCH (n:Person) DELETE n")
        self.env.assertGreaterEqual(result.nodes_deleted, 1)

        message = _wait_for_message(pubsub, timeout=3.0)
        self.env.assertIsNotNone(message)
        if message:
            self.env.assertEquals(message['data'], GRAPH_ID)

        pubsub.punsubscribe()
        pubsub.close()

    def test03_graph_deleted_notification(self):
        """Test that graph.deleted notification is sent when graph is deleted"""
        if not self.notifications_enabled:
            self.env.skip()

        # Create data in the graph first
        self.graph.query("CREATE (n:Person {name: 'Charlie'})")

        pubsub = self.conn.pubsub()
        pubsub.psubscribe("__keyevent@0__:graph.deleted")
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

    def test04_no_notification_on_read_only(self):
        """Test that no notification is sent for read-only queries"""
        if not self.notifications_enabled:
            self.env.skip()

        # Create some data first
        self.graph.query("CREATE (n:Person {name: 'Dave'})")

        pubsub = self.conn.pubsub()
        pubsub.psubscribe("__keyevent@0__:graph.modified")

        # Drain subscription confirmations and any pending messages from the CREATE
        _drain_pubsub(pubsub, timeout=1.5)

        # Perform a read-only query
        self.graph.query("MATCH (n:Person) RETURN n")

        # Verify no notification was received for the read-only query
        message = _wait_for_message(pubsub, timeout=1.0)
        self.env.assertIsNone(message)

        pubsub.punsubscribe()
        pubsub.close()

    def test05_multiple_graph_modifications(self):
        """Test that multiple modifications each generate a notification"""
        if not self.notifications_enabled:
            self.env.skip()

        pubsub = self.conn.pubsub()
        pubsub.psubscribe("__keyevent@0__:graph.modified")
        _drain_pubsub(pubsub)

        num_ops = 3
        for i in range(num_ops):
            self.graph.query(f"CREATE (n:Person {{id: {i}}})")

        # Collect all notifications, expecting one per write operation
        notifications_received = 0
        deadline = time.time() + 10.0
        while notifications_received < num_ops and time.time() < deadline:
            message = pubsub.get_message(timeout=0.5)
            if message and message.get('type') == 'pmessage':
                self.env.assertEquals(message['data'], GRAPH_ID)
                notifications_received += 1

        self.env.assertEquals(notifications_received, num_ops)

        pubsub.punsubscribe()
        pubsub.close()
