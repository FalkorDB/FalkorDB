import time
from common import Env, FalkorDB

GRAPH_ID = "keyspace_notifications_test"

class testKeyspaceNotifications():
    def __init__(self):
        self.env, self.db = Env()
        self.conn = self.env.getConnection()
        self.graph = self.db.select_graph(GRAPH_ID)

        # Enable keyspace notifications
        self.conn.execute_command("CONFIG", "SET", "notify-keyspace-events", "AKE")

    def test01_graph_modified_notification(self):
        """Test that graph.modified notification is sent on write operations"""
        
        # Create a pubsub connection to listen for notifications
        pubsub = self.conn.pubsub()
        pubsub.psubscribe("__keyevent@0__:graph.modified")
        
        # Give pubsub a moment to subscribe
        time.sleep(0.1)
        
        # Create a graph and perform a write operation
        result = self.graph.query("CREATE (n:Person {name: 'Alice'})")
        self.env.assertEquals(result.nodes_created, 1)
        
        # Check for notification
        message = pubsub.get_message(timeout=2.0)
        # Skip the subscribe confirmation message
        if message and message['type'] == 'psubscribe':
            message = pubsub.get_message(timeout=2.0)
        
        # Verify the notification was received
        self.env.assertIsNotNone(message)
        self.env.assertEquals(message['type'], 'pmessage')
        self.env.assertEquals(message['pattern'], '__keyevent@0__:graph.modified')
        self.env.assertEquals(message['channel'], '__keyevent@0__:graph.modified')
        self.env.assertEquals(message['data'], GRAPH_ID)
        
        # Clean up
        pubsub.punsubscribe()
        pubsub.close()

    def test02_graph_modified_on_different_operations(self):
        """Test that graph.modified is sent for various write operations"""
        
        # Create a pubsub connection
        pubsub = self.conn.pubsub()
        pubsub.psubscribe("__keyevent@0__:graph.modified")
        time.sleep(0.1)
        
        # Test CREATE operation
        result = self.graph.query("CREATE (n:Person {name: 'Bob'})")
        self.env.assertEquals(result.nodes_created, 1)
        
        # Check for notification
        message = pubsub.get_message(timeout=2.0)
        if message and message['type'] == 'psubscribe':
            message = pubsub.get_message(timeout=2.0)
        self.env.assertIsNotNone(message)
        self.env.assertEquals(message['data'], GRAPH_ID)
        
        # Test SET operation (property update)
        result = self.graph.query("MATCH (n:Person) SET n.age = 30")
        self.env.assertGreaterEqual(result.properties_set, 1)
        
        message = pubsub.get_message(timeout=2.0)
        self.env.assertIsNotNone(message)
        self.env.assertEquals(message['data'], GRAPH_ID)
        
        # Test DELETE operation
        result = self.graph.query("MATCH (n:Person) DELETE n")
        self.env.assertGreaterEqual(result.nodes_deleted, 1)
        
        message = pubsub.get_message(timeout=2.0)
        self.env.assertIsNotNone(message)
        self.env.assertEquals(message['data'], GRAPH_ID)
        
        # Clean up
        pubsub.punsubscribe()
        pubsub.close()

    def test03_graph_deleted_notification(self):
        """Test that graph.deleted notification is sent when graph is deleted"""

        # Create a graph
        self.graph.query("CREATE (n:Person {name: 'Charlie'})")
        
        # Create a pubsub connection to listen for delete notifications
        pubsub = self.conn.pubsub()
        pubsub.psubscribe("__keyevent@0__:graph.deleted")
        time.sleep(0.1)
        
        # Delete the graph
        self.graph.delete()
        
        # Check for notification
        message = pubsub.get_message(timeout=2.0)
        # Skip the subscribe confirmation message
        if message and message['type'] == 'psubscribe':
            message = pubsub.get_message(timeout=2.0)
        
        # Verify the notification was received
        self.env.assertIsNotNone(message)
        self.env.assertEquals(message['type'], 'pmessage')
        self.env.assertEquals(message['pattern'], '__keyevent@0__:graph.deleted')
        self.env.assertEquals(message['channel'], '__keyevent@0__:graph.deleted')
        self.env.assertEquals(message['data'], GRAPH_ID)
        
        # Clean up
        pubsub.punsubscribe()
        pubsub.close()

    def test04_no_notification_on_read_only(self):
        """Test that no notification is sent for read-only queries"""
        
        # Create some data first
        self.graph.query("CREATE (n:Person {name: 'Dave'})")
        
        # Create a pubsub connection
        pubsub = self.conn.pubsub()
        pubsub.psubscribe("__keyevent@0__:graph.modified")
        time.sleep(0.1)
        
        # Clear any pending messages (from the CREATE above)
        msg = pubsub.get_message(timeout=0.1)
        while msg is not None:
            msg = pubsub.get_message(timeout=0.1)
        
        # Perform a read-only query
        self.graph.query("MATCH (n:Person) RETURN n")
        
        # Check that no notification was received
        message = pubsub.get_message(timeout=1.0)
        self.env.assertIsNone(message)
        
        # Clean up
        pubsub.punsubscribe()
        pubsub.close()

    def test05_notification_disabled_by_default(self):
        """Test that notifications are not sent when notifications are disabled"""
        # Disable keyspace notifications
        self.conn.execute_command("CONFIG", "SET", "notify-keyspace-events", "")
        
        # Create a pubsub connection
        pubsub = self.conn.pubsub()
        pubsub.psubscribe("__keyevent@0__:graph.modified")
        time.sleep(0.1)
        
        # Clear subscribe message
        pubsub.get_message(timeout=0.1)
        
        # Perform a write operation
        self.graph.query("CREATE (n:Person {name: 'Eve'})")
        
        # Check that no notification was received (keyspace notifications disabled)
        message = pubsub.get_message(timeout=1.0)
        self.env.assertIsNone(message)
        
        # Clean up
        pubsub.punsubscribe()
        pubsub.close()

        # Re-enable for other tests
        self.conn.execute_command("CONFIG", "SET", "notify-keyspace-events", "AKE")

    def test06_multiple_graph_modifications(self):
        """Test that multiple modifications generate multiple notifications"""
        
        # Create a pubsub connection
        pubsub = self.conn.pubsub()
        pubsub.psubscribe("__keyevent@0__:graph.modified")
        time.sleep(0.1)
        
        # Clear subscribe message
        pubsub.get_message(timeout=0.1)
        
        # Perform multiple write operations
        for i in range(3):
            self.graph.query(f"CREATE (n:Person {{id: {i}}})")
        
        # Check for 3 notifications
        notifications_received = 0
        for _ in range(3):
            message = pubsub.get_message(timeout=2.0)
            if message and message['type'] == 'pmessage':
                self.env.assertEquals(message['data'], GRAPH_ID)
                notifications_received += 1
        
        self.env.assertEquals(notifications_received, 3)
        
        # Clean up
        pubsub.punsubscribe()
        pubsub.close()

