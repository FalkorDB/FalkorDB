from common import *
import time

GRAPH_ID = "keyspace_test"


class testKeyspaceNotifications(FlowTestsBase):
    def __init__(self):
        self.env, self.db = Env()
        self.redis_con = self.env.getConnection()
        self.graph = self.db.select_graph(GRAPH_ID)
        
        # Enable keyspace notifications for module events
        self.redis_con.config_set('notify-keyspace-events', 'AKd')
        
        # Create a subscriber connection for notifications
        self.pubsub = self.redis_con.pubsub()
    
    def tearDown(self):
        try:
            self.pubsub.close()
        except:
            pass
        try:
            self.graph.delete()
        except:
            pass
    
    def test_graph_create_notification(self):
        """Test that graph.create notification is emitted when a graph is created"""
        # Subscribe to module keyspace events
        self.pubsub.psubscribe('__keyspace@0__:*')
        
        # Clear any existing messages
        for _ in range(10):
            message = self.pubsub.get_message(timeout=0.1)
            if message is None:
                break
        
        # Create a new graph by executing a query
        new_graph_id = "new_test_graph"
        new_graph = self.db.select_graph(new_graph_id)
        new_graph.query("CREATE (:Person {name: 'Alice'})")
        
        # Wait for notification
        time.sleep(0.1)
        
        # Check for graph.create notification
        notifications = []
        for _ in range(10):
            message = self.pubsub.get_message(timeout=0.5)
            if message is None:
                break
            if message['type'] == 'pmessage':
                notifications.append(message['data'])
        
        # Clean up
        try:
            new_graph.delete()
        except:
            pass
        
        # Verify we got graph.create or graph.node.create notification
        assert any(b'graph.create' in n or b'graph.node.create' in n for n in notifications), \
            f"Expected graph.create or graph.node.create notification, got: {notifications}"
    
    def test_node_create_notification(self):
        """Test that graph.node.create notification is emitted when nodes are created"""
        # Subscribe to module keyspace events
        self.pubsub.psubscribe('__keyspace@0__:*')
        
        # Clear any existing messages
        for _ in range(10):
            message = self.pubsub.get_message(timeout=0.1)
            if message is None:
                break
        
        # Create nodes
        self.graph.query("CREATE (:Person {name: 'Bob'})")
        
        # Wait for notification
        time.sleep(0.1)
        
        # Check for notifications
        notifications = []
        for _ in range(10):
            message = self.pubsub.get_message(timeout=0.5)
            if message is None:
                break
            if message['type'] == 'pmessage':
                notifications.append(message['data'])
        
        # Verify we got graph.node.create notification
        assert any(b'graph.node.create' in n for n in notifications), \
            f"Expected graph.node.create notification, got: {notifications}"
    
    def test_edge_create_notification(self):
        """Test that graph.edge.create notification is emitted when edges are created"""
        # Subscribe to module keyspace events
        self.pubsub.psubscribe('__keyspace@0__:*')
        
        # Clear any existing messages
        for _ in range(10):
            message = self.pubsub.get_message(timeout=0.1)
            if message is None:
                break
        
        # Create nodes and relationship
        self.graph.query("CREATE (:Person {name: 'Charlie'})-[:KNOWS]->(:Person {name: 'Dave'})")
        
        # Wait for notification
        time.sleep(0.1)
        
        # Check for notifications
        notifications = []
        for _ in range(10):
            message = self.pubsub.get_message(timeout=0.5)
            if message is None:
                break
            if message['type'] == 'pmessage':
                notifications.append(message['data'])
        
        # Verify we got graph.edge.create notification
        assert any(b'graph.edge.create' in n for n in notifications), \
            f"Expected graph.edge.create notification, got: {notifications}"
    
    def test_node_delete_notification(self):
        """Test that graph.node.delete notification is emitted when nodes are deleted"""
        # Create a node first
        self.graph.query("CREATE (:Person {name: 'Eve'})")
        
        # Subscribe to module keyspace events
        self.pubsub.psubscribe('__keyspace@0__:*')
        
        # Clear any existing messages
        for _ in range(10):
            message = self.pubsub.get_message(timeout=0.1)
            if message is None:
                break
        
        # Delete the node
        self.graph.query("MATCH (n:Person {name: 'Eve'}) DELETE n")
        
        # Wait for notification
        time.sleep(0.1)
        
        # Check for notifications
        notifications = []
        for _ in range(10):
            message = self.pubsub.get_message(timeout=0.5)
            if message is None:
                break
            if message['type'] == 'pmessage':
                notifications.append(message['data'])
        
        # Verify we got graph.node.delete notification
        assert any(b'graph.node.delete' in n for n in notifications), \
            f"Expected graph.node.delete notification, got: {notifications}"
    
    def test_graph_delete_notification(self):
        """Test that graph.delete notification is emitted when a graph is deleted"""
        # Create a graph
        delete_test_graph = "delete_test_graph"
        graph = self.db.select_graph(delete_test_graph)
        graph.query("CREATE (:Person {name: 'Frank'})")
        
        # Subscribe to module keyspace events
        self.pubsub.psubscribe('__keyspace@0__:*')
        
        # Clear any existing messages
        for _ in range(10):
            message = self.pubsub.get_message(timeout=0.1)
            if message is None:
                break
        
        # Delete the graph
        graph.delete()
        
        # Wait for notification
        time.sleep(0.1)
        
        # Check for notifications
        notifications = []
        for _ in range(10):
            message = self.pubsub.get_message(timeout=0.5)
            if message is None:
                break
            if message['type'] == 'pmessage':
                notifications.append(message['data'])
        
        # Verify we got graph.delete notification
        assert any(b'graph.delete' in n for n in notifications), \
            f"Expected graph.delete notification, got: {notifications}"
