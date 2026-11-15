from common import *

# Server event ids agree between Redis and Valkey up to 17 and then part ways:
# Redis uses 18 and 19 for cluster slot migration, Valkey uses 18 for an
# authentication attempt and 19 for its own slot migration. The RediSearch
# code linked into this module subscribed to 18 and 19 under their Redis
# meaning, so on Valkey every AUTH, granted or denied, ran RediSearch's slot
# migration handler on the authentication payload and crashed the server.
# Embedded RediSearch now leaves those events alone on either server.

GRAPH_ID = "authentication"
USER = "graph_auth_user"
PASSWORD = "graph-auth-password"
# RediSearch logs this from the same function, at the same level, just before
# it used to subscribe to slot migration: proof the log was read at all.
CONFIG_SUBSCRIBED = "Subscribe to config changes"
SLOT_MIGRATION_SUBSCRIBED = "Subscribe to cluster slot migration events"
SLOT_MIGRATION_HANDLED = "Got ASM"


class testAuthentication():
    def __init__(self):
        self.env, self.db = Env()
        self.conn = self.env.getConnection()
        self.graph = self.db.select_graph(GRAPH_ID)

    def client(self):
        # RESP2, so the client sends no HELLO of its own on connect and the
        # HELLO below does not switch the protocol under it.
        return redis.Redis(host=self.env.host, port=self.env.port, protocol=2,
                           socket_timeout=SOCKET_TIMEOUT, retry=NO_RETRY,
                           decode_responses=True)

    def server_log(self):
        # None in CI's services mode, where the shared server's log is not
        # mounted; there only the outcome in test02 is checked.
        if self.env.log_path is None:
            return None
        with open(self.env.log_path) as f:
            log = f.read()
        self.env.assertContains(CONFIG_SUBSCRIBED, log)
        return log

    def test01_slot_migration_events_not_subscribed(self):
        log = self.server_log()
        if log is None:
            self.env.skip()
            return
        self.env.assertNotContains(SLOT_MIGRATION_SUBSCRIBED, log)

    def test02_auth_attempts_leave_the_server_up(self):
        self.graph.query("CREATE (:Person {name: 'a'})")

        self.conn.execute_command("ACL", "SETUSER", USER, "reset", "on",
                                  f">{PASSWORD}", "~*", "&*", "+@all")
        try:
            # Valkey fires the event for every attempt, granted or denied,
            # through AUTH and through HELLO alike.
            granted = self.client()
            granted.execute_command("AUTH", USER, PASSWORD)
            self.env.assertEqual(granted.execute_command("ACL", "WHOAMI"), USER)

            denied = self.client()
            try:
                denied.execute_command("AUTH", USER, "wrong-password")
                self.env.assertTrue(False, message="a wrong password was accepted")
            except redis.exceptions.AuthenticationError as e:
                # redis-py's mapping of the server's WRONGPASS reply.
                self.env.assertContains("invalid username-password pair", str(e))

            hello = self.client()
            hello.execute_command("HELLO", 2, "AUTH", USER, PASSWORD)
            self.env.assertEqual(hello.execute_command("ACL", "WHOAMI"), USER)
        finally:
            self.conn.execute_command("ACL", "DELUSER", USER)

        self.env.assertTrue(self.conn.ping())
        res = self.graph.query("MATCH (p:Person {name: 'a'}) RETURN count(p)")
        self.env.assertEqual(res.result_set, [[1]])

        log = self.server_log()
        if log is not None:
            self.env.assertNotContains(SLOT_MIGRATION_HANDLED, log)
