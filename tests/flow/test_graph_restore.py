import struct
from common import Env
from redis import ResponseError

# tests GRAPH.RESTORE with payloads it should reject
#
# the payload normally comes from GRAPH.COPY, but any client can call
# GRAPH.RESTORE directly, so a bad payload has to produce an error reply
# and leave the server running

TYPE_BYTES = b"\x00"
TYPE_UNSIGNED = b"\x04"


def unsigned(v):
    return TYPE_UNSIGNED + struct.pack("<Q", v)


def buffer(data):
    return TYPE_BYTES + struct.pack("<Q", len(data)) + data


def header(name=b"g\x00", relationship_count=0, key_count=1):
    # graph name, node/edge/deleted-node/deleted-edge/label counts,
    # relationship count, one multi-edge flag per relationship, key count
    return (buffer(name)
            + unsigned(0) * 5
            + unsigned(relationship_count)
            + unsigned(0) * relationship_count
            + unsigned(key_count))


class testGraphRestore():
    def __init__(self):
        self.env, self.db = Env()
        self.conn = self.env.getConnection()

    def tearDown(self):
        self.conn.flushall()

    def assert_restore_refused(self, payload):
        try:
            self.conn.execute_command("GRAPH.RESTORE", "restored", payload)
            self.env.assertTrue(False)  # should not get here
        except ResponseError:
            pass
        # server is still up and no key was created
        self.env.assertTrue(self.conn.ping())
        self.env.assertEqual(self.conn.exists("restored"), 0)

    def test01_empty_payload(self):
        # used to crash the server: the decoder hit the end of the buffer
        # and tried to load another chunk with a null IO handle
        self.assert_restore_refused(b"")

    def test02_truncated_payload(self):
        # cut a valid header short at every possible offset
        payload = header()
        for end in range(1, len(payload) + 1):
            self.assert_restore_refused(payload[:end])

    def test03_wrong_type_tag(self):
        self.assert_restore_refused(unsigned(0))

    def test04_huge_relationship_count(self):
        # the count was used as an allocation size without any check
        for count in [1 << 40, 1 << 62, (1 << 64) - 1]:
            payload = buffer(b"g\x00") + unsigned(0) * 5 + unsigned(count)
            self.assert_restore_refused(payload)

    def test05_huge_schema_counts(self):
        huge = (1 << 64) - 1

        # attribute count
        self.assert_restore_refused(header() + unsigned(huge))

        # node schema count
        self.assert_restore_refused(header() + unsigned(0) + unsigned(huge))

        # relationship schema count
        self.assert_restore_refused(header() + unsigned(0) * 2 + unsigned(huge))

        # payload directory count
        self.assert_restore_refused(header() + unsigned(0) * 3 + unsigned(huge))

    def test06_multi_key_payload(self):
        # RESTORE only accepts single key payloads
        self.assert_restore_refused(header(key_count=2))
