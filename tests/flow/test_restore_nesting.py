import socket
import struct

from common import *

GRAPH_ID = "restore_nesting"

# The deepest a value may nest (`MAX_VALUE_NESTING` in graph/src/runtime/value.rs).
MAX_VALUE_NESTING = 4096


def _read_line(f):
    return f.readline()[:-2]


def _read_resp(f):
    t = f.read(1)
    if t == b"*":
        return [_read_resp(f) for _ in range(int(_read_line(f)))]
    if t == b"$":
        data = f.read(int(_read_line(f)))
        f.read(2)
        return data
    if t in (b"+", b"-", b":"):
        return _read_line(f)
    raise RuntimeError(f"unexpected RESP type {t!r}")


def capture_restore_payload(conn, port, src, dst):
    """The `GRAPH.RESTORE` payload `GRAPH.COPY src dst` replicates, read by
    posing as a replica on a raw socket."""
    s = socket.create_connection(("127.0.0.1", port))
    try:
        f = s.makefile("rb")
        s.sendall(b"*3\r\n$8\r\nREPLCONF\r\n$4\r\ncapa\r\n$6\r\npsync2\r\n")
        _read_line(f)
        s.sendall(b"*3\r\n$5\r\nPSYNC\r\n$1\r\n?\r\n$2\r\n-1\r\n")
        line = _read_line(f)
        while not line.startswith(b"$"):
            line = _read_line(f)
        f.read(int(line[1:]))  # the RDB snapshot, which has no trailing CRLF
        conn.execute_command("GRAPH.COPY", src, dst)
        while True:
            cmd = _read_resp(f)
            if isinstance(cmd, list) and cmd and cmd[0].upper() == b"GRAPH.RESTORE":
                return cmd[2]
    finally:
        s.close()


def _unsigned(v):
    # `VecWriter::write_unsigned`: a TYPE_UNSIGNED byte, then 8 bytes LE.
    return b"\x04" + struct.pack("<Q", v)


T_ARRAY = 1 << 3
T_STRING = 1 << 11
T_NULL = 1 << 15
MARKER = b"DEEPMARKER"
# The encoded string property `MARKER`: its tag, then a TYPE_BYTES buffer
# holding the NUL-terminated bytes.
ENCODED_MARKER = (
    _unsigned(T_STRING) + b"\x00" + struct.pack("<Q", len(MARKER) + 1) + MARKER + b"\x00"
)


class testRestoreNesting(FlowTestsBase):
    def __init__(self):
        self.env, self.db = Env(enableDebugCommand=True)
        self.conn = self.env.getConnection()

    def tearDown(self):
        self.conn.flushall()

    def nested_payload(self, depth):
        """A real `GRAPH.RESTORE` payload for one node whose property is
        `depth` nested single-element lists around NULL."""
        self.conn.flushall()
        self.conn.execute_command(
            "GRAPH.QUERY", "src", f"CREATE ({{v: '{MARKER.decode()}'}})"
        )
        payload = capture_restore_payload(self.conn, self.env.port, "src", "dst")
        self.conn.flushall()
        self.env.assertEqual(payload.count(ENCODED_MARKER), 1)
        nested = (_unsigned(T_ARRAY) + _unsigned(1)) * depth + _unsigned(T_NULL)
        return payload.replace(ENCODED_MARKER, nested)

    def test01_nested_payload_restores(self):
        # The crafted payload is well-formed: shallow nesting restores intact.
        payload = self.nested_payload(10)
        self.env.assertEqual(self.conn.execute_command("GRAPH.RESTORE", GRAPH_ID, payload), "OK")
        res = self.conn.execute_command("GRAPH.QUERY", GRAPH_ID, "MATCH (n) RETURN n.v")
        self.env.assertEqual(res[1], [["[" * 10 + "NULL" + "]" * 10]])

    def test02_payload_at_the_cap_restores(self):
        payload = self.nested_payload(MAX_VALUE_NESTING)
        self.env.assertEqual(self.conn.execute_command("GRAPH.RESTORE", GRAPH_ID, payload), "OK")
        res = self.conn.execute_command("GRAPH.QUERY", GRAPH_ID, "MATCH (n) RETURN size(n.v)")
        self.env.assertEqual(res[1], [[1]])

    def test03_payload_past_the_cap_is_refused(self):
        payload = self.nested_payload(MAX_VALUE_NESTING + 1)
        try:
            self.conn.execute_command("GRAPH.RESTORE", GRAPH_ID, payload)
            self.env.assertTrue(False, message="a payload past the cap restored")
        except redis.exceptions.ResponseError as e:
            self.env.assertContains("nesting exceeds the maximum depth", str(e))
        self.env.assertEqual(self.conn.exists(GRAPH_ID), 0)
        self.env.assertTrue(self.conn.ping())

    def test04_deep_restore_payload_is_refused(self):
        # 60,000 levels: deep enough to exhaust the decoder's stack before the
        # depth was bounded.
        payload = self.nested_payload(60_000)
        try:
            self.conn.execute_command("GRAPH.RESTORE", GRAPH_ID, payload)
            self.env.assertTrue(False, message="a 60,000-deep payload restored")
        except redis.exceptions.ResponseError as e:
            self.env.assertContains("nesting exceeds the maximum depth", str(e))
        self.env.assertTrue(self.conn.ping())
        self.env.assertEqual(self.conn.exists(GRAPH_ID), 0)

    def test05_nested_list_properties_survive_reload(self):
        q = "CREATE ({v: [1, [2, [3, ['a', [true, [4.5]]]]], []]})"
        self.conn.execute_command("GRAPH.QUERY", GRAPH_ID, q)
        before = self.conn.execute_command("GRAPH.QUERY", GRAPH_ID, "MATCH (n) RETURN n.v")[1]
        self.conn.execute_command("DEBUG", "RELOAD")
        after = self.conn.execute_command("GRAPH.QUERY", GRAPH_ID, "MATCH (n) RETURN n.v")[1]
        self.env.assertEqual(before, after)
        self.env.assertEqual(after, [["[1, [2, [3, [a, [true, [4.500000]]]]], []]"]])
