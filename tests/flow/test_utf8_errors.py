from common import Env, FlowTestsBase, ResponseError

# An error reply used to be built one `char` at a time with `c as u8`, so any code
# point whose low byte is 0 — '一' (U+4E00), 'Ā' (U+0100) — became a NUL, and the
# `CString::new(..).unwrap()` that followed panicked and took Redis down (#3185).
# Every other non-ASCII character was silently truncated to a garbage byte.
#
# An error reply must keep the UTF-8 bytes of its message verbatim, and must never
# take the server down, whatever the query text holds.
GRAPH = "utf8_errors"


class testUtf8ErrorReplies(FlowTestsBase):
    def __init__(self):
        self.env, self.db = Env()
        self.conn = self.env.getConnection()

    def assert_error(self, *cmd):
        try:
            self.conn.execute_command(*cmd)
            self.env.assertTrue(False)  # unreachable: the query must not parse
        except ResponseError as e:
            msg = str(e)
        # the server must still be there
        self.env.assertTrue(self.conn.ping())
        return msg

    def test01_low_byte_zero_code_points_do_not_crash(self):
        # The originally reported repro.
        self.assert_error("GRAPH.QUERY", GRAPH, "MATCH (n:Person {name:'一'}) RETURN n +")
        self.assert_error("GRAPH.QUERY", GRAPH, "RETURN 'Ā' +")

    def test02_error_text_keeps_utf8_verbatim(self):
        msg = self.assert_error("GRAPH.QUERY", GRAPH, "RETURN '中文' +")
        self.env.assertContains("中文", msg)
        msg = self.assert_error("GRAPH.QUERY", GRAPH, "MATCH (n:Person {name:'一'}) RETURN n +")
        self.env.assertContains("'一'", msg)

    def test03_other_commands(self):
        for cmd in ("GRAPH.QUERY", "GRAPH.RO_QUERY", "GRAPH.EXPLAIN"):
            msg = self.assert_error(cmd, GRAPH, "MATCH (n:Person {name:'一'}) RETURN n +")
            self.env.assertContains("'一'", msg)
            msg = self.assert_error(cmd, GRAPH, "RETURN 'Ā中文' +")
            self.env.assertContains("Ā中文", msg)
