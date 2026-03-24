from common import *
from neo4j import GraphDatabase
from neo4j.spatial import WGS84Point
import neo4j.graph
import socket
import struct
import hashlib
import base64
import os
import time

BOLT_PORT = 7687

def _bolt_setup(env_self):
    """Shared setup: start server and create bolt driver."""
    env_self.env, _ = Env(moduleArgs=f"BOLT_PORT {BOLT_PORT}")
    env_self.bolt_con = GraphDatabase.driver(
        f"bolt://localhost:{BOLT_PORT}", auth=("falkordb", ""))

def _bolt_teardown(env_self):
    """Shared teardown: close bolt driver to release connections."""
    if hasattr(env_self, 'bolt_con') and env_self.bolt_con is not None:
        env_self.bolt_con.close()

def _ws_connect(port):
    """Perform a WebSocket upgrade handshake to the Bolt port.
    Returns (socket, True) on success, (None, False) on failure."""
    s = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    s.settimeout(5)
    s.connect(('127.0.0.1', port))
    key = base64.b64encode(os.urandom(16)).decode()
    request = (
        "GET / HTTP/1.1\r\n"
        f"Host: 127.0.0.1:{port}\r\n"
        "Upgrade: websocket\r\n"
        "Connection: Upgrade\r\n"
        f"Sec-WebSocket-Key: {key}\r\n"
        "Sec-WebSocket-Version: 13\r\n"
        f"Origin: http://127.0.0.1:{port}\r\n"
        "\r\n"
    )
    s.sendall(request.encode())
    # read response byte-by-byte to avoid consuming WS frame data
    response = b''
    while b'\r\n\r\n' not in response:
        b = s.recv(1)
        if not b:
            break
        response += b
    return s, response.startswith(b'HTTP/1.1 101')

def _ws_send_frame(s, data):
    """Send a WebSocket binary frame (masked, as required by client)."""
    mask_key = os.urandom(4)
    masked = bytearray(len(data))
    for i in range(len(data)):
        masked[i] = data[i] ^ mask_key[i % 4]
    header = bytearray()
    header.append(0x82)  # FIN + binary opcode
    length = len(data)
    if length < 126:
        header.append(0x80 | length)  # MASK bit set
    elif length < 65536:
        header.append(0x80 | 126)
        header += struct.pack('>H', length)
    else:
        header.append(0x80 | 127)
        header += struct.pack('>Q', length)
    header += mask_key
    s.sendall(header + masked)

def _ws_recv_frame(s):
    """Receive a WebSocket frame and return the payload bytes."""
    head = s.recv(2)
    if len(head) < 2:
        return b''
    payload_len = head[1] & 0x7F
    if payload_len == 126:
        ext = s.recv(2)
        payload_len = struct.unpack('>H', ext)[0]
    elif payload_len == 127:
        ext = s.recv(8)
        payload_len = struct.unpack('>Q', ext)[0]
    data = bytearray()
    while len(data) < payload_len:
        chunk = s.recv(payload_len - len(data))
        if not chunk:
            break
        data += chunk
    return bytes(data)

def _bolt_handshake_over_ws(s):
    """Send the Bolt handshake over an established WebSocket connection.
    Returns the server version response bytes."""
    # small delay to let the server finish processing the WS upgrade
    time.sleep(0.1)
    # Bolt magic preamble (0x6060B017) + 4 version proposals
    handshake = struct.pack('>I', 0x6060B017)
    # version proposals: [5.7, 5.1, 0, 0]
    handshake += struct.pack('>I', 0x00070005)
    handshake += struct.pack('>I', 0x00010005)
    handshake += struct.pack('>I', 0x00000000)
    handshake += struct.pack('>I', 0x00000000)
    _ws_send_frame(s, handshake)
    response = _ws_recv_frame(s)
    return response


class testBolt():
    def __init__(self):
        _bolt_setup(self)

    def __del__(self):
        _bolt_teardown(self)

    def test01_null(self):
        with self.bolt_con.session() as session:
            result = session.run("RETURN null, $v", {"v": None})
            record = result.single()
            self.env.assertEquals(record[0], None)

    def test02_boolean(self):
        with self.bolt_con.session() as session:
            result = session.run("RETURN true, false, $v_true, $v_false", {"v_true": True, "v_false": False})
            record = result.single()
            self.env.assertEquals(record[0], True)
            self.env.assertEquals(record[1], False)
            self.env.assertEquals(record[2], True)
            self.env.assertEquals(record[3], False)

    def test03_integer(self):
        with self.bolt_con.session() as session:
            result = session.run("RETURN -1, 0, 1, 2, $v", {"v": 3})
            record = result.single()
            self.env.assertEquals(record[0], -1)
            self.env.assertEquals(record[1], 0)
            self.env.assertEquals(record[2], 1)
            self.env.assertEquals(record[3], 2)
            self.env.assertEquals(record[4], 3)

            result = session.run("RETURN 255, 256, 257, $v", {"v": 258})
            record = result.single()
            self.env.assertEquals(record[0], 255)
            self.env.assertEquals(record[1], 256)
            self.env.assertEquals(record[2], 257)
            self.env.assertEquals(record[3], 258)

            result = session.run("RETURN 65535, 65536, 65537, $v", {"v": 65538})
            record = result.single()
            self.env.assertEquals(record[0], 65535)
            self.env.assertEquals(record[1], 65536)
            self.env.assertEquals(record[2], 65537)
            self.env.assertEquals(record[3], 65538)

            result = session.run("RETURN 4294967295, 4294967296, 4294967297, $v", {"v": 4294967298})
            record = result.single()
            self.env.assertEquals(record[0], 4294967295)
            self.env.assertEquals(record[1], 4294967296)
            self.env.assertEquals(record[2], 4294967297)
            self.env.assertEquals(record[3], 4294967298)

            result = session.run("RETURN 9223372036854775807, $v", {"v": 9223372036854775807})
            record = result.single()
            self.env.assertEquals(record[0], 9223372036854775807)
            self.env.assertEquals(record[1], 9223372036854775807)

    def test04_float(self):
        with self.bolt_con.session() as session:
            result = session.run("RETURN 1.23, $v", {"v": 4.56})
            record = result.single()
            self.env.assertEquals(record[0], 1.23)
            self.env.assertEquals(record[1], 4.56)

    def test05_string(self):
        with self.bolt_con.session() as session:
            result = session.run("RETURN '', 'Hello, World!', $v8, $v16", {"v8": 'A' * 255, "v16": 'A' * 256})
            record = result.single()
            self.env.assertEquals(record[0], '')
            self.env.assertEquals(record[1], 'Hello, World!')
            self.env.assertEquals(record[2], 'A' * 255)
            self.env.assertEquals(record[3], 'A' * 256)

    def test06_list(self):
        with self.bolt_con.session() as session:
            result = session.run("RETURN [], [1,2,3], $v8, $v16", {"v8": [1] * 255, "v16": [1] * 256})
            record = result.single()
            self.env.assertEquals(record[0], [])
            self.env.assertEquals(record[1], [1,2,3])
            self.env.assertEquals(record[2], [1] * 255)
            self.env.assertEquals(record[3], [1] * 256)

    def test07_map(self):
        with self.bolt_con.session() as session:
             result = session.run("RETURN {}, {foo:'bar'}, $v8", {"v8": {'foo':'bar'} })
             record = result.single()
             self.env.assertEquals(record[0], {})
             self.env.assertEquals(record[1], {'foo':'bar'})
             self.env.assertEquals(record[2], {'foo':'bar'})

    def test08_point(self):
         with self.bolt_con.session() as session:
             result = session.run("RETURN POINT({longitude:1, latitude:2})")
             record = result.single()
             self.env.assertEquals(record[0], WGS84Point((1, 2)))

    def test09_graph_entities_values(self):
         with self.bolt_con.session() as session:
             result = session.run("""CREATE (a:A {v: 1})-[r1:R1]->(b:B)<-[r2:R2]-(c:C) RETURN a, r1, b, r2, c""")
             record = result.single()
             a:neo4j.graph.Node = record[0]
             r1:neo4j.graph.Relationship = record[1]
             b:neo4j.graph.Node = record[2]
             r2:neo4j.graph.Relationship = record[3]
             c:neo4j.graph.Node = record[4]

             self.env.assertEquals(a.id, 0)
             self.env.assertEquals(a.labels, set(['A']))

             self.env.assertEquals(r1.id, 0)
             self.env.assertEquals(r1.type, 'R1')
             self.env.assertEquals(r1.start_node, a)
             self.env.assertEquals(r1.end_node, b)

             self.env.assertEquals(b.id, 1)
             self.env.assertEquals(b.labels, set(['B']))

             self.env.assertEquals(r2.id, 1)
             self.env.assertEquals(r2.type, 'R2')
             self.env.assertEquals(r2.start_node, c)
             self.env.assertEquals(r2.end_node, b)

             self.env.assertEquals(c.id, 2)
             self.env.assertEquals(c.labels, set(['C']))

             result = session.run("""MATCH p=(:A) RETURN p""")
             record = result.single()
             p:neo4j.graph.Path = record[0]
             self.env.assertEquals(p.start_node.labels, set(['A']))

             result = session.run("""MATCH p=(:A)-[:R1]->(:B) RETURN p""")
             record = result.single()
             p:neo4j.graph.Path = record[0]
             self.env.assertEquals(p.start_node.labels, set(['A']))
             self.env.assertEquals(p.end_node.labels, set(['B']))
             self.env.assertEquals(p.nodes[0].labels, set(['A']))
             self.env.assertEquals(p.nodes[1].labels, set(['B']))
             self.env.assertEquals(p.relationships[0].type, 'R1')

             result = session.run("""MATCH p=(:A)-[:R1]->(:B)<-[:R2]-(:C) RETURN p""")
             record = result.single()
             p:neo4j.graph.Path = record[0]
             self.env.assertEquals(p.start_node.labels, set(['A']))
             self.env.assertEquals(p.end_node.labels, set(['C']))
             self.env.assertEquals(p.nodes[0].labels, set(['A']))
             self.env.assertEquals(p.nodes[1].labels, set(['B']))
             self.env.assertEquals(p.nodes[2].labels, set(['C']))
             self.env.assertEquals(p.relationships[0].type, 'R1')
             self.env.assertEquals(p.relationships[1].type, 'R2')

    def test10_tiny_int_negative_range(self):
        """Verify tiny int encoding for the full range -16..127.
        Before the fix, bolt_reply_int never used the tiny int path
        because the comparison used the unsigned constant 0xF0 (240)."""
        with self.bolt_con.session() as session:
            # test all negative tiny ints: -16 to -1
            vals = list(range(-16, 0))
            placeholders = ', '.join(f'${f"n{abs(v)}"}' for v in vals)
            params = {f"n{abs(v)}": v for v in vals}
            result = session.run(f"RETURN {placeholders}", params)
            record = result.single()
            for i, v in enumerate(vals):
                self.env.assertEquals(record[i], v)

            # test boundaries around tiny int range
            result = session.run("RETURN $a, $b, $c, $d",
                                 {"a": -16, "b": -17, "c": 127, "d": 128})
            record = result.single()
            self.env.assertEquals(record[0], -16)
            self.env.assertEquals(record[1], -17)
            self.env.assertEquals(record[2], 127)
            self.env.assertEquals(record[3], 128)

    def test11_negative_integer_boundaries(self):
        """Test negative values at int8/int16/int32/int64 boundaries."""
        with self.bolt_con.session() as session:
            result = session.run(
                "RETURN $a, $b, $c, $d, $e, $f",
                {"a": -128, "b": -129, "c": -32768,
                 "d": -32769, "e": -2147483648, "f": -2147483649})
            record = result.single()
            self.env.assertEquals(record[0], -128)     # int8 min
            self.env.assertEquals(record[1], -129)     # int16
            self.env.assertEquals(record[2], -32768)   # int16 min
            self.env.assertEquals(record[3], -32769)   # int32
            self.env.assertEquals(record[4], -2147483648)   # int32 min
            self.env.assertEquals(record[5], -2147483649)   # int64

    def test12_point_followed_by_values(self):
        """Verify Point2D doesn't corrupt subsequent values in the record.
        Before the fix, an extra bolt_reply_null shifted all following data."""
        with self.bolt_con.session() as session:
            result = session.run(
                "RETURN POINT({longitude:1.5, latitude:2.5}), 42, 'hello'")
            record = result.single()
            self.env.assertEquals(record[0], WGS84Point((1.5, 2.5)))
            self.env.assertEquals(record[1], 42)
            self.env.assertEquals(record[2], 'hello')

    def test13_multiple_points(self):
        """Test returning multiple Point2D values with float coordinates.
        Uses float32-exact values (multiples of powers of 2) to exercise
        the float serialization path without precision mismatch."""
        with self.bolt_con.session() as session:
            result = session.run(
                "RETURN POINT({longitude:10.5, latitude:20.25}), "
                "POINT({longitude:-30.125, latitude:40.75}), "
                "POINT({longitude:-50.5, latitude:60.875})")
            record = result.single()
            self.env.assertEquals(record[0], WGS84Point((10.5, 20.25)))
            self.env.assertEquals(record[1], WGS84Point((-30.125, 40.75)))
            self.env.assertEquals(record[2], WGS84Point((-50.5, 60.875)))

    def test14_large_string_parameter(self):
        """Test queries with large string parameters.
        Before the fix, a fixed 4096-byte stack buffer could overflow."""
        with self.bolt_con.session() as session:
            large_str = 'X' * 8000
            result = session.run("RETURN $v", {"v": large_str})
            record = result.single()
            self.env.assertEquals(record[0], large_str)

    def test15_many_parameters(self):
        """Test a query with many parameters to stress the parameterized
        query buffer allocation."""
        with self.bolt_con.session() as session:
            params = {f"p{i}": i for i in range(100)}
            placeholders = ', '.join(f'${k}' for k in params)
            result = session.run(f"RETURN {placeholders}", params)
            record = result.single()
            for i in range(100):
                self.env.assertEquals(record[i], i)

    def test16_int16_string16_list16_serialization(self):
        """Verify int16/string16/list16 serialization works correctly."""
        with self.bolt_con.session() as session:
            # int16 range: 128..32767 and -128..-17
            result = session.run("RETURN $a, $b, $c, $d",
                                 {"a": 200, "b": 32767, "c": -100, "d": -32768})
            record = result.single()
            self.env.assertEquals(record[0], 200)
            self.env.assertEquals(record[1], 32767)
            self.env.assertEquals(record[2], -100)
            self.env.assertEquals(record[3], -32768)

            # string16: string with length 256 (requires 16-bit size header)
            s16 = 'B' * 256
            result = session.run("RETURN $v", {"v": s16})
            record = result.single()
            self.env.assertEquals(record[0], s16)

            # list16: list with 256 elements (requires 16-bit size header)
            l16 = list(range(256))
            result = session.run("RETURN $v", {"v": l16})
            record = result.single()
            self.env.assertEquals(record[0], l16)

    def test17_int32_int64_serialization(self):
        """Verify int32/int64 serialization at boundary values."""
        with self.bolt_con.session() as session:
            # int32 range
            result = session.run("RETURN $a, $b, $c, $d",
                                 {"a": 100000, "b": 2147483647,
                                  "c": -100000, "d": -2147483648})
            record = result.single()
            self.env.assertEquals(record[0], 100000)
            self.env.assertEquals(record[1], 2147483647)
            self.env.assertEquals(record[2], -100000)
            self.env.assertEquals(record[3], -2147483648)

            # int64 range
            result = session.run("RETURN $a, $b",
                                 {"a": 9223372036854775807,
                                  "b": -9223372036854775808})
            record = result.single()
            self.env.assertEquals(record[0], 9223372036854775807)
            self.env.assertEquals(record[1], -9223372036854775808)

    def test18_reset_during_session(self):
        """Verify RESET handling works correctly.
        Before the fix, raw pointer arithmetic in the RESET message removal
        could corrupt memory when data spanned buffer chunk boundaries.
        session.reset() sends a RESET message through the bolt protocol."""
        with self.bolt_con.session() as session:
            # run a query, then reset, then run another query
            result = session.run("RETURN 1")
            record = result.single()
            self.env.assertEquals(record[0], 1)

        # after closing and reopening a session, the connection is reused
        # but the state is reset
        with self.bolt_con.session() as session:
            result = session.run("RETURN 'after_reset'")
            record = result.single()
            self.env.assertEquals(record[0], 'after_reset')

    def test19_reset_with_pipelined_queries(self):
        """Test that RESET works correctly when interleaved with queries.
        Uses explicit transaction begin/rollback to trigger RESET."""
        with self.bolt_con.session() as session:
            # begin a transaction then roll it back (triggers RESET-like behavior)
            tx = session.begin_transaction()
            tx.run("RETURN 1").consume()
            tx.rollback()

            # subsequent query on the same session should work correctly
            result = session.run("RETURN 42")
            record = result.single()
            self.env.assertEquals(record[0], 42)

    def test20_rapid_session_cycling(self):
        """Rapidly open/close sessions to stress RESET handling.
        Each session close sends RESET on the pooled connection."""
        for i in range(5):
            with self.bolt_con.session() as session:
                result = session.run("RETURN $i", {"i": i})
                record = result.single()
                self.env.assertEquals(record[0], i)

    def test21_large_string_parameter_value(self):
        """Test parameterized query where write_value must serialize a very
        large string value. Before the fix, write_value wrote into a fixed
        buffer that could overflow; now it uses a growable wbuf_t."""
        with self.bolt_con.session() as session:
            large = 'Z' * 8192
            result = session.run("RETURN $v", {"v": large})
            record = result.single()
            self.env.assertEquals(record[0], large)

    def test22_large_list_parameter_value(self):
        """Test parameterized query with a large list value that forces
        write_value to grow the buffer during recursive serialization."""
        with self.bolt_con.session() as session:
            large_list = list(range(200))
            result = session.run("RETURN $v", {"v": large_list})
            record = result.single()
            self.env.assertEquals(record[0], large_list)

    def test23_nested_map_parameter(self):
        """Test parameterized query with a nested map to exercise
        write_value's recursive map serialization with heap-allocated keys."""
        with self.bolt_con.session() as session:
            nested = {"outer_key": {"inner_key": "inner_value"}}
            result = session.run("RETURN $v", {"v": nested})
            record = result.single()
            self.env.assertEquals(record[0], nested)

    def test24_many_large_parameters(self):
        """Combine many parameters with large values to stress the
        growable query buffer across multiple write_value calls."""
        with self.bolt_con.session() as session:
            params = {f"p{i}": 'V' * 500 for i in range(15)}
            placeholders = ', '.join(f'${k}' for k in params)
            result = session.run(f"RETURN {placeholders}", params)
            record = result.single()
            for i in range(15):
                self.env.assertEquals(record[i], 'V' * 500)

    def test25_multiple_consecutive_rollbacks(self):
        """Test that multiple consecutive transaction rollbacks within the
        same session work correctly. Each rollback sends a RESET-like message;
        the RESET scan loop must continue past the first removed frame to
        handle any subsequent ones in the buffer."""
        with self.bolt_con.session() as session:
            for i in range(3):
                tx = session.begin_transaction()
                tx.run("RETURN $i", {"i": i}).consume()
                tx.rollback()

            # after all rollbacks, the session must still be usable
            result = session.run("RETURN 'after_rollbacks'")
            record = result.single()
            self.env.assertEquals(record[0], 'after_rollbacks')

    def test26_interleaved_rollback_and_queries(self):
        """Stress the RESET scan loop by interleaving rollbacks with
        queries on the same session. Verifies the buffer state is
        consistent after each RESET removal."""
        with self.bolt_con.session() as session:
            for i in range(3):
                # rollback a transaction
                tx = session.begin_transaction()
                tx.run("RETURN 1").consume()
                tx.rollback()

                # immediately run a normal query
                result = session.run("RETURN $v", {"v": i * 100})
                record = result.single()
                self.env.assertEquals(record[0], i * 100)

    def test27_auth_empty_credentials_no_password(self):
        """Verify that authentication with empty credentials still works
        when no requirepass is configured. Before the fix, PING was used
        which always succeeds even when requirepass IS set. Now AUTH with
        empty string is used, which correctly reflects server config."""
        # the test env has no requirepass, so empty auth should still succeed
        driver = GraphDatabase.driver(
            f"bolt://localhost:{BOLT_PORT}", auth=("falkordb", ""))
        with driver.session() as session:
            result = session.run("RETURN 1")
            record = result.single()
            self.env.assertEquals(record[0], 1)
        driver.close()

    def test28_deeply_nested_map_parameter_rejected(self):
        """Test that deeply nested parameters beyond the recursion limit
        cause query construction to fail gracefully rather than crash.
        Before the fix, write_value had no recursion depth limit; a deeply
        nested structure would overflow the C stack."""
        # build a map nested 200 levels deep (exceeds WRITE_VALUE_MAX_DEPTH=128)
        nested = "inner_value"
        for i in range(200):
            nested = {"k": nested}
        with self.bolt_con.session() as session:
            try:
                result = session.run("RETURN $v", {"v": nested})
                result.consume()
                # if server handled it gracefully (returned error), that's fine
            except Exception:
                # a client or server error is acceptable — crash is not
                pass
        # verify the server is still alive after the deeply nested param
        with self.bolt_con.session() as session:
            result = session.run("RETURN 'alive'")
            record = result.single()
            self.env.assertEquals(record[0], 'alive')

    def test29_moderately_nested_map_parameter_succeeds(self):
        """Verify that moderately nested parameters (within the depth
        limit) still work correctly after the recursion limit was added."""
        # 50 levels deep — well within WRITE_VALUE_MAX_DEPTH=128
        nested = "leaf"
        for i in range(50):
            nested = {"k": nested}
        with self.bolt_con.session() as session:
            result = session.run("RETURN $v", {"v": nested})
            record = result.single()
            # verify the innermost value survived the round-trip
            val = record[0]
            for i in range(50):
                self.env.assertContains("k", val)
                val = val["k"]
            self.env.assertEquals(val, "leaf")

    def test30_ws_handshake_upgrade(self):
        """Test that the server accepts a WebSocket upgrade request and
        responds with HTTP 101 Switching Protocols."""
        # close the bolt driver to free server thread-pool threads
        # idle pooled connections block threads preventing new connections
        _bolt_teardown(self)
        s, ok = _ws_connect(BOLT_PORT)
        self.env.assertTrue(ok)
        s.close()

    def test31_ws_bolt_handshake(self):
        """Test Bolt protocol handshake over WebSocket transport.
        After WS upgrade, send the Bolt magic + version proposals and
        verify the server negotiates a valid Bolt version."""
        s, ok = _ws_connect(BOLT_PORT)
        self.env.assertTrue(ok)
        response = _bolt_handshake_over_ws(s)
        # response should be 4 bytes: 2 zero bytes + minor + major
        self.env.assertGreaterEqual(len(response), 4)
        if len(response) >= 4:
            # Bolt version 5.x expected
            self.env.assertEquals(response[-1], 5)
            self.env.assertTrue(response[-2] >= 1)
        s.close()

    def test32_ws_reject_invalid_upgrade(self):
        """Test that the server rejects a non-WebSocket, non-Bolt connection
        by closing the socket."""
        s = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        s.settimeout(5)
        s.connect(('127.0.0.1', BOLT_PORT))
        # send garbage that's neither Bolt handshake nor WS upgrade
        s.sendall(b'INVALID REQUEST\r\n\r\n')
        try:
            data = s.recv(4096)
            # empty response or connection reset both acceptable
            self.env.assertTrue(len(data) == 0 or data == b'')
        except (ConnectionResetError, socket.timeout):
            pass  # expected: server closes connection
        s.close()

    def test33_ws_connection_header_variants(self):
        """Test that WebSocket upgrade works with different Connection
        header formats (comma-separated tokens, mixed case)."""
        s = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        s.settimeout(5)
        s.connect(('127.0.0.1', BOLT_PORT))
        key = base64.b64encode(os.urandom(16)).decode()
        request = (
            "GET / HTTP/1.1\r\n"
            f"Host: 127.0.0.1:{BOLT_PORT}\r\n"
            "Upgrade: websocket\r\n"
            "Connection: keep-alive, Upgrade\r\n"
            f"Sec-WebSocket-Key: {key}\r\n"
            "Sec-WebSocket-Version: 13\r\n"
            f"Origin: http://127.0.0.1:{BOLT_PORT}\r\n"
            "\r\n"
        )
        s.sendall(request.encode())
        response = b''
        try:
            while b'\r\n\r\n' not in response:
                b = s.recv(1)
                if not b:
                    break
                response += b
            self.env.assertTrue(response.startswith(b'HTTP/1.1 101'))
        except socket.timeout:
            self.env.assertTrue(False, "Timeout waiting for WS upgrade response")
        s.close()
