from common import *
import socket
import random
import select
import threading
from neo4j import GraphDatabase

local_port = random.randint(10000, 20000)
# pipe communication between two sockets
def pipe_sockets(source_socket, destination_socket, capture_communication, buffer_size=4096):
    try:
        while True:
            # read from source
            data = source_socket.recv(buffer_size)
            capture_communication['client'].append([data])
            if not data:
                break

            # forward to destination
            destination_socket.sendall(data)

            # read from destination
            data = destination_socket.recv(buffer_size)
            capture_communication['server'].append([data])
            if not data:
                break

            # forward to source
            source_socket.sendall(data)
    except Exception as e:
        pass

# neo4j client sending query to remote server via port forwarding
def neo4j_client(query, params={}):
    print(f"neo4j_client: {query}")
    # use Neo4J bolt driver to connect to our fake bolt server
    bolt_con = GraphDatabase.driver(f"bolt://localhost:{local_port}")

    with bolt_con.session() as session:
        print(f"Connected to bolt server")
        print(f"Running query")
        result = session.run(query, params)
        record = result.single()

# returns bolt communication between client and server
# for the given query
def bolt_communication(query, params={}, bolt_port=6380):
    # Create destination FalkorDB bolt socket
    destination_socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    destination_socket.connect(('localhost', bolt_port))

    # Create a fake local bolt server socket
    server_socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    server_socket.bind(('localhost', local_port))
    server_socket.listen(1)

    client_thread = threading.Thread(target=neo4j_client, args=(query, params))
    client_thread.start()

    # Waiting for connection
    print("Waiting for connection")
    source_socket, source_address = server_socket.accept()

    communication = {'client': [], 'server': []}

    forwarding_thread = threading.Thread(target=pipe_sockets,
    args=(source_socket, destination_socket, communication))
    forwarding_thread.start()

    # Wait for the thread to finish (optional)
    client_thread.join()

    # Close sockets
    source_socket.close()
    destination_socket.close()
    server_socket.close()

    forwarding_thread.join()

    return communication

# try to receive data from socket with timeout
# timeout is in seconds
# returns None if no data is available within the timeout
def recv_with_timeout(sock, timeout=1, buffer_size=1024):
    ready_to_read, _, exceptional = select.select([sock], [], [], timeout)

    if exceptional:
        print("Socket exception")
        return None
    elif ready_to_read:
        data = sock.recv(buffer_size)
        if data == b'':
            print("Socket closed")
            return None
        else:
            print(f"received data: {data}")
            return data
    else:
        print("No data available within timeout")
        return None  # No data available within the specified timeout

class testBoltConnection():
    def __init__(self):
        global bolt_con
        self.bolt_port = 6380
        self.env,_ = Env(moduleArgs=f"BOLT_PORT {self.bolt_port}")

    def test01_handshake(self):
        handshake = b"\x60\x60\xb0\x17\x00\x00\x03\x05\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00"
        for i in range(4):
            new_handshake = bytearray(handshake)
            new_handshake[i] += 1
            with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
                s.connect(("localhost", self.bolt_port))
                # bolt handshake
                s.sendall(new_handshake)
                data = s.recv(1024)
                self.env.assertEquals(data, b"")

    def test02_logon(self):
        with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
            s.connect(("localhost", self.bolt_port))
            # bolt handshake
            s.sendall(b"\x60\x60\xb0\x17\x00\x00\x03\x05\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00")
            data = s.recv(1024)
            self.env.assertEquals(data, b"\x00\x00\x03\x05")

            # hello
            s.sendall(b"\x00\x02\xb0\x01\x00\x00")
            data = s.recv(1024)
            self.env.assertEquals(data[:39], b"\x00K\xb1p\xa2\x86server\x8cNeo4j/5.14.0\x8dconnection_id")

            # logon
            s.sendall(b"\x00\x03\xb1\x6a\xa0\x00\x00")
            data = s.recv(1024)
            self.env.assertEquals(data, b"\x00\x03\xb1p\xa0\x00\x00")

            # run RETURN $a {a: "avi"} pull
            s.sendall(b"\x00\x14\xb3\x10\x89\x52\x45\x54\x55\x52\x4e\x20\x24\x61\xa1\x81\x61\x83\x61\x76\x69\xa0\x00\x00\x00\x08\xb1\x3f\xa1\x81\x6e\xc9\x03\xe8\x00\x00")
            data = s.recv(1024)
            self.env.assertEquals(data, b"\x00\x1e\xb1p\xa3\x87t_first\xc8\x02\x86fields\x91\x82$a\x83qid\xc8\x00\x00\x00\x00\x07\xb1q\x91\x83avi\x00\x00\x00\x0c\xb1p\xa1\x86t_last\xc8\x01\x00\x00")

    def test03_short_read(self):
        # capture communication for the query RETURN 1
        communication = bolt_communication("RETURN 1")

        # all bytes sent by client should be received by server
        sent_data = [msg for submsg in communication['client'] for msg in submsg]
        sent_data = bytes([byte for byte_array in sent_data for byte in byte_array])
        print(f"sent_data: {sent_data}")

        received_data = [msg for submsg in communication['server'] for msg in submsg]
        received_data = bytes([byte for byte_array in received_data for byte in byte_array])
        print(f"received_data: {received_data}")

        total_sent_bytes = len(sent_data)
        total_received_bytes = len(received_data)

        # send up to i bytes in each iteration
        #for i in range(1, total_sent_bytes-1):
        #    print(f"i: {i}")
        #    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        #        s.connect(("localhost", self.bolt_port))
        #        s.setblocking(0)
        #        s.sendall(sent_data[:i])
        #        print(f"Sent {sent_data[:i]}")

        #        # pull data
        #        while recv_with_timeout(s, timeout=0.2, buffer_size=1024) is not None:
        #            pass

        # send entire message
        input("Press Enter to continue...")
        with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
            s.connect(("localhost", self.bolt_port))
            s.sendall(sent_data)

            # read responses
            received = s.recv(total_received_bytes)
            self.env.assertEquals(received, received_data)
