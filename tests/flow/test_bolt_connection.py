from common import *
import socket

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