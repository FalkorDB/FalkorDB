from common import *
from itertools import permutations
from enum import Enum
import random


class Connection(Enum):
    Connected = 1
    Disconnected = 2

# TODO: when introducing new encoder/decoder this needs to be updated consider
# using GRAPH.DEBUG command to be able to get this data
keys = {
    b'x': b'\x07\x81\x82\xb6\xa9\x85\xd6\xadh\n\x05\x02x\x00\x02\x1e\x02\x00\x02\x01\x02\x00\x02\x03\x02\x01\x05\x02v\x00\x02\x01\x02\x00\x05\x02N\x00\x02\x01\x02\x01\x05\x02v\x00\x02\x00\x02\x01\x02\x01\x02\n\x02\x00\x02\x01\x02\x00\x02\x01\x02\x00\x02`\x00\x02\x01\x02\x01\x02\x01\x02\x00\x02\x01\x02\x00\x02`\x00\x02\x02\x02\x02\x02\x01\x02\x00\x02\x01\x02\x00\x02`\x00\x02\x03\x02\x03\x02\x01\x02\x00\x02\x01\x02\x00\x02`\x00\x02\x04\x02\x04\x02\x01\x02\x00\x02\x01\x02\x00\x02`\x00\x02\x05\x02\x05\x02\x01\x02\x00\x02\x01\x02\x00\x02`\x00\x02\x06\x02\x06\x02\x01\x02\x00\x02\x01\x02\x00\x02`\x00\x02\x07\x02\x07\x02\x01\x02\x00\x02\x01\x02\x00\x02`\x00\x02\x08\x02\x08\x02\x01\x02\x00\x02\x01\x02\x00\x02`\x00\x02\t\x02\t\x02\x01\x02\x00\x02\x01\x02\x00\x02`\x00\x02\n\x00\t\x00\x84\xf96Z\xd1\x98\xec\xc0',
    b'{x}x_a244836f-fe81-4f8d-8ee2-83fc3fbcf102': b'\x07\x81\x82\xb6\xa9\x86g\xadh\n\x05\x02x\x00\x02\x1e\x02\x00\x02\x01\x02\x00\x02\x03\x02\x01\x05\x02v\x00\x02\x01\x02\x00\x05\x02N\x00\x02\x01\x02\x01\x05\x02v\x00\x02\x00\x02\x01\x02\x01\x02\n\x02\n\x02\x01\x02\x00\x02\x01\x02\x00\x02`\x00\x02\x0b\x02\x0b\x02\x01\x02\x00\x02\x01\x02\x00\x02`\x00\x02\x0c\x02\x0c\x02\x01\x02\x00\x02\x01\x02\x00\x02`\x00\x02\r\x02\r\x02\x01\x02\x00\x02\x01\x02\x00\x02`\x00\x02\x0e\x02\x0e\x02\x01\x02\x00\x02\x01\x02\x00\x02`\x00\x02\x0f\x02\x0f\x02\x01\x02\x00\x02\x01\x02\x00\x02`\x00\x02\x10\x02\x10\x02\x01\x02\x00\x02\x01\x02\x00\x02`\x00\x02\x11\x02\x11\x02\x01\x02\x00\x02\x01\x02\x00\x02`\x00\x02\x12\x02\x12\x02\x01\x02\x00\x02\x01\x02\x00\x02`\x00\x02\x13\x02\x13\x02\x01\x02\x00\x02\x01\x02\x00\x02`\x00\x02\x14\x00\t\x00\x13H\x11\xb8\x15\xd3\xdc~',
    b'{x}x_53ab30bb-1dbb-47b2-a41d-cac3acd68b8c': b'\x07\x81\x82\xb6\xa9\x86g\xadh\n\x05\x02x\x00\x02\x1e\x02\x00\x02\x01\x02\x00\x02\x03\x02\x01\x05\x02v\x00\x02\x01\x02\x00\x05\x02N\x00\x02\x01\x02\x01\x05\x02v\x00\x02\x00\x02\x05\x02\x01\x02\n\x02\x02\x02\x00\x02\x03\x02\x00\x02\x04\x02\x00\x02\x05\x02\x01\x02\x14\x02\x01\x02\x00\x02\x01\x02\x00\x02`\x00\x02\x15\x02\x15\x02\x01\x02\x00\x02\x01\x02\x00\x02`\x00\x02\x16\x02\x16\x02\x01\x02\x00\x02\x01\x02\x00\x02`\x00\x02\x17\x02\x17\x02\x01\x02\x00\x02\x01\x02\x00\x02`\x00\x02\x18\x02\x18\x02\x01\x02\x00\x02\x01\x02\x00\x02`\x00\x02\x19\x02\x19\x02\x01\x02\x00\x02\x01\x02\x00\x02`\x00\x02\x1a\x02\x1a\x02\x01\x02\x00\x02\x01\x02\x00\x02`\x00\x02\x1b\x02\x1b\x02\x01\x02\x00\x02\x01\x02\x00\x02`\x00\x02\x1c\x02\x1c\x02\x01\x02\x00\x02\x01\x02\x00\x02`\x00\x02\x1d\x02\x1d\x02\x01\x02\x00\x02\x01\x02\x00\x02`\x00\x02\x1e\x00\t\x00\x1b\xa64\xd6\xf5\x0bk\xa6'
}

# test to see if replication works as expected when importing data
# RedisGraph should replicate all data using virtual keys mechanism
# in case we imported part of the data validate that we replicate it correctly.

class testReplicationState():
    def __init__(self):
        # skip test if we're running under Valgrind
        if VALGRIND or SANITIZER:
            Environment.skip(None) # valgrind is not working correctly with replication

        self.env, self.db = Env(useSlaves=True, env='oss', moduleArgs='VKEY_MAX_ENTITY_COUNT 10')
        self.master = self.env.getConnection()
        self.slave = self.env.getSlaveConnection()
        info = self.slave.info("Replication")
        self.master_host = info["master_host"]
        self.master_port = info["master_port"]
        self.connection_state = Connection.Connected

    # check that the expected key count exists in both master and slave
    def _check(self, keys_master, keys_slave):
        if keys_master is not None:
            keys = self.master.keys('*')
            if len(keys) != keys_master:
                return False
        if keys_slave is not None:
            # the WAIT command forces master slave sync to complete
            self.master.execute_command("WAIT", "1", "0")
            keys = self.slave.keys('*')
            if len(keys) != keys_slave:
                return False
            if keys_master is not None:
                keys = self.master.keys('*')
                if len(keys) != keys_master:
                    return False
        return True

    # restore the key data and validate the # of keys
    def _step(self, key, keys_master):
        self.master.restore(key, '0', keys[key])
        return self._check(keys_master, None)

    # validate that the imported data exists in both master and slave
    def _test_data(self):
        expected = [[i] for i in range(1, 31)]
        q = "MATCH (n:N) RETURN n.v"
        result = self.master.execute_command("GRAPH.RO_QUERY", "x", q)
        if result[1] != expected:
            return False

        result = self.slave.execute_command("GRAPH.RO_QUERY", "x", q)
        return result[1] == expected
    
    def _connect_replication(self):
        if self.connection_state == Connection.Disconnected:
            self.slave.slaveof(self.master_host, self.master_port)
            self.connection_state = Connection.Connected

    def _disconnect_replication(self):
        if self.connection_state == Connection.Connected:
            self.slave.slaveof()
            self.connection_state = Connection.Disconnected

    def _connection_permutation(self, state, i):
        if state[i] == 1:
            self._connect_replication()   
        elif state[i] == 0:
            self._disconnect_replication()
    
    def _permutation(self, r, d):
        for i in range(r ** d):
            res = []
            for j in range(d):
                res.append((i >> j) % 2)
            yield res

    def _choose_random(self, iter, k):
        is_random = True
        if is_random:
            return random.choices(list(iter), k=k)
        return iter

    def test_replication_permutations(self):
        for scenario in self._choose_random(permutations(keys.keys()), 2):
            for connection_permutation in self._choose_random(self._permutation(2, 5), 3):
                self.master.flushall()
                if not self._check(0, 0):
                    print(f"scenario: {scenario}")
                    print(f"connection_permutation: {connection_permutation}")
                    self.assertTrue(False)

                self._connection_permutation(connection_permutation, 0)

                aux = self.master.execute_command("GRAPH.DEBUG", "AUX", "START")
                self.env.assertEqual(aux, 1)

                self._connection_permutation(connection_permutation, 1)

                if not self._step(scenario[0], 1):
                    print(f"scenario: {scenario}")
                    print(f"connection_permutation: {connection_permutation}")
                    self.assertTrue(False)

                self._connection_permutation(connection_permutation, 2)

                if not self._step(scenario[1], 2):
                    print(f"scenario: {scenario}")
                    print(f"connection_permutation: {connection_permutation}")
                    self.assertTrue(False)

                self._connection_permutation(connection_permutation, 3)

                if not self._step(scenario[2], 3):
                    print(f"scenario: {scenario}")
                    print(f"connection_permutation: {connection_permutation}")
                    self.assertTrue(False)

                self._connection_permutation(connection_permutation, 4)

                aux = self.master.execute_command("GRAPH.DEBUG", "AUX", "END")
                self.env.assertEqual(aux, 0)

                self._connect_replication()

                if not self._check(1, 1):
                    print(f"scenario: {scenario}")
                    print(f"connection_permutation: {connection_permutation}")
                    self.assertTrue(False)

                if not self._test_data():
                    print(f"scenario: {scenario}")
                    print(f"connection_permutation: {connection_permutation}")
                    self.assertTrue(False)

