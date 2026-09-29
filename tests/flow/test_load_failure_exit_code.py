from common import *

import os
import time
import subprocess

# Redis ends a failed data load with a hard-coded exit code: 1, or 0 when
# redis-check-rdb, which it runs in-process after a failed load from a file,
# finds the file structurally sound. FalkorDB replaces that status with
# EX_DATAERR so an orchestrator can tell a failed load apart from any other exit
LOAD_FAILURE_EXIT_CODE = 65

GRAPH_ID = "g"

# each test ends the server process, so each one lives in its own class / Env
# the graph must stay in a single key: these tests are about the exit code,
# not about how the virtual key decoder copes with a damaged value
MODULE_ARGS = "VKEY_MAX_ENTITY_COUNT 100000"


# CRC-64 (Jones variant) - the checksum Redis stamps at the end of an RDB file
def _crc64(data):
    POLY = 0x95ac9329ac4bc9b5
    crc = 0
    for byte in data:
        crc ^= byte
        for _ in range(8):
            crc = (crc >> 1) ^ POLY if (crc & 1) else (crc >> 1)
    return crc & 0xFFFFFFFFFFFFFFFF


# the module id of the graph key sits right after its RDB type and name:
# RDB_TYPE_MODULE_2, name length, name, then 0x81 + big-endian u64 id whose
# low 10 bits are the encoding version
def _module_id_offset(rdb):
    needle = bytes([7, len(GRAPH_ID)]) + GRAPH_ID.encode() + b"\x81"
    at = rdb.find(needle)
    assert at > 0, "graph key not found in the RDB"
    return at + len(needle)


# FalkorDB refuses the graph (its encoding version is from the future) while
# the file stays structurally sound, with a correct checksum
def _reject_graph(rdb):
    rdb = bytearray(rdb)
    off = _module_id_offset(rdb) + 6
    lo  = int.from_bytes(rdb[off:off + 2], "big")
    rdb[off:off + 2] = ((lo & ~0x3ff) | 0x3ff).to_bytes(2, "big")
    rdb[-8:] = _crc64(bytes(rdb[:-8])).to_bytes(8, "little")
    return bytes(rdb)


# the file ends in the middle of the graph's value
def _truncate(rdb):
    return rdb[:len(rdb) * 57 // 100]


# the data is intact, the checksum is not
def _break_checksum(rdb):
    return rdb[:-1] + bytes([rdb[-1] ^ 0xff])


class LoadExitCodeBase():
    def __init__(self):
        self.env, self.db = Env(moduleArgs=MODULE_ARGS, env='oss')
        self.conn = self.env.getConnection()

        self.conn.execute_command("GRAPH.QUERY", GRAPH_ID,
            "UNWIND range(1, 1000) AS i CREATE (:N {v: i})-[:R {w: i}]->(:M {v: i})")
        self.conn.save()

        self.rdb_dir = self.conn.config_get("dir")["dir"]
        rdb_file = self.conn.config_get("dbfilename")["dbfilename"]
        self.rdb_path = os.path.join(self.rdb_dir, rdb_file)

    # stop the server, apply `damage` to its RDB and boot a new server process
    # with the same command line: the RDB is loaded at startup, as in production
    def boot(self, damage=None):
        runner = self.env.envRunner
        args = runner.masterProcess.args

        try:
            self.conn.execute_command("SHUTDOWN", "NOSAVE")
        except Exception:
            pass # the server closes the connection as it exits
        runner.masterProcess.wait(timeout=60)
        runner.masterProcess = None

        if damage is not None:
            with open(self.rdb_path, "rb") as f:
                rdb = f.read()
            with open(self.rdb_path, "wb") as f:
                f.write(damage(rdb))

        return subprocess.Popen(args, cwd=self.rdb_dir,
                                stdout=subprocess.DEVNULL,
                                stderr=subprocess.DEVNULL)

    # exit status of `process`, negative when it was killed by a signal
    def exit_code(self, process):
        try:
            return process.wait(timeout=120)
        except subprocess.TimeoutExpired:
            process.kill()
            process.wait()
            raise
        finally:
            # the damaged file must not be picked up by anything that follows
            if os.path.exists(self.rdb_path):
                os.remove(self.rdb_path)

    # the server log, when RLTest writes one
    def log(self):
        name = self.env.envRunner._getFileName("master", ".log")
        try:
            with open(os.path.join(self.env.logDir, name)) as f:
                return f.read()
        except FileNotFoundError:
            return None

    def assert_load_failure_exit(self, damage):
        code = self.exit_code(self.boot(damage))
        self.env.assertEquals(code, LOAD_FAILURE_EXIT_CODE)

        # FalkorDB replaced the status, rather than something else exiting 65
        log = self.log()
        if log is not None:
            self.env.assertContains(
                "exiting with code %d" % LOAD_FAILURE_EXIT_CODE, log)


class testRejectedGraphExitCode(LoadExitCodeBase):
    def __init__(self):
        super().__init__()

    # a graph FalkorDB refuses, in a file whose checksum is correct:
    # redis-check-rdb finds nothing wrong, so Redis itself would exit 0
    def test_rejected_graph_exit_code(self):
        self.assert_load_failure_exit(_reject_graph)


class testTruncatedRdbExitCode(LoadExitCodeBase):
    def __init__(self):
        super().__init__()

    # a short read inside the graph's value, Redis itself would exit 1
    def test_truncated_rdb_exit_code(self):
        self.assert_load_failure_exit(_truncate)


class testBadChecksumExitCode(LoadExitCodeBase):
    def __init__(self):
        super().__init__()

    # the graph loads, then the checksum check fails, Redis itself would exit 1
    def test_bad_checksum_exit_code(self):
        self.assert_load_failure_exit(_break_checksum)


class testCleanExitAfterLoad(LoadExitCodeBase):
    def __init__(self):
        super().__init__()

    # a load that succeeded must not leave the process marked as loading:
    # a later shutdown exits 0
    def test_clean_exit_after_load(self):
        process = self.boot()

        # wait for the startup load to finish and the graph to be back
        conn = redis.Redis(port=self.env.envRunner.port, decode_responses=True)
        deadline = time.time() + 60
        while True:
            try:
                res = conn.execute_command("GRAPH.RO_QUERY", GRAPH_ID,
                                           "MATCH (n) RETURN count(n)")
                break
            except (redis.exceptions.ConnectionError,
                    redis.exceptions.BusyLoadingError):
                if time.time() > deadline:
                    raise
                time.sleep(0.1)
        self.env.assertEquals(res[1][0][0], 2000)

        try:
            conn.execute_command("SHUTDOWN", "NOSAVE")
        except Exception:
            pass # the server closes the connection as it exits

        self.env.assertEquals(self.exit_code(process), 0)
