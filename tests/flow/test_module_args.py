import os
import subprocess
import tempfile
from common import *

# Module-load arguments (`--loadmodule falkordb.so NAME VALUE ...`) are parsed
# like C's `Config_Init`: names are case-insensitive, values go through the same
# validation as GRAPH.CONFIG SET, and an invalid value, an unknown name or an
# odd argument count refuses the load (#3268).

# (module args, config to read, expected value)
ACCEPTED = [
    ("CMD_INFO YES",              "CMD_INFO",           1),
    ("cmd_info no",               "CMD_INFO",           0),
    ("DELAY_INDEXING YES",        "DELAY_INDEXING",     1),
    ("cache_size 10",             "CACHE_SIZE",         10),
    ("CACHE_SIZE 5000",           "CACHE_SIZE",         5000),
    ("ASYNC_DELETE no",           "ASYNC_DELETE",       0),
    ("async_delete No",           "ASYNC_DELETE",       0),
    ("timeout_max 7",             "TIMEOUT_MAX",        7),
    ("Thread_Count 3",            "THREAD_COUNT",       3),
    ("max_queued_queries 5",      "MAX_QUEUED_QUERIES", 5),
    ("Vkey_Max_Entity_Count 0",   "VKEY_MAX_ENTITY_COUNT", 0),
    ("node_creation_buffer 600",  "NODE_CREATION_BUFFER", 1024),
    ("TIMEOUT_DEFAULT 5 TIMEOUT_MAX 10", "TIMEOUT_DEFAULT", 5),
]

# (module args, message logged before the load is refused)
REFUSED = [
    ("CMD_INFO garbage",          "Failed setting field 'CMD_INFO'"),
    ("DELAY_INDEXING 1",          "Failed setting field 'DELAY_INDEXING'"),
    ("CACHE_SIZE 0",              "Failed setting field 'CACHE_SIZE'"),
    ("THREAD_COUNT 0",            "Failed setting field 'THREAD_COUNT'"),
    ("OMP_THREAD_COUNT 0",        "Failed setting field 'OMP_THREAD_COUNT'"),
    ("MAX_QUEUED_QUERIES 0",      "Failed setting field 'MAX_QUEUED_QUERIES'"),
    ("JS_HEAP_SIZE 5",            "JS_HEAP_SIZE must be at least 1MB (1048576)"),
    ("FOO 1",                     "Encountered unknown configuration field 'FOO'"),
    ("CACHE_SIZE",                "FalkorDB received 1 arguments, all configurations should be key-value pairs"),
    ("CACHE_SIZE 10 THREAD_COUNT", "FalkorDB received 3 arguments, all configurations should be key-value pairs"),
    ("TIMEOUT 10 TIMEOUT_MAX 10", "The TIMEOUT configuration parameter should be removed"),
    ("TIMEOUT_MAX 10 TIMEOUT 10", "The TIMEOUT configuration parameter is deprecated"),
    ("TIMEOUT_DEFAULT 10 TIMEOUT_MAX 5", "TIMEOUT_MAX configuration parameter cannot be set to a value lower than TIMEOUT_DEFAULT"),
]

# C configs this engine accepts but does not act on
IGNORED = ["ASYNC_DELETE", "DELAY_INDEXING", "DELTA_MAX_PENDING_CHANGES", "EFFECTS_THRESHOLD"]

DEFAULTS = {
    "TIMEOUT": 0,
    "TIMEOUT_DEFAULT": 0,
    "TIMEOUT_MAX": 0,
    "CACHE_SIZE": 25,
    "ASYNC_DELETE": 1,
    "RESULTSET_SIZE": -1,
    "VKEY_MAX_ENTITY_COUNT": 100000,
    "QUERY_MEM_CAPACITY": 0,
    "DELTA_MAX_PENDING_CHANGES": 10000,
    "NODE_CREATION_BUFFER": 16384,
    "CMD_INFO": 1,
    "MAX_INFO_QUERIES": 1000,
    "EFFECTS_THRESHOLD": 300,
    "DELAY_INDEXING": 0,
    "IMPORT_FOLDER": "/var/lib/FalkorDB/import/",
    "TEMP_FOLDER": "/tmp",
    "JS_HEAP_SIZE": 268435456,
    "JS_STACK_SIZE": 1048576,
}


def _load_locally(module_args):
    """Start redis-server with the module and `module_args`, without listening
    on a port (so it exits right after loading the module), and return
    (exit code, or None if it did not exit, and its log)."""
    module = Defaults.module
    if isinstance(module, (list, tuple)):
        module = module[0]
    log_dir = tempfile.mkdtemp()
    log_file = os.path.join(log_dir, "redis.log")
    cmd = [Defaults.binary, "--port", "0", "--save", "", "--dir", log_dir,
           "--logfile", log_file, "--loadmodule", module, *module_args.split()]
    proc = subprocess.Popen(cmd, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    try:
        rc = proc.wait(timeout=20)
    except subprocess.TimeoutExpired:
        proc.kill()
        proc.wait()
        rc = None
    with open(log_file) as f:
        log = f.read()
    return rc, log


class testModuleArgs():
    def __init__(self):
        self.env, self.db = Env()

    def test01_defaults(self):
        """No module arguments: every config keeps its default."""
        for name, expected in DEFAULTS.items():
            self.env.assertEqual(self.db.config_get(name), expected, message=name)

    def test02_accepted(self):
        """Valid arguments take effect, whatever the case of their name."""
        for args, name, expected in ACCEPTED:
            env, db = Env(moduleArgs=args)
            env.assertEqual(db.config_get(name), expected, message=args)

    def test03_refused(self):
        """Invalid arguments refuse the load, as in C."""
        if SANITIZER:
            # a refused load exits with the module half-initialised
            self.env.skip()

        for args, err in REFUSED:
            loaded = True
            try:
                env, _ = Env(moduleArgs=args)
                env.getConnection().ping()
            except Exception:
                loaded = False
            self.env.assertFalse(loaded, message=args)

        # Locally, also check the server exits and logs why.
        if os.getenv("FALKORDB_TEST_IMAGE"):
            return
        for args, err in REFUSED:
            rc, log = _load_locally(args)
            self.env.assertNotEqual(rc, None, message=args)
            self.env.assertNotEqual(rc, 0, message=args)
            self.env.assertContains(err, log, message=args)
            self.env.assertContains("initialization failed. Module not loaded", log, message=args)

    def test04_ignored_configs_logged(self):
        """C configs this engine does not act on are accepted, stored, and
        logged as deprecated."""
        args = "ASYNC_DELETE no delay_indexing yes DELTA_MAX_PENDING_CHANGES 5 EFFECTS_THRESHOLD 7"
        env, db = Env(moduleArgs=args)
        env.assertEqual(db.config_get("ASYNC_DELETE"), 0)
        env.assertEqual(db.config_get("DELAY_INDEXING"), 1)
        env.assertEqual(db.config_get("DELTA_MAX_PENDING_CHANGES"), 5)
        env.assertEqual(db.config_get("EFFECTS_THRESHOLD"), 7)
        if env.log_path is not None:
            with open(env.log_path) as f:
                log = f.read()
            for name in IGNORED:
                env.assertContains(
                    f"configuration field '{name}' is deprecated and has no effect", log)
