import os
import sys
from functools import wraps

from RLTest import Env as Environment, Defaults

import redis
from redis import ResponseError
from falkordb import FalkorDB, Graph, Node, Edge, Path, ExecutionPlan

from base import FlowTestsBase

Defaults.decode_responses = True

SANITIZER     = os.getenv('SANITIZER', '')      != ''
CODE_COVERAGE = os.getenv('CODE_COVERAGE', '0') == '1'

# Normalized OS name for cross-platform test guards (matches the C FalkorDB
# convention used by tests/memcheck): "macos", "linux", "windows", or the
# raw sys.platform value if unrecognized.
OS = {'darwin': 'macos', 'linux': 'linux', 'win32': 'windows'}.get(sys.platform, sys.platform)

# Module configs that can only be set at load time (--loadmodule key value).
# Mirrors src/lib.rs's ConfigurationFlags::IMMUTABLE list. If a test passes any
# of these via moduleArgs and we're under EXISTING_ENV, the helper falls back
# to spawning a private redis via RLTest's env='oss' mode — the running
# service container can't honor these via GRAPH.CONFIG SET.
_LOAD_TIME_CONFIG_KEYS = {
    "CACHE_SIZE", "THREAD_COUNT", "NODE_CREATION_BUFFER",
    "VKEY_MAX_ENTITY_COUNT", "IMPORT_FOLDER", "TEMP_FOLDER",
    "JS_HEAP_SIZE", "JS_STACK_SIZE", "CMD_INFO", "DELAY_INDEXING",
    "BOLT_PORT",  # atomic but doesn't restart the bolt server post-load
}


def _module_args_to_pairs(s):
    """Parse a space-separated 'K1 V1 K2 V2 ...' moduleArgs string into pairs."""
    parts = s.split()
    if len(parts) % 2 != 0:
        raise ValueError(f"moduleArgs has odd token count: {s!r}")
    return list(zip(parts[0::2], parts[1::2]))


def _has_load_time_arg(module_args):
    if not module_args:
        return False
    return any(k.upper() in _LOAD_TIME_CONFIG_KEYS
               for k, _ in _module_args_to_pairs(module_args))


def Env(moduleArgs=None, env='oss', useSlaves=False, enableDebugCommand=False, shardsCount=None):
    # Existing-env mode: connect to an already-running redis (CI services-block).
    # The address comes from flow.sh's --existing-env-addr; FalkorDB client picks
    # up the same host/port from env so it doesn't fall back to env.port (unreliable
    # under existing-env).
    existing_env = os.getenv("EXISTING_ENV", "") == "1"
    # Tracks which moduleArgs (if any) need to be applied to the running redis
    # via GRAPH.CONFIG SET after we connect.
    runtime_config_pairs = []
    if existing_env:
        if useSlaves:
            # The service-container topology is a single redis. Replica-aware
            # tests can't be run against it.
            env_obj = Environment(decodeResponses=True, env=env)
            env_obj.skip()
            return (env_obj, None)
        if _has_load_time_arg(moduleArgs):
            # Test asked for module config that can't be set on a running
            # redis. Fall back to RLTest's spawn mode — it'll start its own
            # redis with --loadmodule <so> + moduleArgs. The .so lives at
            # target/release/libfalkordb.so in CI (extracted from the service
            # container) or wherever cargo built it locally.
            env = 'oss'
        else:
            # All moduleArgs (if any) are runtime-settable. Stash them so we
            # can issue GRAPH.CONFIG SET after the existing-env connect.
            if moduleArgs:
                runtime_config_pairs = _module_args_to_pairs(moduleArgs)
                moduleArgs = None  # don't pass through; existing-env ignores it
            env = 'existing-env'
    env = Environment(decodeResponses=True, moduleArgs=moduleArgs, env=env,
                      useSlaves=useSlaves, enableDebugCommand=enableDebugCommand, shardsCount=shardsCount)
    if existing_env and env.env == 'existing-env':
        # Honor FALKORDB_HOST/PORT from the workflow's env block.
        host = os.getenv("FALKORDB_HOST", "localhost")
        port_str = os.getenv("FALKORDB_PORT", "6379")
        try:
            port = int(port_str)
        except ValueError as e:
            raise RuntimeError(
                f"FALKORDB_PORT must be an integer, got {port_str!r}"
            ) from e
    else:
        # Either we're in local-dev mode, or we fell back to env='oss' because
        # of load-time moduleArgs. Either way, RLTest manages its own port.
        host = "localhost"
        port = env.port
    db  = FalkorDB(host, port)
    # Apply runtime-settable moduleArgs against the now-connected redis. Doing
    # this after the client is constructed (rather than before Environment())
    # is correct: existing-env's redis is already up, runtime configs take
    # effect immediately.
    if runtime_config_pairs:
        conn = db.connection
        for key, value in runtime_config_pairs:
            conn.execute_command("GRAPH.CONFIG", "SET", key, value)
    return (env, db)

def skip():
    def decorate(f):
        @wraps(f)
        def wrapper(x, *args, **kwargs):
            env = x if isinstance(x, Environment) else x.env
            env.skip()
            return f(x, *args, **kwargs)
        return wrapper
    return decorate
