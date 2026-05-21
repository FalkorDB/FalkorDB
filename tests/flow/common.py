import atexit
import os
import socket
import subprocess
import sys
import time
import uuid
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

# ──── Private container orchestration (CI path) ─────────────────────────────
#
# When a test passes load-time moduleArgs and we're in CI (FALKORDB_TEST_IMAGE
# set), we can't reconfigure the shared `master` service. Spin up a dedicated
# container from the same RC image with the desired -e FALKORDB_ARGS=... and
# point the test client at it.
#
# Containers join the same docker network as the job container so they're
# reachable via their `--network-alias`. Cleanup is best-effort via atexit:
# the GHA runner is destroyed at job end regardless, so leaking is harmless,
# but we prefer to remove eagerly.

_SPAWNED_CIDS = []
_JOB_NETWORK = None


def _job_network():
    """The docker network the GHA job container is attached to.
    Spawned containers join this network so the job container can resolve
    them via their --network-alias."""
    global _JOB_NETWORK
    if _JOB_NETWORK is not None:
        return _JOB_NETWORK
    hostname = socket.gethostname()
    out = subprocess.check_output([
        "docker", "inspect", hostname,
        "-f", "{{range $k,$v := .NetworkSettings.Networks}}{{$k}} {{end}}",
    ]).decode().strip().split()
    # Prefer github_network_* if present (the network GHA created for the job);
    # otherwise fall back to whatever's first.
    for n in out:
        if n.startswith("github_network_"):
            _JOB_NETWORK = n
            return n
    if out:
        _JOB_NETWORK = out[0]
        return _JOB_NETWORK
    raise RuntimeError(
        "could not determine the job container's docker network; "
        "is /var/run/docker.sock mounted and is the host running docker?"
    )


def _wait_for_redis(host, port, attempts=100, interval=0.1):
    r = redis.Redis(host=host, port=port, socket_connect_timeout=1)
    for _ in range(attempts):
        try:
            if r.ping():
                return
        except Exception:
            time.sleep(interval)
    raise RuntimeError(f"redis at {host}:{port} did not become ready after {attempts * interval:.1f}s")


def _spawn_falkordb(image, falkordb_args="", redis_args="", alias=None):
    """Start a falkordb container, return (host, port, container_id).

    `falkordb_args` becomes the module's load-time arg string (CACHE_SIZE 16 ...).
    `redis_args` is forwarded as the redis-server CLI args (for --replicaof, etc.)."""
    alias = alias or f"falkordb-{uuid.uuid4().hex[:8]}"
    cmd = [
        "docker", "run", "-d",
        "--network", _job_network(),
        "--network-alias", alias,
        "-e", f"FALKORDB_ARGS={falkordb_args}",
        "-e", f"REDIS_ARGS={redis_args}",
        image,
    ]
    cid = subprocess.check_output(cmd, stderr=subprocess.STDOUT).decode().strip()
    _SPAWNED_CIDS.append(cid)
    _wait_for_redis(alias, 6379)
    return alias, 6379, cid


@atexit.register
def _cleanup_spawned():
    for cid in _SPAWNED_CIDS:
        subprocess.run(
            ["docker", "rm", "-f", cid],
            stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL, check=False,
        )


# ──── Env() — RLTest Environment wrapper ────────────────────────────────────


def Env(moduleArgs=None, env='oss', useSlaves=False, enableDebugCommand=False, shardsCount=None):
    """Construct an RLTest Environment and a FalkorDB client.

    Two execution modes:

    1. Local dev (FALKORDB_TEST_IMAGE unset):
       RLTest spawns its own redis-server with --loadmodule + moduleArgs.
       Behavior identical to the original Env(). Untouched by the CI path.

    2. CI (FALKORDB_TEST_IMAGE set):
       Spawn a dedicated container from the RC image per Env() call, with
       -e FALKORDB_ARGS=... baked from moduleArgs so cross-key validation
       (TIMEOUT vs TIMEOUT_MAX, immutable CACHE_SIZE/THREAD_COUNT/...) fires
       the same way it would under --loadmodule. For useSlaves=True, an
       additional replica container is spawned on the same docker network
       with REDIS_ARGS='--replicaof <alias> 6379'. Containers live for the
       test process's lifetime and are cleaned up by atexit.

       Every Env() call gets a fresh container — full per-class isolation,
       no cross-test state leakage. Each call adds ~1-2s of container
       startup; cells run in parallel so wall-clock impact is small.
    """
    test_image = os.getenv("FALKORDB_TEST_IMAGE", "")

    # Mode 1: local dev — original RLTest spawn behavior, untouched.
    if not test_image:
        env_obj = Environment(decodeResponses=True, moduleArgs=moduleArgs, env=env,
                              useSlaves=useSlaves, enableDebugCommand=enableDebugCommand,
                              shardsCount=shardsCount)
        # Expose host so tests can use BlockingConnectionPool(host=self.env.host, ...)
        # uniformly across local dev and CI without per-mode branches.
        env_obj.host = "localhost"
        db = FalkorDB("localhost", env_obj.port)
        return (env_obj, db)

    # Mode 2: CI. Always private spawn — every Env() call gets a fresh master
    # (plus replica when useSlaves=True). The RLTest Environment is a stub
    # (env='existing-env') so it doesn't try to manage redis lifecycle itself.
    env_obj = Environment(decodeResponses=True, env='existing-env')
    master_alias, master_port, _ = _spawn_falkordb(
        test_image, falkordb_args=moduleArgs or "")
    host, port = master_alias, master_port
    if useSlaves:
        replica_alias, _, _ = _spawn_falkordb(
            test_image,
            falkordb_args=moduleArgs or "",
            redis_args=f"--replicaof {master_alias} 6379",
        )
        _attach_slave(env_obj, replica_alias, 6379)

    # Some flow tests construct their own connection pools from self.env.port
    # (assuming RLTest's spawned redis on localhost). Override the env stub's
    # port/host so those callsites land on our actual instance. envRunner is
    # the underlying ExistsRedis; tests that read self.env.envRunner.port hit
    # it too.
    env_obj.port = port
    env_obj.host = host
    if hasattr(env_obj, "envRunner") and env_obj.envRunner is not None:
        env_obj.envRunner.port = port
        env_obj.envRunner.host = host

    db = FalkorDB(host, port)
    return (env_obj, db)


def _attach_slave(env_obj, host, port):
    """Expose a slave-connection accessor on the RLTest Environment.

    RLTest's env.getSlaveConnection() normally returns a redis client for the
    replica spawned by RLTest. Under our existing-env stub we override it to
    return a client pointing at the supplied (host, port)."""
    def _get_slave_connection(*_args, **_kwargs):
        return redis.Redis(host=host, port=port, decode_responses=True)
    env_obj.getSlaveConnection = _get_slave_connection


def skip():
    def decorate(f):
        @wraps(f)
        def wrapper(x, *args, **kwargs):
            env = x if isinstance(x, Environment) else x.env
            env.skip()
            return f(x, *args, **kwargs)
        return wrapper
    return decorate
