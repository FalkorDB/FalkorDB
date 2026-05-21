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
_JOB_MOUNTS = None


def _job_mounts():
    """The job container's mount table: list of (host_source, in_container_dest)
    tuples, sorted by destination length (longest first) so the deepest mount
    matches before any parent. Used to translate an in-container path to its
    host-side equivalent for bind-mounting into spawned sibling containers.

    The spawned containers go through the host docker daemon (via the bind-
    mounted /var/run/docker.sock), so `-v <src>:<dest>` interprets <src> as a
    *host* path — not as a path inside the job container. Without this
    translation, mounting e.g. /__w/repo/repo/tests/flow/csvs (a job-container
    path) silently creates a fresh empty directory on the host instead of
    surfacing the test's CSV files."""
    global _JOB_MOUNTS
    if _JOB_MOUNTS is not None:
        return _JOB_MOUNTS
    hostname = socket.gethostname()
    # On bare-metal-host dev runs the hostname doesn't match any container,
    # so `docker inspect` exits non-zero. Treat that as "no mount table" —
    # _host_path_for then returns paths unchanged, which is correct because
    # IMPORT_FOLDER is already a host path in that scenario.
    proc = subprocess.run(
        ["docker", "inspect", hostname,
         "-f", "{{range .Mounts}}{{.Source}}={{.Destination}}\n{{end}}"],
        stdout=subprocess.PIPE, stderr=subprocess.DEVNULL, check=False,
    )
    mounts = []
    if proc.returncode == 0:
        for line in proc.stdout.decode().splitlines():
            line = line.strip()
            if "=" not in line:
                continue
            src, dest = line.split("=", 1)
            mounts.append((src, dest))
        mounts.sort(key=lambda m: len(m[1]), reverse=True)
    _JOB_MOUNTS = mounts
    return mounts


def _host_path_for(in_container_path):
    """Translate an in-container absolute path to the host path that backs it,
    via the job container's mount table. If no mount covers the path, return
    it unchanged (local dev: the spawn happens directly on the host, so the
    path is already the host path)."""
    for src, dest in _job_mounts():
        if in_container_path == dest or in_container_path.startswith(dest.rstrip("/") + "/"):
            return src + in_container_path[len(dest):]
    return in_container_path


def _import_folder_arg(falkordb_args):
    """Extract IMPORT_FOLDER's value from a module-args string. moduleArgs is
    a space-separated `KEY VALUE KEY VALUE ...` sequence (matches the format
    redis-server's --loadmodule consumes), so the value is the token directly
    after IMPORT_FOLDER. Returns None if absent."""
    if not falkordb_args:
        return None
    toks = falkordb_args.split()
    for i, t in enumerate(toks):
        if t == "IMPORT_FOLDER" and i + 1 < len(toks):
            return toks[i + 1]
    return None


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


def _wait_for_redis(host, port, cid, attempts=50, interval=0.1):
    """Wait for redis at host:port to PING. Bails out fast if the container
    has already exited (some tests intentionally pass invalid moduleArgs that
    abort module load; we shouldn't wait the full budget for those).

    Note `retry=Retry(NoBackoff(),0)`: redis-py 7.4 enables connection
    retries by default with effectively no upper bound on ConnectionError,
    so a normal `redis.Redis(...).ping()` against a dead container hangs
    indefinitely regardless of socket_connect_timeout. We need an explicit
    no-retry policy to honor our per-attempt budget."""
    from redis.retry import Retry
    from redis.backoff import NoBackoff
    r = redis.Redis(host=host, port=port,
                    socket_connect_timeout=1,
                    retry=Retry(NoBackoff(), 0))
    for _ in range(attempts):
        # If the container exited (e.g. module load failed), don't bother
        # waiting more — raise immediately so the caller's try/except sees
        # the failure as the test expects.
        inspect_out = subprocess.run(
            ["docker", "inspect", "-f", "{{.State.Running}}", cid],
            stdout=subprocess.PIPE, stderr=subprocess.DEVNULL, check=False,
        ).stdout.decode().strip()
        if inspect_out == "false":
            raise RuntimeError(f"container {cid[:12]} for {host}:{port} exited before becoming ready")
        try:
            if r.ping():
                return
        except Exception:
            time.sleep(interval)
    raise RuntimeError(f"redis at {host}:{port} did not become ready after {attempts * interval:.1f}s")


def _spawn_falkordb(image, falkordb_args="", redis_args="", alias=None,
                    enable_debug_command=False):
    """Start a falkordb container, return (host, port, container_id).

    `falkordb_args` becomes the module's load-time arg string (CACHE_SIZE 16 ...).
    `redis_args` is forwarded as the redis-server CLI args (for --replicaof, etc.).
    `enable_debug_command=True` adds `--enable-debug-command yes` to redis-server,
    matching what tests that pass enableDebugCommand=True to Env() expect under
    the old RLTest model (RLTest's --enable-debug-command CLI flag did the same)."""
    alias = alias or f"falkordb-{uuid.uuid4().hex[:8]}"
    if enable_debug_command:
        redis_args = f"--enable-debug-command yes {redis_args}".strip()
    # If moduleArgs sets IMPORT_FOLDER to a job-container path (e.g.
    # tests/flow/test_load_csv.py points it at .../tests/flow/csvs/), bind-mount
    # the matching host path into the spawned container at the same path so
    # LOAD CSV's file:// resolution under that folder sees the test fixtures.
    # _host_path_for translates via the job container's mount table; on local
    # dev (no mounts to traverse) it returns the path unchanged.
    extra_args = []
    import_folder = _import_folder_arg(falkordb_args)
    if import_folder:
        host_path = _host_path_for(import_folder)
        extra_args += ["-v", f"{host_path}:{import_folder}"]
    cmd = [
        "docker", "run", "-d",
        "--network", _job_network(),
        "--network-alias", alias,
        "-e", f"FALKORDB_ARGS={falkordb_args}",
        "-e", f"REDIS_ARGS={redis_args}",
        *extra_args,
        image,
    ]
    cid = subprocess.check_output(cmd, stderr=subprocess.STDOUT).decode().strip()
    _SPAWNED_CIDS.append(cid)
    _wait_for_redis(alias, 6379, cid)
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
    # (plus replica when useSlaves=True). Order matters: spawn the container
    # *first*, then point RLTest's existing-env address at it via Defaults
    # before constructing the Environment. Environment(env='existing-env')'s
    # startEnv() pings the address; if it doesn't resolve, construction
    # raises before we can override anything. flow.sh's --existing-env-addr
    # is a placeholder that gets superseded here.
    master_alias, master_port, _ = _spawn_falkordb(
        test_image, falkordb_args=moduleArgs or "",
        enable_debug_command=enableDebugCommand)
    host, port = master_alias, master_port
    Defaults.external_addr = f"{master_alias}:{master_port}"
    env_obj = Environment(decodeResponses=True, env='existing-env')
    if useSlaves:
        replica_alias, _, _ = _spawn_falkordb(
            test_image,
            falkordb_args=moduleArgs or "",
            redis_args=f"--replicaof {master_alias} 6379",
            enable_debug_command=enableDebugCommand,
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
