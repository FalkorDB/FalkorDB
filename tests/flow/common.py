import atexit
import os
import socket
import subprocess
import sys
import tempfile
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
    via the job container's mount table. Returns (host_path, covered_by_mount).
    When not covered, host_path is the input unchanged — callers must NOT
    bind-mount in that case, because the host daemon would interpret it as a
    fresh path and create an empty directory there."""
    for src, dest in _job_mounts():
        if in_container_path == dest or in_container_path.startswith(dest.rstrip("/") + "/"):
            return src + in_container_path[len(dest):], True
    return in_container_path, False


_PATH_ARG_KEYS = ("IMPORT_FOLDER", "TEMP_FOLDER")


def _path_args_in(falkordb_args):
    """Yield (key, value) for path-shaped module args present in the args
    string. moduleArgs is a space-separated `KEY VALUE KEY VALUE ...`
    sequence (matches what redis-server's --loadmodule consumes), so the
    value is the token directly after the key."""
    if not falkordb_args:
        return
    toks = falkordb_args.split()
    for i, t in enumerate(toks):
        if t in _PATH_ARG_KEYS and i + 1 < len(toks):
            yield t, toks[i + 1]


def mountable_mkdtemp(*args, **kwargs):
    """tempfile.mkdtemp(), but under FALKORDB_TEST_IMAGE the dir is rooted in
    a workspace-relative location that's covered by the job container's
    bind-mount table. Spawned sibling containers then see the same directory
    via _spawn_falkordb's bind-mount logic. Local dev gets plain mkdtemp."""
    if os.getenv("FALKORDB_TEST_IMAGE") and "dir" not in kwargs:
        kwargs["dir"] = _ci_tmpdir()
    return tempfile.mkdtemp(*args, **kwargs)


def mountable_mkstemp(*args, **kwargs):
    """Workspace-rooted tempfile.mkstemp(). See mountable_mkdtemp() for why."""
    if os.getenv("FALKORDB_TEST_IMAGE") and "dir" not in kwargs:
        kwargs["dir"] = _ci_tmpdir()
    return tempfile.mkstemp(*args, **kwargs)


def _ci_tmpdir():
    p = os.path.join(os.path.dirname(os.path.abspath(__file__)), ".ci-tmp")
    os.makedirs(p, exist_ok=True)
    return p


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
    # Bind-mount path-shaped module args (IMPORT_FOLDER, TEMP_FOLDER) so the
    # spawned sibling container resolves them to the same content the test
    # set up on the job-container side. Two guards:
    #   - the path must be covered by the job container's mount table, so
    #     docker run -v <host>:<dest> targets a real host directory rather
    #     than silently creating an empty one
    #   - the path must currently exist, so tests that intentionally pass
    #     non-existent paths (testConfigTempFolder.test_02) still see the
    #     module fail validation
    extra_args = []
    for key, val in _path_args_in(falkordb_args):
        host_path, covered = _host_path_for(val)
        if covered and os.path.exists(val):
            extra_args += ["-v", f"{host_path}:{val}"]
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
    # Capture docker stdout/stderr for each spawned container before tearing
    # it down. tests/flow/logs/ is the directory RLTest uses for its own
    # per-test logs and is uploaded as a GHA artifact on failure, so dropping
    # files there means a failed run carries everything needed for post-mortem:
    #   - redis-server lifecycle messages (default destination: stderr)
    #   - module RedisModule_Log output (routes through redis's logger)
    #   - ASAN/LSan reports (default destination: stderr, no log_path set)
    log_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "logs")
    os.makedirs(log_dir, exist_ok=True)
    for cid in _SPAWNED_CIDS:
        short = cid[:12]
        # Graceful SIGTERM with a 10s grace before SIGKILL gives ASAN's
        # at-exit leak detection time to print before the container dies;
        # `docker rm -f` alone would SIGKILL immediately and drop those.
        subprocess.run(
            ["docker", "stop", "--time", "10", cid],
            stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL, check=False,
        )
        with open(os.path.join(log_dir, f"spawned-{short}.log"), "wb") as f:
            subprocess.run(
                ["docker", "logs", cid],
                stdout=f, stderr=subprocess.STDOUT, check=False,
            )
        subprocess.run(
            ["docker", "rm", "-f", cid],
            stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL, check=False,
        )


# ──── Env() — RLTest Environment wrapper ────────────────────────────────────


def Env(moduleArgs=None, env='oss', useSlaves=False, enableDebugCommand=False, shardsCount=None):
    """Construct an RLTest Environment and a FalkorDB client.

    Three execution modes:

    1. Local dev (FALKORDB_TEST_IMAGE unset):
       RLTest spawns its own redis-server with --loadmodule + moduleArgs.
       Behavior identical to the original Env(). Untouched by the CI path.

    2. CI services mode (FALKORDB_TEST_IMAGE set, FALKORDB_USE_SERVICE=1):
       Connect to a long-running GHA `services:` container at
       FALKORDB_HOST:FALKORDB_PORT — no docker run, no per-class spawn.
       Selected by the matrix for test files whose Env() invocations carry
       no special flags (classified by tests/flow/test_matrix_split.py).
       FLUSHALL between Env() calls gives class-level isolation since the
       service is shared across all classes in the cell.
       Passing any special flag here is a classifier miss and raises.

    3. CI spawn mode (FALKORDB_TEST_IMAGE set, FALKORDB_USE_SERVICE unset):
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
    use_service = os.getenv("FALKORDB_USE_SERVICE", "") != ""

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

    # Mode 2: services — shared GHA service container per matrix cell.
    # enableDebugCommand=True is a no-op here because the services job runs
    # the falkordb container with REDIS_ARGS=--enable-debug-command yes; the
    # flag was historically a spawn-trigger but the classifier no longer
    # routes on it. useSlaves=True is supported via a second `replica`
    # service the job runs with --replicaof falkordb 6379.
    if use_service:
        if env != 'oss' or shardsCount:
            raise RuntimeError(
                "Services-mode Env() doesn't support oss-cluster / "
                "shardsCount. This file was routed to the services bucket "
                "but uses a cluster flag; either reclassify in "
                "tests/flow/test_matrix_split.py or remove the flag. "
                f"Got env={env!r} shardsCount={shardsCount}."
            )
        host = os.getenv("FALKORDB_HOST", "falkordb")
        port = int(os.getenv("FALKORDB_PORT", "6379"))
        # Wipe the shared service back to a clean state so each class starts
        # fresh — mirrors the per-class isolation spawn-mode gets for free.
        redis.Redis(host=host, port=port).flushall()
        # moduleArgs (runtime-mutable keys only — the classifier guarantees
        # immutable keys go to spawn) are applied via GRAPH.CONFIG SET.
        # A failed SET propagates as RuntimeError so tests that expect
        # invalid-args to raise still catch via their try/except wrappers
        # (e.g. test_timeout.test05_invalid_loadtime_config relies on this).
        if moduleArgs:
            _apply_module_args_via_config_set(host, port, moduleArgs)
        Defaults.external_addr = f"{host}:{port}"
        env_obj = Environment(decodeResponses=True, env='existing-env')
        env_obj.port = port
        env_obj.host = host
        if hasattr(env_obj, "envRunner") and env_obj.envRunner is not None:
            env_obj.envRunner.port = port
            env_obj.envRunner.host = host
        if useSlaves:
            replica_host = os.getenv("FALKORDB_REPLICA_HOST", "replica")
            replica_port = int(os.getenv("FALKORDB_REPLICA_PORT", "6379"))
            # Both service containers come up in parallel; the replica may
            # need a few seconds to discover and sync from master before
            # the test starts querying it. Block until INFO replication
            # reports master_link_status=up so the test never hits a
            # half-replicated state.
            _wait_for_replication(replica_host, replica_port)
            _attach_slave(env_obj, replica_host, replica_port)
        db = FalkorDB(host, port)
        return (env_obj, db)

    # Mode 3: CI spawn — every Env() call gets a fresh master container
    # (plus replica when useSlaves=True). Order matters: spawn the container
    # *first*, then point RLTest's existing-env address at it via Defaults
    # before constructing the Environment. Environment(env='existing-env')'s
    # startEnv() pings the address; if it doesn't resolve, construction
    # raises before we can override anything. flow.sh's --existing-env-addr
    # is a placeholder that gets superseded here.
    master_alias, master_port, master_cid = _spawn_falkordb(
        test_image, falkordb_args=moduleArgs or "",
        enable_debug_command=enableDebugCommand)
    host, port = master_alias, master_port
    Defaults.external_addr = f"{master_alias}:{master_port}"
    env_obj = Environment(decodeResponses=True, env='existing-env')
    # Expose a file-handle-like reader over the container's stdout/stderr,
    # which holds the same messages RLTest's master.log would. Tests that
    # parse redis logs (test_encode_decode.test_10) use this when present.
    env_obj.read_log = _make_docker_log_reader(master_cid)
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


def _apply_module_args_via_config_set(host, port, module_args):
    """Apply space-separated `KEY VALUE` pairs via GRAPH.CONFIG SET.

    The classifier (tests/flow/test_matrix_split.py) only routes files to
    services if every moduleArgs key is runtime-mutable. So this can apply
    each pair without checking. If a SET returns an error (e.g. invalid
    value, or a combination the module rejects like TIMEOUT alongside
    TIMEOUT_DEFAULT), we surface it as RuntimeError — tests that
    intentionally expect Env() to raise (test_timeout.test05) still catch
    via their bare except."""
    toks = module_args.split()
    if len(toks) % 2 != 0:
        raise RuntimeError(
            f"moduleArgs must be space-separated KEY VALUE pairs: "
            f"got {module_args!r}"
        )
    conn = redis.Redis(host=host, port=port)
    for i in range(0, len(toks), 2):
        key, value = toks[i], toks[i + 1]
        try:
            conn.execute_command("GRAPH.CONFIG", "SET", key, value)
        except redis.ResponseError as e:
            raise RuntimeError(
                f"GRAPH.CONFIG SET {key} {value} failed: {e}. "
                "If this key is actually immutable, add it to "
                "IMMUTABLE_MODULE_ARGS in tests/flow/test_matrix_split.py "
                "so this file routes to the spawn bucket."
            ) from e


def _wait_for_replication(host, port, attempts=50, interval=0.2):
    """Block until the replica at host:port reports it has finished syncing
    from its master (master_link_status=up, master_sync_in_progress=0).
    Services come up in parallel and the replica's `--replicaof` only
    starts trying to connect once redis is initialized, so the first few
    INFO replication calls may show 'down' or 'connecting'."""
    r = redis.Redis(host=host, port=port)
    for _ in range(attempts):
        try:
            info = r.info('replication')
            if (info.get('master_link_status') == 'up'
                    and int(info.get('master_sync_in_progress', 1)) == 0):
                return
        except Exception:
            pass
        time.sleep(interval)
    raise RuntimeError(
        f"replica at {host}:{port} did not reach master_link_status=up "
        f"in {attempts * interval:.1f}s")


def _make_docker_log_reader(cid):
    """Returns a callable read(role='master') that mimics RLTest's file-handle
    log access: each call returns only the new bytes appended since the prior
    call, decoded. Backed by `docker logs <cid>`, which captures both redis's
    stdout (where it logs by default when no --logfile is given) and the
    module's RedisModule_Log output."""
    state = {"offset": 0}
    def read(_role="master"):
        out = subprocess.run(
            ["docker", "logs", cid],
            stdout=subprocess.PIPE, stderr=subprocess.STDOUT, check=False,
        ).stdout
        new_bytes = out[state["offset"]:]
        state["offset"] = len(out)
        return new_bytes.decode("utf-8", errors="replace")
    return read


def _attach_slave(env_obj, host, port):
    """Expose a slave-connection accessor on the RLTest Environment.

    RLTest's env.getSlaveConnection() normally returns a redis client for the
    replica spawned by RLTest. Under our existing-env stub we override it to
    return a client pointing at the supplied (host, port).

    Also stash `replica_host`/`replica_port` on the env so test code that
    constructs its own clients (FalkorDB, AsyncRedis, etc.) at the replica
    can address it directly. RLTest's original model puts the replica at
    port+1 on localhost — that assumption doesn't hold under docker-per-class
    where the replica is a sibling container on a network alias."""
    def _get_slave_connection(*_args, **_kwargs):
        return redis.Redis(host=host, port=port, decode_responses=True)
    env_obj.getSlaveConnection = _get_slave_connection
    env_obj.replica_host = host
    env_obj.replica_port = port


def skip():
    def decorate(f):
        @wraps(f)
        def wrapper(x, *args, **kwargs):
            env = x if isinstance(x, Environment) else x.env
            env.skip()
            return f(x, *args, **kwargs)
        return wrapper
    return decorate
