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
from redis.retry import Retry
from redis.backoff import NoBackoff
from falkordb import FalkorDB, Graph, Node, Edge, Path, ExecutionPlan

from base import FlowTestsBase

Defaults.decode_responses = True

SANITIZER     = os.getenv('SANITIZER', '')      != ''
CODE_COVERAGE = os.getenv('CODE_COVERAGE', '0') == '1'

# Per-socket read timeout (seconds) applied to redis clients constructed
# from this module. Under ASAN/coverage instrumentation the GraphBLAS+
# falkordb binary runs 3-5x slower (see PR #539); tests that issue heavy
# synthetic workloads (test_index_create.test08_async_index_creation —
# 1 M-node CREATE, test_slowlog.populate_slowlog — 19 concurrent
# UNWIND(250k), test_pending_queries_limit — 16 concurrent UNWIND(1M))
# overrun redis-py's default read deadline and trip a misleading
# "Timeout reading from socket" instead of completing or surfacing the
# real assertion. Bump the deadline generously on instrumented builds;
# leave it unbounded otherwise.
SOCKET_TIMEOUT = 600 if (SANITIZER or CODE_COVERAGE) else None

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


# Path inside the spawned container where redis writes --logfile output
# (and where _log_mount_args bind-mounts the host-side per-spawn dir).
LOG_DIR_IN_CONTAINER = "/var/lib/falkordb/logs"
LOG_FILE_NAME = "redis.log"


def _log_mount_args(alias):
    """Bind-mount a per-spawn logs dir into the spawned container so redis's
    --logfile output is reachable from the test process. Job container and
    spawned container share the host workspace bind-mount, so writing under
    ${GITHUB_WORKSPACE}/redis-logs/<alias>/ lets the test open the file
    directly (no `docker logs` subprocess).

    Returns (docker -v args, in-container path of the log file). The path is
    the value the test process passes to open() — same workspace path the
    container sees, because both sides share the bind-mount. When the
    workspace isn't bind-mountable (local dev, no GITHUB_WORKSPACE), returns
    ([], None) and the caller skips both the mount and --logfile injection,
    leaving redis on its default stdout logger."""
    ws = os.environ.get("GITHUB_WORKSPACE")
    if not ws:
        return [], None
    log_dir = os.path.join(ws, "redis-logs", alias)
    os.makedirs(log_dir, mode=0o777, exist_ok=True)
    host_path, covered = _host_path_for(log_dir)
    if not covered:
        return [], None
    return (
        ["-v", f"{host_path}:{LOG_DIR_IN_CONTAINER}"],
        os.path.join(log_dir, LOG_FILE_NAME),
    )


def _coverage_mount_args():
    """When CODE_COVERAGE=1 under CI, bind-mount a per-spawn subdir of
    ${GITHUB_WORKSPACE}/cov onto /var/lib/falkordb/cov inside the spawned
    container. The coverage image's LLVM_PROFILE_FILE writes profraws there;
    the bind-mount surfaces them on the host runner where _flow-flavour.yml's
    collect step picks them up. Per-spawn UUID subdir prevents PID-1 filename
    collisions across containers in the same cell (every container's main pid
    is 1, so the %p substitution alone isn't unique)."""
    if not CODE_COVERAGE:
        return []
    ws = os.environ.get("GITHUB_WORKSPACE")
    if not ws:
        return []
    in_container = os.path.join(ws, "cov", uuid.uuid4().hex)
    os.makedirs(in_container, mode=0o777, exist_ok=True)
    host_path, covered = _host_path_for(in_container)
    if not covered:
        return []
    return ["-v", f"{host_path}:/var/lib/falkordb/cov"]


def _spawn_falkordb(image, falkordb_args="", redis_args="", alias=None,
                    enable_debug_command=False):
    """Start a falkordb container, return (host, port, container_id, log_path).

    `falkordb_args` becomes the module's load-time arg string (CACHE_SIZE 16 ...).
    `redis_args` is forwarded as the redis-server CLI args (for --replicaof, etc.).
    `enable_debug_command=True` adds `--enable-debug-command yes` to redis-server,
    matching what tests that pass enableDebugCommand=True to Env() expect under
    the old RLTest model (RLTest's --enable-debug-command CLI flag did the same).

    `log_path` is the test-process-visible path of redis's --logfile output,
    or None when the workspace isn't bind-mountable (e.g. bare-metal local
    spawn). Tests that need to read the log (test_encode_decode.test_10) open
    it via env.log_path."""
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
    extra_args += _coverage_mount_args()
    log_mount, log_path = _log_mount_args(alias)
    extra_args += log_mount
    if log_path is not None:
        # --logfile redirects redis's own logger (RedisModule_Log routes
        # through it too) to the bind-mounted file. Stderr-only output —
        # ASAN/LSan reports, Rust panic backtraces, redis pre-init noise —
        # still hits the container's stdout/stderr and is captured by the
        # atexit `docker logs <cid>` cleanup. Both signals coexist.
        redis_args = f"--logfile {LOG_DIR_IN_CONTAINER}/{LOG_FILE_NAME} {redis_args}".strip()
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
    return alias, 6379, cid, log_path


def _spawn_falkordb_cluster(image, shards, falkordb_args="",
                            enable_debug_command=False):
    """Spawn `shards` falkordb sibling containers wired up as a Redis Cluster.

    Each shard runs with --cluster-enabled yes and announces its docker
    --network-alias as the cluster bus hostname so peers reach it by the
    same name the test process uses (no docker-network IP discovery needed).
    Slots are split evenly across shards by `_bootstrap_cluster`; no
    replicas are provisioned (the spawn-mode cluster path is masters-only
    until a test needs replicas).

    Returns the list of per-shard (alias, port, cid, log_path) tuples in
    shard order — caller uses tuple 0 as the "head" shard for the FalkorDB
    client and the RLTest Defaults.external_addr.
    """
    # Pre-allocate aliases so the announce-hostname flag knows the final
    # name before the container is created. Sharing the falkordb-cl-<hex>
    # prefix makes leaked containers easy to spot in `docker ps`.
    aliases = [f"falkordb-cl-{uuid.uuid4().hex[:8]}" for _ in range(shards)]
    nodes = []
    for alias in aliases:
        # cluster-require-full-coverage=no: during bootstrap not every slot
        # is owned yet, so the default 'yes' would refuse client traffic
        # until the very last ADDSLOTSRANGE lands. Off keeps the shards
        # answering CLUSTER MEET / CLUSTER INFO throughout.
        # cluster-announce-hostname + preferred-endpoint-type=hostname:
        # peers receive `<alias>` as the contact address (resolves via the
        # job network's DNS) instead of the container's internal IP, which
        # may not be reachable from sibling containers across docker
        # network reattachments.
        cluster_args = (
            "--cluster-enabled yes "
            f"--cluster-config-file /tmp/nodes-{alias}.conf "
            "--cluster-node-timeout 5000 "
            "--cluster-require-full-coverage no "
            f"--cluster-announce-hostname {alias} "
            "--cluster-preferred-endpoint-type hostname"
        )
        host, port, cid, log_path = _spawn_falkordb(
            image,
            falkordb_args=falkordb_args,
            redis_args=cluster_args,
            alias=alias,
            enable_debug_command=enable_debug_command,
        )
        nodes.append((host, port, cid, log_path))
    _bootstrap_cluster([(a, p) for a, p, _, _ in nodes])
    return nodes


def _bootstrap_cluster(shard_addrs, attempts=100, interval=0.2):
    """Form a Redis cluster from `shard_addrs` (list of (host, port)) using
    pure redis-py — no redis-cli dependency in the job/RC images.

    Three phases:
      1. CLUSTER ADDSLOTSRANGE — split 16384 slots into N contiguous ranges
         and assign one per shard.
      2. CLUSTER MEET — from shard 0, meet every other shard. Cluster bus
         gossip propagates the full topology to the remaining nodes.
      3. Poll CLUSTER INFO on every shard until cluster_state=ok and
         cluster_known_nodes equals the shard count. Bounded retry mirrors
         _wait_for_redis's shape.
    """
    n = len(shard_addrs)
    if n == 0:
        return
    clients = [
        redis.Redis(host=h, port=p, socket_connect_timeout=1,
                    retry=Retry(NoBackoff(), 0))
        for h, p in shard_addrs
    ]
    total = 16384
    for i, c in enumerate(clients):
        # Integer slot split: shard i owns [i*total/n, (i+1)*total/n - 1].
        # Last shard absorbs the rounding remainder so the union covers
        # all 16384 slots without gaps or overlap.
        start = (i * total) // n
        end = ((i + 1) * total) // n - 1
        c.execute_command("CLUSTER", "ADDSLOTSRANGE", start, end)
    head = clients[0]
    for (h, p) in shard_addrs[1:]:
        # CLUSTER MEET requires an IP literal (it rejects hostnames with
        # "Invalid node address"). Resolve the docker --network-alias via
        # DNS first; the gossip layer still propagates the announced
        # hostname (--cluster-announce-hostname) once nodes are paired.
        ip = socket.gethostbyname(h)
        head.execute_command("CLUSTER", "MEET", ip, p)
    for c, (h, p) in zip(clients, shard_addrs):
        last_exc = None
        ok = False
        for _ in range(attempts):
            try:
                info = c.execute_command("CLUSTER", "INFO")
                if isinstance(info, bytes):
                    info = info.decode()
                fields = dict(
                    line.split(":", 1)
                    for line in info.splitlines() if ":" in line
                )
                if (fields.get("cluster_state", "").strip() == "ok"
                        and int(fields.get("cluster_known_nodes", "0")) == n):
                    ok = True
                    break
            except Exception as e:
                last_exc = e
            time.sleep(interval)
        if not ok:
            msg = (f"cluster shard {h}:{p} never reached cluster_state=ok "
                   f"with {n} known nodes in {attempts * interval:.1f}s")
            if last_exc is not None:
                msg += f"; last error: {last_exc!r}"
            raise RuntimeError(msg)


def _wrap_get_connection_with_timeout(env_obj):
    """Wrap env_obj.getConnection so the returned redis.Redis client carries
    SOCKET_TIMEOUT. RLTest's native getConnection ignores socket_timeout, but
    heavy synthetic workloads (e.g. 1M-node CREATE) overrun redis-py's default
    read deadline under ASAN/coverage and surface as misleading socket
    timeouts. No-op when SOCKET_TIMEOUT is None."""
    if SOCKET_TIMEOUT is None:
        return
    inner = env_obj.getConnection
    def _get_connection_with_timeout(*args, **kwargs):
        c = inner(*args, **kwargs)
        try:
            c.connection_pool.connection_kwargs['socket_timeout'] = SOCKET_TIMEOUT
        except Exception:
            pass
        return c
    env_obj.getConnection = _get_connection_with_timeout


def _attach_cluster(env_obj, shard_addrs):
    """Expose getConnection(shardId=N) on the RLTest Environment, matching
    RLTest's native oss-cluster Env contract. shardId is 1-based to mirror
    RLTest's API (test code uses shardId=1..N directly).

    Also stash env.shards as a list of (host, port) so test code that wants
    to build its own clients (FalkorDB per shard, async pools, etc.) can
    address each node without going through getConnection.
    """
    shards = list(shard_addrs)

    def _get_connection(shardId=1, *_args, **_kwargs):
        idx = int(shardId) - 1
        if idx < 0 or idx >= len(shards):
            raise IndexError(
                f"shardId {shardId} out of range; have {len(shards)} shards"
            )
        h, p = shards[idx]
        return redis.Redis(host=h, port=p, decode_responses=True,
                           socket_timeout=SOCKET_TIMEOUT)

    env_obj.getConnection = _get_connection
    env_obj.shards = shards


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
       Selected by the matrix for test files whose Env() invocations stay
       within the runtime-mutable set (classified by
       tests/flow/test_matrix_split.py). FLUSHALL between Env() calls
       gives class-level isolation since the service is shared across all
       classes in the cell.
       Supported flags here:
         - moduleArgs (runtime-mutable keys only — applied via GRAPH.CONFIG SET)
         - useSlaves=True — a second `replica` service container is run by
           the GHA job with --replicaof falkordb 6379
         - enableDebugCommand=True — no-op; the service is launched with
           REDIS_ARGS=--enable-debug-command yes regardless
       Unsupported (classifier miss → raises): env='oss-cluster',
       shardsCount (cluster topology requires per-class spawn).

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
        # log_path mirrors what RLTest writes the master log to — same attribute
        # spawn mode sets — so tests can read self.env.log_path without per-mode
        # branching. Only StandardEnv exposes _getFileName; ClusterEnv (oss-
        # cluster) doesn't have a single master log, so leave log_path=None
        # there (matches what services mode does for cluster-shaped tests).
        if hasattr(env_obj.envRunner, "_getFileName"):
            env_obj.log_path = f"{env_obj.logDir}/{env_obj.envRunner._getFileName('master', '.log')}"
        else:
            env_obj.log_path = None
        _wrap_get_connection_with_timeout(env_obj)
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
        # Bounded connect timeout + no-retry so a not-yet-ready service raises
        # quickly instead of stalling under redis-py's default retry policy.
        redis.Redis(host=host, port=port, socket_connect_timeout=1,
                    retry=Retry(NoBackoff(), 0)).flushall()
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
        # Services-mode shared container doesn't bind-mount its log; any test
        # that needs to read it should route to the spawn bucket instead.
        env_obj.log_path = None
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
        _wrap_get_connection_with_timeout(env_obj)
        db = FalkorDB(host, port)
        return (env_obj, db)

    # Mode 3: CI spawn — every Env() call gets a fresh master container
    # (plus replica when useSlaves=True, or N shards when env='oss-cluster'
    # / shardsCount is set). Order matters: spawn the container *first*,
    # then point RLTest's existing-env address at it via Defaults before
    # constructing the Environment. Environment(env='existing-env')'s
    # startEnv() pings the address; if it doesn't resolve, construction
    # raises before we can override anything. flow.sh's --existing-env-addr
    # is a placeholder that gets superseded here.
    if env not in ('oss', 'oss-cluster'):
        raise RuntimeError(
            f"Spawn-mode Env() doesn't support env={env!r}; only 'oss' "
            "(single-node) and 'oss-cluster' (multi-shard) are wired up."
        )

    cluster = env == 'oss-cluster' or bool(shardsCount)
    if cluster:
        # RLTest's native oss-cluster Env defaults to 3 shards when
        # shardsCount is omitted; mirror that so test classes can write
        # `Env(env='oss-cluster')` without specifying a count.
        nshards = int(shardsCount) if shardsCount else 3
        if useSlaves:
            # Cluster + replicas isn't needed by any current test and adds
            # a non-trivial bootstrap step (CLUSTER REPLICATE per replica,
            # per-shard slave wait). Bail loudly until something needs it.
            raise NotImplementedError(
                "Spawn-mode cluster Env() doesn't support useSlaves=True "
                "(masters-only). Add cluster-replica orchestration to "
                "_spawn_falkordb_cluster when a test needs it."
            )
        nodes = _spawn_falkordb_cluster(
            test_image,
            shards=nshards,
            falkordb_args=moduleArgs or "",
            enable_debug_command=enableDebugCommand,
        )
        head_alias, head_port, _, head_log_path = nodes[0]
        host, port = head_alias, head_port
        Defaults.external_addr = f"{head_alias}:{head_port}"
        env_obj = Environment(decodeResponses=True, env='existing-env')
        env_obj.log_path = head_log_path
        _attach_cluster(env_obj, [(a, p) for a, p, _, _ in nodes])
        env_obj.port = port
        env_obj.host = host
        if hasattr(env_obj, "envRunner") and env_obj.envRunner is not None:
            env_obj.envRunner.port = port
            env_obj.envRunner.host = host
        # FalkorDB(host, port) auto-detects cluster mode via Is_Cluster
        # (checks INFO for cluster_enabled:1) and returns a cluster-aware
        # client, so udf_load / queries route through the cluster.
        db = FalkorDB(host, port)
        return (env_obj, db)

    master_alias, master_port, master_cid, master_log_path = _spawn_falkordb(
        test_image, falkordb_args=moduleArgs or "",
        enable_debug_command=enableDebugCommand)
    host, port = master_alias, master_port
    Defaults.external_addr = f"{master_alias}:{master_port}"
    env_obj = Environment(decodeResponses=True, env='existing-env')
    # log_path points at redis's --logfile output, bind-mounted into the
    # workspace by _log_mount_args. Same attribute name as Mode 1.
    env_obj.log_path = master_log_path
    if useSlaves:
        replica_alias, _, _, _ = _spawn_falkordb(
            test_image,
            falkordb_args=moduleArgs or "",
            redis_args=f"--replicaof {master_alias} 6379",
            enable_debug_command=enableDebugCommand,
        )
        # Replica handshake is async — wait for INFO replication to report
        # master_link_status=up before any test queries hit it, mirroring
        # what services mode does. Without this gate, tests that read from
        # the replica immediately after Env() can observe a half-synced
        # state.
        _wait_for_replication(replica_alias, 6379)
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

    _wrap_get_connection_with_timeout(env_obj)
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
    r = redis.Redis(host=host, port=port, socket_connect_timeout=1,
                    retry=Retry(NoBackoff(), 0))
    # Track the last exception so the final timeout message can show *why*
    # the polls were failing (connection refused vs auth vs INFO parse).
    last_exc = None
    for _ in range(attempts):
        try:
            info = r.info('replication')
            if (info.get('master_link_status') == 'up'
                    and int(info.get('master_sync_in_progress', 1)) == 0):
                return
        except Exception as e:
            last_exc = e
        time.sleep(interval)
    msg = (f"replica at {host}:{port} did not reach master_link_status=up "
           f"in {attempts * interval:.1f}s")
    if last_exc is not None:
        msg += f"; last error: {last_exc!r}"
    raise RuntimeError(msg)


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
        return redis.Redis(host=host, port=port, decode_responses=True,
                           socket_timeout=SOCKET_TIMEOUT)
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
