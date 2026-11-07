"""Docker backends: CI spawn mode and CI external-service mode.

Both exist because CI runs the *published* image rather than a locally built
module. The logic here is lifted from tests/flow/common.py, where it was
already load-bearing for every CI run; it is reorganised into backends but
its behaviour (and the reasoning in its comments) is unchanged.
"""

import atexit
import os
import socket
import subprocess
import tempfile
import time
import uuid

from .clients import probe_client
from .cluster import bootstrap_cluster
from .local import _PATH_ARG_KEYS, wait_for_replication
from .module_args import partition

import redis

CODE_COVERAGE = os.getenv('CODE_COVERAGE', '0') == '1'

# Path inside the spawned container where redis writes --logfile output
# (and where _log_mount_args bind-mounts the host-side per-spawn dir).
LOG_DIR_IN_CONTAINER = "/var/lib/falkordb/logs"
LOG_FILE_NAME = "redis.log"

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


def host_path_for(in_container_path):
    """Translate an in-container absolute path to the host path that backs it,
    via the job container's mount table. Returns (host_path, covered_by_mount).
    When not covered, host_path is the input unchanged — callers must NOT
    bind-mount in that case, because the host daemon would interpret it as a
    fresh path and create an empty directory there."""
    for src, dest in _job_mounts():
        if in_container_path == dest or in_container_path.startswith(dest.rstrip("/") + "/"):
            return src + in_container_path[len(dest):], True
    return in_container_path, False


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
    abort module load; we shouldn't wait the full budget for those)."""
    r = probe_client(host, port)
    for _ in range(attempts):
        inspect_out = subprocess.run(
            ["docker", "inspect", "-f", "{{.State.Running}}", cid],
            stdout=subprocess.PIPE, stderr=subprocess.DEVNULL, check=False,
        ).stdout.decode().strip()
        if inspect_out == "false":
            raise RuntimeError(
                f"container {cid[:12]} for {host}:{port} exited before becoming ready")
        try:
            if r.ping():
                return
        except Exception:
            time.sleep(interval)
    raise RuntimeError(
        f"redis at {host}:{port} did not become ready after {attempts * interval:.1f}s")


def _log_mount_args(alias):
    """Bind-mount a per-spawn logs dir into the spawned container so redis's
    --logfile output is reachable from the test process.

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
    host_path, covered = host_path_for(log_dir)
    if not covered:
        return [], None
    return (
        ["-v", f"{host_path}:{LOG_DIR_IN_CONTAINER}"],
        os.path.join(log_dir, LOG_FILE_NAME),
    )


def _coverage_mount_args():
    """When CODE_COVERAGE=1 under CI, bind-mount a per-spawn subdir of
    ${GITHUB_WORKSPACE}/cov onto /var/lib/falkordb/cov inside the spawned
    container. Per-spawn UUID subdir prevents PID-1 filename collisions
    across containers in the same cell (every container's main pid is 1, so
    the %p substitution alone isn't unique)."""
    if not CODE_COVERAGE:
        return []
    ws = os.environ.get("GITHUB_WORKSPACE")
    if not ws:
        return []
    in_container = os.path.join(ws, "cov", uuid.uuid4().hex)
    os.makedirs(in_container, mode=0o777, exist_ok=True)
    host_path, covered = host_path_for(in_container)
    if not covered:
        return []
    return ["-v", f"{host_path}:/var/lib/falkordb/cov"]


class Container:
    """A spawned falkordb container, restartable in place."""

    def __init__(self, cid, alias, port, log_path):
        self.cid = cid
        self.alias = alias
        self.host = alias
        self.port = port
        self.log_path = log_path

    def stop(self):
        # Graceful SIGTERM with a grace period gives ASAN's at-exit leak
        # detection time to print before the container dies.
        subprocess.run(["docker", "stop", "--time", "10", self.cid],
                       stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL,
                       check=False)

    def start(self):
        subprocess.run(["docker", "start", self.cid],
                       stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL,
                       check=False)
        _wait_for_redis(self.host, self.port, self.cid)

    def restart(self):
        self.stop()
        self.start()

    @property
    def pid(self):
        out = subprocess.run(
            ["docker", "inspect", "-f", "{{.State.Pid}}", self.cid],
            stdout=subprocess.PIPE, stderr=subprocess.DEVNULL, check=False,
        ).stdout.decode().strip()
        return int(out) if out.isdigit() else None


def spawn_falkordb(image, falkordb_args="", redis_args="", alias=None,
                   enable_debug_command=False):
    """Start a falkordb container, return a Container.

    `falkordb_args` becomes the module's load-time arg string (CACHE_SIZE 16 ...).
    `redis_args` is forwarded as the redis-server CLI args (for --replicaof, etc.).
    """
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
        host_path, covered = host_path_for(val)
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
    return Container(cid, alias, 6379, log_path)


def spawn_falkordb_cluster(image, shards, falkordb_args="",
                           enable_debug_command=False):
    """Spawn `shards` falkordb sibling containers wired up as a Redis Cluster.

    Each shard runs with --cluster-enabled yes and announces its docker
    --network-alias as the cluster bus hostname so peers reach it by the
    same name the test process uses (no docker-network IP discovery needed).
    Slots are split evenly by bootstrap_cluster; no replicas are provisioned.
    """
    aliases = [f"falkordb-cl-{uuid.uuid4().hex[:8]}" for _ in range(shards)]
    nodes = []
    for alias in aliases:
        # cluster-require-full-coverage=no: during bootstrap not every slot
        # is owned yet, so the default 'yes' would refuse client traffic
        # until the very last ADDSLOTSRANGE lands.
        # cluster-announce-hostname + preferred-endpoint-type=hostname:
        # peers receive `<alias>` as the contact address (resolves via the
        # job network's DNS) instead of the container's internal IP, which
        # may not be reachable across docker network reattachments.
        cluster_args = (
            "--cluster-enabled yes "
            f"--cluster-config-file /tmp/nodes-{alias}.conf "
            "--cluster-node-timeout 5000 "
            "--cluster-require-full-coverage no "
            f"--cluster-announce-hostname {alias} "
            "--cluster-preferred-endpoint-type hostname"
        )
        nodes.append(spawn_falkordb(
            image, falkordb_args=falkordb_args, redis_args=cluster_args,
            alias=alias, enable_debug_command=enable_debug_command))
    bootstrap_cluster([(n.host, n.port) for n in nodes])
    return nodes


@atexit.register
def _cleanup_spawned():
    # Capture docker stdout/stderr for each spawned container before tearing
    # it down. tests/flow/logs/ is uploaded as a GHA artifact on failure, so
    # dropping files there means a failed run carries everything needed for
    # post-mortem:
    #   - redis-server lifecycle messages (default destination: stderr)
    #   - module RedisModule_Log output (routes through redis's logger)
    #   - ASAN/LSan reports (default destination: stderr, no log_path set)
    log_dir = os.path.join(
        os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "flow", "logs")
    os.makedirs(log_dir, exist_ok=True)
    for cid in _SPAWNED_CIDS:
        short = cid[:12]
        subprocess.run(["docker", "stop", "--time", "10", cid],
                       stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL,
                       check=False)
        with open(os.path.join(log_dir, f"spawned-{short}.log"), "wb") as f:
            subprocess.run(["docker", "logs", cid], stdout=f,
                           stderr=subprocess.STDOUT, check=False)
        subprocess.run(["docker", "rm", "-f", cid],
                       stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL,
                       check=False)


class DockerBackend:
    """CI spawn mode: a dedicated container (or cluster) per Env()."""

    name = "docker"

    def __init__(self, image, module_args=None, use_slaves=False, shards=None,
                 enable_debug_command=False):
        self.module_args = module_args or ""
        self.master = None
        self.replica = None
        self.shards = []

        if shards:
            if use_slaves:
                # Cluster + replicas isn't needed by any current test and adds
                # a non-trivial bootstrap step (CLUSTER REPLICATE per replica,
                # per-shard slave wait). Bail loudly until something needs it.
                raise NotImplementedError(
                    "Spawn-mode cluster Env() doesn't support useSlaves=True "
                    "(masters-only). Add cluster-replica orchestration to "
                    "spawn_falkordb_cluster when a test needs it.")
            self.shards = spawn_falkordb_cluster(
                image, shards=shards, falkordb_args=self.module_args,
                enable_debug_command=enable_debug_command)
            self.master = self.shards[0]
        else:
            self.master = spawn_falkordb(
                image, falkordb_args=self.module_args,
                enable_debug_command=enable_debug_command)
            if use_slaves:
                self.replica = spawn_falkordb(
                    image, falkordb_args=self.module_args,
                    redis_args=f"--replicaof {self.master.alias} 6379",
                    enable_debug_command=enable_debug_command)
                # Replica handshake is async — wait for INFO replication to
                # report master_link_status=up before any test queries hit it.
                wait_for_replication(self.replica.host, self.replica.port)

    @property
    def host(self):
        return self.master.host

    @property
    def port(self):
        return self.master.port

    @property
    def log_path(self):
        return self.master.log_path

    @property
    def replica_addr(self):
        if self.replica is None:
            return None
        return (self.replica.host, self.replica.port)

    @property
    def shard_addrs(self):
        return [(n.host, n.port) for n in self.shards]

    def stop(self):
        self.master.stop()

    def start(self):
        self.master.start()

    def restart_and_reload(self):
        self.master.restart()

    def teardown(self):
        # Containers are torn down by the atexit hook, which also captures
        # their logs. Leaving them up until then keeps a failed run
        # inspectable.
        pass


class ServiceBackend:
    """CI services mode: a long-running GHA service container per matrix cell.

    Shared across every class in the cell, so isolation comes from FLUSHALL
    rather than from a fresh server. Only the runtime-mutable subset of
    moduleArgs is supported; the classifier guarantees files needing more are
    routed to the spawn bucket.
    """

    name = "service"

    def __init__(self, module_args=None, use_slaves=False, shards=None,
                 enable_debug_command=False):
        if shards:
            raise RuntimeError(
                "Services-mode Env() doesn't support oss-cluster / "
                "shardsCount. This file was routed to the services bucket "
                "but uses a cluster flag; either reclassify in "
                "tests/flow/test_matrix_split.py or remove the flag.")
        self.host = os.getenv(
            "FALKORDB_HOST",
            "falkordb" if os.getenv("FALKORDB_TEST_IMAGE") else "localhost")
        self.port = int(os.getenv("FALKORDB_PORT", "6379"))
        self.shards = []
        # Wipe the shared service back to a clean state so each class starts
        # fresh — mirrors the per-class isolation spawn mode gets for free.
        probe_client(self.host, self.port).flushall()
        if module_args:
            apply_module_args_via_config_set(self.host, self.port, module_args)
        self.replica = None
        if use_slaves:
            replica_host = os.getenv("FALKORDB_REPLICA_HOST", "replica")
            replica_port = int(os.getenv("FALKORDB_REPLICA_PORT", "6379"))
            wait_for_replication(replica_host, replica_port)
            self.replica = (replica_host, replica_port)

    # Services-mode shared container doesn't bind-mount its log; any test that
    # needs to read it should route to the spawn bucket instead.
    log_path = None

    @property
    def replica_addr(self):
        return self.replica

    @property
    def shard_addrs(self):
        return []

    def _unsupported(self, what):
        raise RuntimeError(
            f"{what} is not supported against the shared services container. "
            "Route this file to the spawn bucket in "
            "tests/flow/test_matrix_split.py.")

    def stop(self):
        self._unsupported("env.stop()")

    def start(self):
        self._unsupported("env.start()")

    def restart_and_reload(self):
        self._unsupported("env.restart_and_reload()")

    def teardown(self):
        pass


def apply_module_args_via_config_set(host, port, module_args):
    """Apply space-separated `KEY VALUE` pairs via GRAPH.CONFIG SET.

    Reaching here with a load-time-only key means the classifier
    (tests/flow/test_matrix_split.py) should have routed this file to the
    spawn bucket and didn't, so say that rather than surfacing the module's
    generic "cannot be set at run-time".
    """
    mutable, immutable = partition(module_args)
    if immutable:
        raise RuntimeError(
            "load-time-only module args "
            + ", ".join(f"{k}={v}" for k, v in immutable)
            + " cannot be applied to the shared services container. This "
            "file should have been routed to the spawn bucket — check "
            "IMMUTABLE_MODULE_ARGS in tests/infra/module_args.py and the "
            "Env() call this came from.")
    conn = redis.Redis(host=host, port=port)
    for key, value in mutable:
        try:
            conn.execute_command("GRAPH.CONFIG", "SET", key, value)
        except redis.ResponseError as e:
            raise RuntimeError(
                f"GRAPH.CONFIG SET {key} {value} failed: {e}. "
                "If this key is actually immutable, add it to "
                "IMMUTABLE_MODULE_ARGS in tests/infra/module_args.py "
                "so this file routes to the spawn bucket.") from e


def ci_tmpdir():
    p = os.path.join(
        os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
        "flow", ".ci-tmp")
    os.makedirs(p, exist_ok=True)
    return p


def mountable_mkdtemp(*args, **kwargs):
    """tempfile.mkdtemp(), but under FALKORDB_TEST_IMAGE the dir is rooted in
    a workspace-relative location that's covered by the job container's
    bind-mount table. Spawned sibling containers then see the same directory
    via spawn_falkordb's bind-mount logic. Local dev gets plain mkdtemp."""
    if os.getenv("FALKORDB_TEST_IMAGE") and "dir" not in kwargs:
        kwargs["dir"] = ci_tmpdir()
    return tempfile.mkdtemp(*args, **kwargs)


def mountable_mkstemp(*args, **kwargs):
    """Workspace-rooted tempfile.mkstemp(). See mountable_mkdtemp() for why."""
    if os.getenv("FALKORDB_TEST_IMAGE") and "dir" not in kwargs:
        kwargs["dir"] = ci_tmpdir()
    return tempfile.mkstemp(*args, **kwargs)
