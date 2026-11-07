"""Local backend: one redis-server subprocess per test class.

This is the local-development counterpart to the CI docker backends, and it
is deliberately shaped to match CI spawn mode 1:1 — a fresh server per
`Env()` call, module args passed at load time via `--loadmodule`, replicas
and cluster shards as sibling processes. RLTest used to reuse a server across
classes whose parameters matched, which made local runs diverge from CI in
exactly the cases that matter (leftover keys, leftover configs).
"""

import atexit
import itertools
import os
import platform
import shutil
import signal
import subprocess
import tempfile
import time

from .clients import make_client, probe_client
from .ports import reserve_port

FLOW_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
FLOW_DIR = os.path.join(FLOW_DIR, "flow")
LOG_DIR = os.path.join(FLOW_DIR, "logs")
REDIS_CONF = os.path.join(FLOW_DIR, "redis.conf")

# Module args whose value is a filesystem path. Callers write them relative
# to the repo root, but redis chdir()s into `--dir` before loading modules
# (and the docker backends need to bind-mount them), so both backends have
# to recognise them.
_PATH_ARG_KEYS = ("IMPORT_FOLDER", "TEMP_FOLDER")

# Every RedisProcess that has been started and not yet cleaned up. A spawn
# can outlive its owner: if a later step of LocalBackend.__init__ raises
# (replication never comes up, cluster bootstrap fails), Env() never returns,
# so the backend is never registered for teardown and nothing else knows the
# process exists. Orphans survive pytest's exit and keep holding ports out of
# the worker's band, so repeated failures exhaust it. The docker backend has
# the same safety net for containers.
_LIVE_PROCS = []
# Monotonic across the whole worker process. len(self.nodes) is per-backend
# and restarts at 0 for every Env(), so it does not distinguish the servers
# of two classes — or of two environments alive at once within one class,
# which test_config/test_timeout/test_cache/test_graph_copy all create.
_NAME_SEQ = itertools.count()


def _reap_orphans():
    for node in list(_LIVE_PROCS):
        try:
            node.cleanup()
        except Exception:
            pass


atexit.register(_reap_orphans)


def module_path():
    """Absolute path to the module built for the current platform.

    Honours the same TARGET / RELEASE contract as flow.sh and
    tests/common.py so a single build serves every suite.
    """
    explicit = os.environ.get("TARGET")
    if explicit:
        return os.path.abspath(explicit)
    target_dir = os.environ.get("TARGET_DIR")
    if not target_dir:
        release = os.environ.get("RELEASE", "").lower() in ("1", "true", "yes")
        target_dir = os.path.join("target", "release" if release else "debug")
    name = "libfalkordb.dylib" if platform.system() == "Darwin" else "libfalkordb.so"
    repo_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    repo_root = os.path.dirname(repo_root)
    return os.path.abspath(os.path.join(repo_root, target_dir, name))


def _absolutize_paths(toks):
    """Resolve relative path-shaped module args against the caller's cwd.

    redis chdir()s into `--dir` before it loads modules, so a relative
    `IMPORT_FOLDER data/` would resolve against the throwaway working
    directory and fail to canonicalize. Callers mean "relative to where I
    started pytest", so bind that here.
    """
    out = list(toks)
    for i, tok in enumerate(out[:-1]):
        if tok in _PATH_ARG_KEYS and not os.path.isabs(out[i + 1]):
            out[i + 1] = os.path.abspath(out[i + 1])
    return out


class RedisProcess:
    """A single redis-server process with the module loaded.

    Owns everything that must outlive a restart: the port, the working
    directory holding its RDB, and the argv needed to bring it back.
    """

    def __init__(self, name, module_args="", redis_args=(), workdir=None):
        self.name = name
        self.module_args = module_args or ""
        self.redis_args = list(redis_args)
        self.workdir = workdir or tempfile.mkdtemp(prefix=f"falkor-{name}-")
        self.log_path = os.path.join(LOG_DIR, f"{name}.log")
        # redis's --logfile captures only what redis itself writes. A Rust
        # panic payload, its backtrace, and any ASAN/LSan report go to the
        # process's stderr, which would otherwise be discarded and leave a
        # local crash with no diagnostic at all.
        self.stderr_path = os.path.join(LOG_DIR, f"{name}.stderr")
        self.proc = None
        self._stderr = None
        self.port = None
        self.host = "127.0.0.1"
        os.makedirs(LOG_DIR, exist_ok=True)

    def _argv(self, port):
        mod = module_path()
        if not os.path.exists(mod):
            raise RuntimeError(
                f"module not found at {mod}. Build it first "
                "(`cargo build --release` and run with RELEASE=1), or point "
                "TARGET at an existing build."
            )
        argv = ["redis-server"]
        if os.path.exists(REDIS_CONF) and os.path.getsize(REDIS_CONF) > 0:
            argv.append(REDIS_CONF)
        argv += [
            "--port", str(port),
            "--dir", self.workdir,
            "--logfile", self.log_path,
            "--save", "",
            # flow.sh passed --enable-debug-command / --enable-protected-configs
            # to RLTest for every test, so these were always on locally. Keep
            # them unconditionally: the per-Env enableDebugCommand flag then
            # stays a no-op here exactly as it is in CI services mode, rather
            # than being a third behaviour to reason about.
            "--enable-debug-command", "yes",
            "--enable-protected-configs", "yes",
            "--enable-module-command", "yes",
        ]
        argv += self.redis_args
        # --loadmodule must come last: everything after the module path is
        # consumed by the module as its own load-time args.
        argv += ["--loadmodule", mod]
        if self.module_args:
            argv += _absolutize_paths(self.module_args.split())
        return argv

    def start(self, port=None):
        """Spawn redis-server and block until it answers PING."""
        reservation = None
        if port is None:
            reservation = reserve_port()
            port = reservation.port
        self.port = port
        argv = self._argv(port)
        # Release the reservation as late as possible: the moment it is
        # closed the port is claimable by anything else on the machine, so
        # the spawn follows immediately.
        if reservation is not None:
            reservation.release()
        self._stderr = open(self.stderr_path, "ab", buffering=0)
        self.proc = subprocess.Popen(argv, stdout=subprocess.DEVNULL,
                                     stderr=self._stderr)
        # Registered before the readiness wait, not after: a module that is
        # slow to load (coverage, sanitizer) can time out below, and the
        # process is very much alive at that point.
        if self not in _LIVE_PROCS:
            _LIVE_PROCS.append(self)
        self._wait_ready()
        return self

    def _wait_ready(self, attempts=600, interval=0.05):
        r = probe_client(self.host, self.port)
        for _ in range(attempts):
            # A module that rejects its load-time args aborts startup. Detect
            # the dead process instead of burning the whole budget, so tests
            # that intentionally pass invalid moduleArgs fail fast and with a
            # message naming the log.
            if self.proc.poll() is not None:
                raise RuntimeError(
                    f"redis-server for {self.name} exited with code "
                    f"{self.proc.returncode} before becoming ready; "
                    f"see {self.log_path}"
                )
            try:
                if r.ping():
                    return
            except Exception:
                time.sleep(interval)
        raise RuntimeError(
            f"redis at {self.host}:{self.port} ({self.name}) did not become "
            f"ready in {attempts * interval:.1f}s; see {self.log_path}"
        )

    def stop(self, save=False):
        """Stop the server, optionally persisting first.

        `save=True` is what restart_and_reload() needs: the RDB written here
        is what the restarted process loads back.
        """
        if self.proc is None or self.proc.poll() is not None:
            self.proc = None
            self._close_stderr()
            if self in _LIVE_PROCS:
                _LIVE_PROCS.remove(self)
            return
        if save:
            # Servers run with `--save ""`, so SIGTERM writes no RDB and this
            # is the only thing persisting the dataset. Swallowing a failure
            # here would restart onto an empty dir and surface as a baffling
            # "expected N nodes, got 0" several assertions later.
            try:
                make_client(self.host, self.port).save()
            except Exception as e:
                raise RuntimeError(
                    f"SAVE failed for {self.name} at {self.host}:{self.port}; "
                    f"restarting now would silently lose the dataset: {e!r}"
                ) from e
        self.proc.send_signal(signal.SIGTERM)
        try:
            self.proc.wait(timeout=30)
        except subprocess.TimeoutExpired:
            # SIGTERM is preferred so ASAN's at-exit leak report still prints;
            # SIGKILL only as a last resort for a wedged process.
            self.proc.kill()
            self.proc.wait(timeout=10)
        self.proc = None
        self._close_stderr()
        if self in _LIVE_PROCS:
            _LIVE_PROCS.remove(self)

    def _close_stderr(self):
        if self._stderr is not None:
            self._stderr.close()
            self._stderr = None
        # A clean run leaves one empty .stderr per server; keeping them would
        # bury the interesting ones in the CI artifact.
        try:
            if os.path.getsize(self.stderr_path) == 0:
                os.remove(self.stderr_path)
        except OSError:
            pass

    def restart(self, save=True):
        """Stop and start again on the same port, keeping the working dir."""
        self.stop(save=save)
        self.start(port=self.port)

    def cleanup(self):
        try:
            self.stop()
        finally:
            shutil.rmtree(self.workdir, ignore_errors=True)

    @property
    def pid(self):
        return self.proc.pid if self.proc is not None else None

    def wait(self, timeout=None):
        if self.proc is not None:
            return self.proc.wait(timeout=timeout)
        return None


class LocalBackend:
    """Spawns redis-server processes on the host for a single Env()."""

    name = "local"

    def __init__(self, module_args=None, use_slaves=False, shards=None):
        self.module_args = module_args or ""
        self.nodes = []
        self.master = None
        self.replica = None
        self.shards = []

        # Anything raising below leaves servers already running, and Env()
        # will not have registered this backend for teardown yet — so it has
        # to reclaim its own before letting the error out.
        try:
            if shards:
                self._start_cluster(shards)
            else:
                self.master = self._spawn("master")
                if use_slaves:
                    self.replica = self._spawn(
                        "replica",
                        redis_args=["--replicaof", self.master.host,
                                    str(self.master.port)],
                    )
                    wait_for_replication(self.replica.host, self.replica.port)
        except Exception:
            self.teardown()
            raise

    def _spawn(self, role, redis_args=()):
        # Named off a process-wide counter so two environments alive at the
        # same time never share a log file. Several classes hold more than
        # one, and env.log_path readers (test_encode_decode) would otherwise
        # parse another server's lines.
        name = f"{role}-{os.getpid()}-{next(_NAME_SEQ)}"
        node = RedisProcess(name, module_args=self.module_args,
                            redis_args=redis_args)
        # Appended before start() so a failed spawn is still reclaimable by
        # the except clause above.
        self.nodes.append(node)
        node.start()
        return node

    def _start_cluster(self, shards):
        from .cluster import bootstrap_cluster
        for i in range(shards):
            node = self._spawn(
                f"shard{i}",
                redis_args=[
                    "--cluster-enabled", "yes",
                    "--cluster-config-file", f"nodes-{i}.conf",
                    "--cluster-node-timeout", "5000",
                    # During bootstrap not every slot is owned yet, so the
                    # default 'yes' would refuse client traffic until the very
                    # last ADDSLOTSRANGE lands.
                    "--cluster-require-full-coverage", "no",
                ],
            )
            self.shards.append(node)
        self.master = self.shards[0]
        bootstrap_cluster([(n.host, n.port) for n in self.shards])

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
        self.master.start(port=self.master.port)

    def restart_and_reload(self):
        self.master.restart(save=True)

    def teardown(self):
        # Guarded per node: one wedged process must not abandon the rest, or
        # a single TimeoutExpired leaks every server after it in the list.
        errors = []
        for node in self.nodes:
            try:
                node.cleanup()
            except Exception as e:
                errors.append(e)
        self.nodes = []
        if errors:
            raise errors[0]


def wait_for_replication(host, port, attempts=300, interval=0.1):
    """Block until the replica reports it finished syncing from its master.

    A replica's `--replicaof` only starts trying to connect once redis is
    initialized, so the first polls legitimately show 'down' or 'connecting'.
    """
    r = probe_client(host, port)
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
