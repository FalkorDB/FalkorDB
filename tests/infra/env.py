"""FalkorEnv — the handle a test holds onto, and Env() which builds one.

The surface is deliberately the one flow tests already use, so migrating a
test file is an assertion rewrite rather than an API rewrite. Everything here
is measured from real call sites in tests/flow:

    getConnection 81, host 37, skip 34, port 21, getSlaveConnection 11,
    dumpAndReload 11, envRunner 8, stop 6, flush 6, restart_and_reload 4,
    log_path 4, shards/replica_host/replica_port/logDir/log 1 each
"""

import os
import sys

import pytest

from .clients import db_handle, make_client
from .docker import DockerBackend, ServiceBackend
from .local import LocalBackend

# Every environment built during the current test class, so the conftest
# finalizer can reclaim all of them. A registry rather than "tear down
# cls.env" because several tests replace their environment mid-class
# (`self.env, self.db = Env(moduleArgs=...)` in test_timeout, test_cache,
# test_config, test_graph_copy). Tracking only the current attribute would
# leak every superseded server for the rest of the session.
_LIVE_ENVS = []


def teardown_all():
    """Tear down every environment created since the last call."""
    errors = []
    while _LIVE_ENVS:
        env = _LIVE_ENVS.pop()
        try:
            env.teardown()
        except Exception as e:
            errors.append(e)
    if errors:
        raise errors[0]


class _EnvRunnerShim:
    """Backwards-compatible stand-in for RLTest's `env.envRunner`.

    Four call sites reach through it (`.port`, `.masterProcess.pid/.wait`,
    `._getFileName`). Keeping the shim means those tests don't need a special
    case, and it keeps the attribute surface honest: anything not listed here
    was not actually used.
    """

    def __init__(self, env):
        self._env = env

    @property
    def port(self):
        return self._env.port

    @property
    def host(self):
        return self._env.host

    @property
    def masterProcess(self):
        return self._env.masterProcess

    def _getFileName(self, role, ext):
        return os.path.basename(self._env.log_path or f"{role}{ext}")


class FalkorEnv:
    """Backend-agnostic handle over one FalkorDB deployment."""

    def __init__(self, backend):
        self._backend = backend
        self.envRunner = _EnvRunnerShim(self)

    # ── addressing ──────────────────────────────────────────────────────
    @property
    def host(self):
        return self._backend.host

    @property
    def port(self):
        return self._backend.port

    @property
    def log_path(self):
        return self._backend.log_path

    @property
    def logDir(self):
        return os.path.dirname(self._backend.log_path or "")

    @property
    def masterProcess(self):
        """The master's process handle, for the two tests that signal it."""
        proc = getattr(self._backend, "master", None)
        if proc is None:
            raise RuntimeError(
                "this backend has no local master process. A test reaching "
                "for env.masterProcess must run in the spawn bucket — see "
                "LIFECYCLE_RE in tests/flow/test_matrix_split.py."
            )
        return proc

    @property
    def shards(self):
        return self._backend.shard_addrs

    @property
    def replica_host(self):
        addr = self._backend.replica_addr
        return addr[0] if addr else None

    @property
    def replica_port(self):
        addr = self._backend.replica_addr
        return addr[1] if addr else None

    # ── connections ─────────────────────────────────────────────────────
    def getConnection(self, shardId=1):
        """A redis client for the master, or for shard N in cluster mode.

        shardId is 1-based to mirror RLTest's API (test code passes
        shardId=1..N directly).
        """
        shards = self._backend.shard_addrs
        if shards:
            idx = int(shardId) - 1
            if idx < 0 or idx >= len(shards):
                raise IndexError(
                    f"shardId {shardId} out of range; have {len(shards)} shards")
            host, port = shards[idx]
            return make_client(host, port)
        return make_client(self.host, self.port)

    def getSlaveConnection(self):
        addr = self._backend.replica_addr
        if addr is None:
            raise RuntimeError(
                "no replica in this environment; construct it with "
                "Env(useSlaves=True)")
        return make_client(*addr)

    # ── lifecycle ───────────────────────────────────────────────────────
    def stop(self):
        self._backend.stop()

    def start(self):
        self._backend.start()

    def restart_and_reload(self):
        """Restart the server and load its data back from the RDB."""
        self._backend.restart_and_reload()

    def dumpAndReload(self):
        """Round-trip the dataset through the RDB encoder in place.

        DEBUG RELOAD is what RLTest used, and it is the point of the
        persistency tests: it exercises the module's serialize/deserialize
        path without losing the connection.
        """
        self.getConnection().execute_command("DEBUG", "RELOAD")

    def flush(self):
        self.getConnection().flushall()

    def teardown(self):
        self._backend.teardown()

    # ── misc ────────────────────────────────────────────────────────────
    def skip(self, reason=None):
        """Skip the running test.

        The reason names the caller, not this frame: these skips are mostly
        environment-conditional (jemalloc, sanitizer, module build), so a CI
        report saying only "env.py:173" hides which guard actually fired.
        """
        if reason is None:
            caller = sys._getframe(1)
            reason = (f"skipped by {caller.f_code.co_name} at "
                      f"{os.path.basename(caller.f_code.co_filename)}:"
                      f"{caller.f_lineno}")
        pytest.skip(reason)

    def log(self, msg):
        print(msg)


def uses_existing_service():
    """True when Env() will attach to an already-running server.

    `EXISTING_ENV` is the root suites' spelling of "a server is already
    running, connect to it"; `FALKORDB_USE_SERVICE` is the flow suite's.
    They mean the same thing, so accept either. Exposed because callers
    outside Env() have to know: an existing server's load-time args are
    fixed, so module args that aren't runtime-mutable can't be honoured.
    """
    return (os.getenv("FALKORDB_USE_SERVICE", "") != ""
            or os.getenv("EXISTING_ENV", "").lower() == "1")


def Env(moduleArgs=None, env='oss', useSlaves=False, enableDebugCommand=False,
        shardsCount=None):
    """Build a FalkorDB environment and a client for it.

    Returns `(FalkorEnv, FalkorDB)` — the same tuple flow tests already
    unpack.

    Three execution modes, selected by the same environment variables as
    before:

    1. Local (FALKORDB_TEST_IMAGE unset) — a redis-server subprocess per
       call, module args passed via --loadmodule.
    2. CI services (FALKORDB_TEST_IMAGE + FALKORDB_USE_SERVICE=1) — the
       shared GHA service container, isolated by FLUSHALL, module args
       applied via GRAPH.CONFIG SET.
    3. CI spawn (FALKORDB_TEST_IMAGE alone) — a dedicated container per
       call with -e FALKORDB_ARGS so load-time validation fires the same way
       --loadmodule would.

    `enableDebugCommand` is accepted and ignored: every mode now enables the
    DEBUG command unconditionally (local always did, via flow.sh; the CI
    service container is launched with it; spawn passes it through). It stays
    in the signature only so existing call sites keep working.
    """
    if env not in ('oss', 'oss-cluster'):
        raise RuntimeError(
            f"Env() doesn't support env={env!r}; only 'oss' (single-node) "
            "and 'oss-cluster' (multi-shard) are wired up.")

    shards = None
    if env == 'oss-cluster' or shardsCount:
        # RLTest's native oss-cluster Env defaults to 3 shards when
        # shardsCount is omitted; mirror that so test classes can write
        # `Env(env='oss-cluster')` without specifying a count.
        shards = int(shardsCount) if shardsCount else 3

    image = os.getenv("FALKORDB_TEST_IMAGE", "")
    use_service = uses_existing_service()

    if use_service:
        backend = ServiceBackend(module_args=moduleArgs, use_slaves=useSlaves,
                                 shards=shards,
                                 enable_debug_command=enableDebugCommand)
    elif not image:
        backend = LocalBackend(module_args=moduleArgs, use_slaves=useSlaves,
                               shards=shards)
    else:
        backend = DockerBackend(image, module_args=moduleArgs,
                                use_slaves=useSlaves, shards=shards,
                                enable_debug_command=enableDebugCommand)

    fenv = FalkorEnv(backend)
    _LIVE_ENVS.append(fenv)
    return fenv, db_handle(fenv.host, fenv.port)
