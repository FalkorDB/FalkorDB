"""Test orchestration shared by every FalkorDB python suite.

`Env()` returns a `(FalkorEnv, FalkorDB)` pair backed by one of three
backends, chosen by environment variable exactly as before:

    (none)                                  -> local redis-server subprocess
    FALKORDB_TEST_IMAGE + FALKORDB_USE_SERVICE -> shared CI service container
    FALKORDB_TEST_IMAGE                     -> dedicated CI container per Env()
"""

from .clients import (NO_RETRY, SOCKET_TIMEOUT, db_handle, make_client,
                      no_retry_client, probe_client)
from .docker import mountable_mkdtemp, mountable_mkstemp
from .env import Env, FalkorEnv, teardown_all, uses_existing_service
from .module_args import IMMUTABLE_MODULE_ARGS, partition

__all__ = [
    "Env",
    "FalkorEnv",
    "teardown_all",
    "uses_existing_service",
    "IMMUTABLE_MODULE_ARGS",
    "partition",
    "SOCKET_TIMEOUT",
    "NO_RETRY",
    "make_client",
    "probe_client",
    "no_retry_client",
    "db_handle",
    "mountable_mkdtemp",
    "mountable_mkstemp",
]
