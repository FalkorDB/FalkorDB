"""Which FalkorDB module args can be changed at runtime, and which can't.

Deliberately stdlib-only and free of any intra-package imports: the CI
matrix classifier (tests/flow/test_matrix_split.py) runs as a bare
`python3 tests/flow/test_matrix_split.py` step before the test dependencies
are installed, so it cannot afford to pull in `pytest` or `redis` via
`tests/infra/__init__.py`. Import it by path, not as `infra.module_args`.

This is the single source of truth for the split. Two consumers rely on it
and must agree, or a load-time-only arg silently becomes a no-op:

  - the classifier, to route a flow test file to the spawn bucket
  - the service backend / root suites, to know an arg can't be applied to an
    already-running server
"""

# Keys that GRAPH.CONFIG SET rejects at runtime (see the read-only list in
# src/commands/config_cmd.rs). A test needing one of these needs a private
# server loaded with the value baked in.
#
# The list is conservative by design: add to it, don't remove from it. A key
# wrongly listed here only costs a spawned container; a key wrongly *missing*
# produces a runtime error in services mode, or worse, a test that appears to
# configure something and doesn't.
IMMUTABLE_MODULE_ARGS = (
    "CACHE_SIZE",
    "THREAD_COUNT",
    "OMP_THREAD_COUNT",
    "IMPORT_FOLDER",
    "TEMP_FOLDER",
    "NODE_CREATION_BUFFER",
    "BOLT_PORT",
    "EFFECTS_THRESHOLD",
    "MAX_INFO_QUERIES",
    "ASYNC_DELETE",
    "CMD_INFO",
    "DELAY_INDEXING",
    "JS_HEAP_SIZE",
    "JS_STACK_SIZE",
    "TIMEOUT",
)


def split_pairs(module_args):
    """`"K1 V1 K2 V2"` -> `[("K1", "V1"), ("K2", "V2")]`.

    Raises on an odd token count rather than silently dropping the trailing
    key, which would turn a typo into a missing configuration.
    """
    toks = (module_args or "").split()
    if len(toks) % 2 != 0:
        raise RuntimeError(
            f"moduleArgs must be space-separated KEY VALUE pairs: "
            f"got {module_args!r}")
    return list(zip(toks[::2], toks[1::2]))


def partition(module_args):
    """Split module args into `(runtime_mutable, load_time_only)` pair lists."""
    mutable, immutable = [], []
    for key, value in split_pairs(module_args):
        (immutable if key in IMMUTABLE_MODULE_ARGS else mutable).append(
            (key, value))
    return mutable, immutable
