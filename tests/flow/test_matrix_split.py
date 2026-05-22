#!/usr/bin/env python3
"""Split flow_tests_done.txt into two GHA-matrix arrays based on whether each
test file calls Env() with arguments that require a private redis instance.

A file goes to `spawn_files` if ANY Env() invocation passes one of:
  - moduleArgs WITH an immutable key (see IMMUTABLE_MODULE_ARGS below) —
    immutable keys can't be set at runtime via GRAPH.CONFIG SET, so the
    test genuinely needs a private redis loaded with those args
  - env='oss-cluster' — needs a multi-node cluster
  - shardsCount      — same as above

NOT spawn-forcing (the services job handles these directly):
  - enableDebugCommand — services container is launched with
                         REDIS_ARGS=--enable-debug-command yes
  - useSlaves          — services job runs a second `replica` container
                         (common.py services mode + _attach_slave)
  - moduleArgs with only runtime-mutable keys (TIMEOUT*, MAX_QUEUED_QUERIES,
                         RESULTSET_SIZE, QUERY_MEM_CAPACITY,
                         DELTA_MAX_PENDING_CHANGES, VKEY_MAX_ENTITY_COUNT) —
                         common.py applies these via GRAPH.CONFIG SET on the
                         shared service after FLUSHALL.

Outputs are emitted as `services_files=<json>` and `spawn_files=<json>` lines
suitable for `>> $GITHUB_OUTPUT`."""

import json
import os
import re
import sys

# Cluster topology forces spawn regardless of moduleArgs.
CLUSTER_RE = re.compile(
    r"\bEnv\s*\([^)]*\b(oss-cluster|shardsCount)\b",
    re.DOTALL,
)

# moduleArgs keys that can't be changed at runtime — these force spawn so
# the module loads with the right value baked in. Anything not on this list
# is assumed runtime-mutable via GRAPH.CONFIG SET and can ride a shared
# service container. The list is conservative; add to it (don't remove from
# it) if a key turns out to be load-time-only.
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
)

# `Env(...moduleArgs="...string...")` — capture the literal string content.
# Handles f-strings (`f"..."`) and both quote styles. re.DOTALL so the
# literal can span multiple lines.
MODULE_ARGS_RE = re.compile(
    r"\bmoduleArgs\s*=\s*f?['\"]([^'\"]*)['\"]",
    re.DOTALL,
)


def files_for_entry(entry):
    """Resolve a flow_tests_done.txt entry to the list of .py files to scan.

    Entries come in three shapes:
      - `tests/flow/test_xyz.py`    — direct .py file
      - `tests/flow/test_xyz`       — .py file with the extension omitted
      - `tests/flow/test_xyz`       — directory containing *.py files
    """
    if os.path.isfile(entry):
        return [entry]
    if os.path.isfile(entry + ".py"):
        return [entry + ".py"]
    if os.path.isdir(entry):
        out = []
        for root, _, names in os.walk(entry):
            for n in names:
                if n.endswith(".py"):
                    out.append(os.path.join(root, n))
        return out
    return []


def needs_spawn(paths):
    for p in paths:
        try:
            with open(p, encoding="utf-8") as f:
                content = f.read()
        except OSError:
            continue
        # Cluster topology — always spawn.
        if CLUSTER_RE.search(content):
            return True
        # moduleArgs with at least one immutable key — spawn.
        for match in MODULE_ARGS_RE.finditer(content):
            args_str = match.group(1)
            if any(key in args_str for key in IMMUTABLE_MODULE_ARGS):
                return True
    return False


def main():
    list_path = "flow_tests_done.txt"
    if len(sys.argv) > 1:
        list_path = sys.argv[1]

    services, spawn = [], []
    with open(list_path) as f:
        for line in f:
            entry = line.strip()
            if not entry:
                continue
            paths = files_for_entry(entry)
            if not paths:
                # Unresolvable entry — treat as spawn to be safe (better to
                # over-spawn than to miss a moduleArgs file).
                spawn.append(entry)
                continue
            (spawn if needs_spawn(paths) else services).append(entry)

    print(f"services_files={json.dumps(services)}")
    print(f"spawn_files={json.dumps(spawn)}")
    # `test_files` is the union for callers that don't care about the split
    # (coverage-flow runs against a locally-built .so, no docker involved).
    print(f"test_files={json.dumps(services + spawn)}")
    # On stderr so it shows up in the GHA log without polluting $GITHUB_OUTPUT.
    print(f"  services: {len(services)}, spawn: {len(spawn)}", file=sys.stderr)


if __name__ == "__main__":
    main()
