#!/usr/bin/env python3
"""Split flow_tests_done.txt into two GHA-matrix arrays based on whether each
test file calls Env() with arguments that require a private redis instance.

A file goes to `spawn_files` if ANY Env() invocation passes one of:
  - moduleArgs       — load-time module config (some keys like CACHE_SIZE,
                       IMPORT_FOLDER, NODE_CREATION_BUFFER are immutable)
  - env='oss-cluster' — needs a multi-node cluster
  - shardsCount      — same as above

NOT spawn-forcing (the services job handles these directly):
  - enableDebugCommand — the services container is launched with
                         REDIS_ARGS=--enable-debug-command yes, so DEBUG
                         RELOAD / DEBUG SLEEP work out of the box
  - useSlaves          — the services job runs a second `replica` container
                         configured with --replicaof falkordb 6379;
                         common.py services mode wires it through
                         env.replica_host/port + _attach_slave

Otherwise the file goes to `services_files` — it can reuse the shared GHA
service container that's brought up once per matrix cell.

Outputs are emitted as `services_files=<json>` and `spawn_files=<json>` lines
suitable for `>> $GITHUB_OUTPUT`."""

import json
import os
import re
import sys

# `Env(` followed by anything (across lines) that contains one of the
# spawn-forcing keywords. re.DOTALL makes . match newlines for multi-line
# Env() calls. We anchor on Env( to avoid matching the keywords elsewhere.
SPAWN_RE = re.compile(
    r"\bEnv\s*\([^)]*\b(moduleArgs|oss-cluster|shardsCount)\b",
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
                if SPAWN_RE.search(f.read()):
                    return True
        except OSError:
            continue
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
