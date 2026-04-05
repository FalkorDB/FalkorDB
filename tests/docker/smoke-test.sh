#!/usr/bin/env bash
# smoke-test.sh — validate a production Docker image by running flow tests
#
# Usage:
#   ./tests/docker/smoke-test.sh <docker-image>
#
# Example:
#   ./tests/docker/smoke-test.sh falkordb/falkordb-x64
#
# The script starts the image, waits for Redis to become ready, then runs
# the subset of flow tests that work against a single container (no
# cluster, replication, restart, or custom module-args required).
# Designed to run in CI between "docker build" and "docker push".

set -euo pipefail

IMAGE="${1:?Usage: $0 <docker-image>}"
CONTAINER_NAME="falkordb-smoke-$$"
PORT=6399
TIMEOUT=30  # seconds to wait for Redis readiness

HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT="$(cd "$HERE/../.." && pwd)"

# Tests that cannot run against a single production container.
# Reasons: useSlaves, shardsCount, enableDebugCommand, custom moduleArgs,
#          env.stop()/restart()/dumpAndReload(), oss-cluster env,
#          BGSAVE, mutates global ACL/config state, concurrent race conditions,
#          depends on production TIMEOUT config.
EXCLUDED_TESTS=(
    test_acl
    test_bolt
    test_cache
    test_concurrent_query
    test_config
    test_constraint
    test_defrag
    test_effects
    test_encode_decode
    test_entity_update
    test_function_calls
    test_graph_copy
    test_graph_info
    test_intern_string
    test_load_csv
    test_memory_usage
    test_multi_writer
    test_pending_queries_limit
    test_persistency
    test_prev_rdb_decode
    test_query_mem_limit
    test_rdb_load
    test_replication
    test_replication_states
    test_results
    test_ro_query
    test_stress
    test_timeout
    test_udf
)

#------------------------------------------------------------------------------
# helpers
#------------------------------------------------------------------------------

cleanup() {
    docker rm -f "$CONTAINER_NAME" >/dev/null 2>&1 || true
}
trap cleanup EXIT

log() { printf '\033[1;34m[smoke]\033[0m %s\n' "$*"; }

is_excluded() {
    local name="$1"
    for exc in "${EXCLUDED_TESTS[@]}"; do
        [[ "$name" == "$exc" ]] && return 0
    done
    return 1
}

#------------------------------------------------------------------------------
# start container
#------------------------------------------------------------------------------

log "Starting container from image: $IMAGE"
docker run -d --name "$CONTAINER_NAME" -p "$PORT":6379 "$IMAGE" >/dev/null

log "Waiting for Redis to become ready (up to ${TIMEOUT}s)..."
elapsed=0
while ! docker exec "$CONTAINER_NAME" redis-cli PING 2>/dev/null | grep -q PONG; do
    sleep 1
    elapsed=$((elapsed + 1))
    if [ "$elapsed" -ge "$TIMEOUT" ]; then
        log "Redis did not become ready within ${TIMEOUT}s"
        docker logs "$CONTAINER_NAME"
        exit 1
    fi
done
log "Redis is ready (${elapsed}s)"

#------------------------------------------------------------------------------
# build test list
#------------------------------------------------------------------------------

cd "$ROOT/tests/flow"

test_file=$(mktemp "${TMPDIR:-/tmp}/smoke-tests.XXXXXXX")
count=0
for f in test_*.py; do
    name="${f%.py}"
    if ! is_excluded "$name"; then
        echo "$f" >> "$test_file"
        count=$((count + 1))
    fi
done

log "Selected $count flow tests (excluded ${#EXCLUDED_TESTS[@]})"

#------------------------------------------------------------------------------
# run flow tests via RLTest
#------------------------------------------------------------------------------

log "Running flow tests against production image on port $PORT..."

set +e
python3 -m RLTest \
    --env existing-env \
    --existing-env-addr "localhost:$PORT" \
    --test-timeout 180 \
    -f "$test_file"
rc=$?
set -e

rm -f "$test_file"

#------------------------------------------------------------------------------
# diagnostics on failure
#------------------------------------------------------------------------------

if [ "$rc" -ne 0 ]; then
    log "--- container logs (last 40 lines) ---"
    docker logs --tail 40 "$CONTAINER_NAME"
fi

exit "$rc"
