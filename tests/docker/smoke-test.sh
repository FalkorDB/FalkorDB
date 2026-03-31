#!/usr/bin/env bash
# smoke-test.sh — validate a production Docker image before publishing
#
# Usage:
#   ./tests/docker/smoke-test.sh <docker-image>
#
# Example:
#   ./tests/docker/smoke-test.sh falkordb/falkordb-x64
#
# The script starts the image, waits for Redis to become ready, runs a
# series of lightweight checks, and exits 0 on success or 1 on any failure.
# Designed to run in CI between "docker build" and "docker push".

set -euo pipefail

IMAGE="${1:?Usage: $0 <docker-image>}"
CONTAINER_NAME="falkordb-smoke-$$"
PORT=6399
TIMEOUT=30  # seconds to wait for Redis readiness

passed=0
failed=0

#------------------------------------------------------------------------------
# helpers
#------------------------------------------------------------------------------

cleanup() {
    docker rm -f "$CONTAINER_NAME" >/dev/null 2>&1 || true
}
trap cleanup EXIT

log()  { printf '\033[1;34m[smoke]\033[0m %s\n' "$*"; }
pass() { printf '\033[1;32m  ✓ %s\033[0m\n' "$*"; passed=$((passed + 1)); }
fail() { printf '\033[1;31m  ✗ %s\033[0m\n' "$*"; failed=$((failed + 1)); }

run_query() {
    redis-cli -p "$PORT" GRAPH.QUERY smoke "$1" 2>&1
}

#------------------------------------------------------------------------------
# start container
#------------------------------------------------------------------------------

log "Starting container from image: $IMAGE"
docker run -d --name "$CONTAINER_NAME" -p "$PORT":6379 "$IMAGE" \
    redis-server --loadmodule /var/lib/falkordb/bin/falkordb.so \
    >/dev/null

log "Waiting for Redis to become ready (up to ${TIMEOUT}s)..."
elapsed=0
while ! redis-cli -p "$PORT" PING >/dev/null 2>&1; do
    sleep 1
    elapsed=$((elapsed + 1))
    if [ "$elapsed" -ge "$TIMEOUT" ]; then
        fail "Redis did not become ready within ${TIMEOUT}s"
        docker logs "$CONTAINER_NAME"
        exit 1
    fi
done
pass "Redis is ready (${elapsed}s)"

#------------------------------------------------------------------------------
# test: FalkorDB module is loaded
#------------------------------------------------------------------------------

log "Checking FalkorDB module is loaded..."
if redis-cli -p "$PORT" MODULE LIST 2>&1 | grep -qi "graph"; then
    pass "FalkorDB module loaded"
else
    fail "FalkorDB module not found in MODULE LIST"
fi

#------------------------------------------------------------------------------
# test: basic graph CRUD
#------------------------------------------------------------------------------

log "Running basic graph CRUD..."
result=$(run_query "CREATE (n:Person {name: 'smoke'})-[:KNOWS]->(m:Person {name: 'test'}) RETURN count(n)")
if echo "$result" | grep -q "1"; then
    pass "CREATE + RETURN works"
else
    fail "CREATE + RETURN: unexpected output: $result"
fi

result=$(run_query "MATCH (n:Person)-[r:KNOWS]->(m:Person) RETURN n.name, m.name")
if echo "$result" | grep -q "smoke" && echo "$result" | grep -q "test"; then
    pass "MATCH traversal works"
else
    fail "MATCH traversal: unexpected output: $result"
fi

#------------------------------------------------------------------------------
# test: LOAD CSV from HTTPS
#------------------------------------------------------------------------------

log "Testing LOAD CSV from HTTPS..."
csv_url="https://raw.githubusercontent.com/FalkorDB/FalkorDB/refs/heads/master/demo/social/resources/person.csv"
result=$(run_query "LOAD CSV FROM '${csv_url}' AS row RETURN count(row)")
if echo "$result" | grep -qE "[1-9][0-9]*"; then
    pass "LOAD CSV from HTTPS returns data"
else
    fail "LOAD CSV from HTTPS returned no rows: $result"
    # show Redis logs for diagnostics
    log "--- container logs (last 20 lines) ---"
    docker logs --tail 20 "$CONTAINER_NAME"
fi

#------------------------------------------------------------------------------
# test: GRAPH.LIST
#------------------------------------------------------------------------------

log "Testing GRAPH.LIST..."
result=$(redis-cli -p "$PORT" GRAPH.LIST 2>&1)
if echo "$result" | grep -q "smoke"; then
    pass "GRAPH.LIST shows created graph"
else
    fail "GRAPH.LIST: unexpected output: $result"
fi

#------------------------------------------------------------------------------
# test: GRAPH.DELETE
#------------------------------------------------------------------------------

log "Testing GRAPH.DELETE..."
result=$(redis-cli -p "$PORT" GRAPH.DELETE smoke 2>&1)
if echo "$result" | grep -qi "OK\|Graph removed"; then
    pass "GRAPH.DELETE succeeds"
else
    fail "GRAPH.DELETE: unexpected output: $result"
fi

#------------------------------------------------------------------------------
# summary
#------------------------------------------------------------------------------

echo ""
log "Results: $passed passed, $failed failed"
if [ "$failed" -gt 0 ]; then
    exit 1
fi
