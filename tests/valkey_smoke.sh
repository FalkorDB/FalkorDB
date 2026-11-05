#!/usr/bin/env bash
# Smoke test: load the module into a real valkey-server and exercise it.
#
# Valkey dlopen()s modules with RTLD_DEEPBIND, unlike Redis, which changes how
# the module and its libraries (libomp) resolve symbols. That only shows up
# under a real valkey-server, so this cannot be a unit test. It also depends on
# how valkey-server was built: the environ crash (#3493) reproduces with
# Ubuntu's valkey-server package, not with the official valkey/valkey images.
#
# Usage: tests/valkey_smoke.sh [path/to/libfalkordb.so]
# Env:   VALKEY_SERVER, VALKEY_CLI, VALKEY_PORT
set -euo pipefail

MODULE=$(realpath "${1:-target/release/libfalkordb.so}")
SERVER=${VALKEY_SERVER:-valkey-server}
CLI=${VALKEY_CLI:-valkey-cli}
PORT=${VALKEY_PORT:-6390}
WORKDIR=$(mktemp -d)
LOG="$WORKDIR/valkey.log"
PID=

cleanup() {
    if [[ -n "$PID" ]] && kill -0 "$PID" 2>/dev/null; then
        kill "$PID"
        wait "$PID" 2>/dev/null || true
    fi
    rm -rf "$WORKDIR"
}
trap cleanup EXIT

fail() {
    echo "FAIL: $*" >&2
    echo "---- valkey log ----" >&2
    cat "$LOG" >&2 || true
    exit 1
}

cli() {
    "$CLI" -e -p "$PORT" "$@"
}

"$SERVER" --version
"$SERVER" --port "$PORT" --dir "$WORKDIR" --save '' --logfile "$LOG" \
    --loadmodule "$MODULE" &
PID=$!

for _ in $(seq 100); do
    kill -0 "$PID" 2>/dev/null || fail "valkey-server exited while loading the module"
    cli PING 2>/dev/null | grep -q PONG && break
    sleep 0.1
done
cli PING | grep -q PONG || fail "valkey-server did not become ready"

cli GRAPH.QUERY smoke "UNWIND range(1, 1000) AS i CREATE (:P {n: i})-[:K]->(:P {n: i + 1})" > /dev/null
cli GRAPH.QUERY smoke "CREATE INDEX FOR (p:P) ON (p.n)" > /dev/null
count=$(cli GRAPH.QUERY smoke "MATCH (:P)-[:K]->(b:P) RETURN count(b)" | sed -n 2p)
[[ "$count" == "1000" ]] || fail "expected count 1000, got '$count'"

# BGSAVE forks, which runs the module's fork handlers and libomp's atfork state.
cli BGSAVE > /dev/null
for _ in $(seq 100); do
    cli INFO persistence | grep -q 'rdb_bgsave_in_progress:0' && break
    sleep 0.1
done
cli INFO persistence | grep -q 'rdb_bgsave_in_progress:0' || fail "BGSAVE did not finish"
cli INFO persistence | grep -q 'rdb_last_bgsave_status:ok' || fail "BGSAVE failed"

kill -0 "$PID" 2>/dev/null || fail "valkey-server died"
if grep -q 'BUG REPORT' "$LOG"; then
    fail "valkey-server reported a crash"
fi
echo "PASS: module loaded and ran on $("$SERVER" --version | cut -d' ' -f1-3)"
