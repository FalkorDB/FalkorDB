#!/bin/bash
if [[ "$(uname -s)" == "Darwin" ]]; then
  TARGET=libfalkordb.dylib
else
  TARGET=libfalkordb.so
fi

if [[ "$TARGET_DIR" == "" ]]; then
  if [[ "$RELEASE" == 1 ]]; then
    TARGET_DIR=target/release
  else
    TARGET_DIR=target/debug
  fi
fi

if [[ "$VERBOSE" == 1 ]]; then
  V=-v
else
  V=
fi

if [[ "$TESTS_FILE" == "" ]]; then
  TESTS_FILE=flow_tests_done.txt
fi

STOP_ON_FAILURE=""
if [[ "$PARALLELISM" == "" ]]; then
  if [[ "$(uname -s)" == "Darwin" ]]; then
    CORES=$(sysctl -n hw.ncpu)
  else
    CORES=$(nproc)
  fi
  echo "Running with parallelism: $CORES"
  PARALLELISM="--parallelism $CORES"
fi
if [[ "$FAIL_FAST" == 1 ]]; then
	STOP_ON_FAILURE="--stop-on-failure"
	PARALLELISM="--parallelism 1"
fi

# Two execution modes, selected by FALKORDB_TEST_IMAGE:
#
# - CI mode (FALKORDB_TEST_IMAGE set, e.g. ghcr.io/falkordb/falkordb-server:rc-pr-N):
#   tests/flow/common.py Env() handles all redis lifecycle (connects to the
#   shared `master` GHA service, or spawns a private container when the test
#   passes load-time moduleArgs). RLTest gets --env existing-env so it doesn't
#   try to spawn redis itself; the --existing-env-addr is bogus-but-required
#   syntax — the helper bypasses it entirely.
#
# - Local dev (FALKORDB_TEST_IMAGE unset):
#   Original behavior — RLTest spawns redis-server locally with --loadmodule.
#   Cargo-build the .so, run ./flow.sh — works the same as before this PR.
#
# --parallelism 1 in CI because the shared `master` service is process-singleton;
# parallel RLTest classes would collide on its state. Locally RLTest spawns one
# redis per class, so concurrency is safe.
if [[ "$FALKORDB_TEST_IMAGE" != "" ]]; then
    MODULE_ARGS=(--env existing-env --existing-env-addr "${FALKORDB_HOST:-master}:${FALKORDB_PORT:-6379}")
    PARALLELISM="--parallelism 1"
else
    MODULE_ARGS=(--module "$TARGET_DIR/$TARGET")
fi

# Add test filter support
TEST_FILTER=()
if [[ "$TEST" != "" ]]; then
    TEST_FILTER=(-t "$TEST")
fi

# To run specific test files, use:
# TEST="tests/flow/test_function_calls:testFunctionCallsFlow.test89_JOIN" FAIL_FAST=1 ./flow.sh
# To run all tests in a specific file, use:
# TEST="tests/flow/test_function_calls" FAIL_FAST=1 ./flow.sh
# To run against an already-running redis (e.g. a service container):
#   FALKORDB_TEST_IMAGE=ghcr.io/falkordb/falkordb-server:edge-rs \
#   FALKORDB_USE_SERVICE=1 \
#   FALKORDB_HOST=<host> FALKORDB_PORT=<port> \
#   ./flow.sh
# Setting FALKORDB_TEST_IMAGE alone selects spawn mode (Env() docker-runs a
# sibling container per call); add FALKORDB_USE_SERVICE=1 to skip the spawn
# and connect to the existing service at FALKORDB_HOST:FALKORDB_PORT.

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
REDIS_CONF="$SCRIPT_DIR/tests/flow/redis.conf"

if [[ ${#TEST_FILTER[@]} -eq 0 ]]; then
    RLTest -f "$TESTS_FILE" "${MODULE_ARGS[@]}" --no-progress $PARALLELISM $STOP_ON_FAILURE --clear-logs --log-dir tests/flow/logs --enable-debug-command --enable-protected-configs --redis-config-file "$REDIS_CONF" $V
else
    RLTest "${TEST_FILTER[@]}" "${MODULE_ARGS[@]}" --no-progress $PARALLELISM $STOP_ON_FAILURE --clear-logs --log-dir tests/flow/logs --enable-debug-command --enable-protected-configs --redis-config-file "$REDIS_CONF" $V
fi
