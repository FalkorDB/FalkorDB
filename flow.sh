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

# Existing-env mode: RLTest connects to a pre-running redis (typically a CI
# services-block container) instead of spawning one with --loadmodule. Parallel
# tests would share the same redis and collide, so force serial execution.
#
# --parallelism is set safely below to 1. We pass --module too so per-test
# Env(env='oss') overrides (tests/flow/common.py uses this when a test's
# moduleArgs include load-time-immutable config that the running redis can't
# apply on the fly) have the module path to spawn redis with. In CI the .so
# is extracted from the service container before flow.sh runs; locally the
# developer flow still builds it via cargo.
if [[ "$EXISTING_ENV" == 1 ]]; then
    MODULE_ARGS=(--env existing-env --existing-env-addr "${FALKORDB_HOST:-localhost}:${FALKORDB_PORT:-6379}" --module "$TARGET_DIR/$TARGET")
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
# EXISTING_ENV=1 FALKORDB_HOST=falkordb FALKORDB_PORT=6379 ./flow.sh

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
REDIS_CONF="$SCRIPT_DIR/tests/flow/redis.conf"

if [[ ${#TEST_FILTER[@]} -eq 0 ]]; then
    RLTest -f "$TESTS_FILE" "${MODULE_ARGS[@]}" --no-progress $PARALLELISM $STOP_ON_FAILURE --clear-logs --log-dir tests/flow/logs --enable-debug-command --enable-protected-configs --redis-config-file "$REDIS_CONF" $V
else
    RLTest "${TEST_FILTER[@]}" "${MODULE_ARGS[@]}" --no-progress $PARALLELISM $STOP_ON_FAILURE --clear-logs --log-dir tests/flow/logs --enable-debug-command --enable-protected-configs --redis-config-file "$REDIS_CONF" $V
fi
