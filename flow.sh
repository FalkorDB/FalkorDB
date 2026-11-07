#!/bin/bash
# Flow test runner.
#
# A thin wrapper around pytest. Redis lifecycle is owned by tests/infra:
#
# - Local dev (FALKORDB_TEST_IMAGE unset): a redis-server subprocess is
#   spawned per test class with --loadmodule pointing at the built module.
# - CI spawn mode (FALKORDB_TEST_IMAGE set): a sibling container per Env().
# - CI services mode (+ FALKORDB_USE_SERVICE=1): the shared `master` service
#   at FALKORDB_HOST:FALKORDB_PORT.
#
# Interface (unchanged from the RLTest runner):
#   TEST         narrow the run; see the examples below
#   TESTS_FILE   file listing the tests to run (default flow_tests_done.txt)
#   RELEASE=1    use target/release instead of target/debug
#   TARGET_DIR   override the module directory outright
#   FAIL_FAST=1  stop at the first failure, and run serially
#   PARALLELISM  worker count (default: one per core; 1 disables xdist)
#   VERBOSE=1    verbose pytest output
#
# Examples:
#   TEST="tests/flow/test_function_calls" FAIL_FAST=1 ./flow.sh   # one file
#   TEST="tests/flow/test_function_calls:testFunctionCallsFlow.test89_JOIN" ./flow.sh
#   FALKORDB_TEST_IMAGE=ghcr.io/falkordb/falkordb-server:edge-rs \
#     FALKORDB_USE_SERVICE=1 FALKORDB_HOST=<host> FALKORDB_PORT=<port> ./flow.sh
#     # services mode is forced serial: every Env() FLUSHALLs the shared server
set -uo pipefail

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
cd "$SCRIPT_DIR"

if [[ "${TARGET_DIR:-}" == "" ]]; then
  if [[ "${RELEASE:-}" == 1 ]]; then
    TARGET_DIR=target/release
  else
    TARGET_DIR=target/debug
  fi
fi
export TARGET_DIR

ARGS=(-p no:randomly)
[[ "${VERBOSE:-}" == 1 ]] && ARGS+=(-v) || ARGS+=(-q)

# Parallelism. FAIL_FAST forces serial: -x is meaningless when several
# workers are mid-test, and a first failure should be the reported one.
#
# Services mode forces serial too, and for a correctness reason rather than a
# reporting one: every Env() FLUSHALLs the one shared server, so two workers
# pointed at it wipe each other's data mid-test. CI's services cells happen to
# set FAIL_FAST=1, but nothing about the mode itself guarantees that.
if [[ "${FALKORDB_USE_SERVICE:-}" != "" || "${EXISTING_ENV:-}" == "1" ]]; then
  if [[ "${PARALLELISM:-1}" != 1 ]]; then
    echo "flow.sh: services mode shares one server; forcing PARALLELISM=1" >&2
  fi
  PARALLELISM=1
fi
if [[ "${FAIL_FAST:-}" == 1 ]]; then
  ARGS+=(-x)
  PARALLELISM=1
elif [[ "${PARALLELISM:-}" == "" ]]; then
  if [[ "$(uname -s)" == "Darwin" ]]; then
    PARALLELISM=$(sysctl -n hw.ncpu)
  else
    PARALLELISM=$(nproc)
  fi
  echo "Running with parallelism: $PARALLELISM"
fi
if ! [[ "$PARALLELISM" =~ ^[0-9]+$ ]]; then
  echo "flow.sh: PARALLELISM must be a worker count, got '$PARALLELISM'" >&2
  exit 2
fi

# Translate the RLTest selector syntax to pytest node ids:
#   tests/flow/test_x                    -> tests/flow/test_x.py
#   tests/flow/test_x:testCls.test_fn    -> tests/flow/test_x.py::testCls::test_fn
#   test_x                               -> tests/flow/test_x.py
to_nodeid() {
  local entry="$1" file="${1%%:*}" sel=""
  [[ "$entry" == *:* ]] && sel="${entry#*:}"
  [[ "$file" != *.py && "$file" != */ && -d "$file" ]] || file="${file%.py}.py"
  [[ -e "$file" ]] || file="tests/flow/$(basename "${file%.py}").py"
  [[ -n "$sel" ]] && echo "${file}::${sel//./::}" || echo "$file"
}

TARGETS=()
if [[ "${TEST:-}" != "" ]]; then
  TARGETS+=("$(to_nodeid "$TEST")")
else
  TESTS_FILE="${TESTS_FILE:-flow_tests_done.txt}"
  # Without this the redirect below just fails, the loop body never runs, and
  # pytest is exec'd with no node ids at all — which makes it collect the
  # whole rootdir instead of the curated list. Silently running the wrong set
  # of tests is the failure mode this script exists to prevent.
  if [[ ! -r "$TESTS_FILE" ]]; then
    echo "flow.sh: cannot read TESTS_FILE '$TESTS_FILE'" >&2
    exit 2
  fi
  # `|| [[ -n "$entry" ]]` catches a final line with no trailing newline —
  # flow_tests_done.txt has exactly that, so a plain `while read` silently
  # drops its last entry. RLTest read the file in Python and never noticed.
  while read -r entry || [[ -n "$entry" ]]; do
    [[ -z "$entry" ]] && continue
    TARGETS+=("$(to_nodeid "$entry")")
  done < "$TESTS_FILE"
fi

if [[ "${#TARGETS[@]}" -eq 0 ]]; then
  echo "flow.sh: no tests selected" >&2
  exit 2
fi

rm -rf tests/flow/logs && mkdir -p tests/flow/logs

# --dist loadfile keeps a whole file (and so a whole class, with its one
# redis instance) on a single worker. More workers than files therefore buys
# nothing but startup cost, which is the common case in CI: one file per
# matrix cell.
if [[ "$PARALLELISM" -gt "${#TARGETS[@]}" ]]; then
  PARALLELISM=$(( ${#TARGETS[@]} > 1 ? ${#TARGETS[@]} : 1 ))
fi
[[ "$PARALLELISM" != 1 ]] && ARGS+=(-n "$PARALLELISM" --dist loadfile)

PY=${PYTHON:-}
if [[ -z "$PY" ]]; then
  command -v python >/dev/null && PY=python || PY=python3
fi

exec "$PY" -m pytest "${ARGS[@]}" "${TARGETS[@]}"
