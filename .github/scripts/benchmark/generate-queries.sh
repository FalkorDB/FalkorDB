#!/usr/bin/env bash
# Build the FalkorDB/benchmark CLI and generate ONE query-workload file for a
# dataset size at $BENCHMARK_DIR/ab-compare-<size>.
#
# Generated once per benchmark run (in _benchmark.yml's `gen-queries` job) and
# shared to every A/B/C variant so they replay the IDENTICAL workload. The
# generator is unseeded, so generating per-variant would make each variant run
# different queries (same distribution, different draws) — sharing one file
# makes the comparison paired. It is also run-variant.sh's fallback when no
# shared file was provided (a standalone run, or a transient download miss).
#
# generate-queries is purely client-side — it writes a JSONL file (metadata line
# + one PreparedQuery per line) and never touches a database, so this runs on a
# plain runner with no server.
#
# Env:
#   BENCHMARK_DIR  - checkout of FalkorDB/benchmark (required)
#   DATASET_SIZE   - small | medium | large (required)
#   WRITE_RATIO    - fraction of write queries (default 0.0)
#   QUERY_PROFILE  - baseline | extended-core | fixture-dependent (default baseline)
#   QUERIES_COUNT  - override the per-size default count (optional)
set -euo pipefail

: "${BENCHMARK_DIR:?BENCHMARK_DIR (checkout of FalkorDB/benchmark) is required}"
: "${DATASET_SIZE:?DATASET_SIZE is required}"
WRITE_RATIO="${WRITE_RATIO:-0.0}"
QUERY_PROFILE="${QUERY_PROFILE:-baseline}"

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
# shellcheck source=.github/scripts/benchmark/workload-sizing.sh
# shellcheck disable=SC1091  # path resolved at runtime via $SCRIPT_DIR
source "$SCRIPT_DIR/workload-sizing.sh"
workload_sizing "$DATASET_SIZE"   # -> WL_QUERIES_COUNT (env QUERIES_COUNT wins)

QUERIES_NAME="ab-compare-${DATASET_SIZE}"

cd "$BENCHMARK_DIR"
# The tool is checked out inside this repo's tree, whose root Cargo.toml is a
# [workspace]; make it its own workspace root so cargo will build it.
grep -q '^\[workspace\]' Cargo.toml || printf '\n[workspace]\n' >> Cargo.toml

echo "::group::Building benchmark CLI (generate-queries)"
cargo build --release --bin benchmark
echo "::endgroup::"

echo "::group::Generating query workload ${QUERIES_NAME} (size=${DATASET_SIZE} count=${WL_QUERIES_COUNT} write_ratio=${WRITE_RATIO} profile=${QUERY_PROFILE})"
cargo run --release --bin benchmark -- generate-queries \
  --vendor falkor --dataset "$DATASET_SIZE" --size "$WL_QUERIES_COUNT" \
  --name "$QUERIES_NAME" --write-ratio "$WRITE_RATIO" \
  --query-profile "$QUERY_PROFILE"
echo "::endgroup::"
echo "Wrote query workload to $BENCHMARK_DIR/$QUERIES_NAME"
