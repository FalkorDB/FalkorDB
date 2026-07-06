#!/usr/bin/env bash
# Runs the FalkorDB-vs-FalkorDB comparison across two or three variants using
# the FalkorDB/benchmark CLI, and writes a single timestamped UI summary
# JSON to $OUT_JSON.
#
#   A/B   — canonical main-branch trend (C engine `edge` vs Rust `edge-rs`)
#   A/B/C — a PR run, where C is the PR's `rc-pr-<N>` image, so B→C isolates
#           exactly the PR's impact and A/C compares it to the C engine.
#
# Mirrors FalkorDB/benchmark's scripts/run_small_benchmark.sh RUN_FALKOR /
# RUN_FALKOR_2 dance (same vendor "falkor" on every side, one shared query
# workload, results renamed into <name>/ subfolders, then aggregated with
# `aggregate-aws-tests`), but drives throwaway Docker containers instead of a
# locally-running redis-server, and never runs variants concurrently so no
# run's numbers are skewed by another contending for the runner's CPU/memory.
set -euo pipefail

: "${IMAGE_A:?IMAGE_A is required}"
: "${IMAGE_B:?IMAGE_B is required}"
: "${BENCHMARK_DIR:?BENCHMARK_DIR (checkout of FalkorDB/benchmark) is required}"
: "${OUT_JSON:?OUT_JSON output path is required (e.g. /tmp/out/falkordb_vs_falkordb_<epoch>.json)}"

# IMAGE_C is optional — set it (with NAME_C) to add the third, PR variant.
IMAGE_C="${IMAGE_C:-}"
NAME_A="${NAME_A:-falkordb-c}"
NAME_B="${NAME_B:-falkordb-rs}"
NAME_C="${NAME_C:-falkordb-pr}"
DATASET_SIZE="${DATASET_SIZE:-small}"   # small|medium|large — benchmark CLI's Size enum
QUERIES_COUNT="${QUERIES_COUNT:-200000}"
PARALLEL="${PARALLEL:-20}"
MPS="${MPS:-7500}"
BATCH_SIZE="${BATCH_SIZE:-5000}"
WRITE_RATIO="${WRITE_RATIO:-0.0}"
DB_PORT="${DB_PORT:-16379}"
# Fixed, equal resource limits on both containers so A and B are compared
# under identical conditions run-to-run, regardless of what else the
# benchmark client process is doing on the same dedicated runner.
DB_CPUS="${DB_CPUS:-4}"
DB_MEMORY="${DB_MEMORY:-12g}"
RESULTS_DIR="${RESULTS_DIR:-$(pwd)/Results-ab}"
CONTAINER_NAME="bench-db-$$"

QUERIES_NAME="ab-compare-${DATASET_SIZE}"

cleanup() {
  docker rm -f "$CONTAINER_NAME" >/dev/null 2>&1 || true
}
trap cleanup EXIT

wait_for_redis() {
  local tries=60
  until docker exec "$CONTAINER_NAME" redis-cli PING 2>/dev/null | grep -q PONG; do
    tries=$((tries - 1))
    if [ "$tries" -le 0 ]; then
      echo "::error::database container never became ready" >&2
      docker logs "$CONTAINER_NAME" 2>&1 | tail -200 || true
      return 1
    fi
    sleep 2
  done
}

run_variant() {
  local image="$1" name="$2"
  echo "::group::Benchmarking ${name} (${image})"

  docker rm -f "$CONTAINER_NAME" >/dev/null 2>&1 || true
  docker pull "$image"
  docker run -d --name "$CONTAINER_NAME" \
    --cpus="$DB_CPUS" --memory="$DB_MEMORY" \
    -p "${DB_PORT}:6379" \
    "$image" >/dev/null

  wait_for_redis

  # Belt-and-braces wipe — the container is freshly created so this is
  # normally a no-op, but guards against a reused/warm image.
  docker exec "$CONTAINER_NAME" redis-cli GRAPH.DELETE falkor >/dev/null 2>&1 || true
  docker exec "$CONTAINER_NAME" redis-cli DEL falkor >/dev/null 2>&1 || true

  cargo run --release --bin benchmark -- load \
    --vendor falkor --size "$DATASET_SIZE" \
    --endpoint "falkor://127.0.0.1:${DB_PORT}" -b "$BATCH_SIZE"

  cargo run --release --bin benchmark -- run \
    --vendor falkor --name "$QUERIES_NAME" \
    --parallel "$PARALLEL" --mps "$MPS" \
    --endpoint "falkor://127.0.0.1:${DB_PORT}" \
    --results-dir "$RESULTS_DIR"

  mkdir -p "$RESULTS_DIR/$name"
  mv "$RESULTS_DIR"/falkor/* "$RESULTS_DIR/$name"/
  rmdir "$RESULTS_DIR/falkor"

  docker rm -f "$CONTAINER_NAME" >/dev/null 2>&1 || true
  echo "::endgroup::"
}

rm -rf "$RESULTS_DIR"
mkdir -p "$RESULTS_DIR"

cd "$BENCHMARK_DIR"

echo "::group::Building benchmark CLI"
cargo build --release --bin benchmark
echo "::endgroup::"

echo "::group::Generating shared query workload (${QUERIES_NAME})"
cargo run --release --bin benchmark -- generate-queries \
  --vendor falkor --dataset "$DATASET_SIZE" --size "$QUERIES_COUNT" \
  --name "$QUERIES_NAME" --write-ratio "$WRITE_RATIO"
echo "::endgroup::"

# Sequential, never concurrent: each variant fully completes (and its
# container is torn down) before the next starts, so no run's numbers are
# skewed by another competing for CPU/memory/disk cache on the runner.
variants=("${NAME_A}|${IMAGE_A}" "${NAME_B}|${IMAGE_B}")
[ -n "$IMAGE_C" ] && variants+=("${NAME_C}|${IMAGE_C}")
for entry in "${variants[@]}"; do
  run_variant "${entry#*|}" "${entry%%|*}"
done

echo "::group::Aggregating comparison into ${OUT_JSON}"
mkdir -p "$(dirname "$OUT_JSON")"
cargo run --release --bin benchmark -- aggregate-aws-tests \
  --aws-tests-dir "$RESULTS_DIR" --out-path "$OUT_JSON"
echo "::endgroup::"

echo "Wrote $OUT_JSON"
