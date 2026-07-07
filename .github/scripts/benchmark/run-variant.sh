#!/usr/bin/env bash
# Run ONE variant of the FalkorDB benchmark on this VM and leave its raw results
# under $RESULTS_DIR/$DATASET_SIZE/$NAME/ for the merge job to aggregate.
#
# The (size × variant) matrix in _benchmark.yml runs a leg per (size, variant),
# so all variants execute in parallel on separate, identical VMs instead of
# sequentially on one. Each leg is self-contained: build the CLI, generate its
# own query workload (same generation params → same distribution; percentile
# comparison stays valid), load the dataset, run, and hand off the raw results.
# Aggregation happens once, later, in `merge`.
set -euo pipefail

: "${IMAGE:?IMAGE is required}"
: "${NAME:?NAME is required (variant label, e.g. falkordb-c)}"
: "${BENCHMARK_DIR:?BENCHMARK_DIR (checkout of FalkorDB/benchmark) is required}"

DATASET_SIZE="${DATASET_SIZE:-small}"
QUERIES_COUNT="${QUERIES_COUNT:-200000}"
PARALLEL="${PARALLEL:-20}"
MPS="${MPS:-7500}"
BATCH_SIZE="${BATCH_SIZE:-5000}"
WRITE_RATIO="${WRITE_RATIO:-0.0}"
# Server-side per-query timeout (FalkorDB aborts the query and frees the thread
# at this deadline; the benchmark client applies it via ro_query .with_timeout).
# Default 5s, well below the tool's own 180s default: the heavy graph algos
# (maxflow/msf/harmonic) and the unbounded shortestPath otherwise run for
# minutes each, making medium take hours. At 5s they abort fast (recorded as
# timeouts) while the ordinary read shapes still complete and compare cleanly.
export FALKOR_QUERY_TIMEOUT_MS="${FALKOR_QUERY_TIMEOUT_MS:-5000}"
DB_PORT="${DB_PORT:-16379}"
DB_CPUS="${DB_CPUS:-4}"
DB_MEMORY="${DB_MEMORY:-12g}"
RESULTS_DIR="${RESULTS_DIR:-$(pwd)/Results-ab}"
CONTAINER_NAME="bench-db-$$"
QUERIES_NAME="ab-compare-${DATASET_SIZE}"

cleanup() { docker rm -f "$CONTAINER_NAME" >/dev/null 2>&1 || true; }
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

rm -rf "$RESULTS_DIR"
mkdir -p "$RESULTS_DIR"

cd "$BENCHMARK_DIR"
# The tool is checked out inside this repo's tree, whose root Cargo.toml is a
# [workspace]; make it its own workspace root so cargo will build it.
grep -q '^\[workspace\]' Cargo.toml || printf '\n[workspace]\n' >> Cargo.toml

echo "::group::Building benchmark CLI"
cargo build --release --bin benchmark
echo "::endgroup::"

echo "::group::Generating query workload (${QUERIES_NAME})"
cargo run --release --bin benchmark -- generate-queries \
  --vendor falkor --dataset "$DATASET_SIZE" --size "$QUERIES_COUNT" \
  --name "$QUERIES_NAME" --write-ratio "$WRITE_RATIO"
echo "::endgroup::"

echo "::group::Benchmarking ${NAME} (${IMAGE})"
docker rm -f "$CONTAINER_NAME" >/dev/null 2>&1 || true
docker pull "$IMAGE"
docker run -d --name "$CONTAINER_NAME" \
  --cpus="$DB_CPUS" --memory="$DB_MEMORY" \
  -p "${DB_PORT}:6379" \
  "$IMAGE" >/dev/null

wait_for_redis

# Throwaway container: never snapshot to disk — a failed bgsave during the big
# load would trip stop-writes-on-bgsave-error and abort it.
docker exec "$CONTAINER_NAME" redis-cli CONFIG SET save "" >/dev/null 2>&1 || true
docker exec "$CONTAINER_NAME" redis-cli CONFIG SET stop-writes-on-bgsave-error no >/dev/null 2>&1 || true

# Belt-and-braces wipe.
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

# Hand off raw results at <size>/<name>/ so the merge job can reassemble every
# variant of a size into one comparison.
dest="$RESULTS_DIR/$DATASET_SIZE/$NAME"
mkdir -p "$dest"
mv "$RESULTS_DIR"/falkor/* "$dest"/
rmdir "$RESULTS_DIR/falkor"

# Post-run server health, captured INTO the artifact (the live job log isn't
# reliably fetchable). On `large` the server goes unreachable mid-run and ~99.8%
# of queries return NoConnection — this tells us whether it was OOM-killed
# (State.OOMKilled=true / ExitCode 137) or crashed some other way, and the log
# tail shows the last thing it did before dying.
{
  echo "=== docker inspect state ($NAME / $DATASET_SIZE) ==="
  docker inspect \
    --format 'OOMKilled={{.State.OOMKilled}} ExitCode={{.State.ExitCode}} Status={{.State.Status}} Error={{.State.Error}} RestartCount={{.RestartCount}}' \
    "$CONTAINER_NAME" 2>&1
  echo "=== server (container) log tail ==="
  docker logs "$CONTAINER_NAME" 2>&1 | tail -100
} > "$dest/server-diagnostics.txt" 2>&1 || true
echo "::group::Server post-run diagnostics ($NAME / $DATASET_SIZE)"
cat "$dest/server-diagnostics.txt" 2>/dev/null || true
echo "::endgroup::"

docker rm -f "$CONTAINER_NAME" >/dev/null 2>&1 || true
echo "::endgroup::"
echo "Wrote raw results to $dest"
