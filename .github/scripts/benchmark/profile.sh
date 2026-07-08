#!/usr/bin/env bash
# Profile the FalkorDB engine under the benchmark workload and emit a CPU flame
# graph. Reproducible on any Linux box with docker + perf + inferno:
#
#   1. run the FalkorDB image's redis-server (its shipped module already carries
#      function-level symbols for the whole compute path — GraphBLAS ~30k,
#      engine ~1k — so no rebuild is needed for a hot-path flame graph);
#   2. drive the SAME workload as the benchmark (generate → load → run);
#   3. sample the server PID with `perf` for the duration of the run;
#   4. render flamegraph.svg with inferno (+ folded stacks and a perf report).
#
# perf runs on the host against the container's PID (containers aren't
# PID-isolated from the host) and reads the module from the container's rootfs,
# so symbols resolve. redis-server itself is stripped but keeps .eh_frame, so
# perf unwinds through its (opaque) event-loop frames without breaking stacks.
#
# For line-level detail + inlined frames, set SO_PATH to a `profiling`-profile
# build (`cargo build --profile profiling`, needs GraphBLAS/RediSearch built);
# it's bind-mounted over the image's module.
#
# Requires: Linux, docker, perf (linux-perf/linux-tools-*), and inferno
# (`cargo install inferno`). perf needs kernel.perf_event_paranoid <= 1 (the CI
# job sets it; locally: `sudo sysctl kernel.perf_event_paranoid=1`).
set -euo pipefail

: "${IMAGE:?IMAGE is required (FalkorDB server image whose redis-server hosts our module, e.g. ghcr.io/falkordb/falkordb-server:rc-pr-<N>)}"
: "${BENCHMARK_DIR:?BENCHMARK_DIR (checkout of FalkorDB/benchmark) is required}"

DATASET_SIZE="${DATASET_SIZE:-small}"
QUERIES_COUNT="${QUERIES_COUNT:-20000}"
PARALLEL="${PARALLEL:-20}"
MPS="${MPS:-5000}"
WRITE_RATIO="${WRITE_RATIO:-0.0}"      # profiling is read-only by default
BATCH_SIZE="${BATCH_SIZE:-5000}"
export FALKOR_QUERY_TIMEOUT_MS="${FALKOR_QUERY_TIMEOUT_MS:-5000}"
DB_PORT="${DB_PORT:-16379}"
PERF_FREQ="${PERF_FREQ:-997}"          # Hz; 997 avoids lock-step with timers
OUT_DIR="${OUT_DIR:-$(pwd)/profile-out}"
CONTAINER_NAME="prof-db-$$"
QUERIES_NAME="ab-compare-${DATASET_SIZE}"
MODULE_PATH_IN_IMAGE="/var/lib/falkordb/bin/falkordb.so"

cleanup() { docker rm -f "$CONTAINER_NAME" >/dev/null 2>&1 || true; }
trap cleanup EXIT

wait_for_redis() {
  local tries=60
  until docker exec "$CONTAINER_NAME" redis-cli PING 2>/dev/null | grep -q PONG; do
    tries=$((tries - 1))
    if [ "$tries" -le 0 ]; then
      echo "::error::database container never became ready" >&2
      docker logs "$CONTAINER_NAME" 2>&1 | tail -100 || true
      return 1
    fi
    sleep 2
  done
}

rm -rf "$OUT_DIR"; mkdir -p "$OUT_DIR"

# Module symbols: the shipped image's own module already carries function-level
# symbols for the compute path (GraphBLAS ~30k, engine ~1k) — enough for a
# hot-path flame graph with ZERO rebuild. That's the default. Set SO_PATH to a
# `profiling`-profile build (`cargo build --profile profiling`) to bind-mount
# it instead and additionally get DWARF line numbers + inlined frames.
SO="${SO_PATH:-}"
if [ -n "$SO" ]; then
  [ -f "$SO" ] || { echo "::error::SO_PATH set but $SO not found" >&2; exit 1; }
  echo "module: bind-mounting profiling build $SO (adds line-level + inlined frames)"
else
  echo "module: profiling the image's own module as-is (function-level symbols, no rebuild)"
fi

echo "::group::Start server ($IMAGE hosting the profiling module)"
docker rm -f "$CONTAINER_NAME" >/dev/null 2>&1 || true
docker pull "$IMAGE"
# No CPU/mem cap: profiling wants the engine's real hot paths, not a throttled
# comparison. Bind-mount our symbol-ful module over the image's only when
# SO_PATH was given; otherwise the image runs its own (already-symbolized).
run_args=(-d --name "$CONTAINER_NAME" -p "${DB_PORT}:6379")
[ -n "$SO" ] && run_args+=(-v "$SO":"$MODULE_PATH_IN_IMAGE":ro)
docker run "${run_args[@]}" "$IMAGE" >/dev/null
wait_for_redis
docker exec "$CONTAINER_NAME" redis-cli CONFIG SET save "" >/dev/null 2>&1 || true
docker exec "$CONTAINER_NAME" redis-cli CONFIG SET stop-writes-on-bgsave-error no >/dev/null 2>&1 || true
PID="$(docker inspect -f '{{.State.Pid}}' "$CONTAINER_NAME")"
[ -n "$PID" ] && [ "$PID" != "0" ] || { echo "::error::could not resolve server host PID" >&2; exit 1; }
echo "server host PID: $PID"
echo "::endgroup::"

echo "::group::Generate queries + load dataset ($DATASET_SIZE)"
cd "$BENCHMARK_DIR"
grep -q '^\[workspace\]' Cargo.toml || printf '\n[workspace]\n' >> Cargo.toml
cargo build --release --bin benchmark
cargo run --release --bin benchmark -- generate-queries \
  --vendor falkor --dataset "$DATASET_SIZE" --size "$QUERIES_COUNT" \
  --name "$QUERIES_NAME" --write-ratio "$WRITE_RATIO"
cargo run --release --bin benchmark -- load \
  --vendor falkor --size "$DATASET_SIZE" \
  --endpoint "falkor://127.0.0.1:${DB_PORT}" -b "$BATCH_SIZE"
echo "::endgroup::"

echo "::group::perf record (server PID $PID) for the duration of the run"
# `-p PID -- <cmd>`: sample PID until <cmd> (the workload) exits. So we profile
# the server across exactly the run phase (not the load).
perf record -F "$PERF_FREQ" --call-graph dwarf -o "$OUT_DIR/perf.data" -p "$PID" -- \
  cargo run --release --bin benchmark -- run \
    --vendor falkor --name "$QUERIES_NAME" \
    --parallel "$PARALLEL" --mps "$MPS" \
    --endpoint "falkor://127.0.0.1:${DB_PORT}" \
    --results-dir "$OUT_DIR/results"
echo "::endgroup::"

echo "::group::Render flame graph"
# --no-inline: without SO_PATH the module has .symtab but no DWARF, so perf's
# per-address addr2line inline-expansion can't read it and SIGPIPEs mid-stream
# (truncating the render). Function-level names come from .symtab regardless;
# verified locally to symbolize the full engine + GraphBLAS path.
perf script --no-inline -i "$OUT_DIR/perf.data" > "$OUT_DIR/perf.script"
inferno-collapse-perf < "$OUT_DIR/perf.script" > "$OUT_DIR/out.folded"
inferno-flamegraph \
  --title "FalkorDB engine — ${DATASET_SIZE} (${QUERIES_COUNT} queries)" \
  "$OUT_DIR/out.folded" > "$OUT_DIR/flamegraph.svg"
perf report -i "$OUT_DIR/perf.data" --stdio --sort overhead,symbol 2>/dev/null \
  | grep -vE '^\s*#' | head -60 > "$OUT_DIR/perf-report.txt" || true
echo "::endgroup::"

docker rm -f "$CONTAINER_NAME" >/dev/null 2>&1 || true
echo "Wrote $OUT_DIR/flamegraph.svg (+ out.folded, perf-report.txt)"
