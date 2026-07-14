#!/usr/bin/env bash
# Single source of truth for the per-dataset-size benchmark workload knobs:
# query count, client concurrency, and dispatch rate. Sourced by
# generate-queries.sh (which needs the count) and run-variant.sh (which needs
# the concurrency + rate) so the two never drift. An explicit env value always
# wins, so a repo variable or a manual run can still override any knob.
#
#   source "$(dirname "$0")/workload-sizing.sh"
#   workload_sizing "$DATASET_SIZE"   # -> WL_QUERIES_COUNT WL_PARALLEL WL_MPS WL_TIMEOUT_MS
#
# Sizing rationale (2026-07): medium is ~14x small (100k vs 10k vertices,
# ~1.77M vs ~122k edges). At 20 clients its heavier queries pile up behind the
# serialized writer, so medium runs FEWER queries (2000) at FEWER clients (8).
# small (and large, left as-is) keep the original saturation profile. MPS stays
# 5000 everywhere — far above the ~5-15 msg/s the engine sustains, so it just
# keeps the workers fed; the run stays a saturation/throughput test.
#
# Per-query timeout (WL_TIMEOUT_MS): small/large cap at 5s so heavy graph algos
# (maxflow/msf/harmonic) and unbounded shortestPath abort fast instead of the
# tool's 180s default. Medium's ordinary shapes (order_by, 5-hop, aggregates on
# 100k) genuinely need seconds, so a 5s cap turned real work into timeouts;
# medium uses 30s so those complete and report real latency while the run stays
# saturated (throughput still differentiates the engines). Longer run is the
# trade-off.
# WL_* are consumed by the script that sources this file; shellcheck can't see
# that cross-file use.
# shellcheck disable=SC2034
workload_sizing() {
  local size="${1:-small}"
  case "$size" in
    medium)
      WL_QUERIES_COUNT="${QUERIES_COUNT:-2000}"
      WL_PARALLEL="${PARALLEL:-8}"
      WL_TIMEOUT_MS="${FALKOR_QUERY_TIMEOUT_MS:-30000}"
      ;;
    *)
      # small, large, and any unknown size: the original saturation profile.
      WL_QUERIES_COUNT="${QUERIES_COUNT:-20000}"
      WL_PARALLEL="${PARALLEL:-20}"
      WL_TIMEOUT_MS="${FALKOR_QUERY_TIMEOUT_MS:-5000}"
      ;;
  esac
  WL_MPS="${MPS:-5000}"
}
