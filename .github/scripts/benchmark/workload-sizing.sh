#!/usr/bin/env bash
# Single source of truth for the per-dataset-size benchmark workload knobs:
# query count, client concurrency, and dispatch rate. Sourced by
# generate-queries.sh (which needs the count) and run-variant.sh (which needs
# the concurrency + rate) so the two never drift. An explicit env value always
# wins, so a repo variable or a manual run can still override any knob.
#
#   source "$(dirname "$0")/workload-sizing.sh"
#   workload_sizing "$DATASET_SIZE"   # -> WL_QUERIES_COUNT WL_PARALLEL WL_MPS
#
# Sizing rationale (2026-07): medium is ~14x small (100k vs 10k vertices,
# ~1.77M vs ~122k edges). At 20 clients its heavier queries pile up behind the
# serialized writer and the 5s per-query cap, inflating p99 into timeouts, so
# medium runs FEWER queries (2000) at FEWER clients (8). small (and large, left
# as-is) keep the original saturation profile. MPS stays 5000 everywhere — far
# above the ~15 msg/s the engine actually sustains, so it just keeps the workers
# fed; `parallel` is the real lever. See run-variant.sh's FALKOR_QUERY_TIMEOUT_MS
# note for the timeout interaction.
# WL_* are consumed by the script that sources this file; shellcheck can't see
# that cross-file use.
# shellcheck disable=SC2034
workload_sizing() {
  local size="${1:-small}"
  case "$size" in
    medium)
      WL_QUERIES_COUNT="${QUERIES_COUNT:-2000}"
      WL_PARALLEL="${PARALLEL:-8}"
      ;;
    *)
      # small, large, and any unknown size: the original saturation profile.
      WL_QUERIES_COUNT="${QUERIES_COUNT:-20000}"
      WL_PARALLEL="${PARALLEL:-20}"
      ;;
  esac
  WL_MPS="${MPS:-5000}"
}
