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
# ~1.77M vs ~122k edges). Its heavier queries pile up behind the serialized
# writer and the 5s cap, so medium runs FEWER queries (2000) at FEWER clients
# (4). small (and large, left as-is) keep the original 20-client saturation
# profile. MPS stays 5000 everywhere — far above the ~2-15 msg/s the engine
# sustains, so it just keeps the workers fed; `parallel` is the real lever.
#
# Tuning history: 8 clients still timed out on medium; a 30s per-query cap made
# it WORSE (throughput ~3x lower, tail 30-56s) because a few intrinsically-heavy
# shapes (writes through the serialized writer, order_by/aggregate on 100k) hog
# workers regardless of the cap. So we keep the 5s cap (uniform, see
# run-variant.sh) and cut medium to 4 clients to shorten the queue.
# WL_* are consumed by the script that sources this file; shellcheck can't see
# that cross-file use.
# shellcheck disable=SC2034
workload_sizing() {
  local size="${1:-small}"
  case "$size" in
    medium)
      WL_QUERIES_COUNT="${QUERIES_COUNT:-2000}"
      WL_PARALLEL="${PARALLEL:-4}"
      ;;
    *)
      # small, large, and any unknown size: the original saturation profile.
      WL_QUERIES_COUNT="${QUERIES_COUNT:-20000}"
      WL_PARALLEL="${PARALLEL:-20}"
      ;;
  esac
  WL_MPS="${MPS:-5000}"
}
