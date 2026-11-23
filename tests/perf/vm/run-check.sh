#!/usr/bin/env bash
# Run one check the contract's way and record how it ended.
#
#   run-check.sh
#
# Environment: COMMAND (split on spaces, no shell syntax), CHECK, CANDIDATE,
# BASELINE, MODE, optional CANDIDATE_LABEL, BASELINE_LABEL,
# CANDIDATE_COMMIT, BASELINE_COMMIT (each flag passed only when non-empty),
# TIMEOUT_SECONDS, optional PERF_CLIENT_CPUS (the command runs under
# `taskset -c` on them), GITHUB_OUTPUT.
#
# Runs `<command> --candidate … --baseline … --out out --mode …` in the
# current directory, writes outputs started, command_exit and timed_out, and exits
# with the command's status. GitHub runs a step under `bash -e`; this
# script does not stop at a failing command before recording it.
set -uo pipefail
set +e

read -r -a cmd <<< "$COMMAND"
args=(--candidate "$CANDIDATE" --baseline "$BASELINE" --out out --mode "$MODE")
if [[ -n ${CANDIDATE_LABEL:-} ]]; then args+=(--candidate-label "$CANDIDATE_LABEL"); fi
if [[ -n ${BASELINE_LABEL:-} ]]; then args+=(--baseline-label "$BASELINE_LABEL"); fi
if [[ -n ${CANDIDATE_COMMIT:-} ]]; then args+=(--candidate-commit "$CANDIDATE_COMMIT"); fi
if [[ -n ${BASELINE_COMMIT:-} ]]; then args+=(--baseline-commit "$BASELINE_COMMIT"); fi

pin=()
if [[ -n ${PERF_CLIENT_CPUS:-} ]]; then pin=(taskset -c "$PERF_CLIENT_CPUS"); fi

rm -rf out && mkdir out
# Lets the workflow tell "stopped by the step timeout" from "never started".
echo "started=true" >> "$GITHUB_OUTPUT"
echo "running${PERF_CLIENT_CPUS:+ on CPUs $PERF_CLIENT_CPUS}: ${cmd[*]} ${args[*]}"
start=$SECONDS
timeout --kill-after=60 "$TIMEOUT_SECONDS" ${pin[@]+"${pin[@]}"} "${cmd[@]}" "${args[@]}"
rc=$?

timed_out=false
# 124: stopped by TERM at the limit; 137: KILLed after the grace period.
if (( (rc == 124 || rc == 137) && SECONDS - start >= TIMEOUT_SECONDS )); then
  timed_out=true
  echo "::error::$CHECK did not finish within $TIMEOUT_SECONDS seconds"
fi
echo "command_exit=$rc" >> "$GITHUB_OUTPUT"
echo "timed_out=$timed_out" >> "$GITHUB_OUTPUT"
if [[ ! -f out/result.json ]]; then
  echo "::warning::$CHECK wrote no result.json (exit $rc)"
fi
exit "$rc"
