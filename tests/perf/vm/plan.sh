#!/usr/bin/env bash
# Check _perf-vm.yml's inputs and pick what to create: machine type, fallback,
# vPMU, gce-github-runner commit and timeouts, as GitHub outputs (or `error`,
# exit 1). Env: CHECK, MODE, TIMEOUT, ZONES, MACHINE_TYPE, FALLBACK, PMU,
# CANDIDATE, BASELINE (+ _LABEL, _COMMIT), GCE_GITHUB_RUNNER_REF and _PMU_REF.
# Runs in bench's environment (`uv run --project bench`), for falkorbench.kit.
# The rules: tests/perf/vm/README.md.
set -euo pipefail

requested_fallback=${FALLBACK:-}
requested_type=${MACHINE_TYPE:-}

fail() {
  local message=${1//$'\n'/ }  # one line: it becomes a job output
  echo "error=$message" >> "$GITHUB_OUTPUT"
  echo "::error::$message"
  exit 1
}

# The name becomes a GCE label key inside PERF_VM_LABEL (63 chars). `--` is
# reserved for a check's own artifact names (contract v1.5).
[[ $CHECK =~ ^[a-z][a-z0-9-]{0,42}$ && $CHECK != *--* ]] \
  || fail "check '$CHECK' must be lowercase letters, digits and single dashes, at most 43 characters"
# The flags the check will get, judged by the contract's own rule
# (images by digest per IMAGE_RE, 40-hex commits, one-line labels, mode):
# a run that could never give a valid result.json gets no VM.
# shellcheck disable=SC2016 # Python source, not shell expansions
if ! contract_errors=$(python -I -B -c '
import sys
from types import SimpleNamespace
from falkorbench.kit import input_errors
fields = ("candidate", "baseline", "mode", "candidate_label", "baseline_label",
          "candidate_commit", "baseline_commit")
values = dict(zip(fields, sys.argv[1:]))
for name in fields[3:]:
    values[name] = values[name] or None  # an empty input means the flag is not passed
errors = input_errors(SimpleNamespace(**values))
print("; ".join(errors))
sys.exit(1 if errors else 0)
' "$CANDIDATE" "$BASELINE" "$MODE" "${CANDIDATE_LABEL:-}" "${BASELINE_LABEL:-}" \
  "${CANDIDATE_COMMIT:-}" "${BASELINE_COMMIT:-}" 2>&1); then
  fail "${contract_errors:-the input check (falkorbench.kit) did not run}"
fi
if ! [[ $TIMEOUT =~ ^[1-9][0-9]*$ ]]; then
  fail "timeout_minutes '$TIMEOUT' is not a whole number of minutes"
fi
# gce-github-runner's startup script deletes the VM 6 h after boot, whatever
# the job is doing. Booting, setup, the step and job backstops and the upload
# need the last hour, so a check gets at most 5 h.
if (( TIMEOUT > 300 )); then
  fail "timeout_minutes $TIMEOUT is over 300: the VM deletes itself 6 h after boot, and booting, setup and upload need the rest"
fi
[[ -n ${ZONES//[ ,]/} ]] || fail "no zones: set the zones input or vars.GCP_ZONES"

for type in "$requested_type" "$requested_fallback"; do
  [[ -z $type || $type =~ ^[a-z][a-z0-9-]*$ ]] || fail "machine type '$type' is not a GCE machine type name"
  # create-vm.sh boots x86 Debian and the action's x64 runner (--arm=false),
  # which no Arm VM can run: C4A and N4A (Axion), T2A (Ampere), A4X (Grace).
  case $type in
    c4a-* | n4a-* | t2a-* | a4x-*) fail "machine type '$type' is Arm; the VM image and runner are x86 only" ;;
  esac
done

case ${PMU:-false} in
  true)
    pmu=standard
    action_ref=$GCE_GITHUB_RUNNER_PMU_REF
    machine_type=${requested_type:-c4-standard-16}
    fallback=$requested_fallback
    if [[ -z $requested_type && -z $fallback ]]; then fallback=c4-standard-8; fi
    for type in "$machine_type" ${fallback:+"$fallback"}; do
      [[ $type =~ ^(c4|m4)-[a-z0-9-]+$ ]] \
        || fail "pmu needs a C4 or M4 machine type (GCE gives hardware counters to no other x86 family); got '$type'"
    done
    ;;
  false)
    pmu=
    action_ref=$GCE_GITHUB_RUNNER_REF
    machine_type=${requested_type:-n4-standard-16}
    fallback=$requested_fallback
    if [[ -z $fallback && $machine_type == n4-* ]]; then fallback=n2-${machine_type#n4-}; fi
    ;;
  *) fail "pmu '$PMU' is not true or false" ;;
esac
[[ -n $action_ref ]] || fail "no gce-github-runner commit for pmu=$PMU"

{
  echo "zones=$ZONES"
  echo "machine_type=$machine_type"
  echo "fallback=$fallback"
  echo "pmu=$pmu"
  echo "action_ref=$action_ref"
  echo "command_timeout_s=$((TIMEOUT * 60))"
  # The command's own `timeout` fires first; the step and job timeouts are
  # backstops that still leave time for the upload.
  echo "step_timeout=$((TIMEOUT + 5))"
  echo "job_timeout=$((TIMEOUT + 20))"
  # The VM deletes itself this long after the run job starts.
  echo "deadline=$((TIMEOUT + 30))"
} >> "$GITHUB_OUTPUT"
