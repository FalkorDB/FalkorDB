#!/usr/bin/env bash
# Check _perf-vm.yml's inputs and work out what to create.
#
#   plan.sh
#
# Environment: CHECK, MODE, TIMEOUT (minutes), ZONES, MACHINE_TYPE and
# FALLBACK (either may be empty), PMU (true or false), CANDIDATE, BASELINE,
# GITHUB_OUTPUT.
#
# Writes outputs zones, machine_type, fallback, pmu (the vPMU type to ask
# for, empty without counters), command_timeout_s, step_timeout,
# job_timeout and deadline; or error, and exits 1.
#
# Without PMU the VM is n4-standard-16 by default, and an n4-<size> falls
# back to n2-<size>. With PMU it must be a family whose VMs GCE gives
# hardware counters (C4 or M4; C4A and A4X are Arm): c4-standard-16 by
# default, c4-standard-8 as the fallback. A type without counters is never
# tried, so a check that wants counters gets them or gets no VM.
set -euo pipefail

requested_fallback=${FALLBACK:-}
requested_type=${MACHINE_TYPE:-}

fail() {
  echo "error=$1" >> "$GITHUB_OUTPUT"
  echo "::error::$1"
  exit 1
}

# The name becomes a GCE label key inside PERF_VM_LABEL (63 chars). `--` is
# reserved for a check's own artifact names (contract v1.5).
[[ $CHECK =~ ^[a-z][a-z0-9-]{0,42}$ && $CHECK != *--* ]] \
  || fail "check '$CHECK' must be lowercase letters, digits and single dashes, at most 43 characters"
[[ $MODE =~ ^(queue|nightly|long)$ ]] || fail "mode '$MODE' is not queue, nightly or long"
for image in "$CANDIDATE" "$BASELINE"; do
  [[ $image =~ ^[^@[:space:]]+@sha256:[0-9a-f]{64}$ ]] \
    || fail "image '$image' is not pinned by digest (name@sha256:<64 hex>)"
done
if ! [[ $TIMEOUT =~ ^[1-9][0-9]*$ ]]; then
  fail "timeout_minutes '$TIMEOUT' is not a whole number of minutes"
fi
[[ -n ${ZONES//[ ,]/} ]] || fail "no zones: set the zones input or vars.GCP_ZONES"

case ${PMU:-false} in
  true)
    pmu=standard
    machine_type=${requested_type:-c4-standard-16}
    fallback=$requested_fallback
    if [[ -z $fallback && $machine_type != c4-standard-8 ]]; then fallback=c4-standard-8; fi
    for type in "$machine_type" ${fallback:+"$fallback"}; do
      [[ $type =~ ^(c4|m4)- ]] \
        || fail "pmu needs a C4 or M4 machine type (GCE gives hardware counters to no other x86 family); got '$type'"
    done
    ;;
  false)
    pmu=
    machine_type=${requested_type:-n4-standard-16}
    fallback=$requested_fallback
    if [[ -z $fallback && $machine_type == n4-* ]]; then fallback=n2-${machine_type#n4-}; fi
    ;;
  *) fail "pmu '$PMU' is not true or false" ;;
esac

{
  echo "zones=$ZONES"
  echo "machine_type=$machine_type"
  echo "fallback=$fallback"
  echo "pmu=$pmu"
  echo "command_timeout_s=$((TIMEOUT * 60))"
  # The command's own `timeout` fires first; the step and job timeouts are
  # backstops that still leave time for the upload.
  echo "step_timeout=$((TIMEOUT + 5))"
  echo "job_timeout=$((TIMEOUT + 20))"
  # The VM deletes itself this long after the run job starts.
  echo "deadline=$((TIMEOUT + 30))"
} >> "$GITHUB_OUTPUT"
