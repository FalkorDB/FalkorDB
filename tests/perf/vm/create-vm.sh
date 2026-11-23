#!/usr/bin/env bash
# Create the perf runner VM with FalkorDB/gce-github-runner, trying every zone
# for the machine type, then every zone for the fallback type.
#
#   create-vm.sh <action-dir> <label> <zones> <machine-type> [<fallback-type>]
#
# <action-dir> is a checkout of FalkorDB/gce-github-runner. Its action.sh runs
# once per (type, zone), with one zone each time: its own zone loop moves on
# only for ZONE_RESOURCE_POOL_EXHAUSTED_WITH_DETAILS and stops the whole create
# on any other error, such as a plain ZONE_RESOURCE_POOL_EXHAUSTED or a type
# that one zone does not offer. Here any failure moves on to the next zone.
#
# Only a failure to create the VM moves on. A VM that was created but whose
# runner never came up (the action's 7-minute readiness wait) is not a zone
# problem, and neither is a GH_SA_TOKEN that cannot register runners, which
# is checked once up front: both stop the loop rather than cost every zone.
#
# Environment: GH_SA_TOKEN, GCP_PROJECT_ID, GCP_SA_KEY (as for the action),
# RUNNER_VER (runner version, default latest), DISK_SIZE (default 100GB),
# CREATE_BUDGET_S (start no attempt after this many seconds, default 3000),
# GITHUB_OUTPUT. gcloud must already be authenticated, for delete-vms.sh.
#
# Writes outputs vm_name, zone, machine_type, fallback_used on success, or
# create_error, one line naming every attempt and why it failed, when no
# VM came up (then exits 1). create_error never holds the project ID:
# GitHub drops a job output that contains a secret.
set -uo pipefail

here=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
action_dir=${1:?usage: create-vm.sh <action-dir> <label> <zones> <machine-type> [<fallback-type>]}
label=${2:?label}
zones=${3:?zones}
primary=${4:?machine type}
fallback=${5:-}

# n4 and c4 boot only from Hyperdisk; the other families' default is the slow
# pd-standard, so name the disk type either way.
disk_type() {
  case $1 in
    n4-* | n4a-* | n4d-* | c4-* | c4a-* | c4d-*) echo hyperdisk-balanced ;;
    *) echo pd-balanced ;;
  esac
}

read -r -a zone_list <<< "${zones//,/ }"
types=("$primary")
if [[ -n $fallback && $fallback != "$primary" ]]; then
  types+=("$fallback")
fi

errors=()
give_up() {
  local joined
  joined=$(printf '%s; ' ${errors[@]+"${errors[@]}"} "$1")
  create_error="no VM came up: ${joined%; }"
  echo "create_error=$create_error" >> "$GITHUB_OUTPUT"
  echo "::error::$create_error"
  exit 1
}

# What gcloud said, on one line, without the project ID: its first ERROR
# line, the ` - reason` line under it, or the action's own last word.
reason_from() {
  local log=$1 rc=$2 text
  text=$(sed -n '/ERROR/{N;s/\n/ /;p;q;}' "$log")
  [[ -n $text ]] || text=$(tail -n1 "$log")
  text=${text:-exit $rc}
  text=${text//$GCP_PROJECT_ID/…}
  # shellcheck disable=SC2001 # a regex, not a literal
  sed 's|projects/[^/ ]*/|projects/…/|g' <<< "$text" | tr -d '\r' | tr -s ' ' | cut -c1-240
}

# action.sh reads the registration token with `curl` and no -f, so a bad
# token becomes the string "null" and every attempt waits 7 minutes for a
# runner that cannot register.
if ! curl -fsS --retry 3 -o /dev/null -X POST -H "Authorization: Bearer $GH_SA_TOKEN" \
    "https://api.github.com/repos/$GITHUB_REPOSITORY/actions/runners/registration-token"; then
  give_up "GH_SA_TOKEN cannot create a runner registration token for $GITHUB_REPOSITORY"
fi

for type in "${types[@]}"; do
  for zone in "${zone_list[@]}"; do
    if (( SECONDS >= ${CREATE_BUDGET_S:-3000} )); then
      give_up "stopped trying after ${SECONDS}s"
    fi
    # A failed attempt leaves a VM behind only if the action's own delete
    # failed; clear it so at most one VM ever carries the label.
    "$here/delete-vms.sh" "$label" 0 || true

    log=$(mktemp)
    action_out=$(mktemp)
    echo "::group::create $type in $zone"
    # action.sh writes runner_id to GITHUB_OUTPUT, also for an attempt that
    # then fails; keep it apart from this step's outputs.
    GITHUB_OUTPUT=$action_out "$action_dir/action.sh" \
      --command=start \
      --token="$GH_SA_TOKEN" \
      --project_id="$GCP_PROJECT_ID" \
      --service_account_key="$GCP_SA_KEY" \
      --runner_ver="${RUNNER_VER:-latest}" \
      --machine_zone="$zone" \
      --machine_type="$type" \
      --network=gh-runner \
      --disk_size="${DISK_SIZE:-100GB}" \
      --boot_disk_type="$(disk_type "$type")" \
      --image_project=debian-cloud \
      --image_family=debian-12 \
      --scopes=compute-rw \
      --shutdown_timeout=30 \
      --runner_label="$label" \
      --preemptible=false \
      --ephemeral=true \
      --no_external_address=false \
      --actions_preinstalled=false \
      --arm=false \
      </dev/null 2>&1 | tee "$log"
    rc=${PIPESTATUS[0]}
    echo "::endgroup::"

    vm_name=$(sed -n 's/^runner_id=//p' "$action_out" | tail -n1)
    rm -f "$action_out"
    if (( rc == 0 )); then
      vm_type=""
      if [[ -n $vm_name ]]; then
        for _ in 1 2 3; do
          vm_type=$(gcloud compute instances describe "$vm_name" --zone="$zone" \
            --format='value(machineType.basename())') && break
        done
        # The action saw this VM's runner come up; a failed describe must
        # not throw it away.
        vm_type=${vm_type:-$type}
      fi
      if [[ -n $vm_type ]]; then
        vm_zone=$zone
        fallback_used=false
        [[ $vm_type == "$primary" ]] || fallback_used=true
        {
          echo "vm_name=$vm_name"
          echo "zone=$vm_zone"
          echo "machine_type=$vm_type"
          echo "fallback_used=$fallback_used"
        } >> "$GITHUB_OUTPUT"
        echo "VM $vm_name: $vm_type in $vm_zone (fallback used: $fallback_used)"
        exit 0
      fi
      echo "::warning::the action reported success but no runner_id"
    fi

    reason=$(reason_from "$log" "$rc")
    if grep -q 'Waited 7 minutes' "$log"; then
      rm -f "$log"
      give_up "$type in $zone: the VM was created but its runner never came up ($reason)"
    fi
    rm -f "$log"
    errors+=("$type in $zone: $reason")
    echo "::warning::$type in $zone failed: $reason"
  done
done

give_up "every attempt failed"
