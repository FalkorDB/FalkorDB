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
# Environment: GH_SA_TOKEN, GCP_PROJECT_ID, GCP_SA_KEY (as for the action),
# RUNNER_VER (runner version, default latest), DISK_SIZE (default 100GB),
# GITHUB_OUTPUT. gcloud must already be authenticated, for delete-vms.sh.
#
# Writes outputs vm_name, zone, machine_type, fallback_used on success, or
# create_error, one line naming every attempt and why it failed, when no
# VM came up (then exits 1).
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
for type in "${types[@]}"; do
  for zone in "${zone_list[@]}"; do
    # A failed attempt leaves a VM behind only if the action's own delete
    # failed; clear it so at most one VM ever carries the label.
    "$here/delete-vms.sh" "$label" 0 || true

    log=$(mktemp)
    echo "::group::create $type in $zone"
    # action.sh writes runner_id/runner_label to GITHUB_OUTPUT, even for an
    # attempt that then fails; keep those out of this step's outputs.
    GITHUB_OUTPUT=$(mktemp) "$action_dir/action.sh" \
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
      --scopes=cloud-platform \
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

    if (( rc == 0 )); then
      if ! found=$(gcloud compute instances list --filter="labels.${label}=1" \
          --format='value(name,zone.basename(),machineType.basename())'); then
        found=""
      fi
      read -r vm_name vm_zone vm_type <<< "$(head -n1 <<< "$found")"
      if [[ -n ${vm_name:-} ]]; then
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
      echo "::warning::the action reported success, but no VM carries label $label"
    fi

    # The first gcloud ERROR line, or the action's own last word.
    reason=$(sed -n '/ERROR/{p;q;}' "$log")
    [[ -n $reason ]] || reason=$(tail -n1 "$log")
    reason=$(tr -d '\r' <<< "${reason:-exit $rc}" | cut -c1-200)
    errors+=("$type in $zone: $reason")
    echo "::warning::$type in $zone failed: $reason"
    rm -f "$log"
  done
done

joined=$(printf '%s; ' "${errors[@]}")
create_error="no VM came up after ${#errors[@]} attempts: ${joined%; }"
echo "create_error=$create_error" >> "$GITHUB_OUTPUT"
echo "::error::$create_error"
exit 1
