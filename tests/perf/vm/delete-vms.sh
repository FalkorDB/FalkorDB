#!/usr/bin/env bash
# Delete every GCE VM that carries the runner label, in any zone, and fail if
# one is still there at the end.
#
#   delete-vms.sh <label> [grace-seconds]
#
# gce-github-runner tags each VM it creates with `<label>=1`. Several VMs can
# carry one label (a failed attempt whose own delete failed, then the next
# attempt), so this deletes all of them rather than the first match.
#
# grace-seconds keeps re-listing for that long even when nothing is found.
# Pass it when the create job did not succeed: a cancelled `gcloud compute
# instances create` leaves its insert running server-side, and the VM can
# appear after the first empty listing.
#
# Needs an authenticated gcloud with a default project. A failed listing is
# retried, never read as "none". Exits 1 if a VM survives, or listing still
# fails, DELETE_VMS_GIVE_UP_S (600) seconds after the grace period.
set -euo pipefail

label=${1:?usage: delete-vms.sh <label> [grace-seconds]}
grace=${2:-0}
poll=${DELETE_VMS_POLL_S:-10}
give_up=${DELETE_VMS_GIVE_UP_S:-600}

# GCE label keys: lowercase letter first, then lowercase letters, digits,
# `_` or `-`, at most 63 characters. No VM can carry any other key.
if ! [[ $label =~ ^[a-z][a-z0-9_-]{0,62}$ ]]; then
  echo "'$label' is not a valid GCE label key, so no VM carries it"
  exit 0
fi

list() {
  gcloud compute instances list \
    --filter="labels.${label}=1" \
    --format='value(name,zone.basename())'
}

start=$(date +%s)
settle_until=$((start + grace))
give_up_at=$((settle_until + give_up))

while :; do
  now=$(date +%s)
  if ! vms=$(list); then
    echo "::warning::listing VMs failed; will retry"
  elif [[ -z $vms ]]; then
    (( now >= settle_until )) && break
  else
    while read -r name zone; do
      echo "deleting $name ($zone)"
      # A VM that is still provisioning or already being deleted refuses;
      # the next listing retries it.
      gcloud compute instances delete "$name" --zone="$zone" --quiet </dev/null \
        || echo "::warning::delete of $name ($zone) failed; will retry"
    done <<< "$vms"
  fi
  (( now >= give_up_at )) && break
  sleep "$poll"
done

if ! left=$(list); then
  echo "::error::cannot list VMs with label $label, so cannot confirm they are gone"
  exit 1
fi
if [[ -n $left ]]; then
  echo "::error::VM(s) with label $label still exist after cleanup:"
  echo "$left"
  exit 1
fi
echo "no VM carries label $label"
