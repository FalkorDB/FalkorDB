#!/usr/bin/env bash
# Waits until no release-tag.yml run is still pushing its release images.
#
# The published-tag check also checks latest*, and latest* points at the
# newest stable GitHub release. A release is public before its images are, so
# a check that runs in between fails with "has no images here yet" although
# the run being checked published correctly. Callers run this first.
#
#   build/await-release-images.sh [--exclude-run RUN_ID] [--minutes N]
#
# --exclude-run skips the caller's own run (release-tag.yml's own check runs
# after its own image job, but that run is not completed yet). After N minutes
# (default 30) it fails, naming the runs still going, rather than let the
# check fail on someone else's release.
#
# Needs GH_TOKEN (actions: read) and GH_REPO.
set -euo pipefail

exclude=""
minutes=30
while [ $# -gt 0 ]; do
  case "$1" in
    --exclude-run) exclude="$2"; shift 2 ;;
    --minutes) minutes="$2"; shift 2 ;;
    *) echo "usage: $0 [--exclude-run RUN_ID] [--minutes N]" >&2; exit 2 ;;
  esac
done
: "${GH_REPO:?GH_REPO must be set}"

deadline=$(( SECONDS + minutes * 60 ))
while :; do
  busy=""
  # Queued runs count too: a release published from the UI is public before
  # its run starts. Assigned on its own line, so a failed call fails the
  # script instead of reading as "no release running".
  runs=$(gh api "repos/${GH_REPO}/actions/workflows/release-tag.yml/runs?per_page=20" \
           --jq '.workflow_runs[] | select(.status != "completed") | .id')
  for run in $runs; do
    [ "$run" = "$exclude" ] && continue
    # An image job not scheduled yet counts as still to come. A tag on an
    # older version branch runs that branch's release-tag.yml, whose image
    # job is a matrix: "image (falkordb)" and so on.
    states=$(gh api "repos/${GH_REPO}/actions/runs/${run}/jobs?per_page=100" --paginate \
               --jq '.jobs[] | select(.name == "image" or (.name | startswith("image ("))) | .status')
    if [ -z "$states" ] || grep -qv '^completed$' <<<"$states"; then
      busy="$busy https://github.com/${GH_REPO}/actions/runs/${run}"
    fi
  done
  if [ -z "$busy" ]; then
    echo "No release is pushing images."
    exit 0
  fi
  if [ "$SECONDS" -ge "$deadline" ]; then
    echo "::error::release-tag.yml is still pushing images after ${minutes} minutes:${busy}. The tag check would fail on latest* until it finishes; re-run this workflow's failed jobs once it has."
    exit 1
  fi
  echo "waiting for release images:${busy}"
  sleep 30
done
