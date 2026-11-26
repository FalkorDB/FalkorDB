# Sourced by the release-soak run scripts (bash): run one scenario in a process group of
# its own, and afterwards kill what is left of that group. A local-module scenario starts
# its redis-servers as children (soakkit.Server), so they share the scenario's group, and
# a server that outlives its scenario (a crash, a timeout) is killed with the group. A
# server this run did not start (another soak, a developer's redis) is never touched, even
# on the same ports: soakkit refuses to start on a port that already answers.
#
# usage: run_in_group <command> [args...]    returns the command's exit status

SOAK_GROUP=

reap_group() {
  [ -n "$SOAK_GROUP" ] || return 0
  kill -9 -- "-$SOAK_GROUP" 2>/dev/null
  # SIGKILL is asynchronous: wait (at most 10 s) until the group is gone, so the next
  # scenario finds its ports free
  local waited=0
  while kill -0 -- "-$SOAK_GROUP" 2>/dev/null && [ $waited -lt 100 ]; do
    sleep 0.1
    waited=$((waited + 1))
  done
  SOAK_GROUP=
}

# The scenario's group does not get the terminal's Ctrl-C, so the script reaps it on the
# way out, whatever ends it.
trap reap_group EXIT
trap 'reap_group; exit 130' INT
trap 'reap_group; exit 143' TERM

run_in_group() {
  # with job control on, a background job leads a process group of its own (pgid = pid);
  # GNU timeout keeps that group for its command
  set -m
  "$@" &
  SOAK_GROUP=$!
  set +m
  wait "$SOAK_GROUP"
  local rc=$?
  reap_group
  return $rc
}
