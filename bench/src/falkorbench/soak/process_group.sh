# Sourced by the release-soak run scripts (bash): run one scenario in a process group of
# its own, and afterwards kill what is left of that group. A local-module scenario starts
# its redis-servers as children (soakkit.Server), so they share the scenario's group, and
# a server that outlives its scenario (a crash, a timeout) is killed with the group. A
# server this run did not start (another soak, a developer's redis) is never touched, even
# on the same ports: soakkit refuses to start on a port that already answers.
#
# usage: run_in_group <command> [args...]    returns the command's exit status
#        refuse_busy_ports <port>...            at start: 1 (and why) if one already answers
#        fresh_run_dir <dir>                    an empty run directory to start an attempt in
#
# SOAK_GROUP_FILE, when the script sets it, holds the running scenario's group id, so a
# script that was itself SIGKILLed (no trap runs) leaves a way to stop what it started.

SOAK_GROUP=
SOAK_GROUP_FILE=${SOAK_GROUP_FILE:-}

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
  [ -z "$SOAK_GROUP_FILE" ] || rm -f "$SOAK_GROUP_FILE"
}

# A server already answering on a scenario's port is another soak's, a developer's, or
# one an earlier run left when it was SIGKILLed. None is this run's to stop, and every
# scenario would fail on it (soakkit refuses a port that answers) and be marked done:
# refuse to start instead.
refuse_busy_ports() {
  local port busy=
  for port in "$@"; do
    if (exec 3<>"/dev/tcp/127.0.0.1/$port") 2>/dev/null; then busy="$busy $port"; fi
  done
  [ -z "$busy" ] && return 0
  echo "a server already answers on port(s)$busy; stop it first" >&2
  if [ -n "$SOAK_GROUP_FILE" ] && [ -s "$SOAK_GROUP_FILE" ]; then
    echo "an earlier run's scenario may still run: kill -9 -- -$(cat "$SOAK_GROUP_FILE")" >&2
  fi
  return 1
}

# The scenario's group does not get the terminal's Ctrl-C, so the script reaps it on the
# way out, whatever ends it.
trap reap_group EXIT
trap 'reap_group; exit 130' INT
trap 'reap_group; exit 143' TERM

# A run directory still there after the scripts archived what they keep (its archive name
# was taken, or it holds an unfinished run) is moved aside, never reused: a run that fails
# before it writes its own summary.json would be scored on the old one. The name has no
# "@", so the scorer never takes it for an archive.
fresh_run_dir() {
  [ ! -e "$1" ] || mv "$1" "$1-superseded-$(date +%Y%m%dT%H%M%S)-$$"
  mkdir -p "$1"
}

run_in_group() {
  # with job control on, a background job leads a process group of its own (pgid = pid);
  # GNU timeout keeps that group for its command
  set -m
  "$@" &
  SOAK_GROUP=$!
  set +m
  [ -z "$SOAK_GROUP_FILE" ] || echo "$SOAK_GROUP" > "$SOAK_GROUP_FILE"
  wait "$SOAK_GROUP"
  local rc=$?
  reap_group
  return $rc
}
