#!/usr/bin/env bash
# Checks which server a CI job ran its tests against, from the servers' own logs.
#
# Every start of Redis or Valkey logs a banner naming the server and its version:
#
#   Valkey version=9.1.2, bits=64, commit=00000000, modified=0, pid=9, just started
#   Redis version=8.10.2, bits=64, commit=00000000, modified=1, pid=9, just started
#
# This prints each distinct banner found in the given files and directories, so
# the job log shows what ran, and fails unless there is at least one banner and
# every banner names the expected server. A job whose tests passed against the
# wrong server, or that never started one, fails here.
#
# Paths that do not exist are reported and skipped: a job passes every place its
# harness may have written logs, and which of them exist depends on the tests.
#
# Usage: tests/assert_server_log.sh <redis|valkey> <file or directory>...
set -euo pipefail

usage() {
    echo "usage: $0 <redis|valkey> <file or directory>..." >&2
    exit 2
}

[[ $# -ge 2 ]] || usage
case "$1" in
    redis) want=Redis other=Valkey ;;
    valkey) want=Valkey other=Redis ;;
    *) usage ;;
esac
shift

paths=()
for p in "$@"; do
    if [[ -e "$p" ]]; then
        paths+=("$p")
    else
        echo "no such path, skipped: $p"
    fi
done
if [[ ${#paths[@]} -eq 0 ]]; then
    echo "::error::none of the log paths exist, so no server start can be checked" >&2
    exit 1
fi

# -a: a log can carry binary bytes (sanitizer reports); grep would otherwise
# print "binary file matches" instead of the banner. grep exits 1 when nothing
# matched, which the empty check below reports; above 1, a log could not be read.
rc=0
matches=$(grep -rahoE '(Redis|Valkey) version=[^,]+, bits=' "${paths[@]}") || rc=$?
if [[ $rc -gt 1 ]]; then
    echo "::error::grep could not read every log (exit $rc)" >&2
    exit 1
fi
if [[ -z "$matches" ]]; then
    echo "::error::no server start banner in: ${paths[*]}" >&2
    exit 1
fi
banners=$(sort <<<"${matches//, bits=/}" | uniq -c)
echo "server start banners (count, banner):"
echo "$banners"
if grep -q " ${other} version=" <<<"$banners"; then
    echo "::error::expected only ${want}, but ${other} started in this job" >&2
    exit 1
fi
echo "every server this job started is ${want}"
