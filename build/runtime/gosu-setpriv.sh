#!/bin/sh
# gosu(1) for the RPM-based server images (Dockerfile.rhel9,
# Dockerfile.amazonlinux2023), installed there as /usr/local/bin/gosu.
#
# run.sh drops root with `gosu falkordb redis-server ...`. gosu is not packaged
# for RHEL or Amazon Linux; setpriv(1) from util-linux makes the same switch:
# real and effective uid and gid, plus the user's supplementary groups.
# Only the `gosu USER COMMAND...` form run.sh uses is supported.
set -eu

if [ "$#" -lt 2 ]; then
    echo "usage: gosu USER COMMAND [ARG]..." >&2
    exit 1
fi
user="$1"
shift
exec setpriv --reuid="$user" --regid="$user" --init-groups -- "$@"
