#!/bin/sh
# Checks that every Dockerfile that builds a server pins the same Redis and
# Valkey releases, so the toolchain image the tests run on, the release
# images and the coverage/ASAN images all carry the same server revisions.
# Run from anywhere; exits non-zero on the first disagreement.
set -eu

cd "$(dirname "$0")/.."

files="build/Dockerfile
build/runtime/Dockerfile
build/runtime/Dockerfile.rhel9
build/runtime/Dockerfile.amazonlinux2023
build/runtime/Dockerfile.coverage
build/runtime/Dockerfile.asan"

status=0
for arg in REDIS_VERSION VALKEY_VERSION; do
    want=""
    for f in $files; do
        pins=$(sed -n "s/^ARG ${arg}=\([^[:space:]]*\)[[:space:]]*$/\1/p" "$f")
        count=$(printf '%s' "$pins" | grep -c . || true)
        if [ "$count" -ne 1 ]; then
            echo "FAIL: $f: expected one 'ARG ${arg}=<version>' default, found $count" >&2
            status=1
            continue
        fi
        if [ -z "$want" ]; then
            want=$pins
            first=$f
        elif [ "$pins" != "$want" ]; then
            echo "FAIL: $f pins ${arg}=$pins, but $first pins $want" >&2
            status=1
        fi
    done
    [ -z "$want" ] || echo "${arg}=${want}"
done
exit $status
