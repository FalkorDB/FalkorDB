#!/bin/sh

set -e

srcdir=`dirname "$0"`
test -z "$srcdir" && srcdir=.

cd "$srcdir" || exit 1

if ! which autoreconf > /dev/null; then
    echo "Error: autoreconf not found." >&2
    exit 1
fi

autoreconf --install
