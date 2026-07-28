#!/bin/bash

PROGNAME="${BASH_SOURCE[0]}"
HERE="$(cd "$(dirname "$PROGNAME")" &>/dev/null && pwd)"
ROOT=$(cd "$HERE/../.." && pwd)

. "$ROOT/tests/common.sh"

cd "$HERE"

EXT=${EXT:-1}
EXT_HOST=${EXT_HOST:-127.0.0.1}
EXT_PORT=${EXT_PORT:-6379}

if [[ "$EXT" != "1" ]]; then
	echo "tests/jit_flow/tests.sh requires EXT=1 (existing preloaded Redis environment)."
	exit 1
fi

# Use YAML-driven smoke runner that doesn't flush the database
python3 "$HERE/runner.py" --host "$EXT_HOST" --port "$EXT_PORT"
