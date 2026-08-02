#!/bin/bash
# Sample-based hot-stack profile of a single benchmark query.
# Usage: bench/profile.sh <out-name> "<cypher>" [GRAPH.QUERY|GRAPH.RO_QUERY] [port]
# Requires a running server with the bench graph:
#   python3 bench/run_bench.py --keep-server "RETURN 1"   # or --reuse setup
set -euo pipefail
NAME=$1
QUERY=$2
CMD=${3:-GRAPH.RO_QUERY}
PORT=${4:-6399}
OUT=bench/results/sample_$NAME.txt
mkdir -p bench/results

PID=$(redis-cli -p "$PORT" info server | grep -o 'process_id:[0-9]*' | cut -d: -f2)
redis-benchmark -p "$PORT" -c 1 -n 2000000 "$CMD" bench "$QUERY" >/dev/null 2>&1 &
BENCH=$!
sleep 0.5
sample "$PID" 5 -file "$OUT" >/dev/null
kill $BENCH 2>/dev/null || true
wait $BENCH 2>/dev/null || true

echo "== top of stack (hot leaves) =="
awk '/Sort by top of stack/,0' "$OUT" | head -25
echo
echo "full profile: $OUT"
