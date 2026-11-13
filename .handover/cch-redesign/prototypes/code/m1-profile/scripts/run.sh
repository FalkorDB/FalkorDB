#!/bin/bash
# usage: run.sh LOGFILE [samply-out.json.gz|-] VAR=VAL ...   (runs m1_reuse from the frozen binary; RSS watchdog 12 GB)
P=/private/tmp/claude-501/-Users-dvirdukhan-Code-worktrees-FalkorDB-pr-reviews-FalkorDB/f9e19c4b-505c-4a27-80e5-7848348679f5/scratchpad/m1prof
BIN=$P/graph-test
LOG=$1; shift
PROF=$1; shift
{ echo "UPTIME_START $(uptime)"; echo "BIN $(ls -lT $BIN) md5=$(md5 -q $BIN)"; echo "ENV $*"; echo "PROF $PROF"; } > $LOG
if [ "$PROF" = "-" ]; then
  env "$@" $BIN --ignored --exact graph::graphblas::m1_coarsen_bench::reuse::m1_reuse --nocapture >> $LOG 2>&1 &
else
  env "$@" samply record --save-only --unstable-presymbolicate -o $PROF -- $BIN --ignored --exact graph::graphblas::m1_coarsen_bench::reuse::m1_reuse --nocapture >> $LOG 2>&1 &
fi
PID=$!
maxrss=0
while kill -0 $PID 2>/dev/null; do
  for p in $PID $(pgrep -P $PID); do
    rss=$(ps -o rss= -p $p 2>/dev/null | tr -d ' ')
    if [ -n "$rss" ]; then
      [ "$rss" -gt "$maxrss" ] && maxrss=$rss
      if [ "$rss" -gt 12582912 ]; then kill -9 $p; echo "WATCHDOG KILLED pid=$p rss_kb=$rss" >> $LOG; fi
    fi
  done
  sleep 1
done
wait $PID; rc=$?
echo "EXIT $rc MAXRSS_KB $maxrss UPTIME_END $(uptime)" >> $LOG
exit $rc
