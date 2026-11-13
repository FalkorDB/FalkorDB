#!/bin/bash
# usage: run.sh LOGFILE TEST VAR=VAL ...   (frozen audit binary; RSS watchdog 12 GB)
A=/private/tmp/claude-501/-Users-dvirdukhan-Code-worktrees-FalkorDB-pr-reviews-FalkorDB/f9e19c4b-505c-4a27-80e5-7848348679f5/scratchpad/m1audit
BIN=$A/graph-test
LOG=$1; shift
TEST=$1; shift
{ echo "UPTIME_START $(uptime)"; echo "BIN $(ls -lT $BIN) md5=$(md5 -q $BIN)"; echo "TEST $TEST"; echo "ENV $*"; } > $LOG
env "$@" $BIN --ignored --exact $TEST --nocapture >> $LOG 2>&1 &
PID=$!
maxrss=0
while kill -0 $PID 2>/dev/null; do
  rss=$(ps -o rss= -p $PID 2>/dev/null | tr -d ' ')
  if [ -n "$rss" ]; then
    [ "$rss" -gt "$maxrss" ] && maxrss=$rss
    if [ "$rss" -gt 12582912 ]; then kill -9 $PID; echo "WATCHDOG KILLED rss_kb=$rss" >> $LOG; fi
  fi
  sleep 1
done
wait $PID; rc=$?
echo "EXIT $rc MAXRSS_KB $maxrss UPTIME_END $(uptime)" >> $LOG
exit $rc
