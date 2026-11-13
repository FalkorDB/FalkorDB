#!/bin/bash
# run.sh BIN LOG VAR=VAL ...: run m1_reuse with a 12 GB RSS watchdog (copy of m1opt/run.sh, verifier paths)
BIN=$1; shift
LOG=$1; shift
mkdir -p $(dirname $LOG)
{ echo "UPTIME_START $(uptime)"; echo "BIN $BIN $(ls -lT $BIN | awk '{print $6,$7,$8,$9}') md5=$(md5 -q $BIN)"; echo "LIST $($BIN --list --ignored 2>/dev/null | grep m1_reuse)"; echo "ENV $*"; } > $LOG
env M1_JIT=run M1_CT=1 "$@" $BIN --ignored --exact graph::graphblas::m1_coarsen_bench::reuse::m1_reuse --nocapture >> $LOG 2>&1 &
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
