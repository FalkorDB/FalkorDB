#!/bin/bash
# Kill any m1bench process above 12 GB RSS; log the peak seen. Exits when $1 (a pid) is gone.
LOG=/private/tmp/claude-501/-Users-dvirdukhan-Code-worktrees-FalkorDB-pr-reviews-FalkorDB/f9e19c4b-505c-4a27-80e5-7848348679f5/scratchpad/m1reuse/out/watchdog.log
peak=0
while kill -0 "$1" 2>/dev/null; do
  for p in $(pgrep -f 'm1bench'); do
    rss=$(ps -o rss= -p $p 2>/dev/null | tr -d ' ')
    [ -z "$rss" ] && continue
    [ "$rss" -gt "$peak" ] && peak=$rss
    if [ "$rss" -gt 12582912 ]; then echo "$(date) KILL pid=$p rss_kb=$rss" >> $LOG; kill $p; fi
  done
  sleep 1
done
echo "$(date) watchdog end, peak_rss_kb=$peak" >> $LOG
