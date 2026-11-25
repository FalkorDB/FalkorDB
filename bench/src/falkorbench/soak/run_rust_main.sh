#!/bin/bash
# Re-soak every Rust scenario on a newer main, one at a time, only on a quiet machine.
# usage: run_rust_main.sh <module-path> <commit-label>
# The previous run of each scenario is kept as full/<scn>-rust@<its commit>.
# Before each run: wait until 1-min load < 8 for 5 straight minutes.
# During each run: sample load every 30 s; if 1-min load ever exceeds 20 the run is
# moved to full/<scn>-rust-<label>-contended-<n> and repeated (up to 3 attempts).
S=$(cd "$(dirname "$0")" && pwd); mkdir -p "$S/full"
MOD=$1; LABEL=$2
# the scenarios run in bench's environment, with its soak dependency group
PY=(uv run --project "$S/../../.." --frozen --group soak python)
log() { echo "$(date +%T) $*" >> "$S/full/progress.log"; }
# 1-minute load average: /proc/loadavg on Linux, vm.loadavg on macOS
load1() { if [ -r /proc/loadavg ]; then awk '{print $1}' /proc/loadavg; else sysctl -n vm.loadavg | awk '{print $2}'; fi; }
wait_quiet() {
  local q=0 said=0 l
  while [ $q -lt 5 ]; do
    l=$(load1 | awk '{print int($1)}')
    if [ "$l" -lt 8 ]; then q=$((q+1)); else q=0; [ $said = 0 ] && log "WAIT quiet machine before $1 (load $l)" && said=1; fi
    sleep 60
  done
}
log "RUST RERUN START on $LABEL"
for f in "$S"/scenarios/s[0-9][0-9]_*.py; do [ "$(basename "$f")" = s00_pilot.py ] && continue
  n=$(basename "$f" .py)
  o=$S/full/$n-rust
  [ -f "$o/.done" ] && [ "$(cat "$o/.commit" 2>/dev/null)" = "$LABEL" ] && continue
  for attempt in 1 2 3; do
    wait_quiet "$n"
    # archive the previous run (another commit) once
    if [ -d "$o" ]; then old=$(cat "$o/.commit" 2>/dev/null); [ -n "$old" ] && [ "$old" != "$LABEL" ] && [ ! -d "$o@$old" ] && mv "$o" "$o@$old"; fi
    [ -d "$o" ] && [ "$(cat "$o/.commit" 2>/dev/null)" = "$LABEL" ] && mv "$o" "$o-$LABEL-partial-$attempt-$$"
    free_kb=$(df -k / | tail -1 | awk '{print $4}')
    if [ "$free_kb" -lt 3000000 ]; then log "RUST RERUN PAUSED: only $((free_kb/1024)) MB free disk before $n"; exit 1; fi
    mkdir -p "$o"; echo "$LABEL" > "$o/.commit"
    pkill -9 -f 'redis-server \*:19' 2>/dev/null; sleep 1
    log "START $n rust dur=1200 ($LABEL)"
    start=$(date +%s)
    ( while true; do echo "$(date +%s) $(load1)" >> "$o/load.log"; sleep 30; done ) &
    sampler=$!
    timeout 6000 "${PY[@]}" -I "$f" --module "$MOD" --port 19000 --out "$o" --duration 1200 --scale 1.0 > "$o/stdout.log" 2>&1
    rc=$?
    kill $sampler 2>/dev/null
    pkill -9 -f 'redis-server \*:19' 2>/dev/null
    find "$o" -type f \( -name '*.rdb' -o -name '*.aof' -o -path '*/aof/*' -o -name 'appendonly*' \) -delete 2>/dev/null
    maxl=$(awk 'BEGIN{m=0} {if ($2>m) m=$2} END{print int(m)}' "$o/load.log")
    if [ "$maxl" -gt 20 ]; then
      log "END   $n rust rc=contended(max load $maxl) wall=$(( $(date +%s) - start ))s ($LABEL) attempt $attempt"
      mv "$o" "$o-$LABEL-contended-$attempt"
      continue
    fi
    echo "$rc" > "$o/.rc"; touch "$o/.done"
    log "END   $n rust rc=$rc wall=$(( $(date +%s) - start ))s ($LABEL) max load $maxl"
    break
  done
  # after 3 contended attempts the scorer falls back to the archived previous run
  [ -f "$o/.done" ] || log "GAVE UP $n after 3 contended attempts; scoring keeps the previous run"
done
log "RUST RERUN DONE ($LABEL)"
