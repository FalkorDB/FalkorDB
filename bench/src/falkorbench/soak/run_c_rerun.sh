#!/bin/bash
# Rerun every C scenario on a released C module; old runs archived as <dir>@jul-master.
S=$(cd "$(dirname "$0")" && pwd); mkdir -p "$S/full"
MOD=${1:?usage: run_c_rerun.sh <c-module> <label>}; LABEL=${2:?label}
# the scenarios run in bench's environment, with its soak dependency group
PY=(uv run --project "$S/../../.." --frozen --group soak python)
# each scenario runs in a process group of its own, which is killed after it (servers it
# left behind included) and nothing else
SOAK_GROUP_FILE=$S/full/.scenario-group
source "$S/process_group.sh"
refuse_busy_ports 19000 19001 19002 || { echo "$(date +%T) C RERUN REFUSED: a scenario port is in use" >> "$S/full/progress.log"; exit 1; }
echo "$(date +%T) C RERUN START on $LABEL" >> "$S/full/progress.log"
run() { # n suffix dur
  n=$1; o=$S/full/$n-$2; dur=$3; f=$S/scenarios/$n.py
  [ -f "$o/.done" ] && [ "$(cat "$o/.commit" 2>/dev/null)" = "$LABEL" ] && return
  if [ -d "$o" ] && [ ! -d "$o@jul-master" ]; then mv "$o" "$o@jul-master"; fi
  free_kb=$(df -k / | tail -1 | awk '{print $4}')
  if [ "$free_kb" -lt 3000000 ]; then echo "$(date +%T) C RERUN PAUSED low disk before $n" >> "$S/full/progress.log"; exit 1; fi
  fresh_run_dir "$o"; echo "$LABEL" > "$o/.commit"
  echo "$(date +%T) START $n $2 dur=$dur ($LABEL)" >> "$S/full/progress.log"; start=$(date +%s)
  run_in_group timeout $((dur * 3 + 2400)) "${PY[@]}" -I "$f" --module "$MOD" --port 19000 --out "$o" --duration "$dur" --scale 1.0 > "$o/stdout.log" 2>&1
  rc=$?; echo "$rc" > "$o/.rc"; touch "$o/.done"
  echo "$(date +%T) END   $n $2 rc=$rc wall=$(( $(date +%s) - start ))s ($LABEL)" >> "$S/full/progress.log"
  find "$o" -type f \( -name '*.rdb' -o -name '*.aof' -o -path '*/aof/*' -o -name 'appendonly*' \) -delete 2>/dev/null
}
for f in "$S"/scenarios/s[0-9][0-9]_*.py; do n=$(basename "$f" .py); [ $n = s00_pilot ] && continue; run $n c 300; done
run s12_iot_retention c-samewindow 1200
echo "$(date +%T) C RERUN DONE ($LABEL)" >> "$S/full/progress.log"
