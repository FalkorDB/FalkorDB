#!/usr/bin/env bash
# Aggregate one dataset size's variant raw-results into a single summary JSON.
#
# $VARIANTS_DIR contains one subdir per variant (falkordb-c/, falkordb-rs/,
# falkordb-pr/), each holding a variant's raw run output. aggregate-aws-tests is
# a two-vendor tool (picked.truncate(2)), so for three variants we aggregate
# each non-baseline variant against the baseline (A+B, A+C) and merge the
# per-vendor runs into one 3-run summary. Used by the merge job.
set -euo pipefail

: "${VARIANTS_DIR:?VARIANTS_DIR (dir of <variant>/ subdirs) is required}"
: "${OUT_JSON:?OUT_JSON output path is required}"
: "${BENCHMARK_DIR:?BENCHMARK_DIR (checkout of FalkorDB/benchmark) is required}"

NAME_A="${NAME_A:-falkordb-c}"
NAME_B="${NAME_B:-falkordb-rs}"
NAME_C="${NAME_C:-falkordb-pr}"

present=()
for v in "$NAME_A" "$NAME_B" "$NAME_C"; do
  [ -d "$VARIANTS_DIR/$v" ] && present+=("$v")
done
if [ "${#present[@]}" -eq 0 ]; then
  echo "::error::no variant subdirs under $VARIANTS_DIR" >&2
  exit 1
fi

mkdir -p "$(dirname "$OUT_JSON")"
cd "$BENCHMARK_DIR"

if [ "${#present[@]}" -le 2 ]; then
  cargo run --release --bin benchmark -- aggregate-aws-tests \
    --aws-tests-dir "$VARIANTS_DIR" --out-path "$OUT_JSON"
else
  merge_inputs=()
  for other in "$NAME_B" "$NAME_C"; do
    [ -d "$VARIANTS_DIR/$other" ] || continue
    pair="$VARIANTS_DIR/_pair_${other}"
    rm -rf "$pair"; mkdir -p "$pair"
    cp -r "$VARIANTS_DIR/$NAME_A" "$pair/$NAME_A"
    cp -r "$VARIANTS_DIR/$other" "$pair/$other"
    cargo run --release --bin benchmark -- aggregate-aws-tests \
      --aws-tests-dir "$pair" --out-path "$pair/summary.json"
    merge_inputs+=("$pair/summary.json")
  done
  python3 - "$OUT_JSON" "${merge_inputs[@]}" <<'PY'
import json, sys
runs = {}
for path in sys.argv[2:]:
    with open(path, encoding="utf-8") as f:
        for r in json.load(f).get("runs", []):
            runs[r["vendor"]] = r
with open(sys.argv[1], "w", encoding="utf-8") as f:
    json.dump({"runs": list(runs.values())}, f)
PY
fi
echo "Wrote $OUT_JSON"
