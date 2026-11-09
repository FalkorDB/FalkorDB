#!/usr/bin/env bash
# The gate job of .github/workflows/proof-impact.yml: checks an AI re-prove patch
# and runs the proof gates on it. The workflow runs this file, and the
# proof_gate_checks.py, lean_ci.sh and coverage.sh next to it, from the PR's
# base commit, so a PR cannot change the checks that judge it.
#
# Run from the workspace root, which holds:
#   in/         proofs.patch, reprove.md, verdicts.json (re-prove job), impact.json
#   code/       the PR head
#   proofs-src/ the PR head again; the patch is applied and built here
# Writes out/ (the proof-gate artifact: the patch, gates.md, gate_status, head_sha,
# added_theorems.json, logs) and, in $GITHUB_OUTPUT, gates=<status> and lean=<status>.
# Env: PROOFS_SHA (the head the patch is for), PROOF_PATHS and PROOF_DENY (the
# path allowlist and denylist), GITHUB_OUTPUT, GITHUB_WORKSPACE.
# Status: pass, held, incomplete, fail or rejected; always exits 0 once it has one.
set -uo pipefail
GATE_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
mkdir -p out
cp in/proofs.patch out/proofs.patch
cp in/reprove.md out/reprove.md
# The status and the head it is for go into the artifact too: proof-apply.yml
# commits the patch only for a `pass` on the PR's current head.
verdict() { echo "gates=$1" >> "$GITHUB_OUTPUT"; echo "$1" > "$GITHUB_WORKSPACE/out/gate_status"; }
echo "$PROOFS_SHA" > "$GITHUB_WORKSPACE/out/head_sha"
cd proofs-src || exit 1
if [[ -s ../out/proofs.patch ]] && ! git apply --index ../out/proofs.patch 2> ../out/apply.log; then
  { echo "Patch does not apply to the proofs base:"; echo '```'; cat ../out/apply.log; echo '```'; } > ../out/gates.md
  verdict rejected; exit 0
fi
# --no-renames: a rename is listed as its deletion and its addition, so both
# paths are checked. --raw gives each destination mode: only regular
# non-executable files (100644) or deletions (000000) are allowed.
raw=$(git diff --cached --raw --no-renames)
paths=$(git diff --cached --name-only --no-renames)
# One entry per line (separate $(...) would each drop their last newline and
# run the entries together). NF: an empty patch gives one blank line, not a file.
bad=$({
  printf '%s\n' "$paths" | grep -vE "$PROOF_PATHS" | grep -v '^$' || true
  printf '%s\n' "$paths" | grep -E "$PROOF_DENY" || true
  printf '%s\n' "$raw" | awk 'NF && $2 != "100644" && $2 != "000000" { print $NF " (mode " $2 ")" }'
})
[[ -n "$bad" ]] && bad+=$'\n'
# A file in a directory with no Lake configuration is never built, so its
# theorems would go unchecked while coverage.sh still counted its rows.
for p in $paths; do
  [[ "$p" == proofs/FINDINGS.md ]] && continue
  proj=$(cut -d/ -f1-2 <<< "$p")
  if ! git cat-file -e "HEAD:$proj/lakefile.toml" 2>/dev/null && ! git cat-file -e "HEAD:$proj/lakefile.lean" 2>/dev/null; then
    bad+="$p (no Lake project at $proj)"$'\n'
  fi
done
if [[ -n "$bad" ]]; then
  # shellcheck disable=SC2016  # literal Markdown backticks
  { echo "Patch dropped: it touches files outside the proof allowlist:"; echo; printf '%s' "$bad" | sed 's/^/- `/; s/$/`/'; } > ../out/gates.md
  verdict rejected; exit 0
fi
# Checks that need no Lean, so nothing in the patch runs before them:
#   - no build-time code added to a .lean file (#eval, elab, IO, ...): with
#     none, `lake build` only checks proofs and cannot touch this runner;
#   - existing theorem statements unchanged (a weaker one would still build);
#   - an explicit verdict for every changed function.
cp ../in/verdicts.json ../out/verdicts.json
cp ../in/impact.json ../out/impact.json
python3 -I "$GATE_DIR/proof_gate_checks.py" "$(git rev-parse HEAD)" ../in/impact.json ../in/verdicts.json ../out/added_theorems.json > ../out/checks.md 2> ../out/checks.err
rc=$?
if [[ $rc -ne 0 ]]; then
  # A crash is never a pass: nothing below has checked the patch.
  { echo "Patch dropped: the patch checks failed (exit $rc):"; echo '```'; tail -n 20 ../out/checks.err; echo '```'; } > ../out/gates.md
  verdict rejected; exit 0
fi
if grep -q '^REJECT$' ../out/checks.md; then
  sed '$d' ../out/checks.md > ../out/gates.md
  verdict rejected; exit 0
fi
# The gate scripts come from the PR's base, not the PR: copy them over the
# PR's own (left unstaged, so the checks above only saw the patch).
cp "$GATE_DIR/lean_ci.sh" "$GATE_DIR/coverage.sh" proofs/
lean=skipped; cov=pass; ok=pass
if [[ -s ../out/proofs.patch ]]; then
  lean=pass; proofs/lean_ci.sh > ../out/lean_ci.log 2>&1 || { lean=fail; ok=fail; }
  # lake builds a project's default targets and what they import, so a file no
  # target imports is never checked. Each added or changed .lean file must have
  # a fresh .olean (this runner restores no cache).
  unbuilt=""
  for p in $(git diff --cached --name-only --no-renames --diff-filter=AM -- 'proofs/*.lean'); do
    proj=$(cut -d/ -f1-2 <<< "$p"); mod=${p#"$proj"/}
    [[ -f "$proj/.lake/build/lib/lean/${mod%.lean}.olean" ]] || unbuilt+="- \`$p\`"$'\n'
  done
  if [[ -n "$unbuilt" ]]; then
    lean=fail; ok=fail
    { echo; echo "Not built by any target of its project (import it from the project's root module):"; printf '%s' "$unbuilt"; } >> ../out/checks.md.tmp
  fi
fi
COVERAGE_RUST_ROOT="$GITHUB_WORKSPACE/code" bash proofs/coverage.sh --unmatched > ../out/coverage.log 2>&1 || cov=fail
# coverage.sh exits 0 with findings; the counts decide (as in lean-proofs.yml).
unlisted=$(awk '/^TOTAL over all/ { print $(NF-3) }' ../out/coverage.log)
unmatched=$(sed -nE 's/.* ([0-9]+) rows unmatched.*/\1/p' ../out/coverage.log)
badb=$(sed -nE 's/.* ([0-9]+) bad buckets.*/\1/p' ../out/coverage.log)
# Same report lines as lean-proofs.yml's coverage gate.
untracked=$(grep -cE '^(projects without COVERAGE\.tsv|TEST-ONLY rows on fns the scanner sees as production code)' ../out/coverage.log || true)
if [[ "${unlisted:-}" != 0 || "${unmatched:-}" != 0 || "${badb:-}" != 0 || "$untracked" != 0 ]]; then cov=fail; fi
[[ "$cov" == fail ]] && ok=fail
tags=$(tail -n 1 ../out/checks.md)
{
  echo "Gates: \`lean_ci.sh\` **$lean**; \`coverage.sh --unmatched\` **$cov** (unlisted=${unlisted:-?}, unmatched=${unmatched:-?}, bad buckets=${badb:-?}, untracked=${untracked:-?})."
  echo
  sed '$d' ../out/checks.md
  [[ -f ../out/checks.md.tmp ]] && cat ../out/checks.md.tmp
} > ../out/gates.md
if [[ "$ok" == pass && "$tags" == *INCOMPLETE* ]]; then ok=incomplete; fi
if [[ "$ok" == pass && "$tags" == *HELD* ]]; then ok=held; fi
echo "lean=$lean" >> "$GITHUB_OUTPUT"
verdict "$ok"
