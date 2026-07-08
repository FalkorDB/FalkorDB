#!/usr/bin/env bash
# Commit the already-staged changes in the current gh-pages checkout and push
# them to the gh-pages branch, retrying with a rebase when a concurrent
# publisher (another PR view, or the canonical trend) moved the branch first.
#
# Single source of truth for the gh-pages write path, shared by publish.sh and
# benchmark.yml's cleanup-branch-view job — the bot identity, the
# token-authenticated remote, and the rebase-retry loop live here, once.
#
# The caller has already staged its change (`git add` / `git rm`) and cd'd into
# the gh-pages worktree; an empty stage is a no-op (identical output / nothing
# to remove), not an error.
#
# Env:
#   REPO      - owner/name, e.g. FalkorDB/falkordb-rs-next-gen
#   GH_TOKEN  - token with contents:write on REPO
# Args:
#   $1 - commit message
set -euo pipefail

: "${REPO:?REPO is required (owner/name)}"
: "${GH_TOKEN:?GH_TOKEN is required (contents:write token)}"
msg="${1:?commit message is required}"

git config user.name "github-actions[bot]"
git config user.email "41898282+github-actions[bot]@users.noreply.github.com"
git remote set-url origin "https://x-access-token:${GH_TOKEN}@github.com/${REPO}.git"

# GitHub Pages runs Jekyll by default, which drops every underscore-prefixed
# path — including Next.js's _next/ (all the dashboard CSS/JS). A root .nojekyll
# disables Jekyll so those assets are actually served. Ensure it exists on every
# publish (staged here so the very first publish carries it too).
if [ ! -f .nojekyll ]; then
  touch .nojekyll
  git add .nojekyll
fi

if git diff --cached --quiet; then
  echo "no staged changes — nothing to publish"
  exit 0
fi

git commit -q -m "$msg"

attempt=1
max_attempts=5
until git push origin gh-pages -q; do
  if [ "$attempt" -ge "$max_attempts" ]; then
    echo "::error::failed to push to gh-pages after ${max_attempts} attempts" >&2
    exit 1
  fi
  echo "push rejected (attempt ${attempt}/${max_attempts}) — fetching + rebasing and retrying"
  git fetch origin gh-pages -q
  # A conflicting rebase would otherwise leave the checkout mid-rebase and, under
  # `set -e`, abort the script with a raw git error instead of the message below.
  if ! git rebase origin/gh-pages; then
    git rebase --abort || true
    echo "::error::gh-pages rebase hit an unresolvable conflict on origin/gh-pages" >&2
    exit 1
  fi
  attempt=$((attempt + 1))
done
