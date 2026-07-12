#!/usr/bin/env bash
# Canonicalize a git branch/ref name into the URL- and filesystem-safe "view"
# slug used for a benchmark's published path (benchmark/branch/<view>/).
#
# Used by benchmark.yml's `prepare` for the workflow_dispatch path only — PR
# views are keyed off the PR number (pr-<N>) instead, since branch names aren't
# unique across PRs. The `main` remap keeps a dispatch from a branch literally
# named "main" from colliding with the canonical /benchmark/ trend (published
# only by rust-push.yml).
set -euo pipefail

raw="${1:?usage: view-slug.sh <branch-or-ref-name>}"

slug=$(printf '%s' "$raw" \
  | tr '[:upper:]' '[:lower:]' \
  | sed -E 's/[^a-z0-9]+/-/g; s/^-+//; s/-+$//')

# A branch made entirely of non-alphanumerics slugs to "" — refuse it rather
# than publish into benchmark/branch/ (which would collide across PRs and break
# cleanup). The cleanup path invokes this with `|| true` so it no-ops instead.
if [ -z "$slug" ]; then
  echo "::error::view-slug.sh: '$raw' has no alphanumeric characters to form a view slug" >&2
  exit 1
fi

if [ "$slug" = "main" ]; then
  slug="branch-main"
fi

printf '%s\n' "$slug"
