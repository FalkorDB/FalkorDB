#!/usr/bin/env bash
# Canonicalize a git branch/ref name into the URL- and filesystem-safe "view"
# slug used for a benchmark's published path (benchmark/branch/<view>/).
#
# Single source of truth: benchmark.yml's `prepare` job (which publishes a
# view) and its `cleanup-branch-view` job (which removes that view when the PR
# closes) must derive the *exact* same slug, or a closed PR would orphan the
# directory it published. Keeping the transform here — instead of duplicating
# the tr/sed pipeline in both jobs — is what guarantees that.
#
# The `main` remap keeps a branch literally named "main" from colliding with
# the canonical /benchmark/ trend (published only by rust-push.yml, never by
# benchmark.yml).
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
