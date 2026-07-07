#!/usr/bin/env bash
# Publishes one A/B benchmark result into the gh-pages checkout at
# $GH_PAGES_DIR, building the vendored FalkorDB/benchmark dashboard (patched
# for relocatable static export — see patch-ui.sh) so the result is browsable
# with the same rich charts/histograms the upstream UI provides.
#
# Layout on gh-pages:
#   benchmark/                        <- canonical view (IS_CANONICAL=true),
#                                          fed only by the edge-tag-promotion
#                                          trigger. History grows unbounded,
#                                          mirroring the upstream repo's own
#                                          long-running falkordb_vs_falkordb
#                                          manifest.
#   benchmark/branch/<slug>/          <- one view per PR/manual-dispatch
#                                          branch, retention-capped so ad-hoc
#                                          runs don't grow gh-pages forever.
#
# Each view directory is a self-contained static export (its own _next/,
# summaries/, falkordb-compare/) with an index.html that redirects straight
# to falkordb-compare/, since that's the only comparison this repo produces
# (not the upstream repo's neo4j/memgraph/aws-tests pages).
set -euo pipefail

: "${REPO:?REPO is required (e.g. FalkorDB/falkordb-rs-next-gen)}"
: "${VIEW:?VIEW is required (slug, e.g. main or a branch slug)}"
: "${IS_CANONICAL:?IS_CANONICAL is required (true|false)}"
: "${SUMMARY_JSON:?SUMMARY_JSON path is required (falkordb_vs_falkordb_<epoch>.json produced by the merge job's aggregate-variants.sh)}"
: "${UI_DIR:?UI_DIR (vendored FalkorDB/benchmark ui/ checkout) is required}"
: "${GH_PAGES_DIR:?GH_PAGES_DIR (checkout of the gh-pages branch) is required}"

RETENTION="${RETENTION:-10}"
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_NAME="${REPO#*/}"
MANIFEST_KEY="falkordb_vs_falkordb.json"

if [ "$IS_CANONICAL" = "true" ]; then
  TARGET_REL="benchmark"
  EFFECTIVE_RETENTION=0   # canonical trend: keep full history, like upstream does
else
  TARGET_REL="benchmark/branch/${VIEW}"
  EFFECTIVE_RETENTION="$RETENTION"
fi

BASE_PATH="/${REPO_NAME}/${TARGET_REL}"
TARGET_DIR="$GH_PAGES_DIR/$TARGET_REL"
SUMMARIES_DIR="$UI_DIR/public/summaries"

BASENAME="$(basename "$SUMMARY_JSON")"                 # falkordb_vs_falkordb_<epoch>.json
EPOCH="${BASENAME#falkordb_vs_falkordb_}"
EPOCH="${EPOCH%.json}"
case "$EPOCH" in
  ''|*[!0-9]*)
    echo "::error::could not derive an epoch timestamp from SUMMARY_JSON basename '$BASENAME' (expected falkordb_vs_falkordb_<epoch>.json)" >&2
    exit 1
    ;;
esac

echo "::group::Seeding prior history for view '${VIEW}' (canonical=${IS_CANONICAL})"
mkdir -p "$SUMMARIES_DIR"
if [ -d "$TARGET_DIR/summaries" ]; then
  cp -a "$TARGET_DIR/summaries/." "$SUMMARIES_DIR/"
  echo "seeded $(find "$TARGET_DIR/summaries" -name '*.json' | wc -l | tr -d ' ') existing summary file(s)"
else
  echo "no prior published data for this view — starting fresh"
fi
echo "::endgroup::"

echo "::group::Adding new snapshot ($BASENAME, epoch=$EPOCH)"
cp "$SUMMARY_JSON" "$SUMMARIES_DIR/$BASENAME"
cp "$SUMMARY_JSON" "$SUMMARIES_DIR/$MANIFEST_KEY"   # "latest" pointer read by the dashboard by default

python3 "$SCRIPT_DIR/update-manifest.py" \
  --manifest "$SUMMARIES_DIR/manifest.json" \
  --key "$MANIFEST_KEY" \
  --add-filename "$BASENAME" \
  --add-timestamp "$EPOCH" \
  --summaries-dir "$SUMMARIES_DIR" \
  --retention "$EFFECTIVE_RETENTION"
echo "::endgroup::"

echo "::group::Building dashboard for base path ${BASE_PATH}"
UI_DIR="$UI_DIR" BASE_PATH="$BASE_PATH" "$SCRIPT_DIR/patch-ui.sh"
(
  cd "$UI_DIR"
  npm ci
  npm run build
)
echo "::endgroup::"

echo "::group::Publishing to ${TARGET_REL}"
OUT_DIR="$UI_DIR/out"
if [ ! -d "$OUT_DIR/falkordb-compare" ]; then
  echo "::error::static export did not produce out/falkordb-compare — aborting publish" >&2
  exit 1
fi

rm -rf "$TARGET_DIR"
mkdir -p "$TARGET_DIR"
# Publish only what this repo actually produces: the shared Next.js runtime
# assets, the falkordb-compare page, and the (already-filtered) summaries
# data. Deliberately drop the upstream repo's own home/neo4j/memgraph/
# aws-tests pages — we never generate that data and don't want stale
# upstream numbers implied as ours.
cp -r "$OUT_DIR/_next" "$TARGET_DIR/_next"
cp -r "$OUT_DIR/falkordb-compare" "$TARGET_DIR/falkordb-compare"
cp -r "$OUT_DIR/summaries" "$TARGET_DIR/summaries"
[ -f "$OUT_DIR/favicon.ico" ] && cp "$OUT_DIR/favicon.ico" "$TARGET_DIR/favicon.ico"
[ -f "$OUT_DIR/favicon.svg" ] && cp "$OUT_DIR/favicon.svg" "$TARGET_DIR/favicon.svg"

# The view's landing page redirects here. Default is the vendored dashboard;
# a PR view overrides it to our A/B/C impact page below.
REDIRECT_TARGET="falkordb-compare"

# Canonical trend only: build the per-query trend page from the full published
# history (all snapshots in the manifest). PR views are single-run, so a
# per-query-over-time view there would have nothing to trend.
if [ "$IS_CANONICAL" = "true" ]; then
  echo "building per-query trend page (${TARGET_REL}/trend/)"
  mkdir -p "$TARGET_DIR/trend"
  python3 "$SCRIPT_DIR/build-trend.py" \
    --summaries-dir "$TARGET_DIR/summaries" \
    --manifest "$TARGET_DIR/summaries/manifest.json" \
    --out "$TARGET_DIR/trend/trend.json" \
    --name-a "${NAME_A:-falkordb-c}" --name-b "${NAME_B:-falkordb-rs}"
  cp "$SCRIPT_DIR/benchmark-trend.html" "$TARGET_DIR/trend/index.html"
fi

# PR view only: build our A/B/C impact page (in-page size + metric switcher)
# from every per-size summary, and make it the landing page — the vendored
# dashboard (one size, rich histograms) stays linked from it. Skips cleanly on
# a two-variant run (no C), leaving the vendored dashboard as the landing page.
if [ "$IS_CANONICAL" != "true" ] && [ -n "${ABC_SUMMARIES_DIR:-}" ]; then
  echo "building A/B/C impact page (${TARGET_REL}/impact/)"
  mkdir -p "$TARGET_DIR/impact"
  if python3 "$SCRIPT_DIR/build-abc.py" \
       --summaries-dir "$ABC_SUMMARIES_DIR" --out "$TARGET_DIR/impact/abc.json" \
       --name-a "${NAME_A:-falkordb-c}" --name-b "${NAME_B:-falkordb-rs}" --name-c "${NAME_C:-falkordb-pr}"; then
    cp "$SCRIPT_DIR/benchmark-abc.html" "$TARGET_DIR/impact/index.html"
    REDIRECT_TARGET="impact"
  else
    echo "no A/B/C data (two-variant run?) — keeping the vendored dashboard as landing page"
    rmdir "$TARGET_DIR/impact" 2>/dev/null || true
  fi
fi

cat > "$TARGET_DIR/index.html" << EOF
<!DOCTYPE html>
<html lang="en">
<head>
  <meta charset="utf-8">
  <meta http-equiv="refresh" content="0; url=./${REDIRECT_TARGET}/">
  <link rel="canonical" href="./${REDIRECT_TARGET}/">
  <title>FalkorDB benchmark — redirecting</title>
</head>
<body>
  <p>Redirecting to <a href="./${REDIRECT_TARGET}/">the benchmark dashboard</a>...</p>
</body>
</html>
EOF
echo "::endgroup::"

PUBLISHED_URL="https://$(echo "${REPO%%/*}" | tr '[:upper:]' '[:lower:]').github.io/${REPO_NAME}/${TARGET_REL}/${REDIRECT_TARGET}/"
echo "published_url=${PUBLISHED_URL}"
echo "target_rel=${TARGET_REL}"

echo "::group::Committing and pushing gh-pages"
cd "$GH_PAGES_DIR"
git add -A -- "$TARGET_REL"
# Shared identity + token-remote + rebase-retry push (also a no-op when the
# output is byte-identical to what's already published).
"$SCRIPT_DIR/gh-pages-push.sh" "benchmark: update ${VIEW} view (epoch ${EPOCH})"
echo "::endgroup::"

# Surface outputs for the calling workflow (GITHUB_OUTPUT-compatible lines;
# the caller is expected to redirect this script's stdout there, or grep it).
echo "RESULT: published_url=${PUBLISHED_URL}"
