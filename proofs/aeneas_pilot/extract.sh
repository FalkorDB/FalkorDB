#!/usr/bin/env bash
# Regenerate every Lean file under Extracted/ and full/Extracted/ from the REAL
# Rust sources (graph/src/...), via Charon (Rust MIR -> LLBC) and Aeneas
# (LLBC -> Lean 4).  Never hand-edit the outputs; rerun this script.
#
# Needs the prebuilt nightly binaries (no opam/nix/sudo):
#   https://github.com/AeneasVerif/aeneas/releases/tag/nightly-2026.09.24-557f7a1
#     aeneas-macos-aarch64.tar.gz  (ships aeneas + the matching charon 0.1.270)
#   rustup toolchain install nightly-2026-09-17 -c rustc-dev,llvm-tools,rust-src
# and AENEAS_DIR pointing at the unpacked tarball (contains ./aeneas, ./charon).
set -euo pipefail
HERE="$(cd "$(dirname "$0")" && pwd)"
REPO="$(cd "$HERE/../.." && pwd)"
: "${AENEAS_DIR:?set AENEAS_DIR to the unpacked aeneas nightly tarball}"
CHARON="$AENEAS_DIR/charon"; AENEAS="$AENEAS_DIR/aeneas"
LLBC="$HERE/llbc"; mkdir -p "$LLBC"
export CARGO_TARGET_DIR="${CARGO_TARGET_DIR:-$HERE/rust/target}"

# 1. narrow_int alone, from the #[path]-including pilot crate. This is the file
#    the top-level (Mathlib-free) project type-checks against Aeneas.lean shim.
(cd "$HERE/rust" && "$CHARON" cargo --preset=aeneas \
   --start-from 'falkor_aeneas_pilot::narrow_int' --dest-file "$LLBC/narrow_int.llbc")
"$AENEAS" -backend lean -no-progress-bar -dest "$HERE/Extracted" "$LLBC/narrow_int.llbc"

# 2. narrow_int + runtime::bitset (needs the real Aeneas Lean lib: see full/).
(cd "$HERE/rust" && "$CHARON" cargo --preset=aeneas --dest-file "$LLBC/bitset.llbc")
"$AENEAS" -backend lean -no-progress-bar -dest "$HERE/full/Extracted" "$LLBC/bitset.llbc"

# 3. effects::v3::id_list, charon'd inside the real `graph` crate (2 min cold,
#    ~2 GB target dir). The excludes are the items Aeneas cannot translate
#    (dyn Iterator, impl-Iterator returns, early return inside a loop,
#    Iterator::copied/collect) -- see AeneasPilot.lean header.
M='graph::effects::v3::id_list'
if [[ "${SKIP_ID_LIST:-0}" != 1 ]]; then
  (cd "$REPO/graph" && CARGO_TARGET_DIR="${GRAPH_TARGET_DIR:-$CARGO_TARGET_DIR/graph}" \
    "$CHARON" cargo --preset=aeneas --start-from "$M" \
      --exclude "$M::_::iter" \
      --exclude "$M::{core::cmp::PartialEq<$M::IdList, _>}" \
      --exclude "$M::{core::fmt::Debug<$M::IdList>}" \
      --exclude "$M::{core::iter::traits::collect::FromIterator<$M::IdList, _>}" \
      --exclude "$M::{core::convert::From<$M::IdList, _>}" \
      --exclude "$M::read_ids" \
      --exclude "$M::{graph::effects::EffectDecodeSized<$M::IdList, _>}" \
      --dest-file "$LLBC/id_list.llbc" -- --lib)
  # Exit 1 = "partial file generated"; the sorries it leaves are listed in the header.
  "$AENEAS" -backend lean -no-progress-bar -split-files \
    -dest "$HERE/full/Extracted/IdList" "$LLBC/id_list.llbc" || true
fi
