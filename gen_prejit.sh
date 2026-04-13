#!/usr/bin/env bash
# gen_prejit.sh – Regenerate GraphBLAS PreJIT kernels for FalkorDB.
#
# What this does:
#   1. Clears the existing PreJIT kernel sources and the GraphBLAS JIT cache.
#   2. Runs `make jit-warmup` (FalkorDB flow tests + LAGraph algorithm tests)
#      so GraphBLAS JIT-compiles every kernel it needs at runtime.
#   3. Harvests the freshly generated kernel sources from the JIT cache and
#      copies them into deps/GraphBLAS/PreJIT/ so they become compiled-in on
#      the next build instead of JIT-compiled at runtime.
#   4. Rebuilds and tests to verify the harvested kernels load correctly.
#
# NOTE: If you have added a new LAGraph algorithm to FalkorDB, update the
#       LAGRAPH_FALKORDB_TESTS regex in build.sh before running this script
#       so the new algorithm's tests are included in the warm-up.

set -uo pipefail

# ---------------------------------------------------------------------------
# Locate the FalkorDB repository root (directory containing this script).
# ---------------------------------------------------------------------------
FALKORDB_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

# ---------------------------------------------------------------------------
# Detect GraphBLAS version from source headers and derive the SuiteSparse
# JIT cache directory (~/.SuiteSparse/GrBMAJOR.MINOR.SUB).
# ---------------------------------------------------------------------------
GB_HEADER="${FALKORDB_DIR}/deps/GraphBLAS/Include/GraphBLAS.h"
if [[ ! -f "$GB_HEADER" ]]; then
    echo "ERROR: GraphBLAS header not found at ${GB_HEADER}" >&2
    exit 1
fi

GB_MAJOR=$(grep -E "^#define GxB_IMPLEMENTATION_MAJOR" "$GB_HEADER" | awk '{print $3}')
GB_MINOR=$(grep -E "^#define GxB_IMPLEMENTATION_MINOR" "$GB_HEADER" | awk '{print $3}')
GB_SUB=$(grep   -E "^#define GxB_IMPLEMENTATION_SUB"   "$GB_HEADER" | awk '{print $3}')
SUITESPARSE_GRB="${HOME}/.SuiteSparse/GrB${GB_MAJOR}.${GB_MINOR}.${GB_SUB}"

echo "============================================================"
echo " FalkorDB PreJIT kernel regeneration"
echo "============================================================"
echo " FalkorDB root : ${FALKORDB_DIR}"
echo " GraphBLAS     : ${GB_MAJOR}.${GB_MINOR}.${GB_SUB}"
echo " JIT cache     : ${SUITESPARSE_GRB}"
echo "============================================================"
echo ""

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------
die() { echo "ERROR: $*" >&2; exit 1; }

# run_with_retry <label> <command...>
#
# Runs the given command. On failure, asks the user:
#   a) continue  – treat this step as passed and move on
#   b) retry     – re-run the command from scratch
#   c) stop      – abort the script
#
# Returns 0 if the command succeeded or the user chose "continue".
# Exits the script if the user chose "stop".
run_with_retry() {
    local label="$1"
    shift
    while true; do
        # Run without aborting on failure (we handle it ourselves)
        if "$@"; then
            return 0
        fi
        echo ""
        echo "------------------------------------------------------------"
        echo " FAILURE: ${label}"
        echo "------------------------------------------------------------"
        echo " This may be a flaky test or a first-run JIT issue."
        echo " Choose an action:"
        echo "   a) Continue  – ignore this failure and proceed"
        echo "   b) Retry     – re-run this step"
        echo "   c) Stop      – abort the script"
        echo ""
        while true; do
            read -rp " Your choice [a/b/c]: " choice
            case "${choice,,}" in
                a) echo " Continuing..."; return 0 ;;
                b) echo " Retrying ${label}..."; break ;;
                c) echo " Stopping."; exit 1 ;;
                *) echo " Please enter a, b, or c." ;;
            esac
        done
    done
}

# ---------------------------------------------------------------------------
# Step 1 – Reminder: update build.sh if a new LAGraph algorithm was added.
# ---------------------------------------------------------------------------
echo "[Step 1] Reminder: if you added a new LAGraph algorithm, make sure its"
echo "         test is included in LAGRAPH_FALKORDB_TESTS in build.sh."
echo "         Press Enter to continue or Ctrl-C to abort and update it first."
read -r

# ---------------------------------------------------------------------------
# Step 2 – Remove the previous build output so the warm-up is done from
#           scratch (prevents stale kernels from being reused instead of
#           regenerated).
# ---------------------------------------------------------------------------
echo "[Step 2] Removing previous build output (bin/)..."
cd "${FALKORDB_DIR}"
rm -rf bin/

# ---------------------------------------------------------------------------
# Step 3 – Clear the existing PreJIT kernel sources.
# ---------------------------------------------------------------------------
echo "[Step 3] Clearing existing PreJIT kernel sources..."
PREJIT_DIR="${FALKORDB_DIR}/deps/GraphBLAS/PreJIT"
[[ -d "$PREJIT_DIR" ]] || die "PreJIT directory not found: ${PREJIT_DIR}"
# Use find so the command is a no-op (rather than an error) when no .c files exist.
find "${PREJIT_DIR}" -maxdepth 1 -name 'GB*.c' -delete
echo "   PreJIT directory cleared."

# ---------------------------------------------------------------------------
# Step 4 – Clear the GraphBLAS JIT runtime cache so kernels are regenerated
#           fresh rather than loaded from a previous warm-up run.
# ---------------------------------------------------------------------------
echo "[Step 4] Clearing GraphBLAS JIT runtime cache (${SUITESPARSE_GRB})..."
if [[ -d "${SUITESPARSE_GRB}" ]]; then
    rm -rf "${SUITESPARSE_GRB}/tmp" \
           "${SUITESPARSE_GRB}/c"   \
           "${SUITESPARSE_GRB}/lib"
    echo "   JIT cache cleared."
else
    echo "   JIT cache directory does not exist yet – nothing to clear."
fi

# ---------------------------------------------------------------------------
# Step 5 – Run the JIT warm-up: FalkorDB flow tests + LAGraph algorithm tests.
# ---------------------------------------------------------------------------
echo "[Step 5] Running jit-warmup (this will take a while)..."
cd "${FALKORDB_DIR}"
run_with_retry "make jit-warmup" make jit-warmup JIT=1

# ---------------------------------------------------------------------------
# Step 6 – Harvest the generated kernel sources from the JIT cache /c dir.
# ---------------------------------------------------------------------------
echo "[Step 6] Harvesting JIT kernels from cache..."
KERNEL_SRC_DIR="${SUITESPARSE_GRB}/c"
[[ -d "$KERNEL_SRC_DIR" ]] || die "JIT kernel source directory not found: ${KERNEL_SRC_DIR}
Did GraphBLAS JIT run during the warm-up? Check that JIT is not disabled (JIT=0)."

# Safety: confirm we have the right directory before moving anything.
echo "   Source  : ${KERNEL_SRC_DIR}"
echo "   Dest    : ${PREJIT_DIR}"

KERNEL_COUNT=$(find "${KERNEL_SRC_DIR}" -name 'GB*.c' | wc -l)
echo "   Kernels found: ${KERNEL_COUNT}"
[[ "$KERNEL_COUNT" -gt 0 ]] || die "No GB*.c kernel files found in ${KERNEL_SRC_DIR}"

find "${KERNEL_SRC_DIR}" -name 'GB*.c' \
    -exec mv {} "${PREJIT_DIR}/" \;

# ---------------------------------------------------------------------------
# Step 7 – Confirm new files arrived in PreJIT/.
# ---------------------------------------------------------------------------
echo "[Step 7] Verifying harvested kernels..."
NEW_COUNT=$(find "${PREJIT_DIR}" -maxdepth 1 -name 'GB*.c' | wc -l)
echo "   PreJIT kernel sources: ${NEW_COUNT}"
[[ "$NEW_COUNT" -gt 0 ]] || die "No kernel files in PreJIT after harvest – something went wrong."
echo "   OK – ${NEW_COUNT} kernel source(s) in place."

# ---------------------------------------------------------------------------
# Step 8 – Rebuild from scratch so the new PreJIT kernels are compiled in,
#           then run the full test suite to verify them.
# ---------------------------------------------------------------------------
echo "[Step 8] Clearing build output and rebuilding with new PreJIT kernels..."
cd "${FALKORDB_DIR}"
rm -rf bin/

echo "[Step 8] Running make test to verify new kernels..."
run_with_retry "make test" make test

echo ""
echo "============================================================"
echo " PreJIT kernel regeneration complete."
echo " ${NEW_COUNT} kernel(s) committed to deps/GraphBLAS/PreJIT/"
echo "============================================================"
