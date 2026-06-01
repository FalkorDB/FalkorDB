#!/usr/bin/env bash
# gen_prejit.sh – Regenerate GraphBLAS PreJIT kernels for falkordb-rs.
#
# What this does:
#   1. Clears the vendored PreJIT kernel sources (build/graphblas/PreJIT/*.c)
#      and the SuiteSparse JIT runtime cache (~/.SuiteSparse/GrBx.y.z/).
#   2. Rebuilds GraphBLAS + falkordb-rs with the JIT engine ENABLED:
#      - FALKORDB_PREJIT_HARVEST=1 tells graphblas.sh to skip PreJIT
#        vendoring so the build starts with an empty PreJIT directory.
#      - `cargo build --features prejit_harvest` selects GxB_JIT_ON in
#        matrix.rs instead of the default GxB_JIT_RUN. The runtime JIT
#        selection is a Cargo feature, not an env var, so a shipped
#        binary can never accidentally enable on-demand compilation.
#   3. Runs the full test suite so GraphBLAS JIT-compiles every kernel the
#      rust port exercises into ~/.SuiteSparse/GrBx.y.z/c/.
#   4. Harvests those .c kernel sources into build/graphblas/PreJIT/ so
#      the next regular build statically links them into libgraphblas.a.
#   5. Rebuilds normally (no FALKORDB_PREJIT_HARVEST, no
#      --features prejit_harvest) + re-runs the suite to verify the
#      harvested kernels load and serve every op shape.
#
# This is a MANUAL developer tool, not a CI step. Re-run when:
#   * The rust port starts exercising new GraphBLAS op shapes that aren't
#     in the vendored PreJIT set (symptom: GxB_JIT_ERROR / generic-only
#     slowdown on a new query pattern).
#   * GraphBLAS version is bumped in graphblas.sh.
#
# NOTE: PreJIT kernel sources are architecture-independent C; the harvest
# can run on any host. The harvested files become a checked-in artifact.

set -euo pipefail

# ---------------------------------------------------------------------------
# Locate the repo root.
# ---------------------------------------------------------------------------
REPO_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PREJIT_DIR="${REPO_DIR}/build/graphblas/PreJIT"

# ---------------------------------------------------------------------------
# Detect GraphBLAS version (we don't keep a checked-in GraphBLAS source tree,
# so derive it from graphblas.sh's pinned GRAPHBLAS_VERSION default).
# ---------------------------------------------------------------------------
GRAPHBLAS_VERSION="${GRAPHBLAS_VERSION:-$(grep -E '^GRAPHBLAS_VERSION=' "${REPO_DIR}/graphblas.sh" | head -1 | sed -E 's/.*[:-]v([0-9.]+).*/\1/')}"
[[ -n "${GRAPHBLAS_VERSION}" ]] || { echo "ERROR: could not detect GraphBLAS version" >&2; exit 1; }

# ~/.SuiteSparse uses a single dotted directory: GrB<MAJOR>.<MINOR>.<SUB>.
SUITESPARSE_GRB="${HOME}/.SuiteSparse/GrB${GRAPHBLAS_VERSION}"

# Use a writable, out-of-tree install prefix so we don't need sudo for the
# harvest build. graphblas.sh CMAKE_INSTALL_PREFIX + build.rs's
# GRAPHBLAS_LIB_DIR keep both halves of the build in sync.
HARVEST_PREFIX="${HARVEST_PREFIX:-${REPO_DIR}/.graphblas-harvest}"

# On macOS, the system /usr/bin/clang doesn't ship OpenMP support. Point
# CC/CXX at homebrew LLVM if not already set so the GraphBLAS + LAGraph
# cmake invocations find_package(OpenMP) successfully.
if [[ "$(uname -s)" == "Darwin" ]] && [[ -z "${CC:-}" ]]; then
    if command -v brew >/dev/null 2>&1; then
        BREW_LLVM="$(brew --prefix llvm 2>/dev/null || true)"
        if [[ -n "${BREW_LLVM}" && -x "${BREW_LLVM}/bin/clang" ]]; then
            export CC="${BREW_LLVM}/bin/clang"
            export CXX="${BREW_LLVM}/bin/clang++"
            echo "Auto-detected homebrew LLVM: CC=${CC}"
        fi
    fi
fi

echo "============================================================"
echo " falkordb-rs PreJIT kernel regeneration"
echo "============================================================"
echo " Repo root     : ${REPO_DIR}"
echo " GraphBLAS     : v${GRAPHBLAS_VERSION}"
echo " Vendored dir  : ${PREJIT_DIR}"
echo " JIT cache     : ${SUITESPARSE_GRB}"
echo " Harvest prefix: ${HARVEST_PREFIX}"
echo "============================================================"
echo ""

die() { echo "ERROR: $*" >&2; exit 1; }

# run_with_retry <label> <command...>
#   Runs the command. On failure:
#     * If GEN_PREJIT_NONINTERACTIVE=1, log it and continue (the JIT cache
#       gets populated by whatever ops did run, regardless of test pass/
#       fail; failing tests in the harvest are tracked and surfaced at
#       the end but don't block kernel collection).
#     * Otherwise prompt: a)continue b)retry c)stop.
FAILURES=()
run_with_retry() {
    local label="$1"; shift
    while true; do
        if "$@"; then return 0; fi
        echo ""
        echo "------------------------------------------------------------"
        echo " FAILURE: ${label}"
        echo "------------------------------------------------------------"
        if [[ "${GEN_PREJIT_NONINTERACTIVE:-0}" == "1" ]]; then
            FAILURES+=("${label}")
            echo " GEN_PREJIT_NONINTERACTIVE=1 — continuing (failure logged)."
            return 0
        fi
        echo " Choose an action:"
        echo "   a) Continue  – ignore this failure and proceed"
        echo "   b) Retry     – re-run this step"
        echo "   c) Stop      – abort the script"
        while true; do
            read -rp " Your choice [a/b/c]: " choice
            case "${choice,,}" in
                a) echo " Continuing..."; FAILURES+=("${label}"); return 0 ;;
                b) echo " Retrying ${label}..."; break ;;
                c) echo " Stopping."; exit 1 ;;
                *) echo " Please enter a, b, or c." ;;
            esac
        done
    done
}

# Some callers may not have venv on PATH; activate if present.
if [[ -d "${REPO_DIR}/venv" && -z "${VIRTUAL_ENV:-}" ]]; then
    # shellcheck disable=SC1091
    source "${REPO_DIR}/venv/bin/activate"
fi

# ---------------------------------------------------------------------------
# Step 1 – Clear vendored PreJIT sources.
# ---------------------------------------------------------------------------
echo "[Step 1] Clearing vendored PreJIT kernel sources (${PREJIT_DIR})..."
mkdir -p "${PREJIT_DIR}"
find "${PREJIT_DIR}" -maxdepth 1 -name 'GB*.c' -delete
echo "   Done."

# ---------------------------------------------------------------------------
# Step 2 – Clear GraphBLAS JIT runtime cache.
# ---------------------------------------------------------------------------
echo "[Step 2] Clearing GraphBLAS JIT runtime cache (${SUITESPARSE_GRB})..."
[[ -n "${SUITESPARSE_GRB}" ]] || die "SUITESPARSE_GRB empty – refusing to rm -rf"
if [[ -d "${SUITESPARSE_GRB}" ]]; then
    rm -rf "${SUITESPARSE_GRB}/tmp" "${SUITESPARSE_GRB}/c" "${SUITESPARSE_GRB}/lib"
    echo "   Done."
else
    echo "   Cache directory does not exist yet – nothing to clear."
fi

# ---------------------------------------------------------------------------
# Step 3 – Rebuild GraphBLAS + falkordb-rs in HARVEST mode.
#            * FALKORDB_PREJIT_HARVEST=1 tells graphblas.sh to skip vendoring
#              PreJIT *.c (so the GraphBLAS build starts with an empty PreJIT
#              directory; whatever the JIT engine compiles at runtime is what
#              we'll capture). graphblas.sh runs outside cargo, so it stays
#              env-var driven.
#            * --features prejit_harvest selects GxB_JIT_ON in matrix.rs
#              (full JIT — cache load AND compile-on-demand) at build time;
#              a shipped binary without the feature can never enable on-demand
#              compilation, even if the env var leaks into its process.
# ---------------------------------------------------------------------------
echo "[Step 3a] Rebuilding GraphBLAS (HARVEST mode → ${HARVEST_PREFIX})..."
export FALKORDB_PREJIT_HARVEST=1
export GRAPHBLAS_INSTALL_PREFIX="${HARVEST_PREFIX}"
export GRAPHBLAS_LIB_DIR="${HARVEST_PREFIX}/lib"
# RELEASE=1 makes tests/common.py load target/release/libfalkordb.{so,dylib}
# instead of target/debug. Without this, pytest would load a stale debug
# .dylib (built against /usr/local/libgraphblas.a, which has the old
# C-derived PreJIT baked in) and the JIT would rarely fire → empty cache.
export RELEASE=1
mkdir -p "${HARVEST_PREFIX}"
run_with_retry "graphblas.sh" "${REPO_DIR}/graphblas.sh"

echo "[Step 3b] Rebuilding falkordb-rs release (--features prejit_harvest)..."
(cd "${REPO_DIR}" && run_with_retry "cargo build" cargo build --release --features prejit_harvest)

# ---------------------------------------------------------------------------
# Step 4 – Run the full test suite to populate the JIT cache.
# ---------------------------------------------------------------------------
echo "[Step 4a] Running cargo tests (graph crate, --features prejit_harvest)..."
(cd "${REPO_DIR}" && run_with_retry "cargo test -p graph" cargo test -p graph --release --features prejit_harvest)

echo "[Step 4b] Running pytest e2e + functions..."
(cd "${REPO_DIR}" && run_with_retry "pytest e2e+functions" \
    pytest tests/test_e2e.py tests/test_functions.py -vv)

echo "[Step 4c] Running pytest MVCC + concurrency..."
(cd "${REPO_DIR}" && run_with_retry "pytest mvcc+concurrency" \
    pytest tests/test_mvcc.py tests/test_concurrency.py -vv)

echo "[Step 4d] Running TCK suite..."
(cd "${REPO_DIR}" && run_with_retry "pytest tck" \
    env TCK_DONE=tck_done.txt pytest tests/tck/test_tck.py -s)

echo "[Step 4e] Running flow tests..."
(cd "${REPO_DIR}" && run_with_retry "flow.sh" ./flow.sh)

# ---------------------------------------------------------------------------
# Step 5 – Harvest .c kernel sources from the JIT cache.
# ---------------------------------------------------------------------------
echo "[Step 5] Harvesting JIT kernels..."
KERNEL_SRC_DIR="${SUITESPARSE_GRB}/c"
[[ -d "${KERNEL_SRC_DIR}" ]] || die "JIT kernel source dir not found: ${KERNEL_SRC_DIR}
Did the JIT actually run? Check FALKORDB_PREJIT_HARVEST is exported (for graphblas.sh)
and the cargo build was invoked with --features prejit_harvest (for matrix.rs)."

KERNEL_COUNT=$(find "${KERNEL_SRC_DIR}" -name 'GB*.c' | wc -l | tr -d ' ')
echo "   Source : ${KERNEL_SRC_DIR}"
echo "   Dest   : ${PREJIT_DIR}"
echo "   Kernels found: ${KERNEL_COUNT}"
[[ "${KERNEL_COUNT}" -gt 0 ]] || die "No GB*.c kernels found in ${KERNEL_SRC_DIR}"

find "${KERNEL_SRC_DIR}" -name 'GB*.c' -exec cp {} "${PREJIT_DIR}/" \;

NEW_COUNT=$(find "${PREJIT_DIR}" -maxdepth 1 -name 'GB*.c' | wc -l | tr -d ' ')
echo "   Harvested ${NEW_COUNT} kernel(s) into ${PREJIT_DIR}"

# ---------------------------------------------------------------------------
# Step 6 – Rebuild normally (default GxB_JIT_RUN runtime, no --features
#          prejit_harvest) and re-run the suite to verify.
# ---------------------------------------------------------------------------
echo "[Step 6a] Rebuilding GraphBLAS with vendored PreJIT kernels..."
unset FALKORDB_PREJIT_HARVEST
unset GRAPHBLAS_INSTALL_PREFIX
unset GRAPHBLAS_LIB_DIR
run_with_retry "graphblas.sh (normal)" "${REPO_DIR}/graphblas.sh"

echo "[Step 6b] Rebuilding falkordb-rs release (normal)..."
(cd "${REPO_DIR}" && run_with_retry "cargo build" cargo build --release)

echo "[Step 6c] Re-running full test suite to verify..."
(cd "${REPO_DIR}" && run_with_retry "cargo test -p graph" cargo test -p graph --release)
(cd "${REPO_DIR}" && run_with_retry "pytest e2e+functions" \
    pytest tests/test_e2e.py tests/test_functions.py -vv)
(cd "${REPO_DIR}" && run_with_retry "pytest mvcc+concurrency" \
    pytest tests/test_mvcc.py tests/test_concurrency.py -vv)
(cd "${REPO_DIR}" && run_with_retry "pytest tck" \
    env TCK_DONE=tck_done.txt pytest tests/tck/test_tck.py -s)
(cd "${REPO_DIR}" && run_with_retry "flow.sh" ./flow.sh)

echo ""
echo "============================================================"
echo " PreJIT kernel regeneration complete."
echo " ${NEW_COUNT} kernel(s) in build/graphblas/PreJIT/"
if [[ "${#FAILURES[@]}" -gt 0 ]]; then
    echo ""
    echo " The following steps failed during the run (kernels may still"
    echo " have been collected from steps that did run before the failure):"
    for f in "${FAILURES[@]}"; do echo "   * ${f}"; done
fi
echo " Commit them and PR."
echo "============================================================"
