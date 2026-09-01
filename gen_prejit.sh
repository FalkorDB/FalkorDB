#!/usr/bin/env bash
# gen_prejit.sh – Regenerate GraphBLAS PreJIT kernels for falkordb-rs.
#
# What this does:
#   1. Clears the vendored PreJIT kernel sources (build/graphblas/PreJIT/*.c)
#      and the SuiteSparse JIT runtime cache (~/.SuiteSparse/GrBx.y.z/).
#   2. Rebuilds GraphBLAS + falkordb-rs with the JIT engine ENABLED:
#      - FALKORDB_PREJIT_HARVEST=1 tells the native-deps GraphBLAS recipe to skip PreJIT
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
#   * the deps/GraphBLAS submodule pin is bumped.
#
# HARMONIC HLL KERNELS: the LAGraph HyperLogLog dot4 semiring kernels
# (GB_jit__AxB_dot4__*__lg_hll_merge_lg_hll_second, plus lg_hll_count /
# lg_hll_delta) are only harvested because tests/flow/test_harmonic_centrality.py
# exercises algo.HarmonicCentrality on BOTH a tiny graph (A-bitmap `...eca`
# dot4 variant) and a larger sparse graph (test08 -> A-sparse `...ec6` variant,
# the one the benchmark hits). Two conditions are required for these to appear:
#   1. The harmonic iso fast path (algo_procedures.rs feeds LAGraph an ISO bool
#      adjacency via eWiseMult ONEB) must be present -- a non-iso adjacency
#      makes HyperBall punt to the ~3x slower generic dot2 and no dot4 kernel
#      is ever JIT-compiled.
#   2. The flow suite must run under --features prejit_harvest (Step 4e below).
# If harmonic centrality regresses on medium/large graphs, confirm those dot4
# kernels are still present in build/graphblas/PreJIT/ after a re-harvest.
#
# NOTE: kernels must be harvested ON LINUX, inside the Linux Docker toolchain
# image (ghcr.io/falkordb/falkordb-build) — the CI/production target. The JIT
# defn strings embedded in the kernels are captured AFTER host header macro
# expansion, so a macOS harvest can bake in Apple-specific expansions (e.g.
# fortify rewriting memcpy to __builtin___memcpy_chk) that fail the kernels'
# _query hash check on Linux — GraphBLAS then silently falls back to slow
# generic kernels. The harvested files become a checked-in artifact.

set -euo pipefail

# ---------------------------------------------------------------------------
# Locate the repo root.
# ---------------------------------------------------------------------------
REPO_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PREJIT_DIR="${REPO_DIR}/build/graphblas/PreJIT"

# ---------------------------------------------------------------------------
# Detect GraphBLAS version.
#
# Read it from the deps/GraphBLAS submodule, which is now the single source of
# truth for the pin (there is no graphblas.sh to grep any more). NOTE: v10.5.0
# renamed these fields from GraphBLAS_VERSION_* to GraphBLAS_VER_*, so accept
# either rather than silently failing on one side of that bump.
# ---------------------------------------------------------------------------
GB_VERSION_CMAKE="${REPO_DIR}/deps/GraphBLAS/cmake_modules/GraphBLAS_version.cmake"
if [[ -z "${GRAPHBLAS_VERSION:-}" ]]; then
    [[ -f "${GB_VERSION_CMAKE}" ]] || {
        echo "ERROR: ${GB_VERSION_CMAKE} not found." >&2
        echo "       Run: git submodule update --init --recursive deps/GraphBLAS" >&2
        exit 1
    }
    gb_field() {
        grep -E "^set \( GraphBLAS_(VER|VERSION)_$1 " "${GB_VERSION_CMAKE}" \
            | head -1 | grep -oE '[0-9]+' | head -1
    }
    GRAPHBLAS_VERSION="$(gb_field MAJOR).$(gb_field MINOR).$(gb_field SUB)"
fi
[[ "${GRAPHBLAS_VERSION}" =~ ^[0-9]+\.[0-9]+\.[0-9]+$ ]] || {
    echo "ERROR: could not detect GraphBLAS version (got '${GRAPHBLAS_VERSION}')" >&2
    exit 1
}

# ~/.SuiteSparse uses a single dotted directory: GrB<MAJOR>.<MINOR>.<SUB>.
SUITESPARSE_GRB="${HOME}/.SuiteSparse/GrB${GRAPHBLAS_VERSION}"


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

# JIT-compiled kernel .{so,dylib}s link their own copy of libomp; loading
# them next to the module's statically-wired libomp trips OpenMP's
# duplicate-runtime abort (__kmp_serial_initialize). Harmless for a harvest
# run, so tell libomp to tolerate it — without this the server crashes after
# compiling the first kernel and the cache stays nearly empty.
export KMP_DUPLICATE_LIB_OK=TRUE

# In the Linux Docker toolchain image only a static /opt/libomp/lib/libomp.a
# exists, and it's outside the default linker search path. The JIT compile
# line ends with `-fopenmp=libomp`, whose implicit `-lomp` then fails with
# "cannot find -lomp" — every runtime JIT compile errors out (GxB_JIT_ERROR),
# aborting each algorithm at its first uncached kernel so later kernels
# never get generated. Symlink the archive into the default path.
# Doctests are linked by rustdoc, and cargo ignores `rustdocflags` under the
# `[target.'cfg(target_os = "linux")']` table in .cargo/config.toml (unused-key
# warning), so the `--allow-multiple-definition` that the rest of the Linux
# link needs never reaches rustdoc. Without this the two doctests in
# parser/string_escape.rs fail to link and Step 4a is reported as a failure
# even though all 108 unit tests passed. Linux-only: the flag is GNU ld's and
# breaks macOS ld64.
# Appends rather than only setting when empty: a caller who exports
# RUSTDOCFLAGS for an unrelated reason would otherwise silently lose the
# linker flag and get the doctest failure back.
if [[ "$(uname -s)" == "Linux" ]]; then
    export RUSTDOCFLAGS="${RUSTDOCFLAGS:+${RUSTDOCFLAGS} }-C link-arg=-Wl,--allow-multiple-definition"
fi

if [[ "$(uname -s)" == "Linux" && -f /opt/libomp/lib/libomp.a && ! -e /usr/lib/libomp.a ]]; then
    if [[ -w /usr/lib ]]; then
        ln -s /opt/libomp/lib/libomp.a /usr/lib/libomp.a
        echo "Symlinked /opt/libomp/lib/libomp.a -> /usr/lib/libomp.a (JIT link needs -lomp)"
    else
        echo "WARNING: /usr/lib not writable; JIT kernel links may fail with 'cannot find -lomp'" >&2
    fi
fi

echo "============================================================"
echo " falkordb-rs PreJIT kernel regeneration"
echo "============================================================"
echo " Repo root     : ${REPO_DIR}"
echo " GraphBLAS     : v${GRAPHBLAS_VERSION}"
echo " Vendored dir  : ${PREJIT_DIR}"
echo " JIT cache     : ${SUITESPARSE_GRB}"
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
#            * --features prejit_harvest selects GxB_JIT_ON in matrix.rs
#              (full JIT — cache load AND compile-on-demand) at build time;
#              a shipped binary without the feature can never enable on-demand
#              compilation, even if the env var leaks into its process.
# ---------------------------------------------------------------------------
# FALKORDB_PREJIT_HARVEST=1 makes the native-deps GraphBLAS recipe skip
# vendoring PreJIT *.c, so the build starts from an empty PreJIT directory and
# whatever the JIT engine compiles at runtime is what we capture. It is also a
# cache-key input, so the harvest build gets its own entry and can never
# clobber the normal one.
export FALKORDB_PREJIT_HARVEST=1
# RELEASE=1 makes tests/common.py load target/release/libfalkordb.{so,dylib}
# instead of target/debug, so the JIT actually fires against the harvest build.
export RELEASE=1

# cargo build drives the GraphBLAS build itself (graph/build.rs ->
# native_deps::ensure()), so there is no separate dep step to run first.
echo "[Step 3] Rebuilding falkordb-rs release (--features prejit_harvest)..."
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
# Includes tests/flow/test_harmonic_centrality.py, whose tiny + 200-node graph
# shapes JIT-compile the LAGraph HLL dot4 kernels (see header note) into the
# cache so Step 5 can harvest them.
(cd "${REPO_DIR}" && run_with_retry "flow.sh" ./flow.sh)

# ---------------------------------------------------------------------------
# Step 4f – Run the benchmark query set (when present).
# ---------------------------------------------------------------------------
# bench/queries.py carries query shapes the functional suites never issue, so
# without this step their kernels are absent from the harvest and the
# benchmark silently runs on generic kernels — precisely the regression PreJIT
# exists to prevent. `--once` executes each query through redis-cli against
# the bench graph (plus the expected-error set, and with AOF on so the
# replication-effect paths run); unlike the measuring mode it needs neither
# redis-benchmark nor valgrind, neither of which the toolchain image ships.
# Guarded because bench/ is not on every branch.
run_bench_queries() {
    # bench/ used to be a single run_bench.py and is now a uv project exposing a
    # `bench` CLI, where `measure --once` is the same unmeasured coverage pass.
    # Try the current form first and fall back to the old script, so this works
    # whichever shape the branch has.
    if [[ -f "${REPO_DIR}/bench/pyproject.toml" ]] && command -v uv >/dev/null 2>&1; then
        (cd "${REPO_DIR}" && run_with_retry "bench measure --once" \
            uv run --project bench bench measure --once)
    elif [[ -f "${REPO_DIR}/bench/run_bench.py" ]]; then
        (cd "${REPO_DIR}" && run_with_retry "bench --once" \
            python3 bench/run_bench.py --once)
    else
        # Not a benign skip: the benchmark query set carries shapes no functional
        # suite issues, so without it those kernels are missing from the harvest
        # and the benchmark silently runs generic — the exact regression PreJIT
        # exists to prevent. Fail loudly into the summary rather than log a line
        # that scrolls past.
        echo "   WARNING: no runnable bench query set — need bench/pyproject.toml plus" >&2
        echo "   uv on PATH, or bench/run_bench.py. Benchmark-only kernels will be" >&2
        echo "   ABSENT from this harvest." >&2
        FAILURES+=("bench query set (no runner found — harvest is incomplete)")
    fi
}
echo "[Step 4f] Running benchmark query set (--once)..."
run_bench_queries

# ---------------------------------------------------------------------------
# Step 5 – Harvest .c kernel sources from the JIT cache.
# ---------------------------------------------------------------------------
echo "[Step 5] Harvesting JIT kernels..."
KERNEL_SRC_DIR="${SUITESPARSE_GRB}/c"
[[ -d "${KERNEL_SRC_DIR}" ]] || die "JIT kernel source dir not found: ${KERNEL_SRC_DIR}
Did the JIT actually run? Check FALKORDB_PREJIT_HARVEST is exported
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
# Keep the writable harvest prefix for the verify build too — installing to
# /usr/local would need sudo (interactive prompt on dev machines). Only the
# harvest-mode flag is dropped so the vendored kernels get baked in.
unset FALKORDB_PREJIT_HARVEST

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
echo "[Step 6d] Re-running benchmark query set to verify..."
run_bench_queries

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
