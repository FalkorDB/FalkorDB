#!/usr/bin/env bash
set -e

#-----------------------------------------------------------------------------
# FalkorDB Build Script
#
# This script handles building the FalkorDB module and running tests.
# It supports various build configurations and test types.
#-----------------------------------------------------------------------------

# Get the absolute path to script directory
ROOT="$(cd "$(dirname "$0")" && pwd)"
BINROOT="$ROOT/bin"

#-----------------------------------------------------------------------------
# Default configuration values
#-----------------------------------------------------------------------------
DEBUG=0              # Debug build flag
COV=0                # Coverage mode
PROFILE=0            # Profile build flag
FORCE=0              # Force clean build flag
VERBOSE=0            # Verbose output flag
SLOW=0               # Disable parallel build
STATIC_OMP=0         # Link OpenMP statically
VG=0                 # Valgrind mode
SAN=""               # Sanitizer type (address/memory/leak/thread)
VARIANT=""           # Custom variant name
GCC=0                # Force GCC toolchain
CLANG=0              # Force Clang toolchain
JIT=1                # GraphBLAS JIT support (default on)

# Test configuration (0=disabled, 1=enabled)
BUILD_TESTS=0        # Build test binaries
RUN_TESTS=0          # Run all tests
RUN_UNIT_TESTS=0     # Run unit tests
RUN_FLOW_TESTS=0     # Run flow tests
RUN_TCK_TESTS=0      # Run TCK tests
RUN_UPGRADE_TESTS=0  # Run upgrade tests

# Other options
LIST_TESTS=0         # List tests only
TEST_FILTER=""       # Specific test to run
TESTFILE=""          # File with list of tests
FAILFILE=""          # File to write failed tests
PARALLEL=""          # Test parallelism
GDB=0                # Run with GDB

# Build flags
MEMCHECK=0           # Memory checking mode
DEPS_DEBUG=0         # Build deps in debug mode

# Clean options
CLEAN=0              # Run clean operation
CLEAN_ALL=0          # Remove entire bin directory
CLEAN_DEPS=0         # Also clean dependencies
CLEAN_AUTOGEN=0      # Remove autogen files

# Package options
PACK=0               # Build RAMP packages

# Fuzz test options
RUN_FUZZ_TESTS=0     # Run fuzz tests
FUZZ_TIMEOUT=30      # Fuzz test timeout in seconds

# Benchmark options
BENCHMARK=0          # Run benchmarks
BENCHMARK_GROUP=""   # Benchmark group to run (group_a, group_b, or empty for all)

# Run options
RUN=0                # Run redis-server with the module

#-----------------------------------------------------------------------------
# Color definitions for output
#-----------------------------------------------------------------------------
if [[ -t 1 ]]; then
    RED='\033[0;31m'
    GREEN='\033[0;32m'
    YELLOW='\033[1;33m'
    BLUE='\033[0;34m'
    CYAN='\033[0;36m'
    NC='\033[0m' # No Color
else
    RED=''
    GREEN=''
    YELLOW=''
    BLUE=''
    CYAN=''
    NC=''
fi

#-----------------------------------------------------------------------------
# Helper functions
#-----------------------------------------------------------------------------

log_info() {
    echo -e "${CYAN}[INFO]${NC} $*"
}

log_success() {
    echo -e "${GREEN}[SUCCESS]${NC} $*"
}

log_error() {
    echo -e "${RED}[ERROR]${NC} $*" >&2
}

log_warn() {
    echo -e "${YELLOW}[WARN]${NC} $*"
}

start_group() {
    if [[ -n $GITHUB_ACTIONS ]]; then
        echo "::group::$1"
    else
        log_info "==== $1 ===="
    fi
}

end_group() {
    if [[ -n $GITHUB_ACTIONS ]]; then
        echo "::endgroup::"
    fi
}

#-----------------------------------------------------------------------------
# Function: show_help
# Display help information
#-----------------------------------------------------------------------------
show_help() {
    cat <<-'EOF'
FalkorDB Build Script

USAGE:
    ./build.sh [OPTIONS]

BUILD OPTIONS:
    DEBUG=1             Build for debugging
    PROFILE=1           Build with profiling enabled
    COV=1               Build for coverage analysis (implies DEBUG=1)
    VG=1|docker         Build for Valgrind
    SAN=type            Build with LLVM sanitizer (type=address|memory|leak|thread)
    FORCE=1             Force clean build
    SLOW=1              Disable parallel build
    STATIC_OMP=1        Link OpenMP statically
    VARIANT=name        Add 'name' to build products directory
    GCC=1               Build with GCC toolchain (default for Linux)
    CLANG=1             Build with Clang toolchain (default for macOS)
    JIT=0               Disable GraphBLAS JIT support

TEST OPTIONS:
    TESTS=1             Build test binaries
    RUN_TESTS=1         Run all tests
    RUN_UNIT_TESTS=1    Run unit tests only
    RUN_FLOW_TESTS=1    Run flow tests only
    RUN_TCK_TESTS=1     Run TCK tests only
    RUN_UPGRADE_TESTS=1 Run upgrade tests only
    LIST=1              List all tests, do not execute
    TEST=name           Run specific test
    TESTFILE=file       Run tests listed in file
    FAILFILE=file       Write failed tests to file
    PARALLEL=n          Set testing parallelism
    GDB=1               Run with GDB debugger

    RUN_FUZZ_TESTS=1    Run fuzz tests
    TIMEOUT=secs        Fuzz test timeout in seconds (default: 30)

BENCHMARK OPTIONS:
    BENCHMARK=1         Run benchmarks
    BENCHMARK_GROUP=x   Run specific benchmark group (group_a, group_b)

RUN OPTIONS:
    RUN=1               Run redis-server with the FalkorDB module loaded
    GDB=1               Run with GDB debugger (use with RUN=1)

CLEAN OPTIONS:
    CLEAN=1             Clean build products
    ALL=1               Remove entire bin directory (with CLEAN=1)
    DEPS=1              Also clean dependencies (with CLEAN=1)
    AUTOGEN=1           Remove autogen files (with CLEAN=1)

PACKAGE OPTIONS:
    PACK=1              Build RAMP packages

OTHER OPTIONS:
    VERBOSE=1           Show verbose output
    HELP=1              Show this help message

EXAMPLES:
    # Build in debug mode
    ./build.sh DEBUG=1

    # Build and run all tests
    ./build.sh RUN_TESTS=1

    # Build with address sanitizer and run flow tests
    ./build.sh SAN=address RUN_FLOW_TESTS=1

    # Build for coverage and run tests
    ./build.sh COV=1 RUN_TESTS=1

    # Clean build products
    ./build.sh CLEAN=1

    # Full clean (remove entire bin directory)
    ./build.sh CLEAN=1 ALL=1

    # Build RAMP packages
    ./build.sh PACK=1

    # Run fuzz tests with 60 second timeout
    ./build.sh RUN_FUZZ_TESTS=1 TIMEOUT=60

EOF
}

#-----------------------------------------------------------------------------
# Function: parse_arguments
# Parse command-line arguments and set configuration variables
#-----------------------------------------------------------------------------
parse_arguments() {
    for arg in "$@"; do
        # Convert to uppercase for case-insensitive matching
        upper_arg=$(echo "$arg" | tr '[:lower:]' '[:upper:]')

        case $upper_arg in
            HELP=1|--HELP|HELP)
                show_help
                exit 0
                ;;
            DEBUG=1)
                DEBUG=1
                ;;
            PROFILE=1)
                PROFILE=1
                ;;
            COV=1)
                COV=1
                DEBUG=1  # Coverage implies debug
                ;;
            VG=1)
                VG=1
                MEMCHECK=1
                ;;
            SAN=*)
                SAN="${arg#*=}"
                MEMCHECK=1
                ;;
            FORCE=1)
                FORCE=1
                ;;
            VERBOSE=1)
                VERBOSE=1
                ;;
            SLOW=1)
                SLOW=1
                ;;
            STATIC_OMP=1)
                STATIC_OMP=1
                ;;
            VARIANT=*)
                VARIANT="${arg#*=}"
                ;;
            GCC=1)
                GCC=1
                ;;
            CLANG=1)
                CLANG=1
                ;;
            JIT=*)
                JIT="${arg#*=}"
                ;;
            TESTS=1|UNIT_TESTS=1)
                BUILD_TESTS=1
                ;;
            RUN_TESTS=1)
                RUN_TESTS=1
                BUILD_TESTS=1
                ;;
            RUN_UNIT_TESTS=1)
                RUN_UNIT_TESTS=1
                BUILD_TESTS=1
                ;;
            RUN_FLOW_TESTS=1|FLOW=1)
                RUN_FLOW_TESTS=1
                ;;
            RUN_TCK_TESTS=1|TCK=1)
                RUN_TCK_TESTS=1
                ;;
            RUN_UPGRADE_TESTS=1|UPGRADE=1)
                RUN_UPGRADE_TESTS=1
                ;;
            LIST=1)
                LIST_TESTS=1
                ;;
            TEST=*)
                TEST_FILTER="${arg#*=}"
                ;;
            TESTFILE=*)
                TESTFILE="${arg#*=}"
                ;;
            FAILFILE=*)
                FAILFILE="${arg#*=}"
                ;;
            PARALLEL=*)
                PARALLEL="${arg#*=}"
                ;;
            GDB=1)
                GDB=1
                ;;
            # Clean options
            CLEAN=1)
                CLEAN=1
                ;;
            ALL=1)
                CLEAN_ALL=1
                ;;
            DEPS=1)
                CLEAN_DEPS=1
                ;;
            AUTOGEN=1)
                CLEAN_AUTOGEN=1
                ;;
            # Package options
            PACK=1|PACKAGE=1)
                PACK=1
                ;;
            # Fuzz test options
            RUN_FUZZ_TESTS=1|FUZZ=1)
                RUN_FUZZ_TESTS=1
                ;;
            TIMEOUT=*)
                FUZZ_TIMEOUT="${arg#*=}"
                ;;
            # Benchmark options
            BENCHMARK=1)
                BENCHMARK=1
                ;;
            BENCHMARK_GROUP=*)
                BENCHMARK_GROUP="${arg#*=}"
                ;;
            # Run options
            RUN=1)
                RUN=1
                ;;
            *)
                log_warn "Unknown argument: $arg"
                ;;
        esac
    done
}

#-----------------------------------------------------------------------------
# Function: detect_platform
# Detect operating system and architecture
#-----------------------------------------------------------------------------
detect_platform() {
    # Detect OS
    OS_NAME=$(uname -s)
    case "$OS_NAME" in
        Linux*)
            OS="linux"
            ;;
        Darwin*)
            OS="macos"
            ;;
        *)
            log_error "Unsupported OS: $OS_NAME"
            exit 1
            ;;
    esac

    # Detect architecture
    ARCH=$(uname -m)
    case "$ARCH" in
        x86_64)
            ARCH="x64"
            ;;
        aarch64|arm64)
            ARCH="arm64v8"
            ;;
        *)
            log_error "Unsupported architecture: $ARCH"
            exit 1
            ;;
    esac

    # Detect OS nickname (simplified version)
    if [[ "$OS" == "linux" ]]; then
        if [[ -f /etc/os-release ]]; then
            . /etc/os-release
            OSNICK=$(echo "$ID" | tr '[:upper:]' '[:lower:]')
            if [[ -n "$VERSION_ID" ]]; then
                OSNICK="${OSNICK}${VERSION_ID}"
            fi
        else
            OSNICK="linux"
        fi
    elif [[ "$OS" == "macos" ]]; then
        OSNICK="macos"
    fi

    # Detect musl libc (for Alpine and similar)
    USING_MUSL=0
    if command -v ldd &>/dev/null; then
        if ldd 2>&1 | head -1 | grep -qi musl; then
            USING_MUSL=1
        fi
    fi
    if [[ "$OSNICK" =~ alpine ]]; then
        USING_MUSL=1
    fi

    if [[ "$VERBOSE" == "1" ]]; then
        log_info "Platform: OS=$OS, OSNICK=$OSNICK, ARCH=$ARCH"
        log_info "Using musl: $USING_MUSL"
    fi
}

#-----------------------------------------------------------------------------
# Function: setup_build_environment
# Configure the build environment variables
#-----------------------------------------------------------------------------
setup_build_environment() {
    # Determine build flavor
    if [[ "$SAN" == "address" ]]; then
        FLAVOR="debug-asan"
    elif [[ -n "$SAN" ]]; then
        FLAVOR="debug-${SAN}"
    elif [[ "$DEBUG" == "1" ]]; then
        FLAVOR="debug"
    elif [[ "$COV" == "1" ]]; then
        FLAVOR="debug-cov"
    elif [[ "$PROFILE" == "1" ]]; then
        FLAVOR="release-profile"
    else
        FLAVOR="release"
    fi

    # Add custom variant if specified
    if [[ -n "$VARIANT" ]]; then
        FLAVOR="${FLAVOR}.${VARIANT}"
    fi

    # Create full variant string
    FULL_VARIANT="${OS}-${ARCH}-${FLAVOR}"

    # For dependencies, use release unless MEMCHECK is set
    if [[ "$MEMCHECK" == "1" ]]; then
        DEPS_FULL_VARIANT="$FULL_VARIANT"
        DEPS_DEBUG=1
    else
        DEPS_FULL_VARIANT="${OS}-${ARCH}-release"
        DEPS_DEBUG=0
    fi

    # Set build directories
    BINROOT="${ROOT}/bin/${FULL_VARIANT}"
    DEPS_BINDIR="${ROOT}/bin/${DEPS_FULL_VARIANT}"
    TARGET="${BINROOT}/falkordb.so"

    # Set up dependency directories
    export RAX_BINDIR="${DEPS_BINDIR}/rax"
    export LIBXXHASH_BINDIR="${DEPS_BINDIR}/xxHash"
    export LIBCURL_BINDIR="${DEPS_BINDIR}/libcurl"
    export LIBCSV_BINDIR="${DEPS_BINDIR}/libcsv"
    export LIBCYPHER_PARSER_BINDIR="${DEPS_BINDIR}/libcypher-parser"
    export GRAPHBLAS_BINDIR="${DEPS_BINDIR}/GraphBLAS"
    export LAGRAPH_BINDIR="${DEPS_BINDIR}/LAGraph"
    export QUICKJS_BINDIR="${DEPS_BINDIR}/quickjs"
    export UTF8PROC_BINDIR="${DEPS_BINDIR}/utf8proc"
    export ONIGURUMA_BINDIR="${DEPS_BINDIR}/oniguruma"
    export FalkorDBRS_BINDIR="${BINROOT}/FalkorDB-core-rs"

    # Export environment variables for dependencies
    export RAX="${RAX_BINDIR}/librax.a"
    export LIBXXHASH="${LIBXXHASH_BINDIR}/libxxhash.a"
    export LIBCURL="${LIBCURL_BINDIR}/lib/.libs/libcurl.a"
    export LIBCSV="${LIBCSV_BINDIR}/.libs/libcsv.a"
    export LIBCYPHER_PARSER="${LIBCYPHER_PARSER_BINDIR}/lib/src/.libs/libcypher-parser.a"
    export GRAPHBLAS="${GRAPHBLAS_BINDIR}/libgraphblas.a"
    export LAGRAPH="${LAGRAPH_BINDIR}/src/liblagraph.a"
    export LAGRAPHX="${LAGRAPH_BINDIR}/experimental/liblagraphx.a"
    export QUICKJS="${QUICKJS_BINDIR}/libquickjs.a"
    export UTF8PROC="${UTF8PROC_BINDIR}/libutf8proc.a"
    export ONIGURUMA="${ONIGURUMA_BINDIR}/libonig.a"

    # FalkorDB Rust
    # Note: When using sanitizer with nightly Rust, cargo builds with --target <triple>
    # which changes the output path structure to include the target triple
    if [[ -n "$SAN" ]] && rustup run nightly rustc --version &>/dev/null 2>&1; then
        # Sanitizer build with nightly uses explicit target - determine the correct triple
        local rust_target
        if [[ "$ARCH" == "arm64v8" ]]; then
            rust_target="aarch64-unknown-linux-gnu"
        else
            rust_target="x86_64-unknown-linux-gnu"
        fi
        export FalkorDBRS="${FalkorDBRS_BINDIR}/${rust_target}/debug/libFalkorDB_rs.a"
    elif [[ "$DEBUG" == "1" || -n "$SAN" || "$COV" == "1" ]]; then
        # Debug, coverage, or sanitizer fallback (without nightly)
        export FalkorDBRS="${FalkorDBRS_BINDIR}/debug/libFalkorDB_rs.a"
    else
        export FalkorDBRS="${FalkorDBRS_BINDIR}/release/libFalkorDB_rs.a"
    fi

    # Setup compiler flags
    if [[ "$OS" == "macos" ]]; then
        export CC="${CC:-clang}"
        export CXX="${CXX:-clang++}"

        # Set macOS deployment target if not already set
        if [[ -z "$CMAKE_OSX_DEPLOYMENT_TARGET" ]]; then
            # Use reasonable minimum deployment target
            # arm64 (Apple Silicon) requires macOS 12.0+
            # x64 (Intel) can use macOS 10.15+
            if [[ "$ARCH" == "arm64v8" ]]; then
                export CMAKE_OSX_DEPLOYMENT_TARGET="12.0"
            else
                export CMAKE_OSX_DEPLOYMENT_TARGET="10.15"
            fi
        fi

        # Also set MACOSX_DEPLOYMENT_TARGET for compatibility
        export MACOSX_DEPLOYMENT_TARGET="${CMAKE_OSX_DEPLOYMENT_TARGET}"
    elif [[ "$GCC" == "1" ]]; then
        export CC="${CC:-gcc}"
        export CXX="${CXX:-g++}"
    elif [[ "$CLANG" == "1" ]]; then
        export CC="${CC:-clang}"
        export CXX="${CXX:-clang++}"
    fi

    # Export CMake flags for RediSearch compatibility
    # RediSearch's readies expects these environment variables
    # Set to space to avoid string REPLACE errors when empty
    export CMAKE_CC_FLAGS="${CMAKE_CC_FLAGS:- }"
    # Disable VLA warnings and GNU extensions completely for RediSearch macro compatibility
    export CMAKE_CC_C_FLAGS="${CMAKE_CC_C_FLAGS:- -Wno-gnu-zero-variadic-macro-arguments}"
    # Note: -Wno-vla-extension is Clang-specific, GCC doesn't recognize it
    # Using -Wno-vla-extension instead of -Wno-vla-cxx-extension for compatibility with Clang < 18
    if [[ "$CLANG" == "1" ]] || [[ "${CC:-}" == *"clang"* ]]; then
        export CMAKE_CC_CXX_FLAGS="${CMAKE_CC_CXX_FLAGS:- -Wno-vla-extension -Wno-gnu-zero-variadic-macro-arguments}"
    else
        export CMAKE_CC_CXX_FLAGS="${CMAKE_CC_CXX_FLAGS:- -Wno-gnu-zero-variadic-macro-arguments}"
    fi
    export CMAKE_LD_FLAGS="${CMAKE_LD_FLAGS:- }"
    export CMAKE_SO_LD_FLAGS="${CMAKE_SO_LD_FLAGS:- }"
    export CMAKE_EXE_LD_FLAGS="${CMAKE_EXE_LD_FLAGS:- }"
    export CMAKE_LD_FLAGS_LIST="${CMAKE_LD_FLAGS_LIST:- }"
    export CMAKE_SO_LD_FLAGS_LIST="${CMAKE_SO_LD_FLAGS_LIST:- }"
    export CMAKE_EXE_LD_FLAGS_LIST="${CMAKE_EXE_LD_FLAGS_LIST:- }"
    export CMAKE_LD_LIBS="${CMAKE_LD_LIBS:- }"

    # Determine number of parallel jobs (needed for dependency builds)
    if [[ "$SLOW" == "1" ]]; then
        NPROC=1
    elif command -v nproc &> /dev/null; then
        NPROC=$(nproc)
    elif command -v sysctl &> /dev/null && [[ "$OS" == "macos" ]]; then
        NPROC=$(sysctl -n hw.physicalcpu)
    else
        NPROC=4
    fi
    export NPROC

    # Print build configuration
    echo ""
    log_info "Build Configuration:"
    log_info "  OS:           $OS                    # Operating system (linux/macos)"
    log_info "  ARCH:         $ARCH                  # CPU architecture (x64/arm64v8)"
    log_info "  FLAVOR:       $FLAVOR                # Build type (release/debug/debug-asan/etc)"
    log_info "  VARIANT:      $FULL_VARIANT          # Full build variant identifier"
    log_info "  BINROOT:      $BINROOT       # Build output directory"
    log_info "  TARGET:       $TARGET        # Output module path"
    log_info "  CC:           ${CC:-cc}                   # C compiler"
    log_info "  CXX:          ${CXX:-c++}                  # C++ compiler"
    log_info "  NPROC:        $NPROC                     # Parallel build jobs"
    if [[ "$OS" == "macos" ]]; then
        log_info "  MACOSX_TARGET: $MACOSX_DEPLOYMENT_TARGET              # macOS deployment target"
    fi
    if [[ -n "$SAN" ]]; then
        log_info "  SANITIZER:    $SAN                # LLVM sanitizer enabled"
    fi
    if [[ "$DEBUG" == "1" ]]; then
        log_info "  DEBUG:        enabled              # Debug build"
    fi
    if [[ "$COV" == "1" ]]; then
        log_info "  COVERAGE:     enabled              # Code coverage instrumentation"
    fi
    if [[ "$VG" == "1" ]]; then
        log_info "  VALGRIND:     enabled              # Valgrind-compatible build"
    fi
    echo ""
}

#-----------------------------------------------------------------------------
# Function: check_dependencies
# Check if dependencies need to be built
#-----------------------------------------------------------------------------
check_dependencies() {
    MISSING_DEPS=()

    [[ ! -f "$RAX" ]] && MISSING_DEPS+=("rax")
    [[ ! -f "$LIBXXHASH" ]] && MISSING_DEPS+=("xxhash")
    [[ ! -f "$LIBCURL" ]] && MISSING_DEPS+=("libcurl")
    [[ ! -f "$LIBCSV" ]] && MISSING_DEPS+=("libcsv")
    [[ ! -f "$LIBCYPHER_PARSER" ]] && MISSING_DEPS+=("libcypher-parser")
    [[ ! -f "$GRAPHBLAS" ]] && MISSING_DEPS+=("graphblas")
    [[ ! -f "$LAGRAPH" ]] && MISSING_DEPS+=("lagraph")
    [[ ! -f "$QUICKJS" ]] && MISSING_DEPS+=("quickjs")
    [[ ! -f "$UTF8PROC" ]] && MISSING_DEPS+=("utf8proc")
    [[ ! -f "$ONIGURUMA" ]] && MISSING_DEPS+=("oniguruma")
    # RediSearch is now built by CMake as a subdirectory
    [[ ! -f "$FalkorDBRS" ]] && MISSING_DEPS+=("falkordbrs")

    if [[ ${#MISSING_DEPS[@]} -gt 0 ]]; then
        log_info "Missing dependencies: ${MISSING_DEPS[*]}"
        return 1
    fi
    return 0
}

#-----------------------------------------------------------------------------
# Function: build_rax
# Build rax library natively (without readies wrapper)
#-----------------------------------------------------------------------------
build_rax() {
    if [[ -f "$RAX" ]] && [[ "$FORCE" != "1" ]]; then
        return 0
    fi

    start_group "Building rax"
    log_info "Building rax..."

    local src_dir="${ROOT}/deps/rax"
    local bin_dir="$RAX_BINDIR"
    mkdir -p "$bin_dir"

    local cc_flags="-fPIC -std=gnu99 -fvisibility=hidden -I${src_dir}"
    if [[ "$DEPS_DEBUG" == "1" ]]; then
        cc_flags+=" -g -O0"
    else
        cc_flags+=" -O3"
    fi

    # Compile rax.c
    ${CC:-cc} $cc_flags -c "$src_dir/rax.c" -o "$bin_dir/rax.o"

    # Create static library
    ar rcs "$bin_dir/librax.a" "$bin_dir/rax.o"

    log_success "rax built successfully"
    end_group
}

#-----------------------------------------------------------------------------
# Function: build_xxhash
# Build xxHash library natively (without readies wrapper)
#-----------------------------------------------------------------------------
build_xxhash() {
    if [[ -f "$LIBXXHASH" ]] && [[ "$FORCE" != "1" ]]; then
        return 0
    fi

    start_group "Building xxHash"
    log_info "Building xxHash..."

    local src_dir="${ROOT}/deps/xxHash"
    local bin_dir="$LIBXXHASH_BINDIR"
    mkdir -p "$bin_dir"

    local cc_flags="-O3 -fPIC"

    # Compile xxhash.c
    ${CC:-cc} $cc_flags -c "$src_dir/xxhash.c" -o "$bin_dir/xxhash.o"

    # Create static library
    ar rcs "$bin_dir/libxxhash.a" "$bin_dir/xxhash.o"

    log_success "xxHash built successfully"
    end_group
}

#-----------------------------------------------------------------------------
# Function: build_libcypher_parser
# Build libcypher-parser using autotools (configure/make) out-of-source
#-----------------------------------------------------------------------------
build_libcypher_parser() {
    if [[ -f "$LIBCYPHER_PARSER" ]] && [[ "$FORCE" != "1" ]]; then
        return 0
    fi

    start_group "Building libcypher-parser"

    local src_dir="${ROOT}/deps/libcypher-parser"
    local build_dir="$LIBCYPHER_PARSER_BINDIR"

    mkdir -p "$build_dir"

    # Run autoreconf if configure doesn't exist (it's gitignored and must be generated)
    if [[ ! -f "$src_dir/configure" ]]; then
        log_info "Running autoreconf for libcypher-parser..."
        cd "$src_dir"
        autoreconf -fi
        cd "$build_dir"
    else
        cd "$build_dir"
    fi

    # Run configure if Makefile doesn't exist
    if [[ ! -f "$build_dir/Makefile" ]]; then
        log_info "Configuring libcypher-parser..."
        # --disable-tools: don't build cypher-lint (we only need the library)
        # Use -fPIC for position-independent code (required for linking into shared library)
        if ! CFLAGS="-fPIC" "$src_dir/configure" --disable-dependency-tracking --disable-tools; then
            log_error "Failed to configure libcypher-parser"
            cd "$ROOT"
            end_group
            exit 1
        fi
    fi

    log_info "Building libcypher-parser..."
    if ! make -j "$NPROC"; then
        log_error "Failed to build libcypher-parser"
        cd "$ROOT"
        end_group
        exit 1
    fi

    cd "$ROOT"

    log_success "libcypher-parser built successfully"
    end_group
}

#-----------------------------------------------------------------------------
# Function: build_libcurl
# Build libcurl using autotools
#-----------------------------------------------------------------------------
build_libcurl() {
    if [[ -f "$LIBCURL" ]] && [[ "$FORCE" != "1" ]]; then
        return 0
    fi

    start_group "Building libcurl"

    local src_dir="${ROOT}/deps/libcurl"
    local build_dir="$LIBCURL_BINDIR"

    mkdir -p "$build_dir"
    cd "$build_dir"

    # Run autoreconf if configure doesn't exist
    if [[ ! -f "$src_dir/configure" ]]; then
        log_info "Running autoreconf for libcurl..."
        cd "$src_dir"
        autoreconf -fi
        cd "$build_dir"
    fi

    # Run configure if Makefile doesn't exist
    if [[ ! -f "$build_dir/Makefile" ]]; then
        log_info "Configuring libcurl..."
        # Disable all optional dependencies to build a minimal static library
        # Use -fPIC for position-independent code (required for linking into shared library)
        if ! CFLAGS="-fPIC" "$src_dir/configure" --disable-dependency-tracking --disable-shared --enable-static \
            --without-ssl --without-libssh2 --without-librtmp --without-libidn2 \
            --without-nghttp2 --without-brotli --without-zstd --without-libpsl \
            --without-zlib --disable-ldap; then
            log_error "Failed to configure libcurl"
            cd "$ROOT"
            end_group
            exit 1
        fi
    fi

    log_info "Building libcurl..."
    if ! make -j "$NPROC"; then
        log_error "Failed to build libcurl"
        cd "$ROOT"
        end_group
        exit 1
    fi

    cd "$ROOT"

    log_success "libcurl built successfully"
    end_group
}

#-----------------------------------------------------------------------------
# Function: build_libcsv
# Build libcsv using autotools
#-----------------------------------------------------------------------------
build_libcsv() {
    if [[ -f "$LIBCSV" ]] && [[ "$FORCE" != "1" ]]; then
        return 0
    fi

    start_group "Building libcsv"

    local src_dir="${ROOT}/deps/libcsv"
    local build_dir="$LIBCSV_BINDIR"

    mkdir -p "$build_dir"

    # Always run autoreconf to ensure build files are consistent with local automake version
    # This avoids "automake-X.XX not found" errors when the repo's generated files
    # were created with a different automake version
    log_info "Running autoreconf for libcsv..."
    cd "$src_dir"
    autoreconf -fi
    cd "$build_dir"

    # Run configure if Makefile doesn't exist
    if [[ ! -f "$build_dir/Makefile" ]]; then
        log_info "Configuring libcsv..."
        # Use -fPIC for position-independent code (required for linking into shared library)
        if ! CFLAGS="-fPIC" "$src_dir/configure" --disable-dependency-tracking; then
            log_error "Failed to configure libcsv"
            cd "$ROOT"
            end_group
            exit 1
        fi
    fi

    log_info "Building libcsv..."
    if ! make -j "$NPROC"; then
        log_error "Failed to build libcsv"
        cd "$ROOT"
        end_group
        exit 1
    fi

    cd "$ROOT"

    log_success "libcsv built successfully"
    end_group
}

#-----------------------------------------------------------------------------
# Function: build_graphblas
# Build GraphBLAS using cmake
#-----------------------------------------------------------------------------
build_graphblas() {
    if [[ -f "$GRAPHBLAS" ]] && [[ "$FORCE" != "1" ]]; then
        return 0
    fi

    start_group "Building GraphBLAS"
    log_info "Building GraphBLAS..."

    local src_dir="${ROOT}/deps/GraphBLAS"
    local build_dir="$GRAPHBLAS_BINDIR"

    mkdir -p "$build_dir"
    cd "$build_dir"

    local cmake_args=(
        -DSUITESPARSE_USE_FORTRAN=OFF
        -DBUILD_STATIC_LIBS=ON
        -DBUILD_SHARED_LIBS=OFF
        -DGRAPHBLAS_COMPACT=ON
        -DCMAKE_POSITION_INDEPENDENT_CODE=ON
        -DCMAKE_C_FLAGS="-fPIC"
        -DCMAKE_CXX_FLAGS="-fPIC"
    )

    if [[ "$DEPS_DEBUG" == "1" ]]; then
        cmake_args+=(-DCMAKE_BUILD_TYPE=Debug)
    else
        cmake_args+=(-DCMAKE_BUILD_TYPE=Release)
    fi

    if [[ "$JIT" != "" ]]; then
        cmake_args+=(-DGRAPHBLAS_USE_JIT="$JIT")
    fi

    log_info "Configuring GraphBLAS..."
    if ! cmake "$src_dir" "${cmake_args[@]}"; then
        log_error "Failed to configure GraphBLAS"
        cd "$ROOT"
        end_group
        exit 1
    fi

    log_info "Building GraphBLAS..."
    if ! cmake --build . --config Release -j "$NPROC"; then
        log_error "Failed to build GraphBLAS"
        cd "$ROOT"
        end_group
        exit 1
    fi

    cd "$ROOT"

    log_success "GraphBLAS built successfully"
    end_group
}

#-----------------------------------------------------------------------------
# Function: build_lagraph
# Build LAGraph using cmake
#-----------------------------------------------------------------------------
build_lagraph() {
    if [[ -f "$LAGRAPH" ]] && [[ "$FORCE" != "1" ]]; then
        return 0
    fi

    start_group "Building LAGraph"
    log_info "Building LAGraph..."

    local src_dir="${ROOT}/deps/LAGraph"
    local build_dir="$LAGRAPH_BINDIR"

    mkdir -p "$build_dir"
    cd "$build_dir"

    local cmake_args=(
        -DBUILD_STATIC_LIBS=ON
        -DBUILD_SHARED_LIBS=OFF
        -DLIBRARY_ONLY=ON
        -DBUILD_TESTING=OFF
        -DGRAPHBLAS_ROOT="$GRAPHBLAS_BINDIR"
        -DGRAPHBLAS_INCLUDE_DIR="${ROOT}/deps/GraphBLAS/Include"
        -DGRAPHBLAS_LIBRARY="$GRAPHBLAS"
        -DCMAKE_POSITION_INDEPENDENT_CODE=ON
        -DCMAKE_C_FLAGS="-fPIC"
        -DCMAKE_CXX_FLAGS="-fPIC"
    )

    if [[ "$DEPS_DEBUG" == "1" ]]; then
        cmake_args+=(-DCMAKE_BUILD_TYPE=Debug)
    else
        cmake_args+=(-DCMAKE_BUILD_TYPE=Release)
    fi

    log_info "Configuring LAGraph..."
    if ! cmake "$src_dir" "${cmake_args[@]}"; then
        log_error "Failed to configure LAGraph"
        cd "$ROOT"
        end_group
        exit 1
    fi

    log_info "Building LAGraph..."
    if ! cmake --build . --config Release -j "$NPROC"; then
        log_error "Failed to build LAGraph"
        cd "$ROOT"
        end_group
        exit 1
    fi

    cd "$ROOT"

    log_success "LAGraph built successfully"
    end_group
}

#-----------------------------------------------------------------------------
# Function: build_quickjs
# Build quickjs natively
#
# Symbol Conflict Resolution:
#   Both quickjs and Friso (a Chinese text segmentation library used by RediSearch)
#   define a function called `unicode_to_utf8`. This causes a duplicate symbol error
#   at link time when building falkordb.so.
#
#   Solution: Before building quickjs, we rename all occurrences of `unicode_to_utf8`
#   to `quickjs_unicode_to_utf8` in the quickjs source files. This includes:
#     - The function definition in cutils.c
#     - The function declaration in cutils.h
#     - All internal calls in quickjs.c, libregexp.c, quickjs-libc.c, etc.
#
#   This is safe because `unicode_to_utf8` is an internal quickjs utility function,
#   not part of the public quickjs API (quickjs.h). FalkorDB only uses the public API,
#   so this rename is transparent to our code. After building, we restore the original
#   source files to keep the deps/quickjs directory clean.
#-----------------------------------------------------------------------------
build_quickjs() {
    if [[ -f "$QUICKJS" ]] && [[ "$FORCE" != "1" ]]; then
        return 0
    fi

    start_group "Building quickjs"
    log_info "Building quickjs..."

    local src_dir="${ROOT}/deps/quickjs"
    local bin_dir="$QUICKJS_BINDIR"

    mkdir -p "$bin_dir"

    cd "$src_dir"

    # Clean first to ensure fresh build
    make clean 2>/dev/null || true

    # Rename conflicting symbol: unicode_to_utf8 -> quickjs_unicode_to_utf8
    # This renames the function definition, declaration, and all internal calls
    # to avoid duplicate symbol error with Friso's unicode_to_utf8 at link time
    log_info "Renaming conflicting symbols..."
    find . -type f \( -name "*.c" -o -name "*.h" \) -exec \
        sed -i.bak 's/unicode_to_utf8/quickjs_unicode_to_utf8/g' {} \;

    # Build the library with -fPIC
    log_info "Compiling quickjs..."
    if ! CFLAGS="-fPIC" make -j "$NPROC" libquickjs.a; then
        # Restore original files on failure
        find . -name "*.bak" -exec sh -c 'mv "$1" "${1%.bak}"' _ {} \;
        log_error "Failed to build quickjs"
        cd "$ROOT"
        end_group
        exit 1
    fi

    # Copy the library to bin directory
    cp "$src_dir/libquickjs.a" "$bin_dir/"

    # Restore original source files
    log_info "Restoring original source files..."
    find . -name "*.bak" -exec sh -c 'mv "$1" "${1%.bak}"' _ {} \;

    cd "$ROOT"

    log_success "quickjs built successfully"
    end_group
}

#-----------------------------------------------------------------------------
# Function: build_utf8proc
# Build utf8proc natively
#-----------------------------------------------------------------------------
build_utf8proc() {
    if [[ -f "$UTF8PROC" ]] && [[ "$FORCE" != "1" ]]; then
        return 0
    fi

    start_group "Building utf8proc"
    log_info "Building utf8proc..."

    local src_dir="${ROOT}/deps/utf8proc"
    local bin_dir="$UTF8PROC_BINDIR"

    mkdir -p "$bin_dir"

    local cc_flags="-O2 -fPIC -std=c99 -I${src_dir}"
    if [[ "$DEPS_DEBUG" == "1" ]]; then
        cc_flags="-g -O0 -fPIC -std=c99 -I${src_dir}"
    fi

    # Compile utf8proc.c
    ${CC:-cc} $cc_flags -c "$src_dir/utf8proc.c" -o "$bin_dir/utf8proc.o"

    # Create static library
    ar rcs "$bin_dir/libutf8proc.a" "$bin_dir/utf8proc.o"

    log_success "utf8proc built successfully"
    end_group
}

#-----------------------------------------------------------------------------
# Function: build_oniguruma
# Build oniguruma using cmake
#-----------------------------------------------------------------------------
build_oniguruma() {
    if [[ -f "$ONIGURUMA" ]] && [[ "$FORCE" != "1" ]]; then
        return 0
    fi

    start_group "Building oniguruma"
    log_info "Building oniguruma..."

    local src_dir="${ROOT}/deps/oniguruma"
    local build_dir="$ONIGURUMA_BINDIR"

    mkdir -p "$build_dir"
    cd "$build_dir"

    local cmake_args=(
        -DBUILD_SHARED_LIBS=OFF
        -DENABLE_POSIX_API=OFF
        -DBUILD_TEST=OFF
        -DCMAKE_POSITION_INDEPENDENT_CODE=ON
        -DCMAKE_C_FLAGS="-fPIC"
    )

    if [[ "$DEPS_DEBUG" == "1" ]]; then
        cmake_args+=(-DCMAKE_BUILD_TYPE=Debug)
    else
        cmake_args+=(-DCMAKE_BUILD_TYPE=Release)
    fi

    log_info "Configuring oniguruma..."
    if ! cmake "$src_dir" "${cmake_args[@]}"; then
        log_error "Failed to configure oniguruma"
        cd "$ROOT"
        end_group
        exit 1
    fi

    log_info "Building oniguruma..."
    if ! cmake --build . --config Release -j "$NPROC"; then
        log_error "Failed to build oniguruma"
        cd "$ROOT"
        end_group
        exit 1
    fi

    cd "$ROOT"

    log_success "oniguruma built successfully"
    end_group
}

#-----------------------------------------------------------------------------
# Function: build_dependencies
# Build all required dependencies
#-----------------------------------------------------------------------------
build_dependencies() {
    if check_dependencies && [[ "$FORCE" != "1" ]]; then
        log_info "All dependencies present, skipping build"
        return 0
    fi

    log_info "Building dependencies..."

    # Build dependencies using native functions (no readies)
    build_rax
    build_xxhash
    build_libcurl
    build_libcsv
    build_libcypher_parser
    build_graphblas
    build_lagraph
    build_quickjs
    build_utf8proc
    build_oniguruma

    # Build FalkorDB-core-rs (Rust)
    build_falkordbrs

    log_success "All dependencies built successfully"
}

#-----------------------------------------------------------------------------
# Function: build_falkordbrs
# Build the Rust component
#-----------------------------------------------------------------------------
build_falkordbrs() {
    start_group "Building FalkorDB-core-rs (Rust)"

    # Check if cargo is available
    if ! command -v cargo &>/dev/null; then
        log_error "Cargo (Rust build tool) not found"
        log_error "Please install Rust from https://rustup.rs/"
        log_error "Or run: curl --proto '=https' --tlsv1.2 -sSf https://sh.rustup.rs | sh"
        end_group
        exit 1
    fi

    local cargo_flags=()
    local rustflags=""
    local cargo_cmd="cargo"

    # Determine release/debug
    if [[ "$DEBUG" != "1" && -z "$SAN" && "$COV" != "1" ]]; then
        cargo_flags+=(--release)
    fi

    # Handle sanitizer - requires nightly toolchain
    if [[ -n "$SAN" ]]; then
        # Check if nightly toolchain is available
        if rustup run nightly rustc --version &>/dev/null; then
            cargo_cmd="cargo +nightly"
            rustflags="-Zsanitizer=${SAN}"
            # Use the correct target triple for the current architecture
            local rust_target
            if [[ "$ARCH" == "arm64v8" ]]; then
                rust_target="aarch64-unknown-linux-gnu"
            else
                rust_target="x86_64-unknown-linux-gnu"
            fi
            # Ensure the target's standard library is installed for nightly
            if ! rustup +nightly target list --installed | grep -q "$rust_target"; then
                log_info "Installing Rust nightly target: $rust_target"
                rustup +nightly target add "$rust_target"
            fi
            cargo_flags+=(--target "$rust_target")
        else
            log_warn "Rust nightly toolchain not available, skipping sanitizer for Rust build"
            log_warn "Install with: rustup toolchain install nightly"
        fi
    fi

    # Handle coverage - only on Linux (macOS has LLVM profiler runtime linker issues)
    if [[ "$COV" == "1" && "$OS" == "linux" ]]; then
        rustflags="${rustflags:+${rustflags} }-C instrument-coverage"
    fi

    log_info "Building Rust component with flags: ${cargo_flags[*]}"

    cd "${ROOT}/deps/FalkorDB-core-rs"

    if [[ -n "$rustflags" ]]; then
        export RUSTFLAGS="$rustflags"
    fi

    if ! $cargo_cmd build "${cargo_flags[@]}" --features falkordb_allocator --target-dir "$FalkorDBRS_BINDIR"; then
        log_error "Failed to build FalkorDB-core-rs"
        end_group
        exit 1
    fi

    cd "$ROOT"
    log_success "FalkorDB-core-rs built successfully"
    end_group
}

#-----------------------------------------------------------------------------
# Function: prepare_cmake_arguments
# Prepare arguments to pass to CMake
#-----------------------------------------------------------------------------
prepare_cmake_arguments() {
    CMAKE_ARGS=()

    # Build type
    if [[ "$DEBUG" == "1" ]]; then
        CMAKE_ARGS+=(-DCMAKE_BUILD_TYPE=Debug)
    else
        CMAKE_ARGS+=(-DCMAKE_BUILD_TYPE=RelWithDebInfo)
    fi

    # Unit tests
    if [[ "$BUILD_TESTS" == "1" ]]; then
        CMAKE_ARGS+=(-DUNIT_TESTS:BOOL=on)
    fi

    # Coverage
    if [[ "$COV" == "1" ]]; then
        CMAKE_ARGS+=(-DCOV=ON)
    fi

    # Sanitizer
    if [[ -n "$SAN" ]]; then
        CMAKE_ARGS+=(-DSAN="$SAN")
    fi

    # Memory checking
    if [[ "$MEMCHECK" == "1" ]]; then
        CMAKE_ARGS+=(-DMEMCHECK=ON)
    fi

    # Static OpenMP
    if [[ "$STATIC_OMP" == "1" ]]; then
        CMAKE_ARGS+=(-DSTATIC_OMP=ON)
    fi

    # Set compilers if specified
    if [[ -n "$CC" ]]; then
        CMAKE_ARGS+=(-DCMAKE_C_COMPILER="$CC")
    fi
    if [[ -n "$CXX" ]]; then
        CMAKE_ARGS+=(-DCMAKE_CXX_COMPILER="$CXX")
    fi

    # Export platform info
    export OS
    export OSNICK
    export ARCH

    if [[ "$VERBOSE" == "1" ]]; then
        log_info "CMake arguments: ${CMAKE_ARGS[*]}"
    fi
}

#-----------------------------------------------------------------------------
# Function: run_cmake
# Run CMake to configure the build
#-----------------------------------------------------------------------------
run_cmake() {
    start_group "Configuring CMake"

    # Create build directory
    mkdir -p "$BINROOT"

    # Clean if forced
    if [[ "$FORCE" == "1" ]]; then
        log_info "Force clean: removing CMake cache"
        rm -f "$BINROOT/CMakeCache.txt"
        rm -rf "$BINROOT/CMakeFiles"
    fi

    log_info "Configuring build in: $BINROOT"

    cd "$BINROOT"

    if [[ "$VERBOSE" == "1" ]]; then
        cmake "$ROOT" "${CMAKE_ARGS[@]}" --trace-expand
    else
        cmake "$ROOT" "${CMAKE_ARGS[@]}"
    fi

    if [[ $? -ne 0 ]]; then
        log_error "CMake configuration failed"
        end_group
        exit 1
    fi

    cd "$ROOT"
    log_success "CMake configuration complete"
    end_group
}

#-----------------------------------------------------------------------------
# Function: build_project
# Build the main FalkorDB project
#-----------------------------------------------------------------------------
build_project() {
    start_group "Building FalkorDB"

    # Determine number of parallel jobs
    if [[ "$SLOW" == "1" ]]; then
        NPROC=1
    elif command -v nproc &> /dev/null; then
        NPROC=$(nproc)
    elif command -v sysctl &> /dev/null && [[ "$OS" == "macos" ]]; then
        NPROC=$(sysctl -n hw.physicalcpu)
    else
        NPROC=4
    fi

    log_info "Building with $NPROC parallel jobs..."

    cd "$BINROOT"

    if [[ "$VERBOSE" == "1" ]]; then
        make -j "$NPROC" VERBOSE=1
    else
        make -j "$NPROC"
    fi

    if [[ $? -ne 0 ]]; then
        log_error "Build failed"
        end_group
        exit 1
    fi

    cd "$ROOT"

    # Verify output exists
    if [[ ! -f "$TARGET" ]]; then
        log_error "Build succeeded but target not found: $TARGET"
        end_group
        exit 1
    fi

    log_success "Build complete: $TARGET"
    end_group
}

#-----------------------------------------------------------------------------
# Function: run_unit_tests
# Run C/C++ and Rust unit tests
#-----------------------------------------------------------------------------
run_unit_tests() {
    if [[ "$RUN_UNIT_TESTS" != "1" ]]; then
        return 0
    fi

    # Prepare coverage capture if enabled
    prepare_coverage_capture

    start_group "Running Unit Tests"

    export BINROOT

    # Set test filter if provided
    if [[ -n "$TEST_FILTER" ]]; then
        export TEST="$TEST_FILTER"
    fi

    if [[ "$LIST_TESTS" == "1" ]]; then
        export LIST=1
    fi

    if [[ "$VERBOSE" == "1" ]]; then
        export VERBOSE=1
    fi

    if [[ -n "$SAN" ]]; then
        export SAN
    fi

    if [[ "$VG" == "1" ]]; then
        export VG=1
    fi

    if [[ "$GDB" == "1" ]]; then
        export GDB=1
    fi

    # Run C/C++ unit tests
    log_info "Running C/C++ unit tests..."
    if ! "${ROOT}/tests/unit/tests.sh"; then
        log_error "C/C++ unit tests failed"
        end_group
        return 1
    fi

    # Run Rust unit tests
    log_info "Running Rust unit tests..."
    if ! cargo test --lib --target-dir "$FalkorDBRS_BINDIR"; then
        log_error "Rust unit tests failed"
        end_group
        return 1
    fi

    log_success "All unit tests passed"

    # Capture coverage if enabled
    capture_coverage unit

    end_group
}

#-----------------------------------------------------------------------------
# Function: run_flow_tests
# Run Python flow tests
#-----------------------------------------------------------------------------
run_flow_tests() {
    if [[ "$RUN_FLOW_TESTS" != "1" ]]; then
        return 0
    fi

    # Prepare coverage capture if enabled
    prepare_coverage_capture

    start_group "Running Flow Tests"

    export MODULE="$TARGET"
    export BINROOT

    # Set parallelism
    if [[ -n "$PARALLEL" ]]; then
        export PARALLEL
    elif [[ "$SLOW" == "1" ]]; then
        export PARALLEL=0
    else
        export PARALLEL=1
    fi

    # Set test options
    export GEN=1
    export AOF=0
    export TCK=0
    export UPGRADE=0

    if [[ -n "$TEST_FILTER" ]]; then
        export TEST="$TEST_FILTER"
    fi

    if [[ -n "$TESTFILE" ]]; then
        export TESTFILE
    fi

    if [[ -n "$FAILFILE" ]]; then
        export FAILEDFILE="$FAILFILE"
    fi

    if [[ "$GDB" == "1" ]]; then
        export GDB=1
    fi

    if [[ "$VERBOSE" == "1" ]]; then
        export VERBOSE=1
    fi

    log_info "Running flow tests..."
    if ! "${ROOT}/tests/flow/tests.sh"; then
        log_error "Flow tests failed"
        end_group
        return 1
    fi

    log_success "Flow tests passed"

    # Capture coverage if enabled
    capture_coverage flow

    end_group
}

#-----------------------------------------------------------------------------
# Function: run_tck_tests
# Run TCK (Technology Compatibility Kit) tests
#-----------------------------------------------------------------------------
run_tck_tests() {
    if [[ "$RUN_TCK_TESTS" != "1" ]]; then
        return 0
    fi

    # Prepare coverage capture if enabled
    prepare_coverage_capture

    start_group "Running TCK Tests"

    export MODULE="$TARGET"
    export BINROOT

    if [[ -n "$PARALLEL" ]]; then
        export PARALLEL
    elif [[ "$SLOW" == "1" ]]; then
        export PARALLEL=0
    else
        export PARALLEL=1
    fi

    export GEN=0
    export AOF=0
    export TCK=1
    export UPGRADE=0

    if [[ -n "$TEST_FILTER" ]]; then
        export TEST="$TEST_FILTER"
    fi

    log_info "Running TCK tests..."
    if ! "${ROOT}/tests/flow/tests.sh"; then
        log_error "TCK tests failed"
        end_group
        return 1
    fi

    log_success "TCK tests passed"

    # Capture coverage if enabled
    capture_coverage tck

    end_group
}

#-----------------------------------------------------------------------------
# Function: run_upgrade_tests
# Run upgrade tests
#-----------------------------------------------------------------------------
run_upgrade_tests() {
    if [[ "$RUN_UPGRADE_TESTS" != "1" ]]; then
        return 0
    fi

    # Prepare coverage capture if enabled
    prepare_coverage_capture

    start_group "Running Upgrade Tests"

    export MODULE="$TARGET"
    export BINROOT

    if [[ -n "$PARALLEL" ]]; then
        export PARALLEL
    else
        export PARALLEL=0  # Upgrade tests run slowly
    fi

    export GEN=0
    export AOF=0
    export TCK=0
    export UPGRADE=1
    export SLOW=1

    if [[ -n "$TEST_FILTER" ]]; then
        export TEST="$TEST_FILTER"
    fi

    log_info "Running upgrade tests..."
    if ! "${ROOT}/tests/flow/tests.sh"; then
        log_error "Upgrade tests failed"
        end_group
        return 1
    fi

    log_success "Upgrade tests passed"

    # Capture coverage if enabled
    capture_coverage upgrade

    end_group
}

#-----------------------------------------------------------------------------
# Function: run_fuzz_tests
# Run fuzz tests
#-----------------------------------------------------------------------------
run_fuzz_tests() {
    if [[ "$RUN_FUZZ_TESTS" != "1" ]]; then
        return 0
    fi

    start_group "Running Fuzz Tests"

    log_info "Running fuzz tests with timeout=${FUZZ_TIMEOUT}s..."

    cd "${ROOT}/tests/fuzz"

    # Check if required Python packages are installed
    if ! python3 -c "import grammarinator" 2>/dev/null; then
        log_info "Installing grammarinator..."
        pip3 install grammarinator
    fi

    if ! python3 -c "from RLTest import Env" 2>/dev/null; then
        log_info "Installing RLTest..."
        pip3 install RLTest
    fi

    if ! python3 -c "from falkordb import FalkorDB" 2>/dev/null; then
        log_info "Installing falkordb..."
        pip3 install falkordb
    fi

    local fuzz_args=()
    fuzz_args+=("-m" "$TARGET")
    fuzz_args+=("-t" "$FUZZ_TIMEOUT")

    if ! python3 ./process.py "${fuzz_args[@]}"; then
        log_error "Fuzz tests failed"
        cd "$ROOT"
        end_group
        return 1
    fi

    cd "$ROOT"
    log_success "Fuzz tests passed"
    end_group
}

#-----------------------------------------------------------------------------
# Function: do_clean
# Clean build products
#-----------------------------------------------------------------------------
do_clean() {
    start_group "Cleaning Build Products"

    if [[ "$CLEAN_ALL" == "1" ]]; then
        log_info "Removing entire bin directory and deps..."
        rm -rf "${ROOT}/bin" "${DEPS_BINDIR}"

        # Also clean libcypher-parser autogen if AUTOGEN=1
        if [[ "$CLEAN_AUTOGEN" == "1" ]]; then
            log_info "Cleaning libcypher-parser autogen files..."
            cd "${ROOT}/deps/libcypher-parser"
            make distclean 2>/dev/null || true
            cd "$ROOT"
        fi
    else
        # Clean current build variant only
        if [[ -d "$BINROOT" ]]; then
            log_info "Cleaning build in: $BINROOT"
            if [[ -f "$BINROOT/Makefile" ]]; then
                make -C "$BINROOT" clean 2>/dev/null || true
            fi
            rm -f "${TARGET}.debug" "${BINROOT}/CMakeCache.txt"
            rm -rf "${BINROOT}/tests"
        fi

        # Clean dependencies if requested
        if [[ "$CLEAN_DEPS" == "1" ]]; then
            log_info "Cleaning dependencies..."

            # Remove dependency build directories
            rm -rf "$RAX_BINDIR" 2>/dev/null || true
            rm -rf "$LIBXXHASH_BINDIR" 2>/dev/null || true
            rm -rf "$UTF8PROC_BINDIR" 2>/dev/null || true
            rm -rf "$ONIGURUMA_BINDIR" 2>/dev/null || true
            rm -rf "$GRAPHBLAS_BINDIR" 2>/dev/null || true
            rm -rf "$LAGRAPH_BINDIR" 2>/dev/null || true
            rm -rf "$QUICKJS_BINDIR" 2>/dev/null || true
            rm -rf "$LIBCURL_BINDIR" 2>/dev/null || true
            rm -rf "$LIBCSV_BINDIR" 2>/dev/null || true
            rm -rf "$LIBCYPHER_PARSER_BINDIR" 2>/dev/null || true

            # Clean quickjs in-source build
            if [[ -d "${ROOT}/deps/quickjs" ]]; then
                make -C "${ROOT}/deps/quickjs" clean 2>/dev/null || true
            fi

            # Clean Rust target
            if [[ -d "$FalkorDBRS_BINDIR" ]]; then
                rm -rf "$FalkorDBRS_BINDIR"
            fi
        fi
    fi

    log_success "Clean complete"
    end_group
}

#-----------------------------------------------------------------------------
# Function: do_pack
# Build RAMP packages
#-----------------------------------------------------------------------------
do_pack() {
    start_group "Building RAMP Packages"

    if [[ ! -f "$TARGET" ]]; then
        log_error "Target module not found: $TARGET"
        log_error "Please build the project first"
        end_group
        return 1
    fi

    export MODULE="$TARGET"

    log_info "Building packages for: $TARGET"

    if ! "${ROOT}/sbin/pack.sh"; then
        log_error "Package creation failed"
        end_group
        return 1
    fi

    log_success "Packages created successfully"
    end_group
}

#-----------------------------------------------------------------------------
# Function: run_benchmark
# Run benchmarks
#-----------------------------------------------------------------------------
run_benchmark() {
    if [[ "$BENCHMARK" != "1" ]]; then
        return 0
    fi

    start_group "Running Benchmarks"

    log_info "Running benchmarks..."

    cd "${ROOT}/tests/benchmarks"

    # Set up Python virtual environment if needed
    if [[ ! -d "venv" ]]; then
        log_info "Creating Python virtual environment..."
        python3 -m venv venv
    fi

    # Activate venv and install requirements
    source venv/bin/activate
    pip install -q -r benchmarks_requirements.txt

    # Set DB_MODULE environment variable
    export DB_MODULE="$TARGET"

    # Run benchmarks
    if [[ -n "$BENCHMARK_GROUP" ]]; then
        log_info "Running benchmark group: $BENCHMARK_GROUP"
        if ! python3 run_benchmarks.py "$BENCHMARK_GROUP"; then
            log_error "Benchmark group $BENCHMARK_GROUP failed"
            deactivate
            cd "$ROOT"
            end_group
            return 1
        fi
    else
        # Run all benchmark groups
        log_info "Running all benchmark groups..."
        for group in group_a group_b; do
            log_info "Running benchmark group: $group"
            if ! python3 run_benchmarks.py "$group"; then
                log_error "Benchmark group $group failed"
                deactivate
                cd "$ROOT"
                end_group
                return 1
            fi
        done
    fi

    deactivate
    cd "$ROOT"
    log_success "Benchmarks completed successfully"
    end_group
}

#-----------------------------------------------------------------------------
# Function: run_redis
# Run redis-server with the FalkorDB module loaded
#-----------------------------------------------------------------------------
run_redis() {
    if [[ "$RUN" != "1" ]]; then
        return 0
    fi

    # Check if target exists
    if [[ ! -f "$TARGET" ]]; then
        log_error "Target module not found: $TARGET"
        log_error "Please build the project first with: ./build.sh"
        return 1
    fi

    # Check if redis-server is available
    if ! command -v redis-server &>/dev/null; then
        log_error "redis-server not found in PATH"
        return 1
    fi

    log_info "Starting redis-server with FalkorDB module..."
    log_info "Module: $TARGET"

    if [[ "$GDB" == "1" ]]; then
        log_info "Running with GDB debugger..."
        exec gdb --args redis-server --loadmodule "$TARGET"
    else
        exec redis-server --loadmodule "$TARGET"
    fi
}

#-----------------------------------------------------------------------------
# Function: prepare_coverage_capture
# Prepare lcov for coverage capture before running tests
#-----------------------------------------------------------------------------
prepare_coverage_capture() {
    if [[ "$COV" != "1" ]]; then
        return 0
    fi

    start_group "Preparing Coverage Capture"

    log_info "Resetting coverage counters..."
    lcov --zerocounters --directory "$BINROOT" --base-directory "$ROOT" 2>/dev/null || true

    log_info "Capturing baseline coverage..."
    lcov --capture --initial --directory "$BINROOT" --base-directory "$ROOT" -o "$BINROOT/base.info" 2>/dev/null || true

    end_group
}

#-----------------------------------------------------------------------------
# Function: capture_coverage
# Capture coverage data after tests and generate report
# Arguments: $1 - optional name for coverage file (default: cov)
#-----------------------------------------------------------------------------
capture_coverage() {
    if [[ "$COV" != "1" ]]; then
        return 0
    fi

    local name=${1:-cov}

    start_group "Capturing Coverage ($name)"

    log_info "Capturing test coverage..."
    lcov --capture --directory "$BINROOT" --base-directory "$ROOT" -o "$BINROOT/test.info" 2>/dev/null

    if [[ ! -f "$BINROOT/base.info" ]]; then
        log_warn "No baseline coverage file found, using test coverage only"
        cp "$BINROOT/test.info" "$BINROOT/full.info"
    else
        log_info "Merging baseline and test coverage..."
        lcov --add-tracefile "$BINROOT/base.info" --add-tracefile "$BINROOT/test.info" -o "$BINROOT/full.info"
    fi

    log_info "Extracting source file coverage..."
    lcov --output-file "$BINROOT/source.info" --extract "$BINROOT/full.info" \
        "$ROOT/src/*" 2>/dev/null || true

    log_info "Removing test file coverage..."
    lcov -o "$BINROOT/$name.info" --ignore-errors unused --remove "$BINROOT/source.info" \
        "*/tests/*" 2>/dev/null || true

    # Clean up intermediate files
    rm -f "$BINROOT/base.info" "$BINROOT/test.info" "$BINROOT/full.info" "$BINROOT/source.info" 2>/dev/null || true

    log_success "Coverage data saved to: $BINROOT/$name.info"
    end_group
}

#-----------------------------------------------------------------------------
# Function: generate_coverage_report
# Generate HTML coverage report using genhtml
#-----------------------------------------------------------------------------
generate_coverage_report() {
    if [[ "$COV" != "1" ]]; then
        return 0
    fi

    start_group "Generating Coverage Report"

    local cov_file="$BINROOT/cov.info"
    local cov_dir="$BINROOT/coverage"

    if [[ ! -f "$cov_file" ]]; then
        log_warn "No coverage file found at $cov_file"
        end_group
        return 1
    fi

    log_info "Generating HTML coverage report..."
    mkdir -p "$cov_dir"

    if genhtml --legend -o "$cov_dir" "$cov_file" 2>/dev/null; then
        log_success "Coverage report generated: $cov_dir/index.html"
    else
        log_warn "Failed to generate HTML report (genhtml may not be installed)"
    fi

    end_group
}

#-----------------------------------------------------------------------------
# Function: run_all_tests
# Run all test suites
#-----------------------------------------------------------------------------
run_all_tests() {
    local has_failures=0

    run_unit_tests || has_failures=1
    run_flow_tests || has_failures=1
    run_tck_tests || has_failures=1
    run_upgrade_tests || has_failures=1

    if [[ $has_failures -eq 1 ]]; then
        log_error "Some tests failed"
        return 1
    fi

    log_success "All tests passed!"
    return 0
}

#-----------------------------------------------------------------------------
# Main execution flow
#-----------------------------------------------------------------------------

main() {
    # Parse command line arguments
    parse_arguments "$@"

    # Show configuration banner
    log_info "FalkorDB Build Script"
    log_info "====================="

    # Detect platform
    detect_platform

    # Setup build environment
    setup_build_environment

    # Handle clean operation (early exit)
    if [[ "$CLEAN" == "1" ]]; then
        do_clean
        exit 0
    fi

    # Handle pack-only operation (no build needed if target exists)
    if [[ "$PACK" == "1" && "$RUN_TESTS" != "1" && "$RUN_UNIT_TESTS" != "1" && \
          "$RUN_FLOW_TESTS" != "1" && "$RUN_TCK_TESTS" != "1" && \
          "$RUN_UPGRADE_TESTS" != "1" && "$RUN_FUZZ_TESTS" != "1" ]]; then
        # Check if target already exists
        if [[ -f "$TARGET" ]]; then
            do_pack
            exit 0
        fi
        # Otherwise, fall through to build first
    fi

    # Build dependencies
    build_dependencies

    # Prepare CMake arguments
    prepare_cmake_arguments

    # Run CMake configuration
    run_cmake

    # Build the project
    build_project

    # Run pack if requested
    if [[ "$PACK" == "1" ]]; then
        do_pack
    fi

    # Run tests if requested
    if [[ "$RUN_TESTS" == "1" ]]; then
        RUN_UNIT_TESTS=1
        RUN_FLOW_TESTS=1
        RUN_TCK_TESTS=1
        RUN_UPGRADE_TESTS=1
    fi

    local test_result=0
    if [[ "$RUN_UNIT_TESTS" == "1" || "$RUN_FLOW_TESTS" == "1" || "$RUN_TCK_TESTS" == "1" || "$RUN_UPGRADE_TESTS" == "1" ]]; then
        if [[ "$RUN_TESTS" == "1" ]]; then
            run_all_tests || test_result=1
        else
            run_unit_tests || test_result=1
            run_flow_tests || test_result=1
            run_tck_tests || test_result=1
            run_upgrade_tests || test_result=1
        fi
    fi

    # Run fuzz tests if requested
    if [[ "$RUN_FUZZ_TESTS" == "1" ]]; then
        run_fuzz_tests || test_result=1
    fi

    # Run benchmarks if requested
    if [[ "$BENCHMARK" == "1" ]]; then
        run_benchmark || test_result=1
    fi

    # Run redis-server if requested (this will exec and not return)
    if [[ "$RUN" == "1" ]]; then
        run_redis
        # run_redis uses exec, so we won't reach here unless it fails
        exit 1
    fi

    # Generate coverage report if tests were run with coverage
    if [[ "$COV" == "1" ]]; then
        generate_coverage_report
    fi

    # Final summary
    echo ""
    log_info "Build Summary"
    log_info "============="
    log_info "Target: $TARGET"
    log_info "Variant: $FULL_VARIANT"

    if [[ $test_result -eq 0 ]]; then
        log_success "Build completed successfully!"
    else
        log_error "Build completed with test failures"
        exit 1
    fi

    exit 0
}

# Run main function with all arguments
main "$@"
