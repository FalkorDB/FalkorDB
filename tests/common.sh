#!/bin/bash
#
# Common functions and definitions for FalkorDB tests
#

# Get the root directory
if [[ -z "$ROOT" ]]; then
    ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
fi

#----------------------------------------------------------------------------------------------
# Color definitions
#----------------------------------------------------------------------------------------------

if [[ -t 1 ]]; then
    RED='\033[0;31m'
    LIGHTRED='\033[1;31m'
    GREEN='\033[0;32m'
    YELLOW='\033[1;33m'
    BLUE='\033[0;34m'
    CYAN='\033[0;36m'
    NOCOLOR='\033[0m'
else
    RED=''
    LIGHTRED=''
    GREEN=''
    YELLOW=''
    BLUE=''
    CYAN=''
    NOCOLOR=''
fi

#----------------------------------------------------------------------------------------------
# Common helper functions
#----------------------------------------------------------------------------------------------

eprint() {
    echo -e "${RED}[ERROR]${NOCOLOR} $*" >&2
}

is_command() {
    command -v "$1" &>/dev/null
}

separator() {
    echo "--------------------------------------------------------------------------------"
}

# Alias for separator (used in flow tests)
sep1() {
    separator
}

runn() {
    if [[ "$V" == "1" || "$VERBOSE" == "1" ]]; then
        echo "Running: $@"
    fi
    "$@"
}

get_platform_os() {
    case "$(uname -s)" in
        Linux*) echo "linux" ;;
        Darwin*) echo "macos" ;;
        *) echo "unknown" ;;
    esac
}

get_platform_arch() {
    case "$(uname -m)" in
        x86_64) echo "x64" ;;
        aarch64|arm64) echo "arm64v8" ;;
        *) uname -m ;;
    esac
}

get_platform_osnick() {
    if [[ "$(uname -s)" == "Darwin" ]]; then
        echo "macos"
    elif [[ -f /etc/os-release ]]; then
        . /etc/os-release
        echo "${ID}${VERSION_ID}" | tr '[:upper:]' '[:lower:]'
    else
        echo "linux"
    fi
}

get_nproc() {
    if command -v nproc &>/dev/null; then
        nproc
    elif command -v sysctl &>/dev/null; then
        sysctl -n hw.physicalcpu
    else
        echo "4"
    fi
}

is_abspath() {
    case $1 in (/*) pathchk -- "$1";; (*) ! : ;; esac
}

#----------------------------------------------------------------------------------------------
# Memcheck functions - for Valgrind and Sanitizer log analysis
#----------------------------------------------------------------------------------------------

_valgrind_check() {
    local logdir="$1"
    local pattern="$2"
    local type="$3"

    if grep -l "$pattern" "$logdir"/*.valgrind.log &> /dev/null; then
        echo
        echo -e "${LIGHTRED}### Valgrind: ${type} detected:${RED}"
        grep -l "$pattern" "$logdir"/*.valgrind.log
        echo -e "${NOCOLOR}"
        return 1
    fi
    return 0
}

_valgrind_summary() {
    local logdir="$1"
    local E=0

    # Check for memory leaks
    local leaks_head=0
    for file in $(ls "$logdir"/*.valgrind.log 2>/dev/null); do
        # If the last "definitely lost: " line of a logfile has a nonzero value, print the file name
        if tac "$file" | grep -a -m 1 "definitely lost: " | grep "definitely lost: [1-9][0-9,]* bytes" &> /dev/null; then
            if [[ $leaks_head == 0 ]]; then
                echo
                echo -e "${LIGHTRED}### Valgrind: leaks detected:${RED}"
                leaks_head=1
            fi
            echo "$file"
            E=1
        fi
    done

    _valgrind_check "$logdir" "Invalid read" "invalid reads" || E=1
    _valgrind_check "$logdir" "Invalid write" "invalid writes" || E=1

    return $E
}

_sanitizer_check() {
    local logdir="$1"
    local pattern="$2"
    local type="$3"
    local warn_only="${4:-0}"

    if grep -l "$pattern" "$logdir"/*.asan.log* &> /dev/null; then
        echo
        echo -e "${LIGHTRED}### Sanitizer: ${type} detected:${RED}"
        grep -l "$pattern" "$logdir"/*.asan.log*
        echo -e "${NOCOLOR}"
        if [[ $warn_only != 1 ]]; then
            return 1
        fi
    fi
    return 0
}

_sanitizer_summary() {
    local logdir="$1"
    local E=0

    if ! _sanitizer_check "$logdir" "Direct leak" "leaks"; then
        E=1
    elif ! _sanitizer_check "$logdir" "detected memory leaks" "leaks"; then
        E=1
    fi

    _sanitizer_check "$logdir" "dynamic-stack-buffer-overflow" "buffer overflow" || E=1
    _sanitizer_check "$logdir" "memcpy-param-overlap" "memory errors" || E=1
    _sanitizer_check "$logdir" "stack-use-after-scope" "stack use after scope" || E=1
    _sanitizer_check "$logdir" "heap-use-after-free" "use after free" || E=1
    _sanitizer_check "$logdir" "signal 11" "signal 11" 1  # warn only

    return $E
}

# Main memcheck summary function
# Usage: memcheck_summary <test_type> [test_type2 ...]
# test_type can be: unit, flow, tck
# Environment variables:
#   VG=1    - Check valgrind logs
#   SAN=x   - Check sanitizer logs
memcheck_summary() {
    local E=0
    local dirs=("$@")

    if [[ ${#dirs[@]} -eq 0 ]]; then
        echo "# No test directories specified for memcheck"
        return 0
    fi

    for dir in "${dirs[@]}"; do
        local logdir="$ROOT/tests/$dir/logs"

        if [[ ! -d "$logdir" ]]; then
            continue
        fi

        if [[ $VG == 1 ]]; then
            _valgrind_summary "$logdir" || E=1
        elif [[ -n $SAN ]]; then
            _sanitizer_summary "$logdir" || E=1
        fi
    done

    if [[ $E == 0 ]]; then
        echo "# No leaks detected"
    fi

    return $E
}
