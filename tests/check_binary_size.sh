#!/usr/bin/env bash

#-----------------------------------------------------------------------------
# Binary Size Verification Script
#
# Checks that built binary sizes are within ±5% of the reference sizes
# from the v4.16.3 release. This prevents unexpected binary bloat or
# missing components from going unnoticed.
#
# Usage:
#   ./tests/check_binary_size.sh <binary_path>
#   ./tests/check_binary_size.sh <directory>
#
# When given a directory, checks all .so files found recursively.
# When given a file, checks just that file.
#
# Exit codes:
#   0 - All checks passed (or no reference size for binary)
#   1 - One or more checks failed
#-----------------------------------------------------------------------------

set -e

TOLERANCE=5

# Reference binary sizes (bytes) from v4.16.3 release
declare -A REFERENCE_SIZES=(
    # Release binaries
    ["falkordb-x64.so"]=39028928
    ["falkordb-arm64v8.so"]=32698928
    ["falkordb-alpine-x64.so"]=40356232
    ["falkordb-alpine-arm64v8.so"]=33713088
    ["falkordb-rhel8-x64.so"]=39908480
    ["falkordb-rhel9-x64.so"]=39841104
    ["falkordb-amazonlinux2023-x64.so"]=38431640
    ["falkordb-macos-arm64v8.so"]=25065704

    # Debug binaries
    ["falkordb-debug-x64.so"]=171419976
    ["falkordb-debug-arm64v8.so"]=164460304
    ["falkordb-debug-alpine-x64.so"]=173350312
    ["falkordb-debug-alpine-arm64v8.so"]=161408176
    ["falkordb-debug-rhel8-x64.so"]=224266144
    ["falkordb-debug-rhel9-x64.so"]=173339608
    ["falkordb-debug-macos-arm64v8.so"]=28996392
)

check_binary() {
    local binary_path="$1"
    local binary_name
    binary_name=$(basename "$binary_path")

    if [[ ! -f "$binary_path" ]]; then
        echo "ERROR: File not found: $binary_path"
        return 1
    fi

    if [[ -z "${REFERENCE_SIZES[$binary_name]+x}" ]]; then
        echo "WARNING: No reference size for '$binary_name', skipping check"
        return 0
    fi

    local reference_size=${REFERENCE_SIZES[$binary_name]}
    local actual_size
    actual_size=$(stat -c%s "$binary_path" 2>/dev/null || stat -f%z "$binary_path")

    local min_size=$(( reference_size * (100 - TOLERANCE) / 100 ))
    local max_size=$(( reference_size * (100 + TOLERANCE) / 100 ))

    echo "Binary: $binary_name"
    echo "  Reference size: $reference_size bytes"
    echo "  Actual size:    $actual_size bytes"
    echo "  Allowed range:  $min_size - $max_size bytes (±${TOLERANCE}%)"

    if [[ $actual_size -ge $min_size ]] && [[ $actual_size -le $max_size ]]; then
        local diff_pct=$(( (actual_size - reference_size) * 100 / reference_size ))
        echo "  ✅ PASS (${diff_pct}% from reference)"
        return 0
    else
        local diff_pct
        if [[ $actual_size -lt $min_size ]]; then
            diff_pct=$(( (reference_size - actual_size) * 100 / reference_size ))
            echo "  ❌ FAIL - Size is ${diff_pct}% smaller than reference"
        else
            diff_pct=$(( (actual_size - reference_size) * 100 / reference_size ))
            echo "  ❌ FAIL - Size is ${diff_pct}% larger than reference"
        fi
        return 1
    fi
}

#-----------------------------------------------------------------------------
# Main
#-----------------------------------------------------------------------------

if [[ $# -ne 1 ]]; then
    echo "Usage: $0 <binary_path|directory>"
    exit 1
fi

TARGET="$1"
FAILURES=0

if [[ -d "$TARGET" ]]; then
    # Check all .so files in the directory (recursively)
    FOUND=0
    while IFS= read -r -d '' so_file; do
        FOUND=1
        echo "---"
        if ! check_binary "$so_file"; then
            FAILURES=$((FAILURES + 1))
        fi
    done < <(find "$TARGET" -name "*.so" -type f -print0)

    if [[ $FOUND -eq 0 ]]; then
        echo "ERROR: No .so files found in $TARGET"
        exit 1
    fi
elif [[ -f "$TARGET" ]]; then
    if ! check_binary "$TARGET"; then
        FAILURES=$((FAILURES + 1))
    fi
else
    echo "ERROR: $TARGET is not a file or directory"
    exit 1
fi

echo ""
if [[ $FAILURES -gt 0 ]]; then
    echo "❌ $FAILURES binary size check(s) FAILED"
    exit 1
else
    echo "✅ All binary size checks PASSED"
    exit 0
fi
