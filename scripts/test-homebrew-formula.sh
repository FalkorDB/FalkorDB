#!/bin/bash
# Test script for FalkorDB Homebrew formula
# This script helps test the formula locally on macOS

set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"
FORMULA_FILE="${REPO_ROOT}/Formula/falkordb.rb"

echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
echo "FalkorDB Homebrew Formula Test Script"
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
echo

# Check if running on macOS
if [[ "$OSTYPE" != "darwin"* ]]; then
    echo "❌ Error: This script is intended for macOS only"
    exit 1
fi

# Check if Homebrew is installed
if ! command -v brew &> /dev/null; then
    echo "❌ Error: Homebrew is not installed"
    echo "   Install Homebrew from https://brew.sh/"
    exit 1
fi

echo "✓ Homebrew is installed: $(brew --version | head -1)"
echo

# Check if formula file exists
if [[ ! -f "$FORMULA_FILE" ]]; then
    echo "❌ Error: Formula file not found at $FORMULA_FILE"
    exit 1
fi

echo "✓ Formula file found: $FORMULA_FILE"
echo

# Validate Ruby syntax
echo "Validating formula syntax..."
if ruby -c "$FORMULA_FILE" > /dev/null 2>&1; then
    echo "✓ Formula syntax is valid"
else
    echo "❌ Error: Formula has syntax errors"
    exit 1
fi
echo

# Check if all dependencies are available
echo "Checking formula dependencies..."
DEPS=(redis autoconf automake cmake libtool m4 peg python@3.13 gcc libomp make)
MISSING_DEPS=()

for dep in "${DEPS[@]}"; do
    if brew list "$dep" &> /dev/null; then
        echo "  ✓ $dep"
    else
        echo "  ✗ $dep (not installed)"
        MISSING_DEPS+=("$dep")
    fi
done

if [ ${#MISSING_DEPS[@]} -ne 0 ]; then
    echo
    echo "⚠️  Warning: Some dependencies are not installed"
    echo "   Missing: ${MISSING_DEPS[*]}"
    echo "   These will be installed automatically when installing FalkorDB"
fi
echo

# Ask user what they want to do
echo "What would you like to do?"
echo "  1) Install FalkorDB from the formula (recommended)"
echo "  2) Build only (without installing)"
echo "  3) Run brew audit on the formula"
echo "  4) Exit"
echo
read -p "Enter your choice (1-4): " choice

case $choice in
    1)
        echo
        echo "Installing FalkorDB from formula..."
        echo "This may take several minutes to compile..."
        brew install --build-from-source "$FORMULA_FILE"
        
        echo
        echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
        echo "✓ Installation complete!"
        echo
        echo "To use FalkorDB, run:"
        echo "  redis-server --loadmodule \$(brew --prefix)/lib/falkordb.so"
        echo
        echo "Or add to your redis.conf:"
        echo "  loadmodule $(brew --prefix)/lib/falkordb.so"
        echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
        ;;
    2)
        echo
        echo "Building FalkorDB (without installing)..."
        brew install --only-dependencies "$FORMULA_FILE"
        
        TMPDIR=$(mktemp -d)
        cd "$TMPDIR"
        tar -xzf <(curl -L "$(grep 'url "' "$FORMULA_FILE" | sed 's/.*url "\(.*\)"/\1/')")
        cd FalkorDB-*
        
        echo "Building in: $(pwd)"
        if command -v gmake &> /dev/null; then
            gmake
        else
            make
        fi
        
        echo
        echo "✓ Build complete!"
        echo "Binary location: $(find . -name "falkordb.so" -type f)"
        ;;
    3)
        echo
        echo "Running brew audit..."
        brew audit --new "$FORMULA_FILE" || true
        echo
        echo "Note: Some audit warnings are expected for formulas outside of homebrew-core"
        ;;
    4)
        echo "Exiting..."
        exit 0
        ;;
    *)
        echo "Invalid choice"
        exit 1
        ;;
esac
