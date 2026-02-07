---
name: Build FalkorDB
description: Build the FalkorDB Redis module with various configurations (debug, sanitizers, coverage, Valgrind)
---

# Build FalkorDB

Build the FalkorDB Redis module (`falkordb.so`) using Make + CMake + Cargo.

## Usage

Run `make` from the project root. The compiled module is output to `bin/<arch>/src/falkordb.so`.

## Build variants

    # Standard release build
    make

    # Debug build (unoptimized, with symbols)
    make DEBUG=1

    # AddressSanitizer (detects memory errors: buffer overflows, use-after-free)
    make SAN=address

    # ThreadSanitizer (detects data races)
    make SAN=thread

    # Build for Valgrind
    make VG=1

    # Build with code coverage (implies DEBUG=1)
    make COV=1

    # Force GCC toolchain (needed on macOS for OpenMP)
    make GCC=1

## Run the module

    # Start redis-server with falkordb.so loaded
    make run

## Build individual dependencies

    make deps                  # All dependencies
    make graphblas             # GraphBLAS
    make redisearch            # RediSearch
    make libcypher-parser      # Cypher parser
    make falkordbrs            # Rust components

## Clean

    make clean                 # Build products only
    make clean ALL=1           # Everything including bin/
    make clean DEPS=1          # Include dependency artifacts

## Notes

- Submodules must be initialized: `git clone --recurse-submodules -j8`
- macOS requires GCC for OpenMP: `brew install gcc g++`
- Ubuntu deps: `apt-get install build-essential cmake m4 automake peg libtool autoconf python3 python3-pip`
- Output binary location: `bin/<arch>/src/falkordb.so`
- Debug symbols are extracted to `falkordb.so.debug`
