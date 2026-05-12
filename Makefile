# FalkorDB Makefile - Wrapper for build.sh
#
# This Makefile provides a familiar make interface while delegating
# all actual build operations to build.sh

.PHONY: all build deps clean pack package test unit-tests flow-tests tck-tests \
        fuzz-tests benchmark coverage help run

#----------------------------------------------------------------------------------------------
# Default target
#----------------------------------------------------------------------------------------------

all: build

#----------------------------------------------------------------------------------------------
# Help text
#----------------------------------------------------------------------------------------------

define HELPTEXT
make all            # Build everything
  DEBUG=1             # Build for debugging
  SLOW=1              # Disable parallel build
  STATIC_OMP=1        # Link OpenMP statically
  VARIANT=name        # Add `name` to build products directory
  GCC=1               # Build with GCC toolchain (default for Linux)
  CLANG=1             # Build with CLang toolchain (default for macOS)
  COV=1               # Build for coverage analysis (implies DEBUG=1)
  VG=1|docker         # build for Valgrind
  SAN=type            # build with LLVM sanitizer (type=address|memory|leak|thread)
make clean          # Clean build products
  ALL=1               # Completely remove build products
  DEPS=1              # Also clean dependant modules
  AUTOGEN=1           # Remove autogen-generated files
make pack|package   # Build RAMP packages
make run            # Run redis-server with falkordb
  GDB=1               # Run with debugger

make test         # Run tests
  LIST=1            # List all tests, do not execute
  COV=1             # Perform coverage analysis
  SLOW=1            # Do not run in parallel
  PARALLEL=n        # Set testing parallelism
  GDB=1             # RLTest interactive debugging
  TEST=test         # Run specific test
  TESTFILE=file     # Run tests listed in file
  FAILFILE=file     # Write failed tests to file

make unit-tests     # Run unit tests
make flow-tests     # Run flow tests
make tck-tests      # Run TCK tests
make fuzz-tests     # Run fuzz tester
  TIMEOUT=secs      # Timeout in `secs`

make benchmark        # Run benchmarks

make micro_benchmarks # Run micro benchmarks
  BENCH=file          # Run a specific benchmark file

make coverage     # Perform coverage analysis (build & test)

endef

export HELPTEXT

help:
	@echo "$$HELPTEXT"

#----------------------------------------------------------------------------------------------
# Build configuration variables passed to build.sh
#----------------------------------------------------------------------------------------------

BUILD_ARGS :=

ifdef DEBUG
BUILD_ARGS += DEBUG=$(DEBUG)
endif

ifdef COV
BUILD_ARGS += COV=$(COV)
endif

ifdef PROFILE
BUILD_ARGS += PROFILE=$(PROFILE)
endif

ifdef VG
BUILD_ARGS += VG=$(VG)
endif

ifdef SAN
BUILD_ARGS += SAN=$(SAN)
endif

ifdef FORCE
BUILD_ARGS += FORCE=$(FORCE)
endif

ifdef VERBOSE
BUILD_ARGS += VERBOSE=$(VERBOSE)
endif

ifdef SLOW
BUILD_ARGS += SLOW=$(SLOW)
endif

ifdef STATIC_OMP
BUILD_ARGS += STATIC_OMP=$(STATIC_OMP)
endif

ifdef VARIANT
BUILD_ARGS += VARIANT=$(VARIANT)
endif

ifdef GCC
BUILD_ARGS += GCC=$(GCC)
endif

ifdef CLANG
BUILD_ARGS += CLANG=$(CLANG)
endif

ifdef JIT
BUILD_ARGS += JIT=$(JIT)
endif

ifdef PARALLEL
BUILD_ARGS += PARALLEL=$(PARALLEL)
endif

ifdef TEST
BUILD_ARGS += TEST=$(TEST)
endif

ifdef TESTFILE
BUILD_ARGS += TESTFILE=$(TESTFILE)
endif

ifdef FAILFILE
BUILD_ARGS += FAILFILE=$(FAILFILE)
endif

ifdef GDB
BUILD_ARGS += GDB=$(GDB)
endif

ifdef LIST
BUILD_ARGS += LIST=$(LIST)
endif

ifdef TIMEOUT
BUILD_ARGS += TIMEOUT=$(TIMEOUT)
endif

ifdef BENCHMARK_GROUP
BUILD_ARGS += BENCHMARK_GROUP=$(BENCHMARK_GROUP)
endif

ifdef BUILD_BENCHMARKS
BUILD_ARGS += BUILD_BENCHMARKS=$(BUILD_BENCHMARKS)
endif

#----------------------------------------------------------------------------------------------
# Build targets
#----------------------------------------------------------------------------------------------

build:
	@./build.sh $(BUILD_ARGS)

deps:
	@./build.sh $(BUILD_ARGS)

#----------------------------------------------------------------------------------------------
# Clean targets
#----------------------------------------------------------------------------------------------

clean:
ifdef ALL
	@./build.sh CLEAN=1 ALL=1 $(BUILD_ARGS)
else ifdef DEPS
	@./build.sh CLEAN=1 DEPS=1 $(BUILD_ARGS)
else ifdef AUTOGEN
	@./build.sh CLEAN=1 AUTOGEN=1 $(BUILD_ARGS)
else
	@./build.sh CLEAN=1 $(BUILD_ARGS)
endif

#----------------------------------------------------------------------------------------------
# Package targets
#----------------------------------------------------------------------------------------------

pack package:
	@./build.sh PACK=1 $(BUILD_ARGS)

#----------------------------------------------------------------------------------------------
# Test targets
#----------------------------------------------------------------------------------------------

test:
	@./build.sh RUN_TESTS=1 $(BUILD_ARGS)

unit-tests:
	@./build.sh RUN_UNIT_TESTS=1 $(BUILD_ARGS)

flow-tests:
	@./build.sh RUN_FLOW_TESTS=1 $(BUILD_ARGS)

tck-tests:
	@./build.sh RUN_TCK_TESTS=1 $(BUILD_ARGS)

fuzz-tests fuzz:
	@./build.sh RUN_FUZZ_TESTS=1 $(BUILD_ARGS)

#----------------------------------------------------------------------------------------------
# Benchmark and coverage targets
#----------------------------------------------------------------------------------------------

micro-benchmark:
	@./build.sh RUN_MICRO_BENCHMARKS=1 $(BUILD_ARGS)

benchmark:
	@./build.sh BENCHMARK=1 $(BUILD_ARGS)

coverage:
	@./build.sh COV=1 RUN_TESTS=1 $(BUILD_ARGS)

#----------------------------------------------------------------------------------------------
# Run target
#----------------------------------------------------------------------------------------------

run:
	@./build.sh RUN=1 $(BUILD_ARGS)
