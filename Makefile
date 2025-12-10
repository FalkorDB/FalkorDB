
ifeq ($(VG),docker)
override VG:=1
VG_DOCKER=1
endif

ifeq ($(VG),1)
export MEMCHECK=1
endif

ifneq ($(SAN),)
export MEMCHECK=1
endif

#----------------------------------------------------------------------------------------------

.NOTPARALLEL:

ROOT=.

MK.cmake=1
SRCDIR=.

include $(ROOT)/deps/readies/mk/main

MK_ALL_TARGETS=bindirs deps build

MK_CUSTOM_CLEAN=1

BINDIR=$(BINROOT)/src
export TARGET=$(BINROOT)/src/falkordb.so

#----------------------------------------------------------------------------------------------

define HELPTEXT
make all            # Build everything
  DEBUG=1             # Build for debugging
  SLOW=1              # Disable parallel build
  STATIC_OMP=1        # Link OpenMP statically
  VARIANT=name        # Add `name` to build products directory
  GCC=1               # Build with GCC toolchain (default nor Linux)
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
  HELP=1            # Show testing options
  LIST=1            # List all tests, do not execute
  UNIT=1            # Run unit tests
  FLOW=1            # Run flow tests (Python)
  TCK=1             # Run TCK framework tests
  UPGRADE=1         # Run upgrade tests
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
make upgrade-tests  # Run upgrade tests
make fuzz-tests     # Run fuzz tester
  TIMEOUT=secs      # Timeout in `secs`

make benchmark    # Run benchmarks

make coverage     # Perform coverage analysis (build & test)
make cov-upload   # Upload coverage data to codecov.io

make upload-artifacts   # copy snapshot packages to S3
  OSNICK=nick             # copy snapshots for specific OSNICK
make upload-release     # copy release packages to S3

common options for upload operations:
  STAGING=1   # copy to staging lab area (for validation)
  FORCE=1     # allow operation outside CI environment
  VERBOSE=1   # show more details
  NOP=1       # do not copy, just print commands

make docker     # build for specified platform
  OSNICK=nick     # platform to build for (default: host platform)
  TEST=1          # run tests after build
  PACK=1          # create package
  ARTIFACTS=1     # copy artifacts to host

endef

#----------------------------------------------------------------------------------------------

ifeq ($(MEMCHECK),1)
DEPS_BINDIR=$(ROOT)/bin/$(FULL_VARIANT)
DEPS_DEBUG=1
CMAKE_DEFS += MEMCHECK=ON
else
export DEPS_BINDIR:=$(ROOT)/bin/$(FULL_VARIANT.release)
DEPS_DEBUG=
endif

#----------------------------------------------------------------------------------------------

RAX_DIR = $(ROOT)/deps/rax
export RAX_BINDIR=$(DEPS_BINDIR)/rax
include $(ROOT)/build/rax/Makefile.defs

LIBXXHASH_DIR = $(ROOT)/deps/xxHash
export LIBXXHASH_BINDIR=$(DEPS_BINDIR)/xxHash
include $(ROOT)/build/xxHash/Makefile.defs

LIBCURL_DIR = $(ROOT)/deps/libcurl
export LIBCURL_BINDIR=$(DEPS_BINDIR)/libcurl
include $(ROOT)/build/libcurl/Makefile.defs

LIBCSV_DIR = $(ROOT)/deps/libcsv
export LIBCSV_BINDIR=$(DEPS_BINDIR)/libcsv
include $(ROOT)/build/libcsv/Makefile.defs

LIBCYPHER_PARSER_DIR = $(ROOT)/deps/libcypher-parser
LIBCYPHER_PARSER_SRCDIR = $(LIBCYPHER_PARSER_DIR)/lib/src
export LIBCYPHER_PARSER_BINDIR=$(DEPS_BINDIR)/libcypher-parser
include $(ROOT)/build/libcypher-parser/Makefile.defs

GRAPHBLAS_DIR = $(ROOT)/deps/GraphBLAS
export GRAPHBLAS_BINDIR=$(DEPS_BINDIR)/GraphBLAS
include $(ROOT)/build/GraphBLAS/Makefile.defs

LAGRAPH_DIR = $(ROOT)/deps/LAGraph
export LAGRAPH_BINDIR=$(DEPS_BINDIR)/LAGraph
include $(ROOT)/build/LAGraph/Makefile.defs

UTF8PROC_DIR = $(ROOT)/deps/utf8proc
export UTF8PROC_BINDIR=$(DEPS_BINDIR)/utf8proc
include $(ROOT)/build/utf8proc/Makefile.defs

ONIGURUMA_DIR = $(ROOT)/deps/oniguruma
export ONIGURUMA_BINDIR=$(DEPS_BINDIR)/oniguruma
include $(ROOT)/build/oniguruma/Makefile.defs

REDISEARCH_DIR = $(ROOT)/deps/RediSearch
export REDISEARCH_BINROOT=$(BINROOT)
include $(ROOT)/build/RediSearch/Makefile.defs

FalkorDBRS_DIR = $(ROOT)/deps/FalkorDB-core-rs
export FalkorDBRS_BINDIR=$(BINROOT)/FalkorDB-core-rs
include $(ROOT)/build/FalkorDB-core-rs/Makefile.defs

BIN_DIRS += $(REDISEARCH_BINROOT)/search-static

LIBS=$(RAX) $(LIBXXHASH) $(GRAPHBLAS) $(LAGRAPH) $(REDISEARCH_LIBS) $(LIBCURL) $(LIBCSV) $(LIBCYPHER_PARSER) $(UTF8PROC) $(ONIGURUMA) $(FalkorDBRS)

#----------------------------------------------------------------------------------------------

CC_COMMON_H=$(SRCDIR)/src/common.h

CC_C_STD=gnu11
CC_OPENMP=1

# Add zstd library for Alpine/musl builds (libcurl dependency)
# On Alpine and other musl-based systems, libcurl is often built with zstd support
# and we need to explicitly link it. Detect both via OSNICK and libc.
USING_MUSL := $(shell ldd --version 2>&1 | grep -q musl && echo 1 || echo 0)
ifeq ($(USING_MUSL),1)
LD_LIBS.ext += zstd
else ifeq ($(findstring alpine,$(OSNICK)),alpine)
LD_LIBS.ext += zstd
endif

include $(MK)/defs

$(info # Building into $(BINDIR))
$(info # Using CC=$(CC))

ifeq ($(UNIT_TESTS),1)
CMAKE_DEFS += UNIT_TESTS:BOOL=on
endif

#----------------------------------------------------------------------------------------------

MISSING_DEPS:=

ifeq ($(wildcard $(RAX)),)
MISSING_DEPS += $(RAX)
endif

ifeq ($(wildcard $(LIBCURL)),)
MISSING_DEPS += $(LIBCURL)
endif

ifeq ($(wildcard $(LIBCSV)),)
MISSING_DEPS += $(LIBCSV)
endif

ifeq ($(wildcard $(LIBXXHASH)),)
MISSING_DEPS += $(LIBXXHASH)
endif

ifeq ($(wildcard $(GRAPHBLAS)),)
MISSING_DEPS += $(GRAPHBLAS)
endif

ifeq ($(wildcard $(LAGRAPH)),)
MISSING_DEPS += $(LAGRAPH)
endif

ifeq ($(wildcard $(LIBCYPHER_PARSER)),)
MISSING_DEPS += $(LIBCYPHER_PARSER)
endif

ifeq ($(wildcard $(UTF8PROC)),)
MISSING_DEPS += $(UTF8PROC)
endif

ifeq ($(wildcard $(ONIGURUMA)),)
MISSING_DEPS += $(ONIGURUMA)
endif

ifneq ($(call files_missing,$(REDISEARCH_LIBS)),)
MISSING_DEPS += $(REDISEARCH_LIBS)
endif

MISSING_DEPS += falkordbrs

ifneq ($(MISSING_DEPS),)
DEPS=1
endif

DEPENDENCIES=libcypher-parser graphblas lagraph libcurl libcsv redisearch rax libxxhash utf8proc oniguruma falkordbrs

ifneq ($(filter all deps $(DEPENDENCIES) pack,$(MAKECMDGOALS)),)
DEPS=1
endif

.PHONY: deps $(DEPENDENCIES)

#----------------------------------------------------------------------------------------------

RUN_CMD=redis-server --loadmodule $(abspath $(TARGET))

#----------------------------------------------------------------------------------------------

MK_ALL_TARGETS=bindirs deps build

.PHONY: all deps clean lint format pack run tests unit_tests flow_tests docker bindirs

all: bindirs $(TARGET)

include $(MK)/rules

#----------------------------------------------------------------------------------------------

ifeq ($(DEPS),1)

deps: $(LIBCURL) $(LIBCSV) $(LIBCYPHER_PARSER) $(GRAPHBLAS) $(LAGRAPH) $(LIBXXHASH) $(RAX) $(REDISEARCH_LIBS) $(UTF8PROC) $(ONIGURUMA) falkordbrs

libxxhash: $(LIBXXHASH)

$(LIBXXHASH):
	@echo Building $@ ...
	$(SHOW)$(MAKE) --no-print-directory -C $(ROOT)/build/xxHash DEBUG=$(DEPS_DEBUG)

rax: $(RAX)

$(RAX):
	@echo Building $@ ...
	$(SHOW)$(MAKE) --no-print-directory -C $(ROOT)/build/rax DEBUG=$(DEPS_DEBUG)

graphblas: $(GRAPHBLAS)

GRAPHBLAS_MAKE_FLAGS.xenial-x64=CC=gcc-5 CXX=gxx-5

$(GRAPHBLAS):
	@echo Building $@ ...
	$(SHOW)$(MAKE) --no-print-directory -C $(ROOT)/build/GraphBLAS DEBUG=$(DEPS_DEBUG) $(GRAPHBLAS_MAKE_FLAGS.$(OSNICK)-$(ARCH)) JIT=$(JIT)

lagraph: $(LAGRAPH)

$(LAGRAPH):
	@echo Building $@ ...
	$(SHOW)$(MAKE) --no-print-directory -C $(ROOT)/build/LAGraph DEBUG=$(DEPS_DEBUG) $(LAGRAPH_MAKE_FLAGS.$(OSNICK)-$(ARCH))

libcypher-parser: $(LIBCYPHER_PARSER)

$(LIBCYPHER_PARSER):
	@echo Building $@ ...
	$(SHOW)$(MAKE) --no-print-directory -C $(ROOT)/build/libcypher-parser DEBUG=$(DEPS_DEBUG)

libcurl: $(LIBCURL)

$(LIBCURL):
	@echo Building $@ ...
	$(SHOW)$(MAKE) --no-print-directory -C $(ROOT)/build/libcurl autoreconf DEBUG=$(DEPS_DEBUG)
	$(SHOW)$(MAKE) --no-print-directory -C $(ROOT)/build/libcurl DEBUG=$(DEPS_DEBUG)

libcsv: $(LIBCSV)

$(LIBCSV):
	@echo Building $@ ...
	$(SHOW)$(MAKE) --no-print-directory -C $(ROOT)/build/libcsv autoreconf DEBUG=$(DEPS_DEBUG)
	$(SHOW)$(MAKE) --no-print-directory -C $(ROOT)/build/libcsv DEBUG=$(DEPS_DEBUG)

utf8proc: $(UTF8PROC)

$(UTF8PROC):
	@echo Building $@ ...
	$(SHOW)$(MAKE) --no-print-directory -C $(ROOT)/build/utf8proc DEBUG=$(DEPS_DEBUG)

oniguruma: $(ONIGURUMA)

$(ONIGURUMA):
	@echo Building $@ ...
	$(SHOW)$(MAKE) --no-print-directory -C $(ROOT)/build/oniguruma DEBUG=$(DEPS_DEBUG)

redisearch: $(REDISEARCH_LIBS)

$(REDISEARCH_LIBS):
	@echo Building $@ ...
	$(SHOW)$(MAKE) -C $(REDISEARCH_DIR) STATIC=1 BINROOT=$(REDISEARCH_BINROOT) CC=$(CC) CXX=$(CXX)


ifneq ($(DEBUG),1)
CARGO_FLAGS=--release
endif

ifneq ($(SAN),)
export RUSTFLAGS=-Zsanitizer=$(SAN)
CARGO_FLAGS=--target x86_64-unknown-linux-gnu
endif

ifneq ($(COV),)
export RUSTFLAGS=-C instrument-coverage
endif

falkordbrs:
	@echo Building $@ ...
	cd deps/FalkorDB-core-rs && cargo build $(CARGO_FLAGS) --features falkordb_allocator --target-dir $(FalkorDBRS_BINDIR)

.PHONY: libcypher-parser graphblas lagraph libcurl libcsv redisearch libxxhash rax utf8proc oniguruma falkordbrs

#----------------------------------------------------------------------------------------------

else

deps: ;

endif # DEPS

.PHONY: deps

#----------------------------------------------------------------------------------------------

$(LIBCYPHER_PARSER_BINDIR)/lib/src/cypher-parser.h : $(LIBCYPHER_PARSER)

#----------------------------------------------------------------------------------------------

clean:
ifeq ($(ALL),1)
	$(SHOW)-rm -rf $(BINROOT) $(DEPS_BINDIR)
	$(SHOW)$(MAKE) -C $(ROOT)/build/libcypher-parser clean ALL=1
else
	$(SHOW)$(MAKE) -C $(BINDIR) clean
	$(SHOW)-rm -fr $(TARGET).debug $(BINDIR)/CMakeCache.txt $(BINDIR)/tests
ifeq ($(DEPS),1)
	$(SHOW)$(MAKE) -C $(ROOT)/build/rax clean DEBUG=$(DEPS_DEBUG)
	$(SHOW)$(MAKE) -C $(ROOT)/build/xxHash clean DEBUG=$(DEPS_DEBUG)
	$(SHOW)$(MAKE) -C $(ROOT)/build/utf8proc clean DEBUG=$(DEPS_DEBUG)
	$(SHOW)$(MAKE) -C $(ROOT)/build/oniguruma clean DEBUG=$(DEPS_DEBUG)
	$(SHOW)$(MAKE) -C $(ROOT)/build/GraphBLAS clean DEBUG=$(DEPS_DEBUG)
	$(SHOW)$(MAKE) -C $(ROOT)/build/LAGraph clean DEBUG=$(DEPS_DEBUG)
	$(SHOW)$(MAKE) -C $(ROOT)/build/libcurl clean DEBUG=$(DEPS_DEBUG)
	$(SHOW)$(MAKE) -C $(ROOT)/build/libcsv clean DEBUG=$(DEPS_DEBUG)
	$(SHOW)$(MAKE) -C $(ROOT)/build/libcypher-parser clean DEBUG=$(DEPS_DEBUG)
	$(SHOW)$(MAKE) -C $(REDISEARCH_DIR) clean ALL=1 BINROOT=$(REDISEARCH_BINROOT)
endif
endif

clean-libcypher-parser:
	$(SHOW)$(MAKE) -C $(ROOT)/build/libcypher-parser clean ALL=1 AUTOGEN=1

clean-search:
ifeq ($(ALL),1)
	$(SHOW)rm -rf $(REDISEARCH_BINROOT)/search-static
else
	$(SHOW)$(MAKE) -C $(REDISEARCH_DIR) clean BINROOT=$(REDISEARCH_BINROOT)
endif

.PHONY: clean clean-libcypher-parser clean-search

#----------------------------------------------------------------------------------------------

pack package: #$(TARGET)
	@MODULE=$(realpath $(TARGET)) $(ROOT)/sbin/pack.sh

upload-release:
	$(SHOW)RELEASE=1 ./sbin/upload-artifacts

upload-artifacts:
	$(SHOW)SNAPSHOT=1 ./sbin/upload-artifacts

.PHONY: pack package upload-artifacts upload-release

#----------------------------------------------------------------------------------------------

ifeq ($(SLOW),1)
_RLTEST_PARALLEL=0
else ifneq ($(PARALLEL),)
_RLTEST_PARALLEL=$(PARALLEL)
else
_RLTEST_PARALLEL=1
endif

ifneq ($(BUILD),0)
TEST_DEPS=$(TARGET)
endif

test: unit-tests flow-tests tck-tests upgrade-tests

unit-tests:
ifneq ($(BUILD),0)
	$(SHOW)$(MAKE) build FORCE=1 UNIT_TESTS=1
endif
	$(SHOW)BINROOT=$(BINROOT) ./tests/unit/tests.sh
	$(SHOW)BINROOT=$(BINROOT) cargo test --lib --target-dir $(FalkorDBRS_BINDIR)

flow-tests: $(TEST_DEPS)
	$(SHOW)MODULE=$(TARGET) BINROOT=$(BINROOT) PARALLEL=$(_RLTEST_PARALLEL) GEN=$(GEN) AOF=$(AOF) TCK=0 UPGRADE=0 ./tests/flow/tests.sh

upgrade-tests: $(TEST_DEPS)
	$(SHOW)MODULE=$(TARGET) BINROOT=$(BINROOT) PARALLEL=$(_RLTEST_PARALLEL) GEN=0 AOF=0 TCK=0 SLOW=1 UPGRADE=1 ./tests/flow/tests.sh

tck-tests: $(TEST_DEPS)
	$(SHOW)MODULE=$(TARGET) BINROOT=$(BINROOT) PARALLEL=$(_RLTEST_PARALLEL) GEN=0 AOF=0 TCK=1 UPGRADE=0 ./tests/flow/tests.sh

.PHONY: test unit-tests flow-tests tck-tests upgrade-tests

#----------------------------------------------------------------------------------------------

ifneq ($(TIMEOUT),)
FUZZ_ARGS=-t $(TIMEOUT)
endif

fuzz fuzz-tests: $(TARGET)
	$(SHOW)cd tests/fuzz && ./process.py -m $(TARGET) $(FUZZ_ARGS)

.PHONY: fuzz fuzz-tests

#----------------------------------------------------------------------------------------------

benchmark: $(TARGET)
	$(SHOW)cd tests/benchmarks && python3 -m venv venv && source venv/bin/activate && pip install -r benchmarks_requirements.txt && python3 run_benchmarks.py group_a && python3 run_benchmarks.py group_b

.PHONY: benchmark

#----------------------------------------------------------------------------------------------

COV_EXCLUDE_DIRS += \
	deps/GraphBLAS \
	deps/LAGraph \
	deps/libcurl \
	deps/libcsv \
	deps/libcypher-parser \
	deps/oniguruma \
	deps/rax \
	deps/RediSearch \
	deps/utf8proc \
	deps/xxHash \
	src/util/sds \
	tests

COV_EXCLUDE+=$(foreach D,$(COV_EXCLUDE_DIRS),'$(realpath $(ROOT))/$(D)/*')

coverage:
	$(SHOW)$(MAKE) build COV=1
	$(SHOW)$(COVERAGE_RESET.llvm)
	-$(SHOW)$(MAKE) unit-tests COV=1
	-$(SHOW)$(MAKE) flow-tests COV=1
	-$(SHOW)$(MAKE) tck-tests COV=1
	$(SHOW)$(COVERAGE_COLLECT_REPORT.llvm)

.PHONY: coverage

#----------------------------------------------------------------------------------------------

docker:
	$(SHOW)$(MAKE) -C build/docker

ifneq ($(wildcard /w/*),)
SANBOX_ARGS += -v /w:/w
endif

sanbox:
	@docker run -it -v $(PWD):/build -w /build --cap-add=SYS_PTRACE --security-opt seccomp=unconfined $(SANBOX_ARGS) redisfab/clang:16-$(ARCH)-bullseye bash

.PHONY: box sanbox
