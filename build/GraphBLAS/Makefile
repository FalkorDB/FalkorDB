
ROOT=../..

MK.cmake=1

include $(ROOT)/deps/readies/mk/main

#----------------------------------------------------------------------------------------------

define HELPTEXT
make build          # compile and link
  DEBUG=1             # build for debugging
  WHY=1               # explain CMake decisions (in /tmp/cmake-why)
  FORCE=1             # Force CMake rerun (default)
  CMAKE_ARGS=...      # extra arguments to CMake
  SLOW=1              # do not parallelize build (for diagnostics)
make clean         # remove build artifacts
  ALL=1              # remove entire artifacts directory

endef

#----------------------------------------------------------------------------------------------

SRCDIR=$(ROOT)/deps/GraphBLAS

CMAKE_TARGETS=GraphBLAS_static

BINDIR=$(BINROOT)/GraphBLAS

TARGET=$(BINDIR)/libgraphblas.a

#----------------------------------------------------------------------------------------------

include $(MK)/defs

define CMAKE_DEFS +=
	BUILD_STATIC_LIBS=ON
	BUILD_SHARED_LIBS=OFF
	BUILD_TESTING=off
	CMAKE_POSITION_INDEPENDENT_CODE=on
	GRAPHBLAS_USE_JIT=off
	CMAKE_C_FLAGS=-fno-stack-protector
	CMAKE_CXX_FLAGS=-fno-stack-protector
endef

#----------------------------------------------------------------------------------------------

MK_CUSTOM_CLEAN=1

include $(MK)/rules

#----------------------------------------------------------------------------------------------

clean:
ifeq ($(ALL),1)
	$(SHOW)rm -rf $(BINDIR)
else
	$(SHOW)$(MAKE) -C $(BINDIR) clean
endif

.PHONY: clean
