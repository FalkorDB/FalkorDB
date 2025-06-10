
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

SRCDIR=$(ROOT)/deps/LAGraph

# build both liblagraph.a & liblagraphx.a
CMAKE_TARGETS=LAGraph_static LAGraphX_static

BINDIR=$(BINROOT)/LAGraph

TARGET=$(BINDIR)/src/liblagraph.a


#----------------------------------------------------------------------------------------------

include $(MK)/defs

define CMAKE_DEFS +=
	BUILD_STATIC_LIBS=ON
	BUILD_SHARED_LIBS=OFF
	GraphBLAS_ROOT=$(BINROOT)/GraphBLAS
	GRAPHBLAS_ROOT=$(ROOT)/deps/GraphBLAS
	CMAKE_POSITION_INDEPENDENT_CODE=on
	CMAKE_C_FLAGS=-fPIC
	CMAKE_CXX_FLAGS=-fPIC
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
