#-------------------------------------------------------------------------------
# GraphBLAS/GraphBLAS/Makefile: compile libgraphblas_matlab for use in MATLAB
#-------------------------------------------------------------------------------

# SuiteSparse:GraphBLAS, Timothy A. Davis, (c) 2017-2023, All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0

#-------------------------------------------------------------------------------

# simple Makefile for GraphBLAS, relies on cmake to do the actual build.  Use
# the CMAKE_OPTIONS argument to this Makefile to pass options to cmake.

JOBS ?= 8

default: library

# default is to install only in /usr/local
library:
	( cd build && cmake $(CMAKE_OPTIONS) .. && cmake --build . --config Release -j${JOBS} )

# install only in SuiteSparse/lib and SuiteSparse/include
local:
	( cd build && cmake $(CMAKE_OPTIONS) -DLOCAL_INSTALL=1 .. && cmake --build . --config Release -j${JOBS} )

# install only in /usr/local (default)
global:
	( cd build && cmake $(CMAKE_OPTIONS) -DLOCAL_INSTALL=0 .. && cmake --build . --config Release -j${JOBS} )

# compile with -g 
debug:
	( cd build && cmake -DCMAKE_BUILD_TYPE=Debug $(CMAKE_OPTIONS) .. && cmake --build . --config Debug -j$(JOBS) )

# compile without FactoryKernels
compact:
	( cd build && cmake $(CMAKE_OPTIONS) -DCOMPACT=1 .. && cmake --build . --config Release -j$(JOBS) )

# compile with -g, and without FactoryKernels
cdebug:
	( cd build && cmake -DCMAKE_BUILD_TYPE=Debug -DCOMPACT=1 $(CMAKE_OPTIONS) .. && cmake --build . --config Debug -j$(JOBS) )

# just do 'make' in build; do not rerun the cmake script
remake:
	( cd build && cmake --build . -j$(JOBS) )

# just run cmake; do not compile
setup:
	( cd build ; cmake $(CMAKE_OPTIONS) .. ; )

# installs GraphBLAS to the install location defined by cmake, usually
# /usr/local/lib and /usr/local/include
install:
	( cd build && cmake --install . )

# remove any installed libraries and #include files
uninstall:
	- xargs rm < build/install_manifest.txt

clean: distclean

purge: distclean

# remove all files not in the distribution
distclean:
	rm -rf build/*
	( cd test/tcov ; $(MAKE) distclean )
	( cd @GrB/private ; $(MAKE) distclean )

