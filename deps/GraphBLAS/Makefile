#-------------------------------------------------------------------------------
# GraphBLAS/Makefile
#-------------------------------------------------------------------------------

# SuiteSparse:GraphBLAS, Timothy A. Davis, (c) 2017-2025, All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0

#-------------------------------------------------------------------------------

# simple Makefile for GraphBLAS, relies on cmake to do the actual build.  Use
# the CMAKE_OPTIONS argument to this Makefile to pass options to cmake.
# For example, to compile with 40 threads, use:
#
#       make JOBS=40
#
# To compile without using Google's cpu_features package, using 40 threads:
#
#       make CMAKE_OPTIONS='-DGBNCPUFEAT=1' JOBS=40
#
# To use multiple options, separate them by a space.  For example, to build
# just the library but not cpu_features, and to enable AVX2 but not AVX512F:
#
#       make CMAKE_OPTIONS='-DGBNCPUFEAT=1 -DGBAVX2=1' JOBS=40

JOBS ?= 8

F = -DSUITESPARSE_USE_FORTRAN=OFF

default: library

# default is to install only in /usr/local
library:
	( cd build && cmake $(F) $(CMAKE_OPTIONS) .. && cmake --build . --config Release -j${JOBS} )

# install only in SuiteSparse/lib and SuiteSparse/include
local:
	( cd build && cmake $(F) $(CMAKE_OPTIONS) -USUITESPARSE_PKGFILEDIR -DSUITESPARSE_LOCAL_INSTALL=1 .. && cmake --build . --config Release -j${JOBS} )

# install only in /usr/local (default)
global:
	( cd build && cmake $(F) $(CMAKE_OPTIONS) -USUITESPARSE_PKGFILEDIR -DSUITESPARSE_LOCAL_INSTALL=0 .. && cmake --build . --config Release -j${JOBS} )

# enable CUDA (NOTE: not ready for production use)
cuda:
	( cd build && cmake $(F) $(CMAKE_OPTIONS) -DGRAPHBLAS_USE_CUDA=1 .. && cmake --build . --config Release -j$(JOBS) )

# compile with -g 
debug:
	( cd build && cmake -DCMAKE_BUILD_TYPE=Debug $(F) $(CMAKE_OPTIONS) .. && cmake --build . --config Debug -j$(JOBS) )

# compile without FactoryKernels
compact:
	( cd build && cmake $(F) $(CMAKE_OPTIONS) -DGRAPHBLAS_COMPACT=1 .. && cmake --build . --config Release -j$(JOBS) )

# compile with -g, and without FactoryKernels
cdebug:
	( cd build && cmake -DCMAKE_BUILD_TYPE=Debug -DGRAPHBLAS_COMPACT=1 $(F) $(CMAKE_OPTIONS) .. && cmake --build . --config Debug -j$(JOBS) )

# build the dynamic library and the demos
all:
	( cd build && cmake $(F) $(CMAKE_OPTIONS) -DSUITESPARSE_DEMOS=1 .. && cmake --build . --config Release -j$(JOBS) )

# run the demos
demos: all
	./build/grow_demo < Demo/Matrix/west0067
	./build/simple_demo > ./build/simple_demo.out
	./build/complex_demo > ./build/complex_demo_out.m
	./build/complex_demo 1 > ./build/complex_demo_out2.m
	./build/wildtype_demo > ./build/wildtype_demo.out
	./build/gauss_demo > ./build/gauss_demo1.out
	./build/gauss_demo > ./build/gauss_demo.out
	./build/wathen_demo > ./build/wathen_demo.out

# diff the demo output
demodiffs: demos
	echo "Comparing demo output:"
	- diff -w Demo/Output/simple_demo.out       build/simple_demo.out
	- diff -w Demo/Output/complex_demo_out.m    build/complex_demo_out.m
	- diff -w Demo/Output/wildtype_demo.out     build/wildtype_demo.out
	- diff -w Demo/Output/gauss_demo.out        build/gauss_demo.out

# just do 'make' in build; do not rerun the cmake script
remake:
	( cd build && cmake --build . -j$(JOBS) )

# just run cmake; do not compile
setup:
	( cd build && cmake $(F) $(CMAKE_OPTIONS) .. )

# build the static library
static:
	( cd build && cmake $(F) $(CMAKE_OPTIONS) -DBUILD_STATIC_LIBS=ON -DBUILD_SHARED_LIBS=OFF .. && cmake --build . --config Release -j$(JOBS) )

# installs GraphBLAS to the install location defined by cmake, usually
# /usr/local/lib and /usr/local/include
install:
	( cd build && cmake --install . )

# create the Doc/GraphBLAS_UserGuide.pdf
docs:
	( cd Doc && $(MAKE) )

# compile the CUDA kernels
gpu:
	( cd CUDA && cmake --build . --config Release -j${JOBS} )

# remove any installed libraries and #include files
uninstall:
	- xargs rm < build/install_manifest.txt

clean: distclean

purge: distclean

# remove all files not in the distribution
distclean:
	- rm -rf build/* Demo/*.out Demo/complex_demo_out*.m Tcov/log.txt
	- rm -rf Config/*.tmp Source/control.m4
	- rm -rf Doc/html/* Doc/*.tmp
	- rm -rf JITpackage/build/*
	( cd GraphBLAS && $(MAKE) distclean )
	( cd Test && $(MAKE) distclean )
	( cd Tcov && $(MAKE) distclean )
	( cd Doc  && $(MAKE) distclean )

