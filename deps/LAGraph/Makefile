#-------------------------------------------------------------------------------
# LAGraph/Makefile
#-------------------------------------------------------------------------------

# LAGraph, (c) 2021-2022 by The LAGraph Contributors, All Rights Reserved.
# SPDX-License-Identifier: BSD-2-Clause
# See additional acknowledgments in the LICENSE file,
# or contact permission@sei.cmu.edu for the full terms.

#-------------------------------------------------------------------------------

# A simple Makefile for LAGraph, which relies on cmake to do the actual build.
# All the work is done in cmake so this Makefile is just for convenience.

# To compile and run the tests:
#
#       make
#       make test
#
# To compile with an alternate compiler:
#
#       make CC=gcc CXX=g++
#
# To compile/install for system-wide usage (typically in /usr/local):
#
#       make
#       sudo make install
#
# To compile/install for elsewhere (for example, in /home/me/mystuff/lib
# and /home/me/mystuff/include), do not use this Makefile.  Instead, do:
#
#       cd build
#       cmake -DCMAKE_INSTALL_PREFIX="/home/me/mystuff" ..
#       make
#       make install
#
# To clean up the files:
#
#       make clean
#
# To uninstall:
#
#       make uninstall
#
# To compile and run test coverage: use "make cov".  Next, open your browser to
# your local file, LAGraph/build/test_coverage/index.html.  Be sure to do "make
# clean" afterwards, and then "make" to compile without test coverage.

JOBS ?= 8

default: library

library:
	( cd build && cmake $(CMAKE_OPTIONS) .. && cmake --build . --config Release -j${JOBS} )

# install only in SuiteSparse/lib and SuiteSparse/include
local:
	( cd build && cmake $(CMAKE_OPTIONS) -USUITESPARSE_PKGFILEDIR -DSUITESPARSE_LOCAL_INSTALL=1 .. && cmake --build . --config Release -j${JOBS} )

# install CMAKE_INSTALL_PREFIX
global:
	( cd build && cmake $(CMAKE_OPTIONS) -USUITESPARSE_PKGFILEDIR -DSUITESPARSE_LOCAL_INSTALL=0 .. && cmake --build . --config Release -j${JOBS} )

vanilla:
	( cd build && cmake $(CMAKE_OPTIONS) -USUITESPARSE_PKGFILEDIR -DLAGRAPH_VANILLA=1 .. && cmake --build . --config Release -j${JOBS} )

# compile with -g for debugging
debug:
	( cd build && cmake $(CMAKE_OPTIONS) -DCMAKE_BUILD_TYPE=Debug .. && cmake --build . --config Release -j${JOBS} )

all: library

test: library
	( cd build && ctest . || ctest . --rerun-failed --output-on-failure )

verbose_test: library
	( cd build && ctest . --verbose || ctest . --rerun-failed --output-on-failure )

# target used in CI
demos: test

# just compile after running cmake; do not run cmake again
remake:
	( cd build && cmake --build . --config Release -j${JOBS} )

# just run cmake to set things up
setup:
	( cd build && cmake $(CMAKE_OPTIONS) .. )

install:
	( cd build && cmake --install . )

# remove any installed libraries and #include files
uninstall:
	- xargs rm < build/install_manifest.txt

# clean, compile, and run test coverage
cov: distclean
	( cd build && cmake -DCOVERAGE=1 .. && cmake --build . --config Release -j${JOBS} && cmake --build . --target test_coverage )

# remove all files not in the distribution
clean: distclean

purge: distclean

distclean:
	- $(RM) -rf build/* config/*.tmp

