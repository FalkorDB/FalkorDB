#-------------------------------------------------------------------------------
# GraphBLAS/Package/Makefile
#-------------------------------------------------------------------------------

# SuiteSparse:GraphBLAS, Timothy A. Davis, (c) 2017-2023, All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0

#-------------------------------------------------------------------------------

go:
	( cd build && cmake $(CMAKE_OPTIONS) .. && cmake --build . && cmake --build . --target GB_JITpackage )

clean: distclean

purge: distclean

# remove all files not in the distribution
distclean:
	- rm -rf build/*

