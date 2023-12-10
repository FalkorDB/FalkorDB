#-------------------------------------------------------------------------------
# GraphBLAS/Package/Makefile
#-------------------------------------------------------------------------------

# SuiteSparse:GraphBLAS, Timothy A. Davis, (c) 2017-2023, All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0

#-------------------------------------------------------------------------------

go:
	( cd build && cmake $(CMAKE_OPTIONS) .. && cmake --build . )
	./build/grb_jitpackage ../Include/GraphBLAS.h \
            ../Source/Template/*.[ch] ../Source/JitKernels/*.[ch] \
            ../Source/Shared/*.h

clean: distclean

purge: distclean

# remove all files not in the distribution
distclean:
	- rm -rf build/*

