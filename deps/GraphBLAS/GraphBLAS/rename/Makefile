#-------------------------------------------------------------------------------
# GraphBLAS/GraphBLAS/rename/Makefile
#-------------------------------------------------------------------------------

# SuiteSparse:GraphBLAS, Timothy A. Davis, (c) 2017-2023, All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0

#-------------------------------------------------------------------------------

# Linux is required, but the result (GB_rename.h) is then used for all
# platforms.  This Makefile onstructs the GB_rename.h file from the compiled
# ../../build/libgraphblas.so, which is then used to compile
# libgraphblas_matlab.so with renamed symbols, to avoid conflicting with the
# built-in libmwgraphblas.so (v3.3.3) in MATLAB R2021a and later.

# GxB_*Iterator* methods are #define'd as both macros and functions in
# GraphBLAS.h, so they are not renamed.  Functions with "__" in their name are
# renamed using macros internally, and so they do not need to be renamed with
# GB_rename.h.

go:
	nm -gD ../../build/libgraphblas.so | grep -v "__" > lib
	grep " GrB_" lib > temp.h
	grep " GxB_" lib \
            | grep -v Iterator >> temp.h
	grep " GB_"  lib \
            | grep -v Iterator >> temp.h
	awk -f rename.awk < temp.h > temp2.h
	echo "#define GB_complex GM_complex" >> temp2.h
	echo "#define GB_complexf GM_complexf" >> temp2.h
	sort temp2.h > GB_rename.h
	rm temp.h temp2.h lib

# on the Mac, use "make mac"
mac:
	nm -gA ../../build/libgraphblas.dylib | grep -v "__" > lib
	grep " _GrB_" lib > temp.h
	grep " _GxB_" lib \
            | grep -v Iterator >> temp.h
	grep " _GB_"  lib \
            | grep -v Iterator >> temp.h
	awk -f rename.awk < temp.h | sort > GB_rename.h
	rm temp.h lib

