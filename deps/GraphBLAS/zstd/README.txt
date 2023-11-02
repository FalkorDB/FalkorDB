ZSTD Library, Copyright (c) Copyright (c) 2016-present, Facebook, Inc. All rights reserved.
SPDX-License-Identifier: BSD-3-clause

Notes by Tim Davis, on inclusion of ZSTD into SuiteSparse:GraphBLAS:

This directory contains a minimal copy of zstd v1.5.3, from
https://github.com/facebook/zstd.git, by Yann Collet.
See ./LICENSE and ./README_zstd.md for more details.

Files in this folder:

    LICENSE         BSD 3-clause, Copyright (c) 2016-present, Facebook, Inc.
    zstd_subset     a subset of zstd (common, compress, decompress),
                    and their include files
    README_zstd.md  zstd/README.md
    README.txt      this file

When ZSTD is compiled for use in SuiteSparse:GraphBLAS, ZSTD_DEPS_MALLOC is
defined, and ZSTD_malloc, ZSTD_calloc, and ZSTD_free, are provided to ZSTD
instead of have it use the standard C11 malloc/calloc/free.  Those
functions use whatever memory manager is given to GxB_init, or the C11
malloc/calloc/free for GrB_init.

This compile-time change could cause a conflict if ZSTD is also installed as a
system-wide library.  To avoid the conflict, all ZSTD function names are renamed
to GB_ZSTD_*, using #defines in ../Source/GB_zstd.c.  SuiteSparse:GraphBLAS will
use this version of ZSTD, integrated into the libgraphblas.so (.dylib, .dll),
rather than a separate libzstd.so library.

