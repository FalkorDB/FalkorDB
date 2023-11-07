LZ4 Library, Copyright (c) 2011-2016, Yann Collet, All rights reserved.
SPDX-License-Identifier: BSD-2-clause

Notes by Tim Davis, on inclusion of LZ4 into SuiteSparse:GraphBLAS:

This directory contains a minimal copy of lz4 v1.9.3, from
https://github.com/lz4/lz4.git and http://www.lz4.org, by Yann Collet.  See
./LICENSE and ./README.md for more details.  Four source files are used in
SuiteSparse:GraphBLAS: lz4/lib/lz4.[ch], and lz4/lib/lz4hc.[ch] copied here
unmodified from v1.9.3, which are under the BSD-2-clause license.

Files in this folder:

    LICENSE         BSD 2-clause, Copyright (c) 2011-2016, Yann Collet
    lz4.c           extremely fast with good compression 
    lz4.h           include file for lz4.c
    lz4hc.c         slow compression, slightly more compact than lz4
    lz4hc.h         include file for lz4hc.c
    README_lz4.md   lz4/README.md
    README.md       lz4/lib/README.md
    README.txt      this file

When LZ4 is compiled for use in SuiteSparse:GraphBLAS,
LZ4_USER_MEMORY_FUNCTIONS is defined, and LZ4_malloc, LZ4_calloc, and LZ4_free,
are provided to LZ4.  Those functions use whatever memory manager is given to
GxB_init, or the C11 malloc/calloc/free for GrB_init.

This compile-time change could cause a conflict if LZ4 is also installed as a
system-wide library.  To avoid the conflict, all LZ4 function names are renamed
to GB_LZ4_*, using #defines in ../Source/GB_lz4.c.  SuiteSparse:GraphBLAS will
use this version of LZ4, integrated into the libgraphblas.so (.dylib, .dll),
rather than a separate liblz4.so library.

