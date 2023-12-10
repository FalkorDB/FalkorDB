//------------------------------------------------------------------------------
// GB_zstd: wrapper for the ZSTD compression library
//------------------------------------------------------------------------------

// SuiteSparse:GraphBLAS, Timothy A. Davis, (c) 2017-2023, All Rights Reserved.
// SPDX-License-Identifier: Apache-2.0

//------------------------------------------------------------------------------

// GB_zstd is a wrapper for the ZSTD compression library.  The
// ZSTD library is compiled with ZSTD_DEP enabled (which is not
// the default), and configured to use the SuiteSparse:GraphBLAS functions
// in place of malloc/calloc/free.

#include "GB.h"
#include "GB_serialize.h"
#include "GB_zstd.h"

void *ZSTD_malloc (size_t s)
{
    return (GB_Global_malloc_function (s)) ;
}

void *ZSTD_calloc (size_t n, size_t s)
{
    // ns = n*s, the size of the space to allocate
    size_t ns = 0 ;
    bool ok = GB_size_t_multiply (&ns, n, s) ;
    if (!ok) return (NULL) ;
    // malloc the space and then use memset to clear it
    void *p = GB_Global_malloc_function (ns) ;
    if (p != NULL) memset (p, 0, ns) ;
    return (p) ;
}

void ZSTD_free (void *p)
{
    GB_Global_free_function (p) ;
}

// ZSTD uses switch statements with no default case.
#if GB_COMPILER_GCC
#pragma GCC diagnostic ignored "-Wswitch-default"
#endif

// Include the unmodified zstd, version 1.5.3.  This
// allows the ZSTD_* functions to be renamed via GB_zstd.h, and avoids any
// conflict with the original -lzstd, which might be linked in by the user
// application.

#include "zstd_subset/common/debug.c"
#include "zstd_subset/common/entropy_common.c"
#include "zstd_subset/common/error_private.c"
#include "zstd_subset/common/fse_decompress.c"
#include "zstd_subset/common/pool.c"
#include "zstd_subset/common/threading.c"
#include "zstd_subset/common/xxhash.c"
#include "zstd_subset/common/zstd_common.c"

#include "zstd_subset/compress/fse_compress.c"
#include "zstd_subset/compress/hist.c"
#include "zstd_subset/compress/huf_compress.c"
#include "zstd_subset/compress/zstd_compress.c"
#include "zstd_subset/compress/zstd_compress_literals.c"
#include "zstd_subset/compress/zstd_compress_sequences.c"
#include "zstd_subset/compress/zstd_compress_superblock.c"
#include "zstd_subset/compress/zstd_double_fast.c"
#include "zstd_subset/compress/zstd_fast.c"
#include "zstd_subset/compress/zstd_lazy.c"
#include "zstd_subset/compress/zstd_ldm.c"
#include "zstd_subset/compress/zstdmt_compress.c"
#include "zstd_subset/compress/zstd_opt.c"

#include "zstd_subset/decompress/huf_decompress.c"
#include "zstd_subset/decompress/zstd_ddict.c"
#include "zstd_subset/decompress/zstd_decompress_block.c"
#include "zstd_subset/decompress/zstd_decompress.c"


