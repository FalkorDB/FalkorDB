//------------------------------------------------------------------------------
// GB_lz4: wrapper for the LZ4 compression library
//------------------------------------------------------------------------------

// SuiteSparse:GraphBLAS, Timothy A. Davis, (c) 2017-2023, All Rights Reserved.
// SPDX-License-Identifier: Apache-2.0

//------------------------------------------------------------------------------

// GB_lz4 is a wrapper for the LZ4 compression library (lz4.c and lz4.h).  The
// LZ4 library is compiled with LZ4_USER_MEMORY_FUNCTIONS enabled (which is not
// the default), and configured to use the SuiteSparse:GraphBLAS functions
// in place of malloc/calloc/free.

#include "GB.h"
#include "GB_serialize.h"
#include "GB_lz4.h"

void *LZ4_malloc (size_t s)
{
    return (GB_Global_malloc_function (s)) ;
}

void *LZ4_calloc (size_t n, size_t s)
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

void LZ4_free (void *p)
{
    GB_Global_free_function (p) ;
}

// LZ4 uses switch statements with no default case.
#if GB_COMPILER_GCC
#pragma GCC diagnostic ignored "-Wswitch-default"
#endif

// Include the unmodified lz4.c and lz4hc.c source code, version 1.9.3.  This
// allows the LZ4_* functions to be renamed via GB_lz4.h, and avoids any
// conflict with the original -llz4, which might be linked in by the user
// application.

#include "lz4.c"
#include "lz4hc.c"

