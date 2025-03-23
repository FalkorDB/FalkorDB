//------------------------------------------------------------------------------
// GB_xalloc_memory: allocate an array for n entries, or 1 if iso
//------------------------------------------------------------------------------

// SuiteSparse:GraphBLAS, Timothy A. Davis, (c) 2017-2025, All Rights Reserved.
// SPDX-License-Identifier: Apache-2.0

//------------------------------------------------------------------------------

#include "GB.h"

void *GB_xalloc_memory      // return the newly-allocated space
(
    // input
    bool use_calloc,        // if true, use calloc
    bool iso,               // if true, only allocate a single entry
    int64_t n,              // # of entries to allocate if non iso
    size_t type_size,       // size of each entry
    // output
    size_t *size            // resulting size
)
{
    void *p ;
    n = GB_IMAX (n, 1) ;
    GBMDUMP ("xalloc : ") ;
    if (iso)
    { 
        // always calloc the iso entry
        p = GB_calloc_memory (1, type_size, size) ;
    }
    else if (use_calloc)
    { 
        p = GB_calloc_memory (n, type_size, size) ;
    }
    else
    { 
        p = GB_malloc_memory (n, type_size, size) ;
    }
    return (p) ;
}

