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
    uint64_t nentries,      // # of entries to allocate if non iso
    uint64_t sizeof_entry,  // size of each entry
    // input/output
    uint64_t *mem           // memlane on input; resulting memsize and memlane
                            // on output
)
{
    void *p ;
    nentries = GB_IMAX (nentries, 1) ;
    GBMDUMP ("xalloc : ") ;
    if (iso)
    { 
        // always calloc the iso entry
        p = GB_calloc_memory (1, sizeof_entry, mem) ;
    }
    else if (use_calloc)
    { 
        p = GB_calloc_memory (nentries, sizeof_entry, mem) ;
    }
    else
    { 
        p = GB_malloc_memory (nentries, sizeof_entry, mem) ;
    }
    return (p) ;
}

