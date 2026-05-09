//------------------------------------------------------------------------------
// GB_free_memory: wrapper for free
//------------------------------------------------------------------------------

// SuiteSparse:GraphBLAS, Timothy A. Davis, (c) 2017-2025, All Rights Reserved.
// SPDX-License-Identifier: Apache-2.0

//------------------------------------------------------------------------------

// A wrapper for free.  If p is NULL on input, it is not freed.

// The memory is freed using the free() function pointer passed in to GrB_init,
// which is typically the ANSI C free function.

#include "GB.h"

#if 0
void GB_free_memory         /* free memory */
(
    /* input/output */
    void **p,               /* pointer to block of memory to free */
    /* input */
    uint64_t mem            /* memsize (in bytes) and memlane */
)
#endif

GB_CALLBACK_FREE_MEMORY_PROTO (GB_free_memory)
{
    if (p != NULL && (*p) != NULL)
    { 
        uint64_t memsize = GB_memsize (mem) ;
        int memlane = GB_memlane (mem) ;
        MEMTABLE_ASSERT (memsize == GB_Global_memtable_memsize (*p)) ;
        MEMTABLE_ASSERT (memlane == GB_Global_memtable_memlane (*p)) ;
        GB_Global_free_function (*p, memlane) ;
        #ifdef GB_MEMDUMP
        GB_Global_memtable_dump ( ) ;
        #endif
        (*p) = NULL ;
    }
}

