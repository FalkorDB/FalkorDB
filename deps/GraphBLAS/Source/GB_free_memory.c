//------------------------------------------------------------------------------
// GB_free_memory: wrapper for free
//------------------------------------------------------------------------------

// SuiteSparse:GraphBLAS, Timothy A. Davis, (c) 2017-2023, All Rights Reserved.
// SPDX-License-Identifier: Apache-2.0

//------------------------------------------------------------------------------

// A wrapper for free.  If p is NULL on input, it is not freed.

// The memory is freed using the free() function pointer passed in to GrB_init,
// which is typically the ANSI C free function.

#include "GB.h"

GB_CALLBACK_FREE_MEMORY_PROTO (GB_free_memory)
{
    if (p != NULL && (*p) != NULL)
    { 
        ASSERT (size_allocated == GB_Global_memtable_size (*p)) ;
        #ifdef GB_MEMDUMP
        printf ("\nhard free %p %ld\n", *p, size_allocated) ;   // MEMDUMP
        #endif
        GB_Global_free_function (*p) ;
        #ifdef GB_MEMDUMP
        GB_Global_memtable_dump ( ) ;
        #endif
        (*p) = NULL ;
    }
}

