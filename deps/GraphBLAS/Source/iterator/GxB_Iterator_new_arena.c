//------------------------------------------------------------------------------
// GxB_Iterator_new_arena: allocate an iterator
//------------------------------------------------------------------------------

// SuiteSparse:GraphBLAS, Timothy A. Davis, (c) 2017-2025, All Rights Reserved.
// SPDX-License-Identifier: Apache-2.0

//------------------------------------------------------------------------------

#include "GB.h"

#define GB_FREE_ALL ;

GrB_Info GxB_Iterator_new_arena
(
    GxB_Iterator *iterator,
    const int header_arena
)
{

    //--------------------------------------------------------------------------
    // check inputs
    //--------------------------------------------------------------------------

    GrB_Info info ;
    GB_CHECK_INIT ;
    GB_RETURN_IF_NULL (iterator) ;
    (*iterator) = NULL ;
    GB_OK (GB_check_arena (header_arena)) ;

    //--------------------------------------------------------------------------
    // create the iterator
    //--------------------------------------------------------------------------

    uint64_t header_mem = GB_mem (header_arena, 0) ;
    (*iterator) = GB_CALLOC_MEMORY (1, sizeof (struct GB_Iterator_opaque),
        &header_mem) ;
    if (*iterator == NULL)
    { 
        // out of memory
        return (GrB_OUT_OF_MEMORY) ;
    }
    (*iterator)->header_size = (size_t) GB_memsize (header_mem) ;
    (*iterator)->header_arena = header_arena ;
    return (GrB_SUCCESS) ;
}

