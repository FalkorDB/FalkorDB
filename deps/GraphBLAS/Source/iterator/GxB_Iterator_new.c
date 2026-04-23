//------------------------------------------------------------------------------
// GxB_Iterator_new: allocate an iterator
//------------------------------------------------------------------------------

// SuiteSparse:GraphBLAS, Timothy A. Davis, (c) 2017-2025, All Rights Reserved.
// SPDX-License-Identifier: Apache-2.0

//------------------------------------------------------------------------------

#include "GB.h"

GrB_Info GxB_Iterator_new (GxB_Iterator *iterator)
{
    GB_CHECK_INIT ;
    GB_RETURN_IF_NULL (iterator) ;
    uint64_t header_mem = 0 ;   // FIXME memlane
    (*iterator) = GB_CALLOC_MEMORY (1, sizeof (struct GB_Iterator_opaque),
        &header_mem) ;
    if (*iterator == NULL)
    { 
        // out of memory
        return (GrB_OUT_OF_MEMORY) ;
    }
    (*iterator)->header_size = (size_t) GB_memsize (header_mem) ;
    return (GrB_SUCCESS) ;
}

