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
    size_t header_size ;
    (*iterator) = GB_CALLOC_MEMORY (1, sizeof (struct GB_Iterator_opaque),
        &header_size) ;
    if (*iterator == NULL)
    { 
        // out of memory
        return (GrB_OUT_OF_MEMORY) ;
    }
    (*iterator)->header_size = header_size ;
    return (GrB_SUCCESS) ;
}

