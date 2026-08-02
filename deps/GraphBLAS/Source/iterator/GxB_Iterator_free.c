//------------------------------------------------------------------------------
// GxB_Iterator_free: free an iterator
//------------------------------------------------------------------------------

// SuiteSparse:GraphBLAS, Timothy A. Davis, (c) 2017-2025, All Rights Reserved.
// SPDX-License-Identifier: Apache-2.0

//------------------------------------------------------------------------------

#include "GB.h"

GrB_Info GxB_Iterator_free (GxB_Iterator *iterator)
{
    if (iterator != NULL && (*iterator) != NULL)
    {
        uint64_t header_size = (uint64_t) ((*iterator)->header_size) ;
        int header_arena = (*iterator)->header_arena ;
        if (header_size > 0)
        { 
            (*iterator)->header_size = 0 ;
            GB_FREE_MEMORY (iterator, GB_mem (header_arena, header_size)) ;
        }
    }
    return (GrB_SUCCESS) ;
}

