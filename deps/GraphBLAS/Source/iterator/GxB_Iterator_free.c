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
        size_t header_size = (*iterator)->header_size ;
        if (header_size > 0)
        { 
            (*iterator)->header_size = 0 ;
            int memlane = 0 ;       // always using memlane = 0
            GB_FREE_MEMORY (iterator, GB_mem (memlane, header_size)) ;
        }
    }
    return (GrB_SUCCESS) ;
}

