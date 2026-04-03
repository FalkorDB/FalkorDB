//------------------------------------------------------------------------------
// GB_Iterator_rc_bitmap_next: move a row/col iterator to next entry in bitmap
//------------------------------------------------------------------------------

// SuiteSparse:GraphBLAS, Timothy A. Davis, (c) 2017-2025, All Rights Reserved.
// SPDX-License-Identifier: Apache-2.0

//------------------------------------------------------------------------------

#include "GB.h"

GrB_Info GB(Iterator_rc_bitmap_next)(GxB_Iterator iterator)
{
    for ( ; iterator->p < iterator->pend ; iterator->p++)
    {
        if (iterator->Ab [iterator->p])
        { 
            // found the next entry in this vector
            return (GrB_SUCCESS) ;
        }
    }
    return (GrB_NO_VALUE) ;
}

