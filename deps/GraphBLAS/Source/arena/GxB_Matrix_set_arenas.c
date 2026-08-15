//------------------------------------------------------------------------------
// GxB_Matrix_set_arenas: set the arenas (header and data) of a matrix
//------------------------------------------------------------------------------

// SuiteSparse:GraphBLAS, Timothy A. Davis, (c) 2017-2026, All Rights Reserved.
// SPDX-License-Identifier: Apache-2.0

//------------------------------------------------------------------------------

#include "GB.h"

#define GB_FREE_ALL ;

GrB_Info GxB_Matrix_set_arenas
(
    // input/output
    GrB_Matrix *Ahandle,        // handle of matrix to modify
    // input
    const int new_header_arena, // new arena for the header of A
    const int new_data_arena    // new arena for the data content of A
)
{ 

    //--------------------------------------------------------------------------
    // check inputs
    //--------------------------------------------------------------------------

    GB_CHECK_INIT ;
    if (Ahandle == NULL || *Ahandle == NULL)
    { 
        return (GrB_NULL_POINTER) ;
    }

    //--------------------------------------------------------------------------
    // change the arenas of the matrix
    //--------------------------------------------------------------------------

    return (GB_set_arenas (Ahandle, new_header_arena, new_data_arena)) ;
}

