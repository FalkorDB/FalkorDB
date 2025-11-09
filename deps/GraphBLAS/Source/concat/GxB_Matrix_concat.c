//------------------------------------------------------------------------------
// GxB_Matrix_concat: concatenate an array of matrices into a single matrix
//------------------------------------------------------------------------------

// SuiteSparse:GraphBLAS, Timothy A. Davis, (c) 2017-2025, All Rights Reserved.
// SPDX-License-Identifier: Apache-2.0

//------------------------------------------------------------------------------

#include "concat/GB_concat.h"

GrB_Info GxB_Matrix_concat          // concatenate a 2D array of matrices
(
    GrB_Matrix C,                   // input/output matrix for results
    const GrB_Matrix *Tiles,        // 2D row-major array of size m-by-n
    const uint64_t m,
    const uint64_t n,
    const GrB_Descriptor desc       // unused, except threading control
)
{

    //--------------------------------------------------------------------------
    // check inputs
    //--------------------------------------------------------------------------

    GB_RETURN_IF_NULL (C) ;
    GB_RETURN_IF_NULL (Tiles) ;
    GB_RETURN_IF_OUTPUT_IS_READONLY (C) ;
    GB_WHERE1 (C, "GxB_Matrix_concat (C, Tiles, m, n, desc)") ;
    GB_BURBLE_START ("GxB_Matrix_concat") ;

    if (m <= 0 || n <= 0)
    { 
        GB_ERROR (GrB_INVALID_VALUE, "m (" GBd ") and n (" GBd ") must be > 0",
            m, n) ;
    }

    // get the descriptor
    GB_GET_DESCRIPTOR (info, desc, xx1, xx2, xx3, xx4, xx5, xx6, xx7) ;

    //--------------------------------------------------------------------------
    // C = concatenate (Tiles)
    //--------------------------------------------------------------------------

    info = GB_concat (C, Tiles, m, n, Werk) ;
    GB_BURBLE_END ;
    return (info) ;
}

