//------------------------------------------------------------------------------
// GxB_Vector_diag: extract a diagonal (as a vector) from a matrix
//------------------------------------------------------------------------------

// SuiteSparse:GraphBLAS, Timothy A. Davis, (c) 2017-2023, All Rights Reserved.
// SPDX-License-Identifier: Apache-2.0

//------------------------------------------------------------------------------

#include "GB_diag.h"

GrB_Info GxB_Vector_diag    // extract a diagonal from a matrix, as a vector
(
    GrB_Vector v,                   // output vector
    const GrB_Matrix A,             // input matrix
    int64_t k,
    const GrB_Descriptor desc       // unused, except threading control
)
{ 

    //--------------------------------------------------------------------------
    // check inputs
    //--------------------------------------------------------------------------

    GB_WHERE (v, "GxB_Vector_diag (v, A, k, desc)") ;
    GB_BURBLE_START ("GxB_Vector_diag") ;
    GB_RETURN_IF_NULL_OR_FAULTY (A) ;
    GB_RETURN_IF_NULL_OR_FAULTY (v) ;

    // get the descriptor
    GB_GET_DESCRIPTOR (info, desc, xx1, xx2, xx3, xx4, xx5, xx6, xx7) ;

    //--------------------------------------------------------------------------
    // v = diag (A,k)
    //--------------------------------------------------------------------------

    info = GB_Vector_diag ((GrB_Matrix) v, A, k, Werk) ;
    GB_BURBLE_END ;
    return (info) ;
}

