//------------------------------------------------------------------------------
// GrB_Matrix_diag: construct a diagonal matrix from a vector
//------------------------------------------------------------------------------

// SuiteSparse:GraphBLAS, Timothy A. Davis, (c) 2017-2023, All Rights Reserved.
// SPDX-License-Identifier: Apache-2.0

//------------------------------------------------------------------------------

// Similar to GxB_Matrix_diag (C, v, k, NULL), except that C is constructed
// as a new matrix, like GrB_Matrix_new.  C has the same type as v.

#include "GB_diag.h"

GrB_Info GrB_Matrix_diag        // construct a diagonal matrix from a vector
(
    GrB_Matrix *C,              // output matrix
    const GrB_Vector v,         // input vector
    int64_t k
)
{

    //--------------------------------------------------------------------------
    // check inputs
    //--------------------------------------------------------------------------

    GB_WHERE1 ("GrB_Matrix_diag (&C, v, k)") ;
    GB_BURBLE_START ("GrB_Matrix_diag") ;
    GB_RETURN_IF_NULL_OR_FAULTY (v) ;

    //--------------------------------------------------------------------------
    // C = diag (v,k)
    //--------------------------------------------------------------------------

    GrB_Index n = v->vlen + GB_IABS (k) ;
    GrB_Info info = GB_Matrix_new (C, v->type, n, n) ;
    if (info == GrB_SUCCESS)
    { 
        info = GB_Matrix_diag (*C, (GrB_Matrix) v, k, Werk) ;
    }

    GB_BURBLE_END ;
    return (info) ;
}

