//------------------------------------------------------------------------------
// GxB_Matrix_diag: build a diagonal matrix from a vector
//------------------------------------------------------------------------------

// SuiteSparse:GraphBLAS, Timothy A. Davis, (c) 2017-2023, All Rights Reserved.
// SPDX-License-Identifier: Apache-2.0

//------------------------------------------------------------------------------

#include "GB_diag.h"

GrB_Info GxB_Matrix_diag        // build a diagonal matrix from a vector
(
    GrB_Matrix C,               // output matrix
    const GrB_Vector v,         // input vector
    int64_t k,
    const GrB_Descriptor desc   // unused, except threading control
)
{ 

    //--------------------------------------------------------------------------
    // check inputs
    //--------------------------------------------------------------------------

    GB_WHERE (C, "GxB_Matrix_diag (C, v, k, desc)") ;
    GB_BURBLE_START ("GxB_Matrix_diag") ;
    GB_RETURN_IF_NULL_OR_FAULTY (C) ;
    GB_RETURN_IF_NULL_OR_FAULTY (v) ;

    GrB_Type ctype = C->type ;
    GrB_Type vtype = v->type ;
    int64_t nrows = GB_NROWS (C) ;
    int64_t ncols = GB_NCOLS (C) ;
    int64_t n = v->vlen + GB_IABS (k) ;     // C must be n-by-n

    if (nrows != ncols || nrows != n)
    { 
        GB_ERROR (GrB_DIMENSION_MISMATCH,
            "Input matrix is " GBd "-by-" GBd " but must be "
            GBd "-by-" GBd "\n", nrows, ncols, n, n) ;
    }

    if (!GB_Type_compatible (ctype, vtype))
    { 
        GB_ERROR (GrB_DOMAIN_MISMATCH, "Input vector of type [%s] "
            "cannot be typecast to output of type [%s]\n",
            vtype->name, ctype->name) ;
    }

    // get the descriptor
    GB_GET_DESCRIPTOR (info, desc, xx1, xx2, xx3, xx4, xx5, xx6, xx7) ;

    //--------------------------------------------------------------------------
    // C = diag (v,k)
    //--------------------------------------------------------------------------

    info = GB_Matrix_diag (C, (GrB_Matrix) v, k, Werk) ;
    GB_BURBLE_END ;
    return (info) ;
}

