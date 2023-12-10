//------------------------------------------------------------------------------
// gbmdiag: construct a diaogonal matrix from a vector
//------------------------------------------------------------------------------

// SuiteSparse:GraphBLAS, Timothy A. Davis, (c) 2017-2023, All Rights Reserved.
// SPDX-License-Identifier: Apache-2.0

//------------------------------------------------------------------------------

// Usage:

// C = gbmdiag (v, k, desc)

#include "gb_interface.h"

#define USAGE "usage: C = gbmdiag (v, k, desc)"

void mexFunction
(
    int nargout,
    mxArray *pargout [ ],
    int nargin,
    const mxArray *pargin [ ]
)
{

    //--------------------------------------------------------------------------
    // check inputs
    //--------------------------------------------------------------------------

    gb_usage (nargin >= 1 && nargin <= 3 && nargout <= 2, USAGE) ;

    //--------------------------------------------------------------------------
    // get the descriptor
    //--------------------------------------------------------------------------

    base_enum_t base ;
    kind_enum_t kind ;
    GxB_Format_Value fmt ;
    int sparsity ;
    GrB_Descriptor desc = NULL ;
    desc = gb_mxarray_to_descriptor (pargin [nargin-1], &kind, &fmt,
        &sparsity, &base) ;
    // if present, remove the descriptor from consideration
    if (desc != NULL) nargin-- ;

    //--------------------------------------------------------------------------
    // get the inputs
    //--------------------------------------------------------------------------

    GrB_Matrix C = NULL ;
    GrB_Matrix V = gb_get_shallow (pargin [0]) ;
    int64_t k = 0 ;

    int64_t ncols ;
    OK (GrB_Matrix_ncols (&ncols, V)) ;
    CHECK_ERROR (ncols != 1, "v must be a column vector") ;

    int s ;
    OK (GxB_Matrix_Option_get (V, GxB_SPARSITY_STATUS, &s)) ;
    CHECK_ERROR (s == GxB_HYPERSPARSE, "v cannot be hypersparse") ;

    if (nargin > 1)
    { 
        CHECK_ERROR (!gb_mxarray_is_scalar (pargin [1]), "k must be a scalar") ;
        double x = mxGetScalar (pargin [1]) ;
        k = (int64_t) x ;
        CHECK_ERROR ((double) k != x, "k must be an integer scalar") ;
    }

    //--------------------------------------------------------------------------
    // construct C
    //--------------------------------------------------------------------------

    GrB_Type ctype = NULL ;
    int64_t n ;
    OK (GxB_Matrix_type (&ctype, V)) ;
    OK (GrB_Matrix_nrows (&n, V)) ;
    n += GB_IABS (k) ;
    fmt = gb_get_format (n, n, NULL, NULL, fmt) ;
    C = gb_new (ctype, n, n, fmt, 0) ;

    //--------------------------------------------------------------------------
    // compute C = diag (v, k)
    //--------------------------------------------------------------------------

    OK1 (C, GxB_Matrix_diag (C, (GrB_Vector) V, k, desc)) ;

    //--------------------------------------------------------------------------
    // free shallow copies
    //--------------------------------------------------------------------------

    OK (GrB_Matrix_free (&V)) ;
    OK (GrB_Descriptor_free (&desc)) ;

    //--------------------------------------------------------------------------
    // export the output matrix C
    //--------------------------------------------------------------------------

    pargout [0] = gb_export (&C, kind) ;
    pargout [1] = mxCreateDoubleScalar (kind) ;
    GB_WRAPUP ;
}

