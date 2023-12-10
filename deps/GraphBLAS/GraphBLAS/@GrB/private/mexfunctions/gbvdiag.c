//------------------------------------------------------------------------------
// gbvdiag: extract a diaogonal of a matrix, as a vector
//------------------------------------------------------------------------------

// SuiteSparse:GraphBLAS, Timothy A. Davis, (c) 2017-2023, All Rights Reserved.
// SPDX-License-Identifier: Apache-2.0

//------------------------------------------------------------------------------

// Usage:

// v = gbvdiag (A, k, desc)

#include "gb_interface.h"

#define USAGE "usage: v = gbvdiag (A, k, desc)"

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

    GrB_Matrix V = NULL ;
    GrB_Matrix A = gb_get_shallow (pargin [0]) ;
    int64_t k = 0 ;

    if (nargin > 1)
    {
        CHECK_ERROR (!gb_mxarray_is_scalar (pargin [1]), "k must be a scalar") ;
        double x = mxGetScalar (pargin [1]) ;
        k = (int64_t) x ;
        CHECK_ERROR ((double) k != x, "k must be an integer scalar") ;
    }

    //--------------------------------------------------------------------------
    // construct V
    //--------------------------------------------------------------------------

    GrB_Type vtype = NULL ;
    int64_t n, nrows, ncols ;
    OK (GxB_Matrix_type (&vtype, A)) ;
    OK (GrB_Matrix_nrows (&nrows, A)) ;
    OK (GrB_Matrix_ncols (&ncols, A)) ;

    if (k >= ncols || k <= -nrows)
    {
        // output vector V must have zero length
        n = 0 ;
    }
    else if (k >= 0)
    {
        // if k is in range 0 to n-1, V must have length min (m,n-k)
        n = GB_IMIN (nrows, ncols - k) ;
    }
    else
    {
        // if k is in range -1 to -m+1, V must have length min (m+k,n)
        n = GB_IMIN (nrows + k, ncols) ;
    }

    V = gb_new (vtype, n, 1, GxB_BY_COL, 0) ;

    //--------------------------------------------------------------------------
    // compute v = diag (A, k)
    //--------------------------------------------------------------------------

    OK1 (V, GxB_Vector_diag ((GrB_Vector) V, A, k, desc)) ;

    //--------------------------------------------------------------------------
    // free shallow copies
    //--------------------------------------------------------------------------

    OK (GrB_Matrix_free (&A)) ;
    OK (GrB_Descriptor_free (&desc)) ;

    //--------------------------------------------------------------------------
    // export the output matrix V
    //--------------------------------------------------------------------------

    pargout [0] = gb_export (&V, kind) ;
    pargout [1] = mxCreateDoubleScalar (kind) ;
    GB_WRAPUP ;
}

