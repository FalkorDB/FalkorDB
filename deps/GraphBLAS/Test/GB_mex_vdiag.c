//------------------------------------------------------------------------------
// GB_mex_vdiag: compute v=diag(A,k)
//------------------------------------------------------------------------------

// SuiteSparse:GraphBLAS, Timothy A. Davis, (c) 2017-2023, All Rights Reserved.
// SPDX-License-Identifier: Apache-2.0

//------------------------------------------------------------------------------

// v = diag (A,k,vtype)

#include "GB_mex.h"

#define USAGE "v = GB_mex_vdiag (A,k,vtype)"

#define FREE_ALL                        \
{                                       \
    GrB_Vector_free_(&V) ;              \
    GrB_Matrix_free_(&A) ;              \
    GB_mx_put_global (true) ;           \
}


void mexFunction
(
    int nargout,
    mxArray *pargout [ ],
    int nargin,
    const mxArray *pargin [ ]
)
{

    bool malloc_debug = GB_mx_get_global (true) ;
    GrB_Matrix A = NULL ;
    GrB_Vector V = NULL ;

    // check inputs
    if (nargout > 1 || nargin < 1 || nargin > 3)
    {
        mexErrMsgTxt ("Usage: " USAGE) ;
    }

    #define GET_DEEP_COPY ;
    #define FREE_DEEP_COPY ;

    // get A
    A = GB_mx_mxArray_to_Matrix (pargin [0], "A", false, true) ;
    if (A == NULL)
    {
        FREE_ALL ;
        mexErrMsgTxt ("A failed") ;
    }

    // get k
    int64_t k = 0 ;
    if (nargin > 1)
    {
        k = (int64_t) mxGetScalar (pargin [1]) ;
    }

    // get the type
    GrB_Type vtype ;
    GxB_Matrix_type (&vtype, A) ;
    vtype = GB_mx_string_to_Type (PARGIN (2), vtype) ;

    // construct V
    int64_t n, nrows, ncols ;
    GrB_Matrix_nrows (&nrows, A) ;
    GrB_Matrix_ncols (&ncols, A) ;
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

    #undef GET_DEEP_COPY
    #undef FREE_DEEP_COPY

    #define GET_DEEP_COPY  GrB_Vector_new (&V, vtype, n) ;
    #define FREE_DEEP_COPY GrB_Vector_free_(&V) ;

    GET_DEEP_COPY ;

    // V = diag (A,k)
    METHOD (GxB_Vector_diag (V, A, k, NULL)) ;

    // return V as a struct
    pargout [0] = GB_mx_Matrix_to_mxArray ((GrB_Matrix *) &V, "V=diag(A,k)",
        true) ;
    FREE_ALL ;
}

