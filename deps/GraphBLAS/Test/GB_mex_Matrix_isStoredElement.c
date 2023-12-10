//------------------------------------------------------------------------------
// GB_mex_Matrix_isStoredElement: interface for x = A(i,j)
//------------------------------------------------------------------------------

// SuiteSparse:GraphBLAS, Timothy A. Davis, (c) 2017-2023, All Rights Reserved.
// SPDX-License-Identifier: Apache-2.0

//------------------------------------------------------------------------------

// x = A (i,j), where i and j are zero-based.  If i and j arrays, then
// x (k) = A (i (k), j (k)) is done for all k.

// I and J and zero-based

#include "GB_mex.h"

#define USAGE "x = GB_mex_Matrix_isStoredElement (A, I, J)"

#define FREE_ALL                        \
{                                       \
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
    GrB_Index *I = NULL, ni = 0, I_range [3] ;
    GrB_Index *J = NULL, nj = 0, J_range [3] ;
    bool is_list ;

    // check inputs
    if (nargout > 1 || nargin < 3 || nargin > 5)
    {
        mexErrMsgTxt ("Usage: " USAGE) ;
    }

    #define GET_DEEP_COPY ;
    #define FREE_DEEP_COPY ;

    // get A (shallow copy)
    A = GB_mx_mxArray_to_Matrix (pargin [0], "A input", false, true) ;
    if (A == NULL)
    {
        FREE_ALL ;
        mexErrMsgTxt ("A failed") ;
    }

    // get I
    if (!GB_mx_mxArray_to_indices (&I, pargin [1], &ni, I_range, &is_list))
    {
        FREE_ALL ;
        mexErrMsgTxt ("I failed") ;
    }
    if (!is_list)
    {
        mexErrMsgTxt ("I is invalid; must be a list") ;
    }

    // get J
    if (!GB_mx_mxArray_to_indices (&J, pargin [2], &nj, J_range, &is_list))
    {
        FREE_ALL ;
        mexErrMsgTxt ("J failed") ;
    }
    if (!is_list)
    {
        mexErrMsgTxt ("J is invalid; must be a list") ;
    }

    if (ni != nj)
    {
        FREE_ALL ;
        mexErrMsgTxt ("I and J must be the same size") ;
    }

    // create output X
    pargout [0] = GB_mx_create_full (ni, 1, GrB_BOOL) ;
    bool *X = (bool *) mxGetData (pargout [0]) ;

    // x = A (i,j)
    for (int64_t k = 0 ; k < ni ; k++)
    {
        GrB_Info info = GxB_Matrix_isStoredElement (A, I [k], J [k]) ;
        X [k] = (info == GrB_SUCCESS) ;
    }

    FREE_ALL ;
}

