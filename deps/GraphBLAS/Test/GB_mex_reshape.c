//------------------------------------------------------------------------------
// GB_mex_reshape: reshape a matrix
//------------------------------------------------------------------------------

// SuiteSparse:GraphBLAS, Timothy A. Davis, (c) 2017-2023, All Rights Reserved.
// SPDX-License-Identifier: Apache-2.0

//------------------------------------------------------------------------------

#include "GB_mex.h"

#define USAGE "C = GB_mex_reshape (A, nrows_new, ncols_new, by_col, in_place)"

#define FREE_ALL                        \
{                                       \
    GrB_Matrix_free_(&A) ;              \
    GrB_Matrix_free_(&C) ;              \
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
    GrB_Matrix C = NULL, A = NULL ;

    // check inputs
    if (nargout > 1 || nargin != 5)
    {
        mexErrMsgTxt ("Usage: " USAGE) ;
    }

    #define FREE_DEEP_COPY GrB_Matrix_free_(&C) ;

    GrB_Index nrows_new = (GrB_Index) mxGetScalar (pargin [1]) ;
    GrB_Index ncols_new = (GrB_Index) mxGetScalar (pargin [2]) ;
    bool      by_col    = (bool     ) mxGetScalar (pargin [3]) ;
    bool      in_place  = (bool     ) mxGetScalar (pargin [4]) ;

    // reshape the matrix
    if (in_place)
    {
        // in-place reshape of C
        #define GET_DEEP_COPY \
        C = GB_mx_mxArray_to_Matrix (pargin [0], "C input", true, true) ;
        GET_DEEP_COPY ;
        if (C == NULL)
        {
            FREE_ALL ;
            mexErrMsgTxt ("C failed") ;
        }
        METHOD (GxB_Matrix_reshape (C, by_col, nrows_new, ncols_new, NULL)) ;
    }
    else
    {
        // C is a new matrix created from the input matrix A
        #undef  GET_DEEP_COPY
        #define GET_DEEP_COPY ;
        A = GB_mx_mxArray_to_Matrix (pargin [0], "A", false, true) ;
        if (A == NULL)
        {
            FREE_ALL ;
            mexErrMsgTxt ("A failed") ;
        }
        METHOD (GxB_Matrix_reshapeDup (&C, A, by_col, nrows_new, ncols_new,
            NULL)) ;
    }

    // return C as a struct and free the GraphBLAS C
    pargout [0] = GB_mx_Matrix_to_mxArray (&C, "C output", true) ;

    FREE_ALL ;
}

