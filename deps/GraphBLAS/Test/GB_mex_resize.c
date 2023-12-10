//------------------------------------------------------------------------------
// GB_mex_resize: resize a matrix
//------------------------------------------------------------------------------

// SuiteSparse:GraphBLAS, Timothy A. Davis, (c) 2017-2023, All Rights Reserved.
// SPDX-License-Identifier: Apache-2.0

//------------------------------------------------------------------------------

#include "GB_mex.h"

#define USAGE "C = GB_mex_resize (A, nrows_new, ncols_new)"

#define FREE_ALL                        \
{                                       \
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
    GrB_Matrix C = NULL ;

    // check inputs
    if (nargout > 1 || nargin < 1 || nargin > 3)
    {
        mexErrMsgTxt ("Usage: " USAGE) ;
    }

    #define GET_DEEP_COPY \
    C = GB_mx_mxArray_to_Matrix (pargin [0], "C input", true, true) ;
    #define FREE_DEEP_COPY GrB_Matrix_free_(&C) ;

    GET_DEEP_COPY ;
    if (C == NULL)
    {
        FREE_ALL ;
        mexErrMsgTxt ("C failed") ;
    }

    // get vlen_new
    int64_t GET_SCALAR (1, int64_t, vlen_new, C->vlen) ;

    // get vdim_new
    int64_t GET_SCALAR (2, int64_t, vdim_new, C->vdim) ;

    // resize the matrix
    if (vlen_new % 5 == 0)
    {
        // test the old GxB functions
        if (GB_VECTOR_OK (C) && vdim_new == 1)
        {
            // resize C as a vector
            METHOD (GxB_Vector_resize ((GrB_Vector) C, vlen_new)) ;
        }
        else
        {
            // resize C as a matrix
            METHOD (GxB_Matrix_resize (C, vlen_new, vdim_new)) ;
        }
    }
    else
    {
        // test the new GrB functions
        if (GB_VECTOR_OK (C) && vdim_new == 1)
        {
            // resize C as a vector
            METHOD (GrB_Vector_resize_((GrB_Vector) C, vlen_new)) ;
        }
        else
        {
            // resize C as a matrix
            METHOD (GrB_Matrix_resize_(C, vlen_new, vdim_new)) ;
        }
    }

    // return C as a struct and free the GraphBLAS C
    pargout [0] = GB_mx_Matrix_to_mxArray (&C, "C output", true) ;

    FREE_ALL ;
}

