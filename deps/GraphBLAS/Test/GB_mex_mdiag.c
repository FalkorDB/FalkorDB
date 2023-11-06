//------------------------------------------------------------------------------
// GB_mex_mdiag: compute C=diag(v,k)
//------------------------------------------------------------------------------

// SuiteSparse:GraphBLAS, Timothy A. Davis, (c) 2017-2023, All Rights Reserved.
// SPDX-License-Identifier: Apache-2.0

//------------------------------------------------------------------------------

// C = diag (v,k,ctype)

#include "GB_mex.h"

#define USAGE "C = GB_mex_mdiag (v,k,ctype,csc)"

#define FREE_ALL                        \
{                                       \
    GrB_Matrix_free_(&V) ;              \
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
    GrB_Matrix V = NULL, C = NULL ;

    // check inputs
    if (nargout > 1 || nargin < 1 || nargin > 4)
    {
        mexErrMsgTxt ("Usage: " USAGE) ;
    }

    #define GET_DEEP_COPY ;
    #define FREE_DEEP_COPY ;

    // get V
    V = GB_mx_mxArray_to_Matrix (pargin [0], "V", false, true) ;
    if (V == NULL)
    {
        FREE_ALL ;
        mexErrMsgTxt ("V failed") ;
    }

    if (!GB_VECTOR_OK (V))
    {
        FREE_ALL ;
        mexErrMsgTxt ("V must be a column vector") ;
    }

    // get k
    int64_t GET_SCALAR (1, int64_t, k, 0) ;

    // get the type
    GrB_Type ctype, vtype ;
    GxB_Matrix_type (&vtype, V) ;
    ctype = GB_mx_string_to_Type (PARGIN (2), vtype) ;

    // get fmt
    int GET_SCALAR (3, int, fmt, GxB_BY_COL) ;

    // construct C
    int64_t n ;
    GrB_Matrix_nrows (&n, V) ;
    n += GB_IABS (k) ;

    #undef FREE_DEEP_COPY
    #define FREE_DEEP_COPY GrB_Matrix_free_(&C) ;

    // C = diag (v,k), using either GrB_Matrix_diag or GxB_Matrix_diag.
    // The two methods do the same thing.  This is just to test.
    if (k % 2 == 0 && ctype == vtype)
    {
        // GrB_Matrix_diag does not handle typecasting
        METHOD (GrB_Matrix_diag (&C, (GrB_Vector) V, k)) ;
    }
    else
    {
        #undef GET_DEEP_COPY
        #define GET_DEEP_COPY                               \
            GrB_Matrix_new (&C, ctype, n, n) ;              \
            GxB_Matrix_Option_set (C, GxB_FORMAT, fmt) ;
        GET_DEEP_COPY ;
        METHOD (GxB_Matrix_diag (C, (GrB_Vector) V, k, NULL)) ;
    }

    // return C as a struct
    pargout [0] = GB_mx_Matrix_to_mxArray (&C, "C=diag(v,k)", true) ;
    FREE_ALL ;
}

