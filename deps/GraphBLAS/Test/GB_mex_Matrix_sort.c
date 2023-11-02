//------------------------------------------------------------------------------
// GB_mex_Matrix_sort: [C,P] = sort (A)
//------------------------------------------------------------------------------

// SuiteSparse:GraphBLAS, Timothy A. Davis, (c) 2017-2023, All Rights Reserved.
// SPDX-License-Identifier: Apache-2.0

//------------------------------------------------------------------------------

#include "GB_mex.h"

#define USAGE \
    "[C,P] = GB_mex_Matrix_sort (op, A, desc, arg1)"

#define FREE_ALL                        \
{                                       \
    GrB_Matrix_free_(&A) ;              \
    GrB_Matrix_free_(&P) ;              \
    GrB_Matrix_free_(&C) ;              \
    GrB_Descriptor_free_(&desc) ;       \
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
    GrB_Matrix A = NULL, C = NULL, P = NULL ;
    GrB_Descriptor desc = NULL ;

    // check inputs
    if (nargout > 2 || nargin < 2 || nargin > 4)
    {
        mexErrMsgTxt ("Usage: " USAGE) ;
    }

    // get A (shallow copy)
    A = GB_mx_mxArray_to_Matrix (pargin [1], "A input", false, true) ;
    if (A == NULL)
    {
        FREE_ALL ;
        mexErrMsgTxt ("A failed") ;
    }

    // get operator
    bool user_complex = (Complex != GxB_FC64) && (A->type == Complex) ;
    GrB_BinaryOp op ;
    if (!GB_mx_mxArray_to_BinaryOp (&op, pargin [0], "op",
        A->type, user_complex) || op == NULL)
    {
        FREE_ALL ;
        mexErrMsgTxt ("add failed") ;
    }

    // get desc
    if (!GB_mx_mxArray_to_Descriptor (&desc, PARGIN (2), "desc"))
    {
        FREE_ALL ;
        mexErrMsgTxt ("desc failed") ;
    }

    // create C and P
    GrB_Index nrows, ncols ;
    GrB_Matrix_nrows (&nrows, A) ;
    GrB_Matrix_ncols (&ncols, A) ;

    // P = GB_mex_Matrix_sort (op, A, desc, 1): do not compute C
    bool P_only = (nargin == 4 && nargout == 1) ;
    if (!P_only)
    {
        GrB_Matrix_new (&C, A->type, nrows, ncols) ;
    }
    if (P_only || nargout > 1)
    {
        GrB_Matrix_new (&P, GrB_INT64, nrows, ncols) ;
    }

    if (nargout == 1 && !P_only)
    {
        GrB_Matrix_free (&C) ;
        #define FREE_DEEP_COPY GrB_Matrix_free (&C) ;
        #define GET_DEEP_COPY  GrB_Matrix_dup (&C, A) ;
        GET_DEEP_COPY ;
        METHOD (GxB_Matrix_sort (C, NULL, op, C, desc)) ;
    }
    else
    {
        // [C,P] = sort(op,A,desc)
        #undef  FREE_DEEP_COPY
        #define FREE_DEEP_COPY ;
        #undef  GET_DEEP_COPY
        #define GET_DEEP_COPY ;
        METHOD (GxB_Matrix_sort (C, P, op, A, desc)) ;
    }

    if (P_only)
    {
        // return P as a struct and free the GraphBLAS P
        pargout [0] = GB_mx_Matrix_to_mxArray (&P, "P output", true) ;
    }
    else
    {
        // return C as a struct and free the GraphBLAS C
        pargout [0] = GB_mx_Matrix_to_mxArray (&C, "C output", true) ;
        if (nargout > 1)
        {
            // return P as a struct and free the GraphBLAS P
            pargout [1] = GB_mx_Matrix_to_mxArray (&P, "P output", true) ;
        }
    }

    FREE_ALL ;
}

