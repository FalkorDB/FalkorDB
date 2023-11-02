//------------------------------------------------------------------------------
// GB_mex_Vector_sort: [C,P] = sort (A)
//------------------------------------------------------------------------------

// SuiteSparse:GraphBLAS, Timothy A. Davis, (c) 2017-2023, All Rights Reserved.
// SPDX-License-Identifier: Apache-2.0

//------------------------------------------------------------------------------

#include "GB_mex.h"

#define USAGE \
    "[C,P] = GB_mex_Vector_sort (op, A, desc)"

#define FREE_ALL                        \
{                                       \
    GrB_Matrix_free_(&A) ;              \
    GrB_Vector_free_(&P) ;              \
    GrB_Vector_free_(&C) ;              \
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
    GrB_Matrix A = NULL ;
    GrB_Vector C = NULL, P = NULL ;
    GrB_Descriptor desc = NULL ;

    // check inputs
    if (nargout > 2 || nargin < 2 || nargin > 3)
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
    if (ncols != 1)
    {
        mexErrMsgTxt ("A must be a column vector") ;
    }

    GrB_Vector_new (&C, A->type, nrows) ;
    if (nargout > 1)
    {
        GrB_Vector_new (&P, GrB_INT64, nrows) ;
    }

    GrB_Vector u = (GrB_Vector) A ;
    if (!GB_VECTOR_OK (u))
    {
        mexErrMsgTxt ("invalid input vector") ;
    }

    // [C,P] = sort(op,A,desc)
    #define FREE_DEEP_COPY ;
    #define GET_DEEP_COPY ;
    METHOD (GxB_Vector_sort (C, P, op, u, desc)) ;

    // return C as a struct and free the GraphBLAS C
    pargout [0] = GB_mx_Vector_to_mxArray (&C, "C output", true) ;

    if (nargout > 1)
    {
        // return P as a struct and free the GraphBLAS P
        pargout [1] = GB_mx_Vector_to_mxArray (&P, "P output", true) ;
    }

    FREE_ALL ;
}

