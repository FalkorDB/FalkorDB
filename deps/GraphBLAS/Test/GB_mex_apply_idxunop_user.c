//------------------------------------------------------------------------------
// GB_mex_apply_idxunop_user: C = idxunop (A)
//------------------------------------------------------------------------------

// SuiteSparse:GraphBLAS, Timothy A. Davis, (c) 2017-2025, All Rights Reserved.
// SPDX-License-Identifier: Apache-2.0

//------------------------------------------------------------------------------

// Apply a user-defined index unary operator to a matrix.

#include "GB_mex.h"

#define USAGE "C = GB_mex_apply_idxunop_user (A)"

#define FREE_ALL                        \
{                                       \
    GrB_Matrix_free_(&C) ;              \
    GrB_Matrix_free_(&A) ;              \
    GrB_IndexUnaryOp_free_(&op) ;       \
    GB_mx_put_global (true) ;           \
}

GrB_Matrix C = NULL ;
GrB_Matrix A = NULL ;
GrB_IndexUnaryOp op = NULL ;

void idx2 (int64_t *z, const void *x, uint64_t i, uint64_t j, const void *y) ;
void idx2 (int64_t *z, const void *x, uint64_t i, uint64_t j, const void *y)
{
    uint64_t thunk = *((uint64_t *) y) ;
    (*z) = i + j + thunk ;
}

//------------------------------------------------------------------------------

void mexFunction
(
    int nargout,
    mxArray *pargout [ ],
    int nargin,
    const mxArray *pargin [ ]
)
{

    bool malloc_debug = GB_mx_get_global (true) ;

    // check inputs
    if (nargout > 1 || nargin != 1)
    {
        mexErrMsgTxt ("Usage: " USAGE) ;
    }

    // get A (shallow copy)
    A = GB_mx_mxArray_to_Matrix (pargin [0], "A input", false, true) ;
    if (A == NULL || A->magic != GB_MAGIC)
    {
        FREE_ALL ;
        mexErrMsgTxt ("A failed") ;
    }

    // create the op
    GrB_IndexUnaryOp_new (&op, (GxB_index_unary_function) idx2,
        GrB_UINT64, GrB_UINT64, GrB_UINT64) ;
    if (op == NULL)
    {
        FREE_ALL ;
        mexErrMsgTxt ("IndexUnaryOp failed") ;
    }

    // C = op (A), where thunk = 1
    uint64_t nrows, ncols ;
    GrB_Matrix_nrows (&nrows, A) ;
    GrB_Matrix_ncols (&ncols, A) ;
    GrB_Matrix_new (&C, GrB_FP64, nrows, ncols) ;
    GrB_Matrix_apply_IndexOp_UINT64 (C, NULL, NULL, op, A, 1, NULL) ;

    // return C as a struct and free the GraphBLAS C
    pargout [0] = GB_mx_Matrix_to_mxArray (&C, "C output", false) ;
    FREE_ALL ;
}

