//------------------------------------------------------------------------------
// GB_mex_subref_symbolic: S=A(I,J)
//------------------------------------------------------------------------------

// SuiteSparse:GraphBLAS, Timothy A. Davis, (c) 2017-2025, All Rights Reserved.
// SPDX-License-Identifier: Apache-2.0

//------------------------------------------------------------------------------

#include "GB_mex.h"

#define USAGE "C = GB_mex_subref_symbolic (A, I, J)"

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
    struct GB_Matrix_opaque C_header ;
    GrB_Matrix C = GB_clear_matrix_header (&C_header) ;

    bool malloc_debug = GB_mx_get_global (true) ;
    GrB_Matrix A = NULL ;
    uint64_t *I = NULL, ni = 0, I_range [3] ;
    uint64_t *J = NULL, nj = 0, J_range [3] ;
    bool ignore ;

    // check inputs
    GB_WERK (USAGE) ;
    if (nargout > 1 || nargin != 3)
    {
        mexErrMsgTxt ("Usage: " USAGE) ;
    }

    #define GET_DEEP_COPY ;
    #define FREE_DEEP_COPY ;

    // get A (shallow copy)
    A = GB_mx_mxArray_to_Matrix (pargin [0], "A", false, true) ;
    if (A == NULL)
    {
        FREE_ALL ;
        mexErrMsgTxt ("A failed") ;
    }

    // get I
    if (!GB_mx_mxArray_to_indices (pargin [1], &I, &ni, I_range, &ignore, NULL))
    {
        FREE_ALL ;
        mexErrMsgTxt ("I failed") ;
    }

    // get J
    if (!GB_mx_mxArray_to_indices (pargin [2], &J, &nj, J_range, &ignore, NULL))
    {
        FREE_ALL ;
        mexErrMsgTxt ("J failed") ;
    }

    // symbolic subref is not needed when A is bitmap.
    int sparsity = 0 ;
    GxB_Matrix_Option_get_(A, GxB_SPARSITY_STATUS, &sparsity) ;
    if (sparsity == GxB_BITMAP)
    {
        mexErrMsgTxt ("A failed: cannot be bitmap") ;
    }

    bool I_is_32 = false ;
    bool J_is_32 = false ;

    // C = A(I,J) or A(J,I)', no need to check dimensions of C; symbolic
    METHOD (GB_subref (C, false, true, A, I, I_is_32, ni, J, J_is_32, nj,
        true, Werk)) ;

    // return C as a struct
    pargout [0] = GB_mx_Matrix_to_mxArray (&C, "C subref symbolic", true) ;

    FREE_ALL ;
}

