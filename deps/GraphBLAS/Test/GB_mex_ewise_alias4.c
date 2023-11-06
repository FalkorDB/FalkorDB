//------------------------------------------------------------------------------
// GB_mex_ewise_alias4: C<M> = M+M
//------------------------------------------------------------------------------

// SuiteSparse:GraphBLAS, Timothy A. Davis, (c) 2017-2023, All Rights Reserved.
// SPDX-License-Identifier: Apache-2.0

//------------------------------------------------------------------------------

#include "GB_mex.h"

#define USAGE "C = GB_mex_ewise_alias4 (C, M, op, desc)"

#define FREE_ALL                            \
{                                           \
    GrB_Matrix_free_(&M) ;                  \
    GrB_Matrix_free_(&C) ;                  \
    GrB_Descriptor_free_(&desc) ;           \
    GB_mx_put_global (true) ;               \
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
    GrB_Matrix C = NULL, M = NULL ;
    GrB_Descriptor desc = NULL ;

    // check inputs
    if (nargout > 1 || nargin < 3 || nargin > 4)
    {
        mexErrMsgTxt ("Usage: " USAGE) ;
    }

    // get C (make a deep copy)
    #define GET_DEEP_COPY \
    C = GB_mx_mxArray_to_Matrix (pargin [0], "C input", true, true) ;
    #define FREE_DEEP_COPY GrB_Matrix_free_(&C) ;
    GET_DEEP_COPY ;
    if (C == NULL)
    {
        FREE_ALL ;
        mexErrMsgTxt ("C failed") ;
    }

    // get M (shallow copy)
    M = GB_mx_mxArray_to_Matrix (pargin [1], "M input", false, true) ;
    if (M == NULL)
    {
        FREE_ALL ;
        mexErrMsgTxt ("M failed") ;
    }

    // get op
    bool user_complex = (Complex != GxB_FC64) && (C->type == Complex) ;
    GrB_BinaryOp op ;
    if (!GB_mx_mxArray_to_BinaryOp (&op, pargin [2], "op",
        C->type, user_complex) || op == NULL)
    {
        FREE_ALL ;
        mexErrMsgTxt ("op failed") ;
    }

    // get desc
    if (!GB_mx_mxArray_to_Descriptor (&desc, PARGIN (3), "desc"))
    {
        FREE_ALL ;
        mexErrMsgTxt ("desc failed") ;
    }

    // C<M> = M+M
    METHOD (GrB_Matrix_eWiseAdd_BinaryOp_(C, M, NULL, op, M, M, desc)) ;

    // return C as a struct and free the GraphBLAS C
    pargout [0] = GB_mx_Matrix_to_mxArray (&C, "C output", true) ;

    FREE_ALL ;
}

