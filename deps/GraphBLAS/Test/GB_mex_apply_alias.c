//------------------------------------------------------------------------------
// GB_mex_apply_alias: C = op(C) using a unary op
//------------------------------------------------------------------------------

// SuiteSparse:GraphBLAS, Timothy A. Davis, (c) 2017-2023, All Rights Reserved.
// SPDX-License-Identifier: Apache-2.0

//------------------------------------------------------------------------------

#include "GB_mex.h"

#define USAGE "C = GB_mex_apply_alias (C, op, desc)"

#define FREE_ALL                            \
{                                           \
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
    GrB_Matrix C = NULL ;
    GrB_UnaryOp op = NULL ;
    GrB_Descriptor desc = NULL ;
    bool ignore ;

    // check inputs
    if (nargout > 1 || nargin < 2 || nargin > 3)
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

    // get op
    if (!GB_mx_mxArray_to_UnaryOp (&op, pargin [1], "op", C->type, false)
        || op == NULL)
    {
        FREE_ALL ;
        mexErrMsgTxt ("UnaryOp failed") ;
    }

    // get desc
    if (!GB_mx_mxArray_to_Descriptor (&desc, PARGIN (2), "desc"))
    {
        FREE_ALL ;
        mexErrMsgTxt ("desc failed") ;
    }

    // C = op(C)
    METHOD (GrB_Matrix_apply_(C, NULL, NULL, op, C, desc)) ;

    // return C as a struct and free the GraphBLAS C
    pargout [0] = GB_mx_Matrix_to_mxArray (&C, "C output", true) ;

    FREE_ALL ;
}

