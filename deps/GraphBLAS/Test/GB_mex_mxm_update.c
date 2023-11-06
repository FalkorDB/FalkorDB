//------------------------------------------------------------------------------
// GB_mex_mxm_update: C += A*B where the accum is the same as the monoid
//------------------------------------------------------------------------------

// SuiteSparse:GraphBLAS, Timothy A. Davis, (c) 2017-2023, All Rights Reserved.
// SPDX-License-Identifier: Apache-2.0

//------------------------------------------------------------------------------

#include "GB_mex.h"

#define USAGE "C = GB_mex_mxm_update (C, semiring, A, B, desc)"

#define FREE_ALL                            \
{                                           \
    GrB_Matrix_free_(&A) ;                  \
    GrB_Matrix_free_(&B) ;                  \
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
    GrB_Matrix C = NULL, A = NULL, B = NULL ;
    GrB_Semiring semiring = NULL ;
    GrB_Descriptor desc = NULL ;
    GrB_BinaryOp accum = NULL ;

    // check inputs
    if (nargout > 1 || nargin < 4 || nargin > 5)
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

    // get semiring and accum
    bool user_complex = (Complex != GxB_FC64) && (C->type == Complex) ;
    if (!GB_mx_mxArray_to_Semiring (&semiring, pargin [1], "semiring",  
        C->type, user_complex))
    {
        FREE_ALL ;
        mexErrMsgTxt ("semiring failed") ;
    }
    accum = semiring->add->op ;

    // get A (shallow copy)
    A = GB_mx_mxArray_to_Matrix (pargin [2], "A input", false, true) ;
    if (A == NULL)
    {
        FREE_ALL ;
        mexErrMsgTxt ("A failed") ;
    }

    // get B (shallow copy)
    B = GB_mx_mxArray_to_Matrix (pargin [3], "B input", false, true) ;
    if (B == NULL)
    {
        FREE_ALL ;
        mexErrMsgTxt ("B failed") ;
    }

    // get desc
    if (!GB_mx_mxArray_to_Descriptor (&desc, PARGIN (4), "desc"))
    {
        FREE_ALL ;
        mexErrMsgTxt ("desc failed") ;
    }

    // C += A*B
    METHOD (GrB_mxm (C, NULL, accum, semiring, A, B, desc)) ;

    // return C as a struct and free the GraphBLAS C
    pargout [0] = GB_mx_Matrix_to_mxArray (&C, "C output", true) ;

    FREE_ALL ;
}

