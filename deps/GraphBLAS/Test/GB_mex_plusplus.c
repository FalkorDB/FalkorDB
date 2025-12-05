//------------------------------------------------------------------------------
// GB_mex_plusplus: C<M> = accum(C,A*B) with user-defined plus_plus_fp32
//------------------------------------------------------------------------------

// SuiteSparse:GraphBLAS, Timothy A. Davis, (c) 2017-2025, All Rights Reserved.
// SPDX-License-Identifier: Apache-2.0

//------------------------------------------------------------------------------

#include "GB_mex.h"

#define USAGE "C = GB_mex_plusplus (C, M, accum, [ ], A, B, desc)"

#define FREE_ALL                                    \
{                                                   \
    GrB_Matrix_free_(&A) ;                          \
    GrB_Matrix_free_(&B) ;                          \
    GrB_Matrix_free_(&C) ;                          \
    GrB_Matrix_free_(&M) ;                          \
    GrB_BinaryOp_free_(&MyPlus) ;                   \
    GrB_Monoid_free_(&MyAdd) ;                      \
    GrB_Semiring_free_(&MyPlusPlus) ;               \
    GrB_Descriptor_free_(&desc) ;                   \
    GB_mx_put_global (true) ;                       \
}

void gb_myplus (float *z, const float *x, const float *y) ;
void gb_myplus (float *z, const float *x, const float *y) { (*z) = (*x)+(*y) ; }
#define MYPLUS_DEFN \
"void gb_myplus (float *z, const float *x, const float *y) { (*z) = (*x)+(*y) ; }"

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
    GrB_Matrix B = NULL ;
    GrB_Matrix C = NULL ;
    GrB_Matrix M = NULL ;
    GrB_Semiring MyPlusPlus = NULL ;
    GrB_Descriptor desc = NULL ;
    GrB_BinaryOp MyPlus = NULL ;
    GrB_Monoid MyAdd = NULL ;

    // check inputs
    if (nargout > 1 || nargin < 6 || nargin > 7)
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
    M = GB_mx_mxArray_to_Matrix (pargin [1], "M", false, false) ;
    if (M == NULL && !mxIsEmpty (pargin [1]))
    {
        FREE_ALL ;
        mexErrMsgTxt ("M failed") ;
    }

    // get A (shallow copy)
    A = GB_mx_mxArray_to_Matrix (pargin [4], "A input", false, true) ;
    if (A == NULL)
    {
        FREE_ALL ;
        mexErrMsgTxt ("A failed") ;
    }

    // get B (shallow copy)
    B = GB_mx_mxArray_to_Matrix (pargin [5], "B input", false, true) ;
    if (B == NULL)
    {
        FREE_ALL ;
        mexErrMsgTxt ("B failed") ;
    }

    // create the semiring
    GxB_BinaryOp_new (&MyPlus, (GxB_binary_function) gb_myplus,
        GrB_FP32, GrB_FP32, GrB_FP32, "gb_myplus", MYPLUS_DEFN) ;
    float zero = 0 ;
    GrB_Monoid_new (&MyAdd, MyPlus, zero) ;
    GrB_Semiring_new (&MyPlusPlus, MyAdd, MyPlus) ;

    // get accum, if present
    GrB_BinaryOp accum ;
    if (!GB_mx_mxArray_to_BinaryOp (&accum, pargin [2], "accum",
        C->type, false))
    {
        FREE_ALL ;
        mexErrMsgTxt ("accum failed") ;
    }

    // get desc
    if (!GB_mx_mxArray_to_Descriptor (&desc, PARGIN (6), "desc"))
    {
        FREE_ALL ;
        mexErrMsgTxt ("desc failed") ;
    }

    // C<M> = accum(C,A*B)
    METHOD (GrB_mxm (C, M, accum, MyPlusPlus, A, B, desc)) ;

    // return C as a struct and free the GraphBLAS C
    pargout [0] = GB_mx_Matrix_to_mxArray (&C, "C output from GrB_mxm", true) ;
    FREE_ALL ;
}

