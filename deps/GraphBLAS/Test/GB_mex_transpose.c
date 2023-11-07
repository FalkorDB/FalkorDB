//------------------------------------------------------------------------------
// GB_mex_transpose: transpose a sparse matrix
//------------------------------------------------------------------------------

// SuiteSparse:GraphBLAS, Timothy A. Davis, (c) 2017-2023, All Rights Reserved.
// SPDX-License-Identifier: Apache-2.0

//------------------------------------------------------------------------------

// C<M> = accum (C,A') or accum (C,A)

#include "GB_mex.h"

#define USAGE "C = GB_mex_transpose (C, M, accum, A, desc, test)"

#define FREE_ALL                        \
{                                       \
    bool A_is_M = (A == M) ;            \
    bool A_is_C = (A == C) ;            \
    bool C_is_M = (C == M) ;            \
    GrB_Matrix_free_(&A) ;              \
    if (A_is_C) C = NULL ;              \
    if (A_is_M) M = NULL ;              \
    GrB_Matrix_free_(&C) ;              \
    if (C_is_M) M = NULL ;              \
    GrB_Matrix_free_(&M) ;              \
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
    GrB_Matrix C = NULL ;
    GrB_Matrix M = NULL ;
    GrB_Descriptor desc = NULL ;

    // check inputs
    if (nargout > 1 || nargin < 4 || nargin > 6)
    {
        mexErrMsgTxt ("Usage: " USAGE) ;
    }

    // get M (shallow copy)
    if (!mxIsChar (pargin [1]))
    {
        M = GB_mx_mxArray_to_Matrix (pargin [1], "M", false, false) ;
        if (M == NULL && !mxIsEmpty (pargin [1]))
        {
            FREE_ALL ;
            mexErrMsgTxt ("M failed") ;
        }
    }

    // get A (shallow copy)
    if (!mxIsChar (pargin [3]))
    {
        A = GB_mx_mxArray_to_Matrix (pargin [3], "A input", false, true) ;
        if (A == NULL)
        {
            FREE_ALL ;
            mexErrMsgTxt ("A failed") ;
        }
    }

    // get C (make a deep copy) and get any aliases for M and A
    #define GET_DEEP_COPY                                                   \
    {                                                                       \
        C = GB_mx_mxArray_to_Matrix (pargin [0], "C input", true, true) ;   \
        if (nargin > 5 && C != NULL)                                        \
        {                                                                   \
            C->nvec_nonempty = -1 ;  /* for testing */                      \
        }                                                                   \
        if (mxIsChar (pargin [1]))                                          \
        {                                                                   \
            M = GB_mx_alias ("M", pargin [1], "C", C, "A", A) ;             \
        }                                                                   \
        if (mxIsChar (pargin [3]))                                          \
        {                                                                   \
            A = GB_mx_alias ("A", pargin [3], "C", C, "M", M) ;             \
        }                                                                   \
    }

    #define FREE_DEEP_COPY          \
    {                               \
        if (A == C) A = NULL ;      \
        if (M == C) M = NULL ;      \
        GrB_Matrix_free_(&C) ;      \
    }

    GET_DEEP_COPY ;
    if (C == NULL)
    {
        FREE_ALL ;
        mexErrMsgTxt ("C failed") ;
    }

    // get accum; default: NOP, default is C->type
    bool user_complex = (Complex != GxB_FC64)
        && (C->type == Complex || A->type == Complex) ;
    GrB_BinaryOp accum ;
    if (!GB_mx_mxArray_to_BinaryOp (&accum, pargin [2], "accum",
        C->type, user_complex))
    {
        FREE_ALL ;
        mexErrMsgTxt ("accum failed") ;
    }

    // get desc
    if (!GB_mx_mxArray_to_Descriptor (&desc, PARGIN (4), "desc"))
    {
        FREE_ALL ;
        mexErrMsgTxt ("desc failed") ;
    }

    // just for testing
    if (nargin > 5)
    {
        if (M != NULL) M->nvec_nonempty = -1 ;
        A->nvec_nonempty = -1 ;
        C->nvec_nonempty = -1 ;
    }

    // C<M> = op(C,A') or op(C,A)
    METHOD (GrB_transpose (C, M, accum, A, desc)) ;

    // return C as a struct and free the GraphBLAS C
    if (C == A) A = NULL ;      // do not free A if it is aliased to C
    if (C == M) M = NULL ;      // do not free M if it is aliased to C
    pargout [0] = GB_mx_Matrix_to_mxArray (&C, "C output", true) ;
    // C is now NULL

    FREE_ALL ;
}

