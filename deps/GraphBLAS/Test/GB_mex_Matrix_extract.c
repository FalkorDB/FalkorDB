//------------------------------------------------------------------------------
// GB_mex_Matrix_extract: interface for C<Mask> = accum (C,A(I,J))
//------------------------------------------------------------------------------

// SuiteSparse:GraphBLAS, Timothy A. Davis, (c) 2017-2025, All Rights Reserved.
// SPDX-License-Identifier: Apache-2.0

//------------------------------------------------------------------------------

#include "GB_mex.h"

#define USAGE "C = GB_mex_Matrix_extract"   \
    " (C, Mask, accum, A, I, J, desc, method)"

#define FREE_ALL                        \
{                                       \
    GrB_Matrix_free_(&C) ;              \
    GrB_Matrix_free_(&Mask) ;           \
    GrB_Matrix_free_(&A) ;              \
    GrB_Vector_free_(&I_vector) ;       \
    GrB_Vector_free_(&J_vector) ;       \
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
    GrB_Matrix C = NULL ;
    GrB_Matrix Mask = NULL ;
    GrB_Matrix A = NULL ;
    GrB_Vector I_vector = NULL, J_vector = NULL ;
    GrB_Descriptor desc = NULL ;
    uint64_t *I = NULL, ni = 0, I_range [3] ;
    uint64_t *J = NULL, nj = 0, J_range [3] ;
    bool ignore ;

    // check inputs
    if (nargout > 1 || nargin < 6 || nargin > 8)
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

    // get Mask (shallow copy)
    Mask = GB_mx_mxArray_to_Matrix (pargin [1], "Mask", false, false) ;
    if (Mask == NULL && !mxIsEmpty (pargin [1]))
    {
        FREE_ALL ;
        mexErrMsgTxt ("Mask failed") ;
    }

    // get A (shallow copy)
    A = GB_mx_mxArray_to_Matrix (pargin [3], "A input", false, true) ;
    if (A == NULL)
    {
        FREE_ALL ;
        mexErrMsgTxt ("A failed") ;
    }

    // get accum, if present
    bool user_complex = (Complex != GxB_FC64)
        && (C->type == Complex || A->type == Complex) ;
    GrB_BinaryOp accum ;
    if (!GB_mx_mxArray_to_BinaryOp (&accum, pargin [2], "accum",
        C->type, user_complex))
    {
        FREE_ALL ;
        mexErrMsgTxt ("accum failed") ;
    }

    // get the method:  0: use (uint64_t *), 1: use GrB_Vector for I,J
    int GET_SCALAR (7, int, method, 0) ;

    // get I
    if (!GB_mx_mxArray_to_indices (pargin [4], &I, &ni, I_range, &ignore,
        (method == 0) ? NULL : &I_vector))
    {
        FREE_ALL ;
        mexErrMsgTxt ("I failed") ;
    }
    // get J
    if (!GB_mx_mxArray_to_indices (pargin [5], &J, &nj, J_range, &ignore,
        (method == 0) ? NULL : &J_vector))
    {
        FREE_ALL ;
        mexErrMsgTxt ("J failed") ;
    }

    // get desc
    if (!GB_mx_mxArray_to_Descriptor (&desc, PARGIN (6), "desc"))
    {
        FREE_ALL ;
        mexErrMsgTxt ("desc failed") ;
    }

    // C<Mask> = accum (C,A(I,J))
    if (method == 0)
    {
        METHOD (GrB_Matrix_extract_(C, Mask, accum, A, I, ni, J, nj, desc)) ;
    }
    else
    {
        METHOD (GxB_Matrix_extract_Vector_(C, Mask, accum, A,
            I_vector, J_vector, desc)) ;
    }

    // return C as a struct and free the GraphBLAS C
    pargout [0] = GB_mx_Matrix_to_mxArray (&C, "C output", true) ;

    FREE_ALL ;
}

