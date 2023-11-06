//------------------------------------------------------------------------------
// GB_mex_subassign_scalar: C<Mask>(I,J) = accum (C (I,J), Scalar)
//------------------------------------------------------------------------------

// SuiteSparse:GraphBLAS, Timothy A. Davis, (c) 2017-2023, All Rights Reserved.
// SPDX-License-Identifier: Apache-2.0

//------------------------------------------------------------------------------

#include "GB_mex.h"

#define USAGE "C = GB_mex_subassign_scalar(C,Mask,acc,S,I,J,desc)"

#define FREE_ALL                        \
{                                       \
    GrB_Matrix_free_(&A) ;              \
    GrB_Matrix_free_(&Mask) ;           \
    GrB_Matrix_free_(&C) ;              \
    GrB_Descriptor_free_(&desc) ;       \
    GB_mx_put_global (true) ;           \
}

#define GET_DEEP_COPY \
    C = GB_mx_mxArray_to_Matrix (pargin [0], "C input", true, true) ;

#define FREE_DEEP_COPY GrB_Matrix_free_(&C) ;

GrB_Matrix C = NULL ;
GrB_Matrix Mask = NULL ;
GrB_Matrix A = NULL ;
GrB_Descriptor desc = NULL ;
GrB_BinaryOp accum = NULL ;
GrB_Index *I = NULL, ni = 0, I_range [3] ;
GrB_Index *J = NULL, nj = 0, J_range [3] ;
bool ignore ;
bool malloc_debug = false ;
GrB_Info info = GrB_SUCCESS ;
int C_sparsity_control ;

//------------------------------------------------------------------------------
// assign: perform a single assignment
//------------------------------------------------------------------------------

#define OK(method)                      \
{                                       \
    info = method ;                     \
    if (info != GrB_SUCCESS)            \
    {                                   \
        return (info) ;                 \
    }                                   \
}


//------------------------------------------------------------------------------
// GB_mex_subassign_scalar mexFunction
//------------------------------------------------------------------------------

void mexFunction
(
    int nargout,
    mxArray *pargout [ ],
    int nargin,
    const mxArray *pargin [ ]
)
{

    C = NULL ;
    Mask = NULL ;
    A = NULL ;
    desc = NULL ;
    accum = NULL ;
    I = NULL ; ni = 0 ;
    J = NULL ; nj = 0 ;
    info = GrB_SUCCESS ;

    //--------------------------------------------------------------------------
    // check inputs
    //--------------------------------------------------------------------------

    malloc_debug = GB_mx_get_global (true) ;
    A = NULL ;
    C = NULL ;
    Mask = NULL ;
    desc = NULL ;

    // check inputs
    if (nargout > 1 || nargin != 7 )
    {
        mexErrMsgTxt ("Usage: " USAGE) ;
    }

    // get C (make a deep copy)
    GET_DEEP_COPY ;
    if (C == NULL)
    {
        FREE_ALL ;
        mexErrMsgTxt ("C failed") ;
    }

    // get Mask (deep copy)
    Mask = GB_mx_mxArray_to_Matrix (pargin [1], "Mask", true, false) ;
    if (Mask == NULL && !mxIsEmpty (pargin [1]))
    {
        FREE_ALL ;
        mexErrMsgTxt ("Mask failed") ;
    }

    // get A (deep copy)
    A = GB_mx_mxArray_to_Matrix (pargin [3], "S", true, true) ;
    if (A == NULL)
    {
        FREE_ALL ;
        mexErrMsgTxt ("S failed") ;
    }
    if (GB_NROWS (A) != 1 || GB_NCOLS (A) != 1)
    {
        mexErrMsgTxt ("S must be a scalar") ;
    }

    // get accum, if present
    bool user_complex = (Complex != GxB_FC64)
        && (C->type == Complex || A->type == Complex) ;
    accum = NULL ;
    if (!GB_mx_mxArray_to_BinaryOp (&accum, pargin [2], "accum",
        C->type, user_complex))
    {
        FREE_ALL ;
        mexErrMsgTxt ("accum failed") ;
    }

    // get I
    if (!GB_mx_mxArray_to_indices (&I, pargin [4], &ni, I_range, &ignore))
    {
        FREE_ALL ;
        mexErrMsgTxt ("I failed") ;
    }

    // get J
    if (!GB_mx_mxArray_to_indices (&J, pargin [5], &nj, J_range, &ignore))
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

    // use GxB_Matrix_subassign_Scalar or GxB_Vector_subassign_Scalar
    GrB_Scalar S = (GrB_Scalar) A ;
    if (GB_VECTOR_OK (C) && (Mask == NULL || GB_VECTOR_OK (Mask)))
    {
        METHOD (GxB_Vector_subassign_Scalar ((GrB_Vector) C, (GrB_Vector) Mask,
            accum, S, I, ni, desc)) ;
    }
    else
    {
        METHOD (GxB_Matrix_subassign_Scalar ((GrB_Matrix) C, (GrB_Matrix) Mask,
            accum, S, I, ni, J, nj, desc)) ;
    }

    //--------------------------------------------------------------------------
    // return C as a struct
    //--------------------------------------------------------------------------

    pargout [0] = GB_mx_Matrix_to_mxArray (&C, "C assign result", true) ;
    FREE_ALL ;
}

