//------------------------------------------------------------------------------
// GB_mex_select: C<M> = accum(C,select(A,k)) or select(A',k)
//------------------------------------------------------------------------------

// SuiteSparse:GraphBLAS, Timothy A. Davis, (c) 2017-2023, All Rights Reserved.
// SPDX-License-Identifier: Apache-2.0

//------------------------------------------------------------------------------

// Apply a select operator to a matrix

#include "GB_mex.h"

#define USAGE "C = GB_mex_select (C, M, accum, op, A, Thunk, desc, test)"

#define FREE_ALL                        \
{                                       \
    GrB_Scalar_free_(&Thunk) ;          \
    GrB_Matrix_free_(&C) ;              \
    GrB_Matrix_free_(&M) ;              \
    GrB_Matrix_free_(&A) ;              \
    GrB_IndexUnaryOp_free_(&isnanop) ;  \
    GrB_Descriptor_free_(&desc) ;       \
    GB_mx_put_global (true) ;           \
}

#if 0
bool isnan64 (GrB_Index i, GrB_Index j, const void *x, const void *b) ;
bool isnan64 (GrB_Index i, GrB_Index j, const void *x, const void *b)
{ 
    double aij = * ((double *) x) ;
    return (isnan (aij)) ;
}
#else
void isnan64 (bool *z, const void *x, GrB_Index i, GrB_Index j, const void *b) ;
void isnan64 (bool *z, const void *x, GrB_Index i, GrB_Index j, const void *b)
{ 
    double aij = * ((double *) x) ;
    (*z) = (isnan (aij)) ;
}
#endif

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
    GrB_Matrix M = NULL ;
    GrB_Matrix A = NULL ;
    GrB_Descriptor desc = NULL ;
    GrB_Scalar Thunk = NULL ;
    GrB_IndexUnaryOp isnanop = NULL ;

    // check inputs
    if (nargout > 1 || nargin < 5 || nargin > 8)
    {
        mexErrMsgTxt ("Usage: " USAGE) ;
    }

    // get C (make a deep copy)
    #define GET_DEEP_COPY \
    C = GB_mx_mxArray_to_Matrix (pargin [0], "C input", true, true) ;   \
    if (nargin > 7 && C != NULL) C->nvec_nonempty = -1 ;
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

    // get select operator; must be present
    GxB_SelectOp op ;
    if (!GB_mx_mxArray_to_SelectOp (&op, pargin [3], "op"))
    {
        FREE_ALL ;
        mexErrMsgTxt ("SelectOp failed") ;
    }

    if (op == NULL)
    {
        // user-defined isnan operator, with no Thunk
        GrB_IndexUnaryOp_new (&isnanop, (void *) isnan64,
            GrB_BOOL, GrB_FP64, GrB_FP64) ;
    }
    else if (nargin > 5)
    {
        // get Thunk (shallow copy)
        if (mxIsSparse (pargin [5]))
        {
            Thunk = (GrB_Scalar) GB_mx_mxArray_to_Matrix (pargin [5],
                "Thunk input", false, false) ;
            if (Thunk == NULL)
            {
                FREE_ALL ;
                mexErrMsgTxt ("Thunk failed") ;
            }
            if (!GB_SCALAR_OK (Thunk))
            { 
                FREE_ALL ;
                mexErrMsgTxt ("Thunk not a valid scalar") ;
            }
        }
        else
        {
            // get k
            Thunk = GB_mx_get_Scalar (pargin [5]) ;
        }
    }

    // get desc
    if (!GB_mx_mxArray_to_Descriptor (&desc, PARGIN (6), "desc"))
    {
        FREE_ALL ;
        mexErrMsgTxt ("desc failed") ;
    }

    // just for testing
    if (nargin > 7)
    {
        if (M != NULL) M->nvec_nonempty = -1 ;
        A->nvec_nonempty = -1 ;
        C->nvec_nonempty = -1 ;
    }

    // C<M> = accum(C,op(A))
    if (C->vdim == 1 && (desc == NULL || desc->in0 == GxB_DEFAULT))
    {
        // this is just to test the Vector version
        if (op == NULL)
        {
            METHOD (GrB_Vector_select_FP64 ((GrB_Vector) C, (GrB_Vector) M,
                accum, isnanop, (GrB_Vector) A, 0, desc)) ;
        }
        else
        {
            // GxB_print (op, 3) ; GxB_print (Thunk, 3) ;
            METHOD (GxB_Vector_select_((GrB_Vector) C, (GrB_Vector) M, accum,
                op, (GrB_Vector) A, Thunk, desc)) ;
        }
    }
    else
    {
        if (op == NULL)
        {
            METHOD (GrB_Matrix_select_FP64 (C, M, accum, isnanop, A, 0, desc)) ;
        }
        else
        {
            // GxB_print (op, 3) ; GxB_print (Thunk, 3) ;
            METHOD (GxB_Matrix_select_(C, M, accum, op, A, Thunk, desc)) ;
        }
    }

    // return C as a struct and free the GraphBLAS C
    pargout [0] = GB_mx_Matrix_to_mxArray (&C, "C output", true) ;

    FREE_ALL ;
}

