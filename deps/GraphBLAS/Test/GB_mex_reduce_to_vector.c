//------------------------------------------------------------------------------
// GB_mex_reduce_to_vector: c = accum(c,reduce_to_vector(A))
//------------------------------------------------------------------------------

// SuiteSparse:GraphBLAS, Timothy A. Davis, (c) 2017-2023, All Rights Reserved.
// SPDX-License-Identifier: Apache-2.0

//------------------------------------------------------------------------------

// Reduce a matrix to a vector: w<mask> = accum (w, reduce_to_vector (A))

// Test interface to GrB_reduce, which relies on GrB_Matrix_reduce_BinaryOp
// and GrB_Matrix_reduce_Monoid to reduce a matrix to a vector.

#include "GB_mex.h"

#define USAGE "w = GB_mex_reduce_to_vector (w, mask, accum, reduce, A, desc)"

#define FREE_ALL                        \
{                                       \
    GrB_Matrix_free_(&A) ;              \
    GrB_Vector_free_(&w) ;              \
    GrB_Vector_free_(&mask) ;           \
    GrB_Descriptor_free_(&desc) ;       \
    if (!user_complex)                  \
    {                                   \
        GrB_Monoid_free_(&reduce) ;     \
    }                                   \
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
    GrB_Vector w = NULL ;
    GrB_Vector mask = NULL ;
    GrB_Descriptor desc = NULL ;
    GrB_Monoid reduce = NULL ;
    bool user_complex = false ;

    // check inputs
    if (nargout > 1 || nargin < 5 || nargin > 6)
    {
        mexErrMsgTxt ("Usage: " USAGE) ;
    }

    // get w (make a deep copy)
    #define GET_DEEP_COPY \
    w = GB_mx_mxArray_to_Vector (pargin [0], "w input", true, true) ;
    #define FREE_DEEP_COPY GrB_Vector_free_(&w) ;
    GET_DEEP_COPY ;
    if (w == NULL)
    {
        FREE_ALL ;
        mexErrMsgTxt ("w failed") ;
    }

    // get mask (shallow copy)
    mask = GB_mx_mxArray_to_Vector (pargin [1], "mask", false, false) ;
    if (mask == NULL && !mxIsEmpty (pargin [1]))
    {
        FREE_ALL ;
        mexErrMsgTxt ("mask failed") ;
    }

    // get A (shallow copy)
    A = GB_mx_mxArray_to_Matrix (pargin [4], "A input", false, true) ;
    if (A == NULL)
    {
        FREE_ALL ;
        mexErrMsgTxt ("A failed") ;
    }

    // get reduce operator
    user_complex = (Complex != GxB_FC64) && (A->type == Complex) ;
    GrB_BinaryOp op ;
    if (!GB_mx_mxArray_to_BinaryOp (&op, pargin [3], "op",
        w->type, user_complex) || op == NULL)
    {
        FREE_ALL ;
        mexErrMsgTxt ("op failed") ;
    }

    // get the reduce monoid
    if (user_complex)
    {
        if (op == Complex_plus)
        {
            reduce = Complex_plus_monoid ;
        }
        else if (op == Complex_times)
        {
            reduce = Complex_times_monoid ;
        }
        else
        {
            FREE_ALL ;
            mexErrMsgTxt ("reduce failed") ;
        }
    }
    else
    {
        // create the reduce monoid
        if (!GB_mx_Monoid (&reduce, op, malloc_debug))
        {
            FREE_ALL ;
            mexErrMsgTxt ("reduce failed") ;
        }
    }

    // get accum, if present
    user_complex = (Complex != GxB_FC64) && (w->type == Complex) ;
    GrB_BinaryOp accum ;
    if (!GB_mx_mxArray_to_BinaryOp (&accum, pargin [2], "accum",
        w->type, user_complex))
    {
        FREE_ALL ;
        mexErrMsgTxt ("accum failed") ;
    }

    // get desc
    if (!GB_mx_mxArray_to_Descriptor (&desc, PARGIN (5), "desc"))
    {
        FREE_ALL ;
        mexErrMsgTxt ("desc failed") ;
    }

    // test GrB_Matrix_reduce_BinaryOp, if possible

    if (op == GrB_MIN_INT8
    ||  op == GrB_MIN_INT16
    ||  op == GrB_MIN_INT32
    ||  op == GrB_MIN_INT64
    ||  op == GrB_MIN_UINT8
    ||  op == GrB_MIN_UINT16
    ||  op == GrB_MIN_UINT32
    ||  op == GrB_MIN_UINT64
    ||  op == GrB_MIN_FP32
    ||  op == GrB_MIN_FP64
    ||  op == GrB_MAX_INT8
    ||  op == GrB_MAX_INT16
    ||  op == GrB_MAX_INT32
    ||  op == GrB_MAX_INT64
    ||  op == GrB_MAX_UINT8
    ||  op == GrB_MAX_UINT16
    ||  op == GrB_MAX_UINT32
    ||  op == GrB_MAX_UINT64
    ||  op == GrB_MAX_FP32
    ||  op == GrB_MAX_FP64
    ||  op == GrB_PLUS_INT8
    ||  op == GrB_PLUS_INT16
    ||  op == GrB_PLUS_INT32
    ||  op == GrB_PLUS_INT64
    ||  op == GrB_PLUS_UINT8
    ||  op == GrB_PLUS_UINT16
    ||  op == GrB_PLUS_UINT32
    ||  op == GrB_PLUS_UINT64
    ||  op == GrB_PLUS_FP32
    ||  op == GrB_PLUS_FP64
    ||  op == GxB_PLUS_FC32
    ||  op == GxB_PLUS_FC64
    ||  op == GrB_TIMES_INT8
    ||  op == GrB_TIMES_INT16
    ||  op == GrB_TIMES_INT32
    ||  op == GrB_TIMES_INT64
    ||  op == GrB_TIMES_UINT8
    ||  op == GrB_TIMES_UINT16
    ||  op == GrB_TIMES_UINT32
    ||  op == GrB_TIMES_UINT64
    ||  op == GrB_TIMES_FP32
    ||  op == GrB_TIMES_FP64
    ||  op == GxB_TIMES_FC32
    ||  op == GxB_TIMES_FC64
    ||  op == GxB_ANY_BOOL
    ||  op == GxB_ANY_INT8
    ||  op == GxB_ANY_INT16
    ||  op == GxB_ANY_INT32
    ||  op == GxB_ANY_INT64
    ||  op == GxB_ANY_UINT8
    ||  op == GxB_ANY_UINT16
    ||  op == GxB_ANY_UINT32
    ||  op == GxB_ANY_UINT64
    ||  op == GxB_ANY_FP32
    ||  op == GxB_ANY_FP64
    ||  op == GxB_ANY_FC32
    ||  op == GxB_ANY_FC64
    ||  op == GrB_LOR
    ||  op == GrB_LAND
    ||  op == GrB_LXOR
    ||  op == GrB_LXNOR
    ||  op == GrB_BOR_UINT8
    ||  op == GrB_BOR_UINT16
    ||  op == GrB_BOR_UINT32
    ||  op == GrB_BOR_UINT64
    ||  op == GrB_BAND_UINT8
    ||  op == GrB_BAND_UINT16
    ||  op == GrB_BAND_UINT32
    ||  op == GrB_BAND_UINT64
    ||  op == GrB_BXOR_UINT8
    ||  op == GrB_BXOR_UINT16
    ||  op == GrB_BXOR_UINT32
    ||  op == GrB_BXOR_UINT64
    ||  op == GrB_BXNOR_UINT8
    ||  op == GrB_BXNOR_UINT16
    ||  op == GrB_BXNOR_UINT32
    ||  op == GrB_BXNOR_UINT64)
    {
        // w<mask> = accum (w, reduce_to_vector (A)) using a binary op
        METHOD (GrB_Matrix_reduce_BinaryOp_(w, mask, accum, op, A, desc)) ;
    }
    else
    {
        // w<mask> = accum (w, reduce_to_vector (A)) using a monoid
        METHOD (GrB_Matrix_reduce_Monoid_(w, mask, accum, reduce, A, desc)) ;
    }

    // return w as a struct and free the GraphBLAS w
    pargout [0] = GB_mx_Vector_to_mxArray (&w, "w output", true) ;

    FREE_ALL ;
}

