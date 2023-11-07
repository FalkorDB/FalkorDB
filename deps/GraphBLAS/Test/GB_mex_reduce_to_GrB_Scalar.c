//------------------------------------------------------------------------------
// GB_mex_reduce_to_GrB_Scalar: S = accum(S,reduce_to_scalar(A))
//------------------------------------------------------------------------------

// SuiteSparse:GraphBLAS, Timothy A. Davis, (c) 2017-2023, All Rights Reserved.
// SPDX-License-Identifier: Apache-2.0

//------------------------------------------------------------------------------

// Reduce a matrix or vector to a scalar

#include "GB_mex.h"

#define USAGE "S = GB_mex_reduce_to_GrB_Scalar (S, accum, reduce, A)"

#define FREE_ALL                        \
{                                       \
    GrB_Matrix_free_(&S) ;              \
    GrB_Matrix_free_(&A) ;              \
    if (reduce_monoid_allocated)        \
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
    GrB_Matrix S = NULL ;
    GrB_Monoid reduce = NULL ;
    bool reduce_is_complex = false ;
    bool reduce_monoid_allocated = false ;

    // check inputs
    if (nargout > 1 || nargin != 4)
    {
        mexErrMsgTxt ("Usage: " USAGE) ;
    }

    // get the GrB_Scalar S as a GrB_Matrix
    #define GET_DEEP_COPY \
    S = GB_mx_mxArray_to_Matrix (pargin [0], "S input", true, true) ;
    #define FREE_DEEP_COPY GrB_Matrix_free_(&S) ;
    GET_DEEP_COPY ;
    if (S == NULL)
    {
        FREE_ALL ;
        mexErrMsgTxt ("S failed") ;
    }
    int64_t Snrows, Sncols ;
    GrB_Matrix_nrows (&Snrows, S) ;
    GrB_Matrix_ncols (&Sncols, S) ;
    if (Snrows != 1 || Sncols != 1)
    { 
        mexErrMsgTxt ("S must be a scalar") ;
    }
    GrB_Type stype = S->type ;

    // get A (shallow copy)
    A = GB_mx_mxArray_to_Matrix (pargin [3], "A input", false, true) ;
    if (A == NULL)
    {
        FREE_ALL ;
        mexErrMsgTxt ("A failed") ;
    }

    // get reduce
    bool user_complex = (Complex != GxB_FC64) && (stype == Complex) ;
    GrB_BinaryOp reduceop ;
    if (!GB_mx_mxArray_to_BinaryOp (&reduceop, pargin [2], "reduceop",
        stype, user_complex) || reduceop == NULL) 
    {
        FREE_ALL ;
        mexErrMsgTxt ("reduceop failed") ;
    }

    // get the reduce monoid
    if (user_complex)
    {
        if (reduceop == Complex_plus)
        {
            reduce = Complex_plus_monoid ;
        }
        else if (reduceop == Complex_times)
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
        if (!GB_mx_Monoid (&reduce, reduceop, malloc_debug))
        {
            FREE_ALL ;
            mexErrMsgTxt ("reduce failed") ;
        }
        reduce_monoid_allocated = true ;
    }

    // get accum, if present
    GrB_BinaryOp accum ;
    if (!GB_mx_mxArray_to_BinaryOp (&accum, pargin [1], "accum",
        stype, user_complex))
    {
        FREE_ALL ;
        mexErrMsgTxt ("accum failed") ;
    }

    // S = accum(S,A*B)

    if (reduceop == GrB_MIN_INT8
    ||  reduceop == GrB_MIN_INT16
    ||  reduceop == GrB_MIN_INT32
    ||  reduceop == GrB_MIN_INT64
    ||  reduceop == GrB_MIN_UINT8
    ||  reduceop == GrB_MIN_UINT16
    ||  reduceop == GrB_MIN_UINT32
    ||  reduceop == GrB_MIN_UINT64
    ||  reduceop == GrB_MIN_FP32
    ||  reduceop == GrB_MIN_FP64
    ||  reduceop == GrB_MAX_INT8
    ||  reduceop == GrB_MAX_INT16
    ||  reduceop == GrB_MAX_INT32
    ||  reduceop == GrB_MAX_INT64
    ||  reduceop == GrB_MAX_UINT8
    ||  reduceop == GrB_MAX_UINT16
    ||  reduceop == GrB_MAX_UINT32
    ||  reduceop == GrB_MAX_UINT64
    ||  reduceop == GrB_MAX_FP32
    ||  reduceop == GrB_MAX_FP64
    ||  reduceop == GrB_PLUS_INT8
    ||  reduceop == GrB_PLUS_INT16
    ||  reduceop == GrB_PLUS_INT32
    ||  reduceop == GrB_PLUS_INT64
    ||  reduceop == GrB_PLUS_UINT8
    ||  reduceop == GrB_PLUS_UINT16
    ||  reduceop == GrB_PLUS_UINT32
    ||  reduceop == GrB_PLUS_UINT64
    ||  reduceop == GrB_PLUS_FP32
    ||  reduceop == GrB_PLUS_FP64
    ||  reduceop == GxB_PLUS_FC32
    ||  reduceop == GxB_PLUS_FC64
    ||  reduceop == GrB_TIMES_INT8
    ||  reduceop == GrB_TIMES_INT16
    ||  reduceop == GrB_TIMES_INT32
    ||  reduceop == GrB_TIMES_INT64
    ||  reduceop == GrB_TIMES_UINT8
    ||  reduceop == GrB_TIMES_UINT16
    ||  reduceop == GrB_TIMES_UINT32
    ||  reduceop == GrB_TIMES_UINT64
    ||  reduceop == GrB_TIMES_FP32
    ||  reduceop == GrB_TIMES_FP64
    ||  reduceop == GxB_TIMES_FC32
    ||  reduceop == GxB_TIMES_FC64
    ||  reduceop == GxB_ANY_BOOL
    ||  reduceop == GxB_ANY_INT8
    ||  reduceop == GxB_ANY_INT16
    ||  reduceop == GxB_ANY_INT32
    ||  reduceop == GxB_ANY_INT64
    ||  reduceop == GxB_ANY_UINT8
    ||  reduceop == GxB_ANY_UINT16
    ||  reduceop == GxB_ANY_UINT32
    ||  reduceop == GxB_ANY_UINT64
    ||  reduceop == GxB_ANY_FP32
    ||  reduceop == GxB_ANY_FP64
    ||  reduceop == GxB_ANY_FC32
    ||  reduceop == GxB_ANY_FC64
    ||  reduceop == GrB_LOR
    ||  reduceop == GrB_LAND
    ||  reduceop == GrB_LXOR
    ||  reduceop == GrB_LXNOR
    ||  reduceop == GrB_BOR_UINT8
    ||  reduceop == GrB_BOR_UINT16
    ||  reduceop == GrB_BOR_UINT32
    ||  reduceop == GrB_BOR_UINT64
    ||  reduceop == GrB_BAND_UINT8
    ||  reduceop == GrB_BAND_UINT16
    ||  reduceop == GrB_BAND_UINT32
    ||  reduceop == GrB_BAND_UINT64
    ||  reduceop == GrB_BXOR_UINT8
    ||  reduceop == GrB_BXOR_UINT16
    ||  reduceop == GrB_BXOR_UINT32
    ||  reduceop == GrB_BXOR_UINT64
    ||  reduceop == GrB_BXNOR_UINT8
    ||  reduceop == GrB_BXNOR_UINT16
    ||  reduceop == GrB_BXNOR_UINT32
    ||  reduceop == GrB_BXNOR_UINT64)
    {
        // S = accum (S, reduce (A)) using a binary op
        if (GB_VECTOR_OK (A))
        {
            METHOD (GrB_Vector_reduce_BinaryOp_Scalar_((GrB_Scalar) S, accum, reduceop, (GrB_Vector) A, NULL)) ;
        }
        else
        {
            METHOD (GrB_Matrix_reduce_BinaryOp_Scalar_((GrB_Scalar) S, accum, reduceop, A, NULL)) ;
        }
    }
    else
    {
        // S = accum (S, reduce (A)) using a monoid
        if (GB_VECTOR_OK (A))
        {
            METHOD (GrB_Vector_reduce_Monoid_Scalar_((GrB_Scalar) S, accum, reduce, (GrB_Vector) A, NULL)) ;
        }
        else
        {
            METHOD (GrB_Matrix_reduce_Monoid_Scalar_((GrB_Scalar) S, accum, reduce, A, NULL)) ;
        }
    }

    // return S as struct
    pargout [0] = GB_mx_Matrix_to_mxArray (&S, "S", true) ;
    FREE_ALL ;
}

