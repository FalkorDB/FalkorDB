//------------------------------------------------------------------------------
// GB_mex_reduce_to_scalar: c = accum(c,reduce_to_scalar(A))
//------------------------------------------------------------------------------

// SuiteSparse:GraphBLAS, Timothy A. Davis, (c) 2017-2023, All Rights Reserved.
// SPDX-License-Identifier: Apache-2.0

//------------------------------------------------------------------------------

// Reduce a matrix or vector to a scalar

#include "GB_mex.h"

#define USAGE "c = GB_mex_reduce_to_scalar (c, accum, reduce, A)"

#define FREE_ALL                        \
{                                       \
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
    GrB_Monoid reduce = NULL ;
    bool reduce_is_complex = false ;
    bool reduce_monoid_allocated = false ;

    // check inputs
    if (nargout > 1 || nargin != 4)
    {
        mexErrMsgTxt ("Usage: " USAGE) ;
    }

    #define GET_DEEP_COPY ;
    #define FREE_DEEP_COPY ;

    // get the scalar c
    GB_void *c ;
    int64_t cnrows, cncols ;
    GrB_Type ctype ;

    GB_mx_mxArray_to_array (pargin [0], &c, &cnrows, &cncols, &ctype) ;
    if (cnrows != 1 || cncols != 1)
    {
        mexErrMsgTxt ("c must be a scalar") ;
    }
    if (ctype == NULL)
    {
        mexErrMsgTxt ("c must be numeric") ;
    }

    // get A (shallow copy)
    A = GB_mx_mxArray_to_Matrix (pargin [3], "A input", false, true) ;
    if (A == NULL)
    {
        FREE_ALL ;
        mexErrMsgTxt ("A failed") ;
    }

    // get reduce
    bool user_complex = (Complex != GxB_FC64) && (ctype == Complex) ;
    GrB_BinaryOp reduceop ;
    if (!GB_mx_mxArray_to_BinaryOp (&reduceop, pargin [2], "reduceop",
        ctype, user_complex) || reduceop == NULL) 
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
        ctype, user_complex))
    {
        FREE_ALL ;
        mexErrMsgTxt ("accum failed") ;
    }

    // c = accum(C,A*B)

    // test both Vector and Matrix methods.  The typecast is not necessary,
    // just to test.

    if (user_complex)
    {
        if (A->vdim == 1)
        {
            GrB_Vector V ;
            V = (GrB_Vector) A ;
            METHOD (GrB_Vector_reduce_UDT (c, accum, reduce, V, NULL)) ;
        }
        else
        {
            METHOD (GrB_Matrix_reduce_UDT (c, accum, reduce, A, NULL)) ;
        }
    }
    else
    {

        #define REDUCE(prefix,suffix,type)              \
        if (A->vdim == 1)                               \
        {                                               \
            GrB_Vector V ;                              \
            V = (GrB_Vector) A ;                        \
            METHOD (prefix ## Vector_reduce ## suffix   \
                ((type *) c, accum, reduce, V, NULL)) ; \
        }                                               \
        else                                            \
        {                                               \
            METHOD (prefix ## Matrix_reduce ## suffix   \
                ((type *) c, accum, reduce, A, NULL)) ; \
        }

        switch (ctype->code)
        {
            case GB_BOOL_code   : REDUCE (GrB_, _BOOL,   bool    ) ; break ;
            case GB_INT8_code   : REDUCE (GrB_, _INT8,   int8_t  ) ; break ;
            case GB_INT16_code  : REDUCE (GrB_, _INT16,  int16_t ) ; break ;
            case GB_INT32_code  : REDUCE (GrB_, _INT32,  int32_t ) ; break ;
            case GB_INT64_code  : REDUCE (GrB_, _INT64,  int64_t ) ; break ;
            case GB_UINT8_code  : REDUCE (GrB_, _UINT8,  uint8_t ) ; break ;
            case GB_UINT16_code : REDUCE (GrB_, _UINT16, uint16_t) ; break ;
            case GB_UINT32_code : REDUCE (GrB_, _UINT32, uint32_t) ; break ;
            case GB_UINT64_code : REDUCE (GrB_, _UINT64, uint64_t) ; break ;
            case GB_FP32_code   : REDUCE (GrB_, _FP32,   float   ) ; break ;
            case GB_FP64_code   : REDUCE (GrB_, _FP64,   double  ) ; break ;
            case GB_FC32_code   : REDUCE (GxB_, _FC32, GxB_FC32_t) ; break ;
            case GB_FC64_code   : REDUCE (GxB_, _FC64, GxB_FC64_t) ; break ;
            default             :
                FREE_ALL ;
                mexErrMsgTxt ("unknown type: reduce to scalar") ;
        }
    }

    // return C as a scalar
    pargout [0] = GB_mx_create_full (1, 1, ctype) ;
    GB_void *p = mxGetData (pargout [0]) ;
    memcpy (p, c, ctype->size) ;
    FREE_ALL ;
}

