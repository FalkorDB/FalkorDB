//------------------------------------------------------------------------------
// GB_mex_expand: C<M,struct> = scalar
//------------------------------------------------------------------------------

// SuiteSparse:GraphBLAS, Timothy A. Davis, (c) 2017-2023, All Rights Reserved.
// SPDX-License-Identifier: Apache-2.0

//------------------------------------------------------------------------------

#include "GB_mex.h"
#include "matrix.h"

#define USAGE "C = GB_mex_expand (M, scalar)"

#define FREE_ALL                        \
{                                       \
    GrB_Matrix_free_(&M) ;              \
    GrB_Matrix_free_(&C) ;              \
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

    GrB_Info info ;
    bool malloc_debug = GB_mx_get_global (true) ;
    GrB_Matrix C = NULL, M = NULL ;

    if (nargin != 2 || nargout > 1)
    {
        mexErrMsgTxt ("Usage: " USAGE) ;
    }

    // get M (shallow copy)
    M = GB_mx_mxArray_to_Matrix (pargin [0], "M", false, false) ;
    if (M == NULL && !mxIsEmpty (pargin [0]))
    {
        FREE_ALL ;
        mexErrMsgTxt ("M failed") ;
    }

    GrB_Index nrows, ncols ;
    GrB_Matrix_nrows (&nrows, M) ;
    GrB_Matrix_ncols (&ncols, M) ;

    // get scalar
    if (!mxIsScalar (pargin [1]))
    {
        FREE_ALL ;
        mexErrMsgTxt ("scalar failed") ;
    }

    // C<M,struct> = scalar
    if (mxIsSparse (pargin [1]))
    {
        mexErrMsgTxt ("scalar must not be sparse") ;
    }

    GrB_Type ctype = GB_mx_Type (pargin [1]) ;
    GrB_Matrix_new (&C, ctype, nrows, ncols) ;

    if (ctype == Complex && Complex != GxB_FC64)
    {
        // user-defined complex case
        GxB_FC64_t *scalar = (GxB_FC64_t *) mxGetComplexDoubles (pargin [1]) ;
        info = GxB_Matrix_subassign_UDT_(C, M, NULL, (void *) scalar,
            GrB_ALL, nrows, GrB_ALL, ncols, GrB_DESC_RS) ;
        if (info != GrB_SUCCESS)
        {
            mexErrMsgTxt ("GxB_Matrix_subassign_[complex] failed") ;
        }
    }
    else
    {
        // built-in GraphBLAS types

        #define CREATE(suffix,c_type)                               \
        {                                                           \
            c_type *scalar = (c_type *) mxGetData (pargin [1]) ;    \
            GxB_Matrix_subassign ## suffix (C, M, NULL, *scalar,    \
                GrB_ALL, nrows, GrB_ALL, ncols, GrB_DESC_RS) ;      \
        }                                                           \
        break ;

        switch (ctype->code)
        {
            case GB_BOOL_code   : CREATE (_BOOL,   bool     ) ;
            case GB_INT8_code   : CREATE (_INT8,   int8_t   ) ;
            case GB_INT16_code  : CREATE (_INT16,  int16_t  ) ;
            case GB_INT32_code  : CREATE (_INT32,  int32_t  ) ;
            case GB_INT64_code  : CREATE (_INT64,  int64_t  ) ;
            case GB_UINT8_code  : CREATE (_UINT8,  uint8_t  ) ;
            case GB_UINT16_code : CREATE (_UINT16, uint16_t ) ;
            case GB_UINT32_code : CREATE (_UINT32, uint32_t ) ;
            case GB_UINT64_code : CREATE (_UINT32, uint64_t ) ;
            case GB_FP32_code   : CREATE (_FP32,   float    ) ;
            case GB_FP64_code   : CREATE (_FP64,   double   ) ;
            case GB_FC32_code   : CREATE (_FC32,   GxB_FC32_t) ;
            case GB_FC64_code   : CREATE (_FC64,   GxB_FC64_t) ;
            default: mexErrMsgTxt ("scalar type not supported") ;
        }
    }

    // return result
    pargout [0] = GB_mx_Matrix_to_mxArray (&C, "C result", true) ;
    FREE_ALL ;
}

