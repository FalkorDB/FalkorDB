//------------------------------------------------------------------------------
// GrB_Matrix_build: build a sparse GraphBLAS matrix
//------------------------------------------------------------------------------

// SuiteSparse:GraphBLAS, Timothy A. Davis, (c) 2017-2023, All Rights Reserved.
// SPDX-License-Identifier: Apache-2.0

//------------------------------------------------------------------------------

// If dup is NULL: any duplicates result in an error.
// If dup is GxB_IGNORE_DUP: duplicates are ignored, which is not an error.
// If dup is a valid binary operator, it is used to reduce any duplicates to
// a single value.

#include "GB_build.h"

#define GB_MATRIX_BUILD(prefix,type,T,xtype)                                  \
GrB_Info GB_EVAL3 (prefix, _Matrix_build_, T) /* build a matrix from tuples */\
(                                                                             \
    GrB_Matrix C,                   /* matrix to build                    */  \
    const GrB_Index *I,             /* array of row indices of tuples     */  \
    const GrB_Index *J,             /* array of column indices of tuples  */  \
    const type *X,                  /* array of values of tuples          */  \
    GrB_Index nvals,                /* number of tuples                   */  \
    const GrB_BinaryOp dup          /* binary op to assemble duplicates   */  \
)                                                                             \
{                                                                             \
    GB_WHERE (C, GB_STR(prefix) "_Matrix_build_" GB_STR(T)                    \
        " (C, I, J, X, nvals, dup)") ;                                        \
    GB_BURBLE_START ("GrB_Matrix_build_" GB_STR(T)) ;                         \
    GB_RETURN_IF_NULL_OR_FAULTY (C) ;  /* check now so C->type can be done */ \
    GrB_Info info = GB_build (C, I, J, X, nvals, dup,                         \
        xtype, true, false, Werk) ;                                           \
    GB_BURBLE_END ;                                                           \
    return (info) ;                                                           \
}

GB_MATRIX_BUILD (GrB, bool      , BOOL   , GrB_BOOL  )
GB_MATRIX_BUILD (GrB, int8_t    , INT8   , GrB_INT8  )
GB_MATRIX_BUILD (GrB, int16_t   , INT16  , GrB_INT16 )
GB_MATRIX_BUILD (GrB, int32_t   , INT32  , GrB_INT32 )
GB_MATRIX_BUILD (GrB, int64_t   , INT64  , GrB_INT64 )
GB_MATRIX_BUILD (GrB, uint8_t   , UINT8  , GrB_UINT8 )
GB_MATRIX_BUILD (GrB, uint16_t  , UINT16 , GrB_UINT16)
GB_MATRIX_BUILD (GrB, uint32_t  , UINT32 , GrB_UINT32)
GB_MATRIX_BUILD (GrB, uint64_t  , UINT64 , GrB_UINT64)
GB_MATRIX_BUILD (GrB, float     , FP32   , GrB_FP32  )
GB_MATRIX_BUILD (GrB, double    , FP64   , GrB_FP64  )
GB_MATRIX_BUILD (GxB, GxB_FC32_t, FC32   , GxB_FC32  )
GB_MATRIX_BUILD (GxB, GxB_FC64_t, FC64   , GxB_FC64  )
GB_MATRIX_BUILD (GrB, void      , UDT    , C->type   )

