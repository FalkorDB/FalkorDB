//------------------------------------------------------------------------------
// GrB_Vector_build: build a sparse GraphBLAS vector
//------------------------------------------------------------------------------

// SuiteSparse:GraphBLAS, Timothy A. Davis, (c) 2017-2023, All Rights Reserved.
// SPDX-License-Identifier: Apache-2.0

//------------------------------------------------------------------------------

// If dup is NULL: any duplicates result in an error.
// If dup is GxB_IGNORE_DUP: duplicates are ignored, which is not an error.
// If dup is a valid binary operator, it is used to reduce any duplicates to
// a single value.

#include "GB_build.h"

#define GB_VECTOR_BUILD(prefix,type,T,xtype)                                  \
GrB_Info GB_EVAL3 (prefix, _Vector_build_, T) /* build a vector from tuples*/ \
(                                                                             \
    GrB_Vector w,                   /* vector to build                    */  \
    const GrB_Index *I,             /* array of row indices of tuples     */  \
    const type *X,                  /* array of values of tuples          */  \
    GrB_Index nvals,                /* number of tuples                   */  \
    const GrB_BinaryOp dup          /* binary op to assemble duplicates   */  \
)                                                                             \
{                                                                             \
    GB_WHERE (w, "GrB_Vector_build_" GB_STR(T) " (w, I, X, nvals, dup)") ;    \
    GB_BURBLE_START ("GrB_Vector_build_" GB_STR(T)) ;                         \
    GB_RETURN_IF_NULL_OR_FAULTY (w) ;  /* check now so w->type can be done */ \
    ASSERT (GB_VECTOR_OK (w)) ;                                               \
    GrB_Info info = GB_build ((GrB_Matrix) w, I, NULL, X, nvals, dup,         \
        xtype, false, false, Werk) ;                                       \
    GB_BURBLE_END ;                                                           \
    return (info) ;                                                           \
}

GB_VECTOR_BUILD (GrB, bool      , BOOL   , GrB_BOOL  )
GB_VECTOR_BUILD (GrB, int8_t    , INT8   , GrB_INT8  )
GB_VECTOR_BUILD (GrB, int16_t   , INT16  , GrB_INT16 )
GB_VECTOR_BUILD (GrB, int32_t   , INT32  , GrB_INT32 )
GB_VECTOR_BUILD (GrB, int64_t   , INT64  , GrB_INT64 )
GB_VECTOR_BUILD (GrB, uint8_t   , UINT8  , GrB_UINT8 )
GB_VECTOR_BUILD (GrB, uint16_t  , UINT16 , GrB_UINT16)
GB_VECTOR_BUILD (GrB, uint32_t  , UINT32 , GrB_UINT32)
GB_VECTOR_BUILD (GrB, uint64_t  , UINT64 , GrB_UINT64)
GB_VECTOR_BUILD (GrB, float     , FP32   , GrB_FP32  )
GB_VECTOR_BUILD (GrB, double    , FP64   , GrB_FP64  )
GB_VECTOR_BUILD (GxB, GxB_FC32_t, FC32   , GxB_FC32  )
GB_VECTOR_BUILD (GxB, GxB_FC64_t, FC64   , GxB_FC64  )

// for user-defined types, X is assumed to have the same type as w
GB_VECTOR_BUILD (GrB, void      , UDT    , w->type   )

