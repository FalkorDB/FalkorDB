//------------------------------------------------------------------------------
// GrB_Scalar_setElement: set an entry in a GrB_Scalar
//------------------------------------------------------------------------------

// SuiteSparse:GraphBLAS, Timothy A. Davis, (c) 2017-2025, All Rights Reserved.
// SPDX-License-Identifier: Apache-2.0

//------------------------------------------------------------------------------

// Set a single scalar, s = x, typecasting from the type of x to
// the type of s as needed.

#include "GB.h"

#define GB_SET(type,T,ampersand)                                            \
GrB_Info GB_EVAL2 (GRB (Scalar_setElement_), T)    /* s = x */              \
(                                                                           \
    GrB_Scalar s,                       /* GrB_Scalar to modify       */    \
    type x                              /* user scalar to assign to s */    \
)                                                                           \
{                                                                           \
    GB_RETURN_IF_NULL (s) ;                                                 \
    GB_WHERE1 (s, "GrB_Scalar_setElement_" GB_STR(T) " (w, x)") ;           \
    ASSERT (GB_SCALAR_OK (s)) ;                                             \
    return (GB_setElement ((GrB_Matrix) s, NULL, ampersand x, 0, 0,         \
        GB_ ## T ## _code, Werk)) ;                                         \
}

GB_SET (bool      , BOOL   , &)
GB_SET (int8_t    , INT8   , &)
GB_SET (int16_t   , INT16  , &)
GB_SET (int32_t   , INT32  , &)
GB_SET (int64_t   , INT64  , &)
GB_SET (uint8_t   , UINT8  , &)
GB_SET (uint16_t  , UINT16 , &)
GB_SET (uint32_t  , UINT32 , &)
GB_SET (uint64_t  , UINT64 , &)
GB_SET (float     , FP32   , &)
GB_SET (double    , FP64   , &)
GB_SET (void *    , UDT    ,  )

//------------------------------------------------------------------------------
// GxB_Scalar_setElement for complex types
//------------------------------------------------------------------------------

#define GXB_SET(type,T,ampersand)                                           \
GrB_Info GB_EVAL2 (GXB (Scalar_setElement_), T)    /* s = x */              \
(                                                                           \
    GrB_Scalar s,                       /* GrB_Scalar to modify       */    \
    type x                              /* user scalar to assign to s */    \
)                                                                           \
{                                                                           \
    GB_RETURN_IF_NULL (s) ;                                                 \
    GB_WHERE1 (s, "GxB_Scalar_setElement_" GB_STR(T) " (w, x)") ;           \
    ASSERT (GB_SCALAR_OK (s)) ;                                             \
    return (GB_setElement ((GrB_Matrix) s, NULL, ampersand x, 0, 0,         \
        GB_ ## T ## _code, Werk)) ;                                         \
}

GXB_SET (GxB_FC32_t, FC32  , &)
GXB_SET (GxB_FC64_t, FC64  , &)

//------------------------------------------------------------------------------
// GxB_Scalar_setElement: set an entry in a GrB_Scalar (historical)
//------------------------------------------------------------------------------

GrB_Info GxB_Scalar_setElement_BOOL   (GrB_Scalar s, bool     x) { return (GrB_Scalar_setElement_BOOL   (s,x)) ; }
GrB_Info GxB_Scalar_setElement_INT8   (GrB_Scalar s, int8_t   x) { return (GrB_Scalar_setElement_INT8   (s,x)) ; }
GrB_Info GxB_Scalar_setElement_INT16  (GrB_Scalar s, int16_t  x) { return (GrB_Scalar_setElement_INT16  (s,x)) ; }
GrB_Info GxB_Scalar_setElement_INT32  (GrB_Scalar s, int32_t  x) { return (GrB_Scalar_setElement_INT32  (s,x)) ; }
GrB_Info GxB_Scalar_setElement_INT64  (GrB_Scalar s, int64_t  x) { return (GrB_Scalar_setElement_INT64  (s,x)) ; }
GrB_Info GxB_Scalar_setElement_UINT8  (GrB_Scalar s, uint8_t  x) { return (GrB_Scalar_setElement_UINT8  (s,x)) ; }
GrB_Info GxB_Scalar_setElement_UINT16 (GrB_Scalar s, uint16_t x) { return (GrB_Scalar_setElement_UINT16 (s,x)) ; }
GrB_Info GxB_Scalar_setElement_UINT32 (GrB_Scalar s, uint32_t x) { return (GrB_Scalar_setElement_UINT32 (s,x)) ; }
GrB_Info GxB_Scalar_setElement_UINT64 (GrB_Scalar s, uint64_t x) { return (GrB_Scalar_setElement_UINT64 (s,x)) ; }
GrB_Info GxB_Scalar_setElement_FP32   (GrB_Scalar s, float    x) { return (GrB_Scalar_setElement_FP32   (s,x)) ; }
GrB_Info GxB_Scalar_setElement_FP64   (GrB_Scalar s, double   x) { return (GrB_Scalar_setElement_FP64   (s,x)) ; }
GrB_Info GxB_Scalar_setElement_UDT    (GrB_Scalar s, void    *x) { return (GrB_Scalar_setElement_UDT    (s,x)) ; }

