//------------------------------------------------------------------------------
// GrB_Vector_setElement: set an entry in a vector, w (row) = x
//------------------------------------------------------------------------------

// SuiteSparse:GraphBLAS, Timothy A. Davis, (c) 2017-2023, All Rights Reserved.
// SPDX-License-Identifier: Apache-2.0

//------------------------------------------------------------------------------

// Set a single scalar, w(row) = x, typecasting from the type of x to
// the type of w as needed.

#define GB_FREE_ALL ;
#include "GB.h"

#define GB_SET(prefix,type,T,ampersand)                                     \
GrB_Info GB_EVAL3 (prefix, _Vector_setElement_, T)    /* w(row) = x */      \
(                                                                           \
    GrB_Vector w,                       /* vector to modify           */    \
    type x,                             /* scalar to assign to w(row) */    \
    GrB_Index row                       /* row index                  */    \
)                                                                           \
{                                                                           \
    GB_WHERE (w, "GrB_Vector_setElement_" GB_STR(T) " (w, x, row)") ;       \
    GB_RETURN_IF_NULL_OR_FAULTY (w) ;                                       \
    ASSERT (GB_VECTOR_OK (w)) ;                                             \
    return (GB_setElement ((GrB_Matrix) w, NULL, ampersand x, row, 0,       \
        GB_ ## T ## _code, Werk)) ;                                      \
}

GB_SET (GrB, bool      , BOOL   , &)
GB_SET (GrB, int8_t    , INT8   , &)
GB_SET (GrB, int16_t   , INT16  , &)
GB_SET (GrB, int32_t   , INT32  , &)
GB_SET (GrB, int64_t   , INT64  , &)
GB_SET (GrB, uint8_t   , UINT8  , &)
GB_SET (GrB, uint16_t  , UINT16 , &)
GB_SET (GrB, uint32_t  , UINT32 , &)
GB_SET (GrB, uint64_t  , UINT64 , &)
GB_SET (GrB, float     , FP32   , &)
GB_SET (GrB, double    , FP64   , &)
GB_SET (GxB, GxB_FC32_t, FC32   , &)
GB_SET (GxB, GxB_FC64_t, FC64   , &)
GB_SET (GrB, void *    , UDT    ,  )

//------------------------------------------------------------------------------
// GrB_Vector_setElement_Scalar: set an entry in a vector from a GrB_Scalar
//------------------------------------------------------------------------------

// When the GrB_Scalar has a single entry, this method is just like the
// setElement methods defined above.  If the GrB_Scalar has no entry, then it
// acts like GrB_Vector_removeElement.

GrB_Info GrB_Vector_setElement_Scalar
(
    GrB_Vector w,                       // vector to modify
    GrB_Scalar scalar,                  // scalar to assign to w(row)
    GrB_Index row                       // row index
)
{

    //--------------------------------------------------------------------------
    // check inputs
    //--------------------------------------------------------------------------

    GB_WHERE (w, "GrB_Vector_setElement_Scalar (w, x, row)") ;
    GB_RETURN_IF_NULL_OR_FAULTY (w) ;
    GB_RETURN_IF_NULL_OR_FAULTY (scalar) ;
    ASSERT (GB_VECTOR_OK (w)) ;

    //--------------------------------------------------------------------------
    // set or remove the element
    //--------------------------------------------------------------------------

    GB_MATRIX_WAIT (scalar) ;
    if (GB_nnz ((GrB_Matrix) scalar) > 0)
    { 
        // set the element: w(row) = scalar
        return (GB_setElement ((GrB_Matrix) w, NULL, scalar->x, row, 0,
            scalar->type->code, Werk)) ;
    }
    else
    { 
        // delete the w(row) element
        return (GB_Vector_removeElement (w, row, Werk)) ;
    }
}

