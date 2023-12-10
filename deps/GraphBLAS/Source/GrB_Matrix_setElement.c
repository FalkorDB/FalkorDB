//------------------------------------------------------------------------------
// GrB_Matrix_setElement: set an entry in a matrix, C(row,col) = x
//------------------------------------------------------------------------------

// SuiteSparse:GraphBLAS, Timothy A. Davis, (c) 2017-2023, All Rights Reserved.
// SPDX-License-Identifier: Apache-2.0

//------------------------------------------------------------------------------

// Set a single entry in a matrix, C(row,col) = x,
// typecasting from the type of x to the type of C, as needed.

#define GB_FREE_ALL ;
#include "GB.h"

#define GB_SET(prefix,type,T,ampersand)                                     \
GrB_Info GB_EVAL3 (prefix, _Matrix_setElement_, T) /* C (row,col) = x */    \
(                                                                           \
    GrB_Matrix C,                       /* matrix to modify               */\
    type x,                             /* scalar to assign to C(row,col) */\
    GrB_Index row,                      /* row index                      */\
    GrB_Index col                       /* column index                   */\
)                                                                           \
{                                                                           \
    GB_WHERE (C, GB_STR(prefix) "_Matrix_setElement_" GB_STR(T)             \
        " (C, row, col, x)") ;                                              \
    GB_RETURN_IF_NULL_OR_FAULTY (C) ;                                       \
    return (GB_setElement (C, NULL, ampersand x, row, col,                  \
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
// GrB_Matrix_setElement_Scalar: set an entry in a matrix from a GrB_Scalar
//------------------------------------------------------------------------------

// When the GrB_Scalar has a single entry, this method is just like the
// setElement methods defined above.  If the GrB_Scalar has no entry, then it
// acts like GrB_Matrix_removeElement.

GrB_Info GrB_Matrix_setElement_Scalar
(
    GrB_Matrix C,                       // matrix to modify
    GrB_Scalar scalar,                  // scalar to assign to C(row,col)
    GrB_Index row,                      // row index
    GrB_Index col                       // column index
)
{

    //--------------------------------------------------------------------------
    // check inputs
    //--------------------------------------------------------------------------

    GB_WHERE (C, "GrB_Matrix_setElement_Scalar (C, x, row, col)") ;
    GB_RETURN_IF_NULL_OR_FAULTY (C) ;
    GB_RETURN_IF_NULL_OR_FAULTY (scalar) ;

    //--------------------------------------------------------------------------
    // set or remove the element
    //--------------------------------------------------------------------------

    GB_MATRIX_WAIT (scalar) ;
    if (GB_nnz ((GrB_Matrix) scalar) > 0)
    { 
        // set the element: C(row,col) = scalar
        return (GB_setElement (C, NULL, scalar->x, row, col,
            scalar->type->code, Werk)) ;
    }
    else
    { 
        // delete the C(row,col) element
        return (GB_Matrix_removeElement (C, row, col, Werk)) ;
    }
}

