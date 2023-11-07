//------------------------------------------------------------------------------
// gb_binop_to_monoid: convert a binary operator to the corresponding monoid
//------------------------------------------------------------------------------

// SuiteSparse:GraphBLAS, Timothy A. Davis, (c) 2017-2023, All Rights Reserved.
// SPDX-License-Identifier: Apache-2.0

//------------------------------------------------------------------------------

#include "gb_interface.h"

GrB_Monoid gb_binop_to_monoid           // return monoid from a binary op
(
    GrB_BinaryOp op
)
{ 

    GrB_Monoid monoid ;

    // MIN monoids:
         if (op == GrB_MIN_INT8   ) monoid = GrB_MIN_MONOID_INT8 ;
    else if (op == GrB_MIN_INT16  ) monoid = GrB_MIN_MONOID_INT16 ;
    else if (op == GrB_MIN_INT32  ) monoid = GrB_MIN_MONOID_INT32 ;
    else if (op == GrB_MIN_INT64  ) monoid = GrB_MIN_MONOID_INT64 ;
    else if (op == GrB_MIN_UINT8  ) monoid = GrB_MIN_MONOID_UINT8 ;
    else if (op == GrB_MIN_UINT16 ) monoid = GrB_MIN_MONOID_UINT16 ;
    else if (op == GrB_MIN_UINT32 ) monoid = GrB_MIN_MONOID_UINT32 ;
    else if (op == GrB_MIN_UINT64 ) monoid = GrB_MIN_MONOID_UINT64 ;
    else if (op == GrB_MIN_FP32   ) monoid = GrB_MIN_MONOID_FP32 ;
    else if (op == GrB_MIN_FP64   ) monoid = GrB_MIN_MONOID_FP64 ;

    // MAX monoids:
    else if (op == GrB_MAX_INT8   ) monoid = GrB_MAX_MONOID_INT8 ;
    else if (op == GrB_MAX_INT16  ) monoid = GrB_MAX_MONOID_INT16 ;
    else if (op == GrB_MAX_INT32  ) monoid = GrB_MAX_MONOID_INT32 ;
    else if (op == GrB_MAX_INT64  ) monoid = GrB_MAX_MONOID_INT64 ;
    else if (op == GrB_MAX_UINT8  ) monoid = GrB_MAX_MONOID_UINT8 ;
    else if (op == GrB_MAX_UINT16 ) monoid = GrB_MAX_MONOID_UINT16 ;
    else if (op == GrB_MAX_UINT32 ) monoid = GrB_MAX_MONOID_UINT32 ;
    else if (op == GrB_MAX_UINT64 ) monoid = GrB_MAX_MONOID_UINT64 ;
    else if (op == GrB_MAX_FP32   ) monoid = GrB_MAX_MONOID_FP32 ;
    else if (op == GrB_MAX_FP64   ) monoid = GrB_MAX_MONOID_FP64 ;

    // PLUS monoids:
    else if (op == GrB_PLUS_INT8  ) monoid = GrB_PLUS_MONOID_INT8 ;
    else if (op == GrB_PLUS_INT16 ) monoid = GrB_PLUS_MONOID_INT16 ;
    else if (op == GrB_PLUS_INT32 ) monoid = GrB_PLUS_MONOID_INT32 ;
    else if (op == GrB_PLUS_INT64 ) monoid = GrB_PLUS_MONOID_INT64 ;
    else if (op == GrB_PLUS_UINT8 ) monoid = GrB_PLUS_MONOID_UINT8 ;
    else if (op == GrB_PLUS_UINT16) monoid = GrB_PLUS_MONOID_UINT16 ;
    else if (op == GrB_PLUS_UINT32) monoid = GrB_PLUS_MONOID_UINT32 ;
    else if (op == GrB_PLUS_UINT64) monoid = GrB_PLUS_MONOID_UINT64 ;
    else if (op == GrB_PLUS_FP32  ) monoid = GrB_PLUS_MONOID_FP32 ;
    else if (op == GrB_PLUS_FP64  ) monoid = GrB_PLUS_MONOID_FP64 ;

    // PLUS monoids for complex:
    else if (op == GxB_PLUS_FC32  ) monoid = GxB_PLUS_FC32_MONOID ;
    else if (op == GxB_PLUS_FC64  ) monoid = GxB_PLUS_FC64_MONOID ;

    // TIMES monoids:
    else if (op == GrB_TIMES_INT8  ) monoid = GrB_TIMES_MONOID_INT8 ;
    else if (op == GrB_TIMES_INT16 ) monoid = GrB_TIMES_MONOID_INT16 ;
    else if (op == GrB_TIMES_INT32 ) monoid = GrB_TIMES_MONOID_INT32 ;
    else if (op == GrB_TIMES_INT64 ) monoid = GrB_TIMES_MONOID_INT64 ;
    else if (op == GrB_TIMES_UINT8 ) monoid = GrB_TIMES_MONOID_UINT8 ;
    else if (op == GrB_TIMES_UINT16) monoid = GrB_TIMES_MONOID_UINT16 ;
    else if (op == GrB_TIMES_UINT32) monoid = GrB_TIMES_MONOID_UINT32 ;
    else if (op == GrB_TIMES_UINT64) monoid = GrB_TIMES_MONOID_UINT64 ;
    else if (op == GrB_TIMES_FP32  ) monoid = GrB_TIMES_MONOID_FP32 ;
    else if (op == GrB_TIMES_FP64  ) monoid = GrB_TIMES_MONOID_FP64 ;

    // TIMES monoids for complex:
    else if (op == GxB_TIMES_FC32  ) monoid = GxB_TIMES_FC32_MONOID ;
    else if (op == GxB_TIMES_FC64  ) monoid = GxB_TIMES_FC64_MONOID ;

    // ANY monoids:
    else if (op == GxB_ANY_BOOL  ) monoid = GxB_ANY_BOOL_MONOID ;
    else if (op == GxB_ANY_INT8  ) monoid = GxB_ANY_INT8_MONOID ;
    else if (op == GxB_ANY_INT16 ) monoid = GxB_ANY_INT16_MONOID ;
    else if (op == GxB_ANY_INT32 ) monoid = GxB_ANY_INT32_MONOID ;
    else if (op == GxB_ANY_INT64 ) monoid = GxB_ANY_INT64_MONOID ;
    else if (op == GxB_ANY_UINT8 ) monoid = GxB_ANY_UINT8_MONOID ;
    else if (op == GxB_ANY_UINT16) monoid = GxB_ANY_UINT16_MONOID ;
    else if (op == GxB_ANY_UINT32) monoid = GxB_ANY_UINT32_MONOID ;
    else if (op == GxB_ANY_UINT64) monoid = GxB_ANY_UINT64_MONOID ;
    else if (op == GxB_ANY_FP32  ) monoid = GxB_ANY_FP32_MONOID ;
    else if (op == GxB_ANY_FP64  ) monoid = GxB_ANY_FP64_MONOID ;
    else if (op == GxB_ANY_FC32  ) monoid = GxB_ANY_FC32_MONOID ;
    else if (op == GxB_ANY_FC64  ) monoid = GxB_ANY_FC64_MONOID ;

    // Boolean monoids:
    else if (op == GrB_LOR         ) monoid = GrB_LOR_MONOID_BOOL ;
    else if (op == GrB_LAND        ) monoid = GrB_LAND_MONOID_BOOL ;
    else if (op == GrB_LXOR        ) monoid = GrB_LXOR_MONOID_BOOL ;
    // these two operators are identical:
    else if (op == GrB_EQ_BOOL     ||
             op == GrB_LXNOR       ) monoid = GrB_LXNOR_MONOID_BOOL ;

    // BOR monoids (bitwise or):
    else if (op == GrB_BOR_UINT8   ) monoid = GxB_BOR_UINT8_MONOID ;
    else if (op == GrB_BOR_UINT16  ) monoid = GxB_BOR_UINT16_MONOID ;
    else if (op == GrB_BOR_UINT32  ) monoid = GxB_BOR_UINT32_MONOID ;
    else if (op == GrB_BOR_UINT64  ) monoid = GxB_BOR_UINT64_MONOID ;

    // BAND monoids (bitwise and):
    else if (op == GrB_BAND_UINT8   ) monoid = GxB_BAND_UINT8_MONOID ;
    else if (op == GrB_BAND_UINT16  ) monoid = GxB_BAND_UINT16_MONOID ;
    else if (op == GrB_BAND_UINT32  ) monoid = GxB_BAND_UINT32_MONOID ;
    else if (op == GrB_BAND_UINT64  ) monoid = GxB_BAND_UINT64_MONOID ;

    // BXOR monoids (bitwise xor):
    else if (op == GrB_BXOR_UINT8   ) monoid = GxB_BXOR_UINT8_MONOID ;
    else if (op == GrB_BXOR_UINT16  ) monoid = GxB_BXOR_UINT16_MONOID ;
    else if (op == GrB_BXOR_UINT32  ) monoid = GxB_BXOR_UINT32_MONOID ;
    else if (op == GrB_BXOR_UINT64  ) monoid = GxB_BXOR_UINT64_MONOID ;

    // BXNOR monoids (bitwise xnor):
    else if (op == GrB_BXNOR_UINT8   ) monoid = GxB_BXNOR_UINT8_MONOID ;
    else if (op == GrB_BXNOR_UINT16  ) monoid = GxB_BXNOR_UINT16_MONOID ;
    else if (op == GrB_BXNOR_UINT32  ) monoid = GxB_BXNOR_UINT32_MONOID ;
    else if (op == GrB_BXNOR_UINT64  ) monoid = GxB_BXNOR_UINT64_MONOID ;

    else
    { 
        ERROR ("not a valid monoid") ;
    }

    return (monoid) ;
}

