//------------------------------------------------------------------------------
// GB_monoid_name_get: get the name of a built-in monoid
//------------------------------------------------------------------------------

// SuiteSparse:GraphBLAS, Timothy A. Davis, (c) 2017-2025, All Rights Reserved.
// SPDX-License-Identifier: Apache-2.0

//------------------------------------------------------------------------------

#include "get_set/GB_get_set.h"

const char *GB_monoid_name_get (GrB_Monoid monoid)
{

    if (monoid->user_name_size > 0)
    { 
        // user-defined monoid, with name defined by GrB_set
        return (monoid->user_name) ;
    }

    GB_Opcode opcode = monoid->op->opcode ;
    GB_Type_code zcode = monoid->op->ztype->code ;

    switch (opcode)
    {

        case GB_ANY_binop_code       :   // z = x or y

            switch (zcode)
            {
                case GB_BOOL_code    : return ("GxB_ANY_BOOL_MONOID"  ) ;
                case GB_INT8_code    : return ("GxB_ANY_INT8_MONOID"  ) ;
                case GB_INT16_code   : return ("GxB_ANY_INT16_MONOID" ) ;
                case GB_INT32_code   : return ("GxB_ANY_INT32_MONOID" ) ;
                case GB_INT64_code   : return ("GxB_ANY_INT64_MONOID" ) ;
                case GB_UINT8_code   : return ("GxB_ANY_UINT8_MONOID" ) ;
                case GB_UINT16_code  : return ("GxB_ANY_UINT16_MONOID") ;
                case GB_UINT32_code  : return ("GxB_ANY_UINT32_MONOID") ;
                case GB_UINT64_code  : return ("GxB_ANY_UINT64_MONOID") ;
                case GB_FP32_code    : return ("GxB_ANY_FP32_MONOID"  ) ;
                case GB_FP64_code    : return ("GxB_ANY_FP64_MONOID"  ) ;
                case GB_FC32_code    : return ("GxB_ANY_FC32_MONOID"  ) ;
                case GB_FC64_code    : return ("GxB_ANY_FC64_MONOID"  ) ;
                default :;
            }
            break ;

        case GB_MIN_binop_code       :   // z = min(x,y)

            switch (zcode)
            {
                case GB_INT8_code    : return ("GrB_MIN_MONOID_INT8"  ) ;
                case GB_INT16_code   : return ("GrB_MIN_MONOID_INT16" ) ;
                case GB_INT32_code   : return ("GrB_MIN_MONOID_INT32" ) ;
                case GB_INT64_code   : return ("GrB_MIN_MONOID_INT64" ) ;
                case GB_UINT8_code   : return ("GrB_MIN_MONOID_UINT8" ) ;
                case GB_UINT16_code  : return ("GrB_MIN_MONOID_UINT16") ;
                case GB_UINT32_code  : return ("GrB_MIN_MONOID_UINT32") ;
                case GB_UINT64_code  : return ("GrB_MIN_MONOID_UINT64") ;
                case GB_FP32_code    : return ("GrB_MIN_MONOID_FP32"  ) ;
                case GB_FP64_code    : return ("GrB_MIN_MONOID_FP64"  ) ;
                default :;
            }
            break ;

        case GB_MAX_binop_code       :   // z = max(x,y)

            switch (zcode)
            {
                case GB_INT8_code    : return ("GrB_MAX_MONOID_INT8"  ) ;
                case GB_INT16_code   : return ("GrB_MAX_MONOID_INT16" ) ;
                case GB_INT32_code   : return ("GrB_MAX_MONOID_INT32" ) ;
                case GB_INT64_code   : return ("GrB_MAX_MONOID_INT64" ) ;
                case GB_UINT8_code   : return ("GrB_MAX_MONOID_UINT8" ) ;
                case GB_UINT16_code  : return ("GrB_MAX_MONOID_UINT16") ;
                case GB_UINT32_code  : return ("GrB_MAX_MONOID_UINT32") ;
                case GB_UINT64_code  : return ("GrB_MAX_MONOID_UINT64") ;
                case GB_FP32_code    : return ("GrB_MAX_MONOID_FP32"  ) ;
                case GB_FP64_code    : return ("GrB_MAX_MONOID_FP64"  ) ;
                default :;
            }
            break ;

        case GB_PLUS_binop_code      :   // z = x + y

            switch (zcode)
            {
                case GB_INT8_code    : return ("GrB_PLUS_MONOID_INT8"  ) ;
                case GB_INT16_code   : return ("GrB_PLUS_MONOID_INT16" ) ;
                case GB_INT32_code   : return ("GrB_PLUS_MONOID_INT32" ) ;
                case GB_INT64_code   : return ("GrB_PLUS_MONOID_INT64" ) ;
                case GB_UINT8_code   : return ("GrB_PLUS_MONOID_UINT8" ) ;
                case GB_UINT16_code  : return ("GrB_PLUS_MONOID_UINT16") ;
                case GB_UINT32_code  : return ("GrB_PLUS_MONOID_UINT32") ;
                case GB_UINT64_code  : return ("GrB_PLUS_MONOID_UINT64") ;
                case GB_FP32_code    : return ("GrB_PLUS_MONOID_FP32"  ) ;
                case GB_FP64_code    : return ("GrB_PLUS_MONOID_FP64"  ) ;
                case GB_FC32_code    : return ("GxB_PLUS_FC32_MONOID"  ) ;
                case GB_FC64_code    : return ("GxB_PLUS_FC64_MONOID"  ) ;
                default :;
            }
            break ;

        case GB_TIMES_binop_code     :   // z = x * y

            switch (zcode)
            {
                case GB_INT8_code    : return ("GrB_TIMES_MONOID_INT8"  ) ;
                case GB_INT16_code   : return ("GrB_TIMES_MONOID_INT16" ) ;
                case GB_INT32_code   : return ("GrB_TIMES_MONOID_INT32" ) ;
                case GB_INT64_code   : return ("GrB_TIMES_MONOID_INT64" ) ;
                case GB_UINT8_code   : return ("GrB_TIMES_MONOID_UINT8" ) ;
                case GB_UINT16_code  : return ("GrB_TIMES_MONOID_UINT16") ;
                case GB_UINT32_code  : return ("GrB_TIMES_MONOID_UINT32") ;
                case GB_UINT64_code  : return ("GrB_TIMES_MONOID_UINT64") ;
                case GB_FP32_code    : return ("GrB_TIMES_MONOID_FP32"  ) ;
                case GB_FP64_code    : return ("GrB_TIMES_MONOID_FP64"  ) ;
                case GB_FC32_code    : return ("GxB_TIMES_FC32_MONOID"  ) ;
                case GB_FC64_code    : return ("GxB_TIMES_FC64_MONOID"  ) ;
                default :;
            }
            break ;

        case GB_LOR_binop_code       :   // z = (x != 0) || (y != 0)

            switch (zcode)
            {
                case GB_BOOL_code    : return ("GrB_LOR_MONOID_BOOL") ;
                default :;
            }
            break ;

        case GB_LAND_binop_code      :   // z = (x != 0) && (y != 0)

            switch (zcode)
            {
                case GB_BOOL_code    : return ("GrB_LAND_MONOID_BOOL") ;
                default :;
            }
            break ;

        case GB_LXOR_binop_code      :   // z = (x != 0) != (y != 0)

            switch (zcode)
            {
                case GB_BOOL_code    : return ("GrB_LXOR_MONOID_BOOL") ;
                default :;
            }
            break ;

        case GB_EQ_binop_code        :  // z = (x == y), is LXNOR for bool

            switch (zcode)
            {
                case GB_BOOL_code    : return ("GrB_LXNOR_MONOID_BOOL") ;
                default :;
            }
            break ;

        case GB_BOR_binop_code       :   // z = (x | y), bitwise or

            switch (zcode)
            {
                case GB_UINT8_code   : return ("GxB_BOR_UINT8_MONOID" ) ;
                case GB_UINT16_code  : return ("GxB_BOR_UINT16_MONOID") ;
                case GB_UINT32_code  : return ("GxB_BOR_UINT32_MONOID") ;
                case GB_UINT64_code  : return ("GxB_BOR_UINT64_MONOID") ;
                default :;
            }
            break ;

        case GB_BAND_binop_code      :   // z = (x & y), bitwise and

            switch (zcode)
            {
                case GB_UINT8_code   : return ("GxB_BAND_UINT8_MONOID" ) ;
                case GB_UINT16_code  : return ("GxB_BAND_UINT16_MONOID") ;
                case GB_UINT32_code  : return ("GxB_BAND_UINT32_MONOID") ;
                case GB_UINT64_code  : return ("GxB_BAND_UINT64_MONOID") ;
                default :;
            }
            break ;

        case GB_BXOR_binop_code      :   // z = (x ^ y), bitwise xor

            switch (zcode)
            {
                case GB_UINT8_code   : return ("GxB_BXOR_UINT8_MONOID" ) ;
                case GB_UINT16_code  : return ("GxB_BXOR_UINT16_MONOID") ;
                case GB_UINT32_code  : return ("GxB_BXOR_UINT32_MONOID") ;
                case GB_UINT64_code  : return ("GxB_BXOR_UINT64_MONOID") ;
                default :;
            }
            break ;

        case GB_BXNOR_binop_code     :   // z = ~(x ^ y), bitwise xnor

            switch (zcode)
            {
                case GB_UINT8_code   : return ("GxB_BXNOR_UINT8_MONOID" ) ;
                case GB_UINT16_code  : return ("GxB_BXNOR_UINT16_MONOID") ;
                case GB_UINT32_code  : return ("GxB_BXNOR_UINT32_MONOID") ;
                case GB_UINT64_code  : return ("GxB_BXNOR_UINT64_MONOID") ;
                default :;
            }
            break ;

        default: ;
    }

    return (NULL) ;
}

