//------------------------------------------------------------------------------
// GB_enumify_identity: return ecode for identity value of an op of a monoid 
//------------------------------------------------------------------------------

// SuiteSparse:GraphBLAS, Timothy A. Davis, (c) 2017-2023, All Rights Reserved.
// SPDX-License-Identifier: Apache-2.0

//------------------------------------------------------------------------------

#include "GB.h"
#include "GB_stringify.h"

void GB_enumify_identity
(
    // output:
    int *ecode,             // enumerated identity, 0 to 31
    // inputs:
    GB_Opcode opcode,       // built-in binary opcode of a monoid
    GB_Type_code zcode      // type code of the operator
)
{ 

    int e = 31 ;            // default: use the monoid->identity bytes

    switch (opcode)
    {

        case GB_PLUS_binop_code     : e = 0 ; break ; // 0

        case GB_TIMES_binop_code    : e = 1 ; break ; // 1

        case GB_LAND_binop_code     : 
        case GB_EQ_binop_code       :   // LXNOR
            e = (zcode == GB_BOOL_code) ? 2 : (31) ; break ; // true

        case GB_LOR_binop_code      : 
        case GB_LXOR_binop_code     : 
            e = (zcode == GB_BOOL_code) ? 3 : (31) ; break ; // false

        case GB_MIN_binop_code :

            switch (zcode)
            {
                case GB_BOOL_code   : e =  2 ; break ; // true
                case GB_INT8_code   : e =  4 ; break ; // INT8_MAX
                case GB_INT16_code  : e =  5 ; break ; // INT16_MAX
                case GB_INT32_code  : e =  6 ; break ; // INT32_MAX
                case GB_INT64_code  : e =  7 ; break ; // INT64_MAX
                case GB_UINT8_code  : e =  8 ; break ; // UINT8_MAX
                case GB_UINT16_code : e =  9 ; break ; // UINT16_MAX
                case GB_UINT32_code : e = 10 ; break ; // UINT32_MAX
                case GB_UINT64_code : e = 11 ; break ; // UINT64_MAX
                case GB_FP32_code   : 
                case GB_FP64_code   : e = 12 ; break ; // INFINITY
                default: ;
            }
            break ;

        case GB_MAX_binop_code :

            switch (zcode)
            {
                case GB_BOOL_code   : e =  3 ; break ; // false
                case GB_INT8_code   : e = 13 ; break ; // INT8_MIN
                case GB_INT16_code  : e = 14 ; break ; // INT16_MIN
                case GB_INT32_code  : e = 15 ; break ; // INT32_MIN
                case GB_INT64_code  : e = 16 ; break ; // INT64_MIN
                case GB_UINT8_code  : 
                case GB_UINT16_code : 
                case GB_UINT32_code : 
                case GB_UINT64_code : e =  0 ; break ; // 0
                case GB_FP32_code   : 
                case GB_FP64_code   : e = 17 ; break ; // -INFINITY
                default: ;
            }
            break ;

        case GB_ANY_binop_code      : e = 18 ; break ; // 0, for ANY op only

        case GB_BOR_binop_code      : 
        case GB_BXOR_binop_code     : e = 0 ; break ; // 0

        case GB_BAND_binop_code     : 
        case GB_BXNOR_binop_code    : 

            switch (zcode)
            {
                case GB_UINT8_code  : e = 19 ; break ; // 0xFF
                case GB_UINT16_code : e = 20 ; break ; // 0xFFFF
                case GB_UINT32_code : e = 21 ; break ; // 0xFFFFFFFF
                case GB_UINT64_code : e = 22 ; break ; // 0xFFFFFFFFFFFFFFFF
                default: ;
            }
            break ;

        default: ;
    }

    (*ecode) = e ;
}

