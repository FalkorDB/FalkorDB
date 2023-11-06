//------------------------------------------------------------------------------
// GB_enumify_terminal: return enum of terminal value
//------------------------------------------------------------------------------

// SuiteSparse:GraphBLAS, Timothy A. Davis, (c) 2017-2023, All Rights Reserved.
// SPDX-License-Identifier: Apache-2.0

//------------------------------------------------------------------------------

#include "GB.h"
#include "GB_stringify.h"

void GB_enumify_terminal        // enumify the terminal value
(
    // output:
    int *ecode,                 // enumerated terminal, 0 to 31
    // input:
    GB_Opcode opcode,           // built-in binary opcode of a monoid
    GB_Type_code zcode          // type code of the operator
)
{ 

    int e = 31 ;                // default is a non-terminal monoid

    switch (opcode)
    {

        case GB_PLUS_binop_code : 

            // boolean PLUS (or) is terminal (true), others are not terminal
            e = (zcode == GB_BOOL_code) ? 2 : 31 ;
            break ;

        case GB_TIMES_binop_code : 

            switch (zcode)
            {
                case GB_BOOL_code   : 
                    e = 3 ;             // false (boolean AND)
                    break ;
                case GB_INT8_code   : 
                case GB_INT16_code  : 
                case GB_INT32_code  : 
                case GB_INT64_code  : 
                case GB_UINT8_code  : 
                case GB_UINT16_code : 
                case GB_UINT32_code : 
                case GB_UINT64_code : 
                    e = 0 ;             // 0
                    break ;
                default: ;              // builtin with no terminal value
            }
            break ;

        case GB_LOR_binop_code      : 

                e = 2 ;                 // true
                break ;

        case GB_LAND_binop_code     : 

                e = 3 ;                 // false
                break ;

        case GB_MIN_binop_code : 

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

        case GB_MAX_binop_code : 

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

        case GB_ANY_binop_code : 

            e = 18 ;                    // no specific terminal value
            break ;

        case GB_LXOR_binop_code     : 
        case GB_EQ_binop_code       :   // LXNOR
        case GB_BXOR_binop_code     : 
        case GB_BXNOR_binop_code    : 
        default                     :   // builtin with no terminal value
            break ;

        case GB_BOR_binop_code      : 

            switch (zcode)
            {
                case GB_UINT8_code  : e = 19 ; break ; // 0xFF
                case GB_UINT16_code : e = 20 ; break ; // 0xFFFF
                case GB_UINT32_code : e = 21 ; break ; // 0xFFFFFFFF
                case GB_UINT64_code : e = 22 ; break ; // 0xFFFFFFFFFFFFFFFF
                default: ;
            }
            break ;

        case GB_BAND_binop_code     : e = 0  ; break ; // 0

        case GB_USER_binop_code     :   // user-defined monoid

            e = 30 ;
            break ;

    }

    (*ecode) = e ;
}

