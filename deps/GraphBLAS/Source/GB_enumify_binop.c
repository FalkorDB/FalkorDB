//------------------------------------------------------------------------------
// GB_enumify_binop: convert binary opcode and xcode into a single enum
//------------------------------------------------------------------------------

// SuiteSparse:GraphBLAS, Timothy A. Davis, (c) 2017-2023, All Rights Reserved.
// SPDX-License-Identifier: Apache-2.0

//------------------------------------------------------------------------------

// ecodes 0 to 31 can be used as a monoid, but only 0:22 are currently in use.
// ecodes 32 and up are not valid for use in a monoid; only 32:139 are in use.

#include "GB.h"
#include "GB_stringify.h"

void GB_enumify_binop
(
    // output:
    int *ecode,         // enumerated operator, range 0 to 255
    // input:
    GB_Opcode opcode,   // opcode of GraphBLAS operator to convert into a macro
    GB_Type_code xcode, // op->xtype->code of the operator
    bool for_semiring   // true for A*B, false for A+B or A.*B
)
{ 

    int e = 0 ;

    switch (opcode)
    {

        //----------------------------------------------------------------------
        // user-defined operator
        //----------------------------------------------------------------------

        case GB_USER_binop_code : 

            e = 0 ; break ;

        //----------------------------------------------------------------------
        // built-in ops that can be used in a monoid or for GrB_build methods:
        //----------------------------------------------------------------------

        case GB_FIRST_binop_code :  // z = x, can be used as the ANY monoid

            e = 1 ; break ;

        case GB_ANY_binop_code :    // z = y (same as SECOND)
        case GB_SECOND_binop_code : // z = y (same as ANY, but not a monoid)

            e = 2 ; break ;

        case GB_MIN_binop_code :    // z = min(x,y)

            switch (xcode)
            {
                case GB_BOOL_code   : e = 18 ; break ; // x && y
                case GB_FP32_code   : e =  3 ; break ; // fminf (x,y)
                case GB_FP64_code   : e =  4 ; break ; // fmin (x,y)
                default             : e =  5 ; break ; // GB_IMIN (x,y)
            }
            break ;

        case GB_MAX_binop_code :    // z = max(x,y)

            switch (xcode)
            {
                case GB_BOOL_code   : e = 17 ; break ; // x || y
                case GB_FP32_code   : e =  6 ; break ; // fmaxf (x,y)
                case GB_FP64_code   : e =  7 ; break ; // fmax (x,y)
                default             : e =  8 ; break ; // GB_IMAX (x,y)
            }
            break ;

        case GB_PLUS_binop_code :   // z = x + y

            switch (xcode)
            {
                case GB_BOOL_code   : e = 17 ; break ; // x || y
                case GB_FC32_code   : e =  9 ; break ; // GB_FC32_add(x,y)
                case GB_FC64_code   : e = 10 ; break ; // GB_FC64_add(x,y)
                default             : e = 11 ; break ; // x + y
            }

            break ;

        case GB_TIMES_binop_code :  // z = x * y

            switch (xcode)
            {
                case GB_BOOL_code   : e = 18 ; break ; // x && y
                case GB_FC32_code   : e = 12 ; break ; // GB_FC32_mul(x,y)
                case GB_FC64_code   : e = 13 ; break ; // GB_FC64_mul(x,y)
                default             : e = 14 ; break ; // x * y
            }
            break ;

        case GB_EQ_binop_code :     // z = (x == y), the LXNOR monoid for bool

            // only a monoid for bool (lxnor)
            switch (xcode)
            {
                case GB_BOOL_code   : e = 15 ; break ; // x == y
                case GB_FC32_code   : e = 32 ; break ; // GB_FC32_eq (x,y)
                case GB_FC64_code   : e = 33 ; break ; // GB_FC64_eq (x,y)
                default             : e = 15 ; break ; // x == y, not a monoid
            }
            break ;

        case GB_NE_binop_code :     // z = (x != y), the LXOR monoid for bool

            // only a monoid for bool (lxor)
            switch (xcode)
            {
                case GB_BOOL_code   : e = 16 ; break ; // x ^ y
                case GB_FC32_code   : e = 36 ; break ; // GB_FC32_ne (x,y)
                case GB_FC64_code   : e = 37 ; break ; // GB_FC64_ne (x,y)
                default             : e =141 ; break ; // x != y
            }
            break ;

        case GB_LOR_binop_code :    // z = (x || y)

            switch (xcode)
            {
                case GB_BOOL_code  : e = 17 ; break ; // x || y
                default            : e = 40 ; break ; // not a monoid
            }
            break ;

        case GB_LAND_binop_code :   // z = (x && y)

            switch (xcode)
            {
                case GB_BOOL_code   : e = 18 ; break ; // x && y
                default             : e = 41 ; break ; // not a monoid
            }
            break ;

        case GB_LXOR_binop_code :   // z = (x != y)

            switch (xcode)
            {
                case GB_BOOL_code   : e = 16 ; break ; // x ^ y
                default             : e = 42 ; break ; // not a monoid
            }
            break ;

        case GB_BOR_binop_code :    // z = (x | y), bitwise or

            if (xcode >= GB_INT8_code && xcode <= GB_UINT64_code) e = 19 ;
            break ;

        case GB_BAND_binop_code :   // z = (x & y), bitwise and

            if (xcode >= GB_INT8_code && xcode <= GB_UINT64_code) e = 20 ;
            break ;

        case GB_BXOR_binop_code :   // z = (x ^ y), bitwise xor

            if (xcode >= GB_INT8_code && xcode <= GB_UINT64_code) e = 21 ;
            break ;

        case GB_BXNOR_binop_code :  // z = ~(x ^ y), bitwise xnor

            if (xcode >= GB_INT8_code && xcode <= GB_UINT64_code) e = 22 ;
            break ;

        //----------------------------------------------------------------------
        // built-in ops that cannot be used in a monoid:
        //----------------------------------------------------------------------

        case GB_ISEQ_binop_code :   // z = (x == y), but not a monoid

            switch (xcode)
            {
                case GB_BOOL_code   : e = 15 ; break ; // x == y
                case GB_FC32_code   : e = 34 ; break ; // GB_FC32_iseq(x,y)
                case GB_FC64_code   : e = 35 ; break ; // GB_FC64_iseq(x,y)
                default             : e =142 ; break ; // (ztype) (x == y)
            }
            break ;

        case GB_ISNE_binop_code :   // z = (x != y), but not a monoid

            switch (xcode)
            {
                case GB_BOOL_code   : e = 16 ; break ; // x != y
                case GB_FC32_code   : e = 38 ; break ; // GB_FC32_isne(x,y)
                case GB_FC64_code   : e = 39 ; break ; // GB_FC64_isne(x,y)
                default             : e =143 ; break ; // (ztype) (x != y)
            }
            break ;

        case GB_MINUS_binop_code :  // z = x - y

            switch (xcode)
            {
                case GB_BOOL_code   : e = 16 ; break ; // x != y
                case GB_FC32_code   : e = 43 ; break ; // GB_FC32_minus(x,y)
                case GB_FC64_code   : e = 44 ; break ; // GB_FC64_minus(x,y)
                default             : e = 45 ; break ; // x - y
            }
            break ;

        case GB_RMINUS_binop_code : // z = y - x

            switch (xcode)
            {
                case GB_BOOL_code   : e = 16 ; break ; // x != y
                case GB_FC32_code   : e = 46 ; break ; // GB_FC32_minus(y,x)
                case GB_FC64_code   : e = 47 ; break ; // GB_FC64_minus(y,x)
                default             : e = 48 ; break ; // y - x
            }
            break ;

        case GB_DIV_binop_code :    // z = x / y ;

            switch (xcode)
            {
                case GB_BOOL_code   : e =  1 ; break ; // x
                case GB_INT8_code   : e = 49 ; break ; // GJ_idiv_int8 (x,y)
                case GB_INT16_code  : e = 50 ; break ; // GJ_idiv_int16 (x,y)
                case GB_INT32_code  : e = 51 ; break ; // GJ_idiv_int32 (x,y)
                case GB_INT64_code  : e = 52 ; break ; // GJ_idiv_int64 (x,y)
                case GB_UINT8_code  : e = 53 ; break ; // GJ_idiv_uint8 (x,y)
                case GB_UINT16_code : e = 54 ; break ; // GJ_idiv_uint16 (x,y)
                case GB_UINT32_code : e = 55 ; break ; // GJ_idiv_uint32 (x,y)
                case GB_UINT64_code : e = 56 ; break ; // GJ_idiv_uint64 (x,y)
                case GB_FC32_code   : e = 57 ; break ; // GJ_FC32_div(x,y)
                case GB_FC64_code   : e = 58 ; break ; // GJ_FC64_div(x,y)
                default             : e = 59 ; break ; // (x) / (y)
            }
            break ;

        case GB_RDIV_binop_code :   // z = y / x ;

            switch (xcode)
            {
                case GB_BOOL_code   : e =  2 ; break ; // y
                case GB_INT8_code   : e = 60 ; break ; // GJ_idiv_int8 (y,x)
                case GB_INT16_code  : e = 61 ; break ; // GJ_idiv_int16 (y,x)
                case GB_INT32_code  : e = 62 ; break ; // GJ_idiv_int32 (y,x)
                case GB_INT64_code  : e = 63 ; break ; // GJ_idiv_int64 (y,x)
                case GB_UINT8_code  : e = 64 ; break ; // GJ_idiv_uint8 (y,x)
                case GB_UINT16_code : e = 65 ; break ; // GJ_idiv_uint16 (y,x)
                case GB_UINT32_code : e = 66 ; break ; // GJ_idiv_uint32 (y,x)
                case GB_UINT64_code : e = 67 ; break ; // GJ_idiv_uint64 (y,x)
                case GB_FC32_code   : e = 68 ; break ; // GJ_FC32_div(y,x)
                case GB_FC64_code   : e = 69 ; break ; // GJ_FC64_div(y,x)
                default             : e = 70 ; break ; // (y) / (x)
            }
            break ;

        case GB_GT_binop_code :     // z = (x > y)

            e = 71 ; break ;

        case GB_ISGT_binop_code :   // z = (ztype) (x > y)

            e = 144 ; break ;

        case GB_LT_binop_code :     // z = (x < y)

            e = 72 ; break ;

        case GB_ISLT_binop_code :   // z = (ztype) (x < y)

            e = 145 ; break ;

        case GB_GE_binop_code :     // z = (x >= y)

            e = 73 ; break ;

        case GB_ISGE_binop_code :   // z = (ztype) (x >= y)

            e = 146 ; break ;

        case GB_LE_binop_code :     // z = (x <= y)

            e = 74 ; break ;

        case GB_ISLE_binop_code :   // z = (ztype) (x <= y)

            e =147 ; break ;

        case GB_BGET_binop_code :   // z = bitget (x,y)

            switch (xcode)
            {
                case GB_INT8_code   : e = 75 ; break ;  // bget
                case GB_INT16_code  : e = 76 ; break ;
                case GB_INT32_code  : e = 77 ; break ;
                case GB_INT64_code  : e = 78 ; break ;
                case GB_UINT8_code  : e = 79 ; break ;
                case GB_UINT16_code : e = 80 ; break ;
                case GB_UINT32_code : e = 81 ; break ;
                case GB_UINT64_code : e = 82 ; break ;
                default: ;
            }
            break ;

        case GB_BSET_binop_code :   // z = bitset (x,y)

            switch (xcode)
            {
                case GB_INT8_code   : e = 83 ; break ;  // bset
                case GB_INT16_code  : e = 84 ; break ;
                case GB_INT32_code  : e = 85 ; break ;
                case GB_INT64_code  : e = 86 ; break ;
                case GB_UINT8_code  : e = 87 ; break ;
                case GB_UINT16_code : e = 88 ; break ;
                case GB_UINT32_code : e = 89 ; break ;
                case GB_UINT64_code : e = 90 ; break ;
                default: ;
            }
            break ;

        case GB_BCLR_binop_code :   // z = bitclr (x,y)

            switch (xcode)
            {
                case GB_INT8_code   : e = 91 ; break ;  // bclr
                case GB_INT16_code  : e = 92 ; break ;
                case GB_INT32_code  : e = 93 ; break ;
                case GB_INT64_code  : e = 94 ; break ;
                case GB_UINT8_code  : e = 95 ; break ;
                case GB_UINT16_code : e = 96 ; break ;
                case GB_UINT32_code : e = 97 ; break ;
                case GB_UINT64_code : e = 98 ; break ;
                default: ;
            }
            break ;

        case GB_BSHIFT_binop_code : // z = bitshift (x,y)

            switch (xcode)
            {
                case GB_INT8_code   : e =  99 ; break ; // bshift
                case GB_INT16_code  : e = 100 ; break ;
                case GB_INT32_code  : e = 101 ; break ;
                case GB_INT64_code  : e = 102 ; break ;
                case GB_UINT8_code  : e = 103 ; break ;
                case GB_UINT16_code : e = 104 ; break ;
                case GB_UINT32_code : e = 105 ; break ;
                case GB_UINT64_code : e = 106 ; break ;
                default: ;
            }
            break ;

        case GB_POW_binop_code :    // z = pow (x,y)

            switch (xcode)
            {
                case GB_BOOL_code   : e =  71 ; break ; // x >= y
                case GB_INT8_code   : e = 107 ; break ; // pow
                case GB_INT16_code  : e = 108 ; break ;
                case GB_INT32_code  : e = 109 ; break ;
                case GB_INT64_code  : e = 110 ; break ;
                case GB_UINT8_code  : e = 111 ; break ;
                case GB_UINT16_code : e = 112 ; break ;
                case GB_UINT32_code : e = 113 ; break ;
                case GB_UINT64_code : e = 114 ; break ;
                case GB_FP32_code   : e = 115 ; break ;
                case GB_FP64_code   : e = 116 ; break ;
                case GB_FC32_code   : e = 117 ; break ;
                case GB_FC64_code   : e = 118 ; break ;
                default: ;
            }
            break ;

        case GB_ATAN2_binop_code :  // z = atan2 (x,y)

            switch (xcode)
            {
                case GB_FP32_code   : e = 119 ; break ;
                case GB_FP64_code   : e = 120 ; break ;
                default: ;
            }
            break ;

        case GB_HYPOT_binop_code :  // z = hypot (x,y)

            switch (xcode)
            {
                case GB_FP32_code   : e = 121 ; break ;
                case GB_FP64_code   : e = 122 ; break ;
                default: ;
            }
            break ;

        case GB_FMOD_binop_code :   // z = fmod (x,y)

            switch (xcode)
            {
                case GB_FP32_code   : e = 123 ; break ;
                case GB_FP64_code   : e = 124 ; break ;
                default: ;
            }
            break ;

        case GB_REMAINDER_binop_code :  // z = remainder (x,y)

            switch (xcode)
            {
                case GB_FP32_code   : e = 125 ; break ;
                case GB_FP64_code   : e = 126 ; break ;
                default: ;
            }
            break ;

        case GB_COPYSIGN_binop_code :   // z = copysign (x,y)

            switch (xcode)
            {
                case GB_FP32_code   : e = 127 ; break ;
                case GB_FP64_code   : e = 128 ; break ;
                default: ;
            }
            break ;

        case GB_LDEXP_binop_code :  // z = ldexp (x,y)

            switch (xcode)
            {
                case GB_FP32_code   : e = 129 ; break ;
                case GB_FP64_code   : e = 130 ; break ;
                default: ;
            }
            break ;

        case GB_CMPLX_binop_code :  // z = cmplx (x,y)

            switch (xcode)
            {
                case GB_FP32_code   : e = 131 ; break ; // cmplx
                case GB_FP64_code   : e = 132 ; break ; // cmplx
                default: ;
            }
            break ;

        case GB_PAIR_binop_code :   // z = 1

            switch (xcode)
            {
                case GB_FC32_code   : e = 148 ; break ; // z = GxB_CMPLXF (1,0)
                case GB_FC64_code   : e = 149 ; break ; // z = GxB_CMPLX (1,0)
                default             : e = 133 ; break ; // z = 1
            }
            break ;

        //----------------------------------------------------------------------
        // positional ops
        //----------------------------------------------------------------------

        case GB_FIRSTI_binop_code :     // z = i

            e = 134 ; break ;

        case GB_FIRSTI1_binop_code :    // z = i+1

            e = 137 ; break ;

        case GB_FIRSTJ_binop_code :     // z = for_semiring ? (k) : (j)

            e = for_semiring ? 135 : 136 ; break ;

        case GB_FIRSTJ1_binop_code :    // z = for_semiring ? (k+1) : (j+1)

            e = for_semiring ? 138 : 139 ; break ;

        case GB_SECONDI_binop_code :    // z = for_semiring ? (k) : (i)

            e = for_semiring ? 135 : 134 ; break ;

        case GB_SECONDI1_binop_code :   // z = for_semiring ? (k+1) : (i+1)

            e = for_semiring ? 138 : 137 ; break ;

        case GB_SECONDJ_binop_code :    // z = j

            e = 136 ; break ;

        case GB_SECONDJ1_binop_code :   // z = j+1

            e = 139 ; break ;

        case GB_NOP_code :              // no operator present

            e = 255 ; break ;

        default: ;
    }

    (*ecode) = e ;
}

