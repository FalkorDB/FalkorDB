//------------------------------------------------------------------------------
// GB_red_factory.c: switch factory for reduction operators
//------------------------------------------------------------------------------

// SuiteSparse:GraphBLAS, Timothy A. Davis, (c) 2017-2025, All Rights Reserved.
// SPDX-License-Identifier: Apache-2.0

//------------------------------------------------------------------------------

// This is a generic body of code for creating hard-coded versions of code for
// 61 combinations of associative operators and built-in types:

//  20:  min, max: 10 non-boolean real types
//  24:  plus, times:  12 non-boolean types
//  4:   lor, land, eq (same as lxnor), lxor for boolean
//  13:  any: for all 13 types

#if defined (GxB_NO_INT8)
#define GB_CASE_INT8(op)
#else
#define GB_CASE_INT8(op) \
    case GB_INT8_code:   GB_RED_WORKER (op, _int8  , int8_t  )
#endif

#if defined (GxB_NO_INT16)
#define GB_CASE_INT16(op)
#else
#define GB_CASE_INT16(op) \
    case GB_INT16_code:  GB_RED_WORKER (op, _int16 , int16_t )
#endif

#if defined (GxB_NO_INT32)
#define GB_CASE_INT32(op)
#else
#define GB_CASE_INT32(op) \
    case GB_INT32_code:  GB_RED_WORKER (op, _int32 , int32_t )
#endif

#if defined (GxB_NO_INT64)
#define GB_CASE_INT64(op)
#else
#define GB_CASE_INT64(op) \
    case GB_INT64_code:  GB_RED_WORKER (op, _int64 , int64_t )
#endif

#if defined (GxB_NO_UINT8)
#define GB_CASE_UINT8(op)
#else
#define GB_CASE_UINT8(op) \
    case GB_UINT8_code:  GB_RED_WORKER (op, _uint8 , uint8_t )
#endif

#if defined (GxB_NO_UINT16)
#define GB_CASE_UINT16(op)
#else
#define GB_CASE_UINT16(op) \
    case GB_UINT16_code: GB_RED_WORKER (op, _uint16, uint16_t)
#endif

#if defined (GxB_NO_UINT32)
#define GB_CASE_UINT32(op)
#else
#define GB_CASE_UINT32(op) \
    case GB_UINT32_code: GB_RED_WORKER (op, _uint32, uint32_t)
#endif

#if defined (GxB_NO_UINT64)
#define GB_CASE_UINT64(op)
#else
#define GB_CASE_UINT64(op) \
    case GB_UINT64_code: GB_RED_WORKER (op, _uint64, uint64_t)
#endif

#if defined (GxB_NO_FP32)
#define GB_CASE_FP32(op)
#else
#define GB_CASE_FP32(op) \
    case GB_FP32_code:   GB_RED_WORKER (op, _fp32  , float   )
#endif

#if defined (GxB_NO_FP64)
#define GB_CASE_FP64(op)
#else
#define GB_CASE_FP64(op) \
    case GB_FP64_code:   GB_RED_WORKER (op, _fp64  , double   )
#endif

#if defined (GxB_NO_FC32)
#define GB_CASE_FC32(op)
#else
#define GB_CASE_FC32(op) \
    case GB_FC32_code:   GB_RED_WORKER (op, _fc32  , GxB_FC32_t)
#endif

#if defined (GxB_NO_FC64)
#define GB_CASE_FC64(op)
#else
#define GB_CASE_FC64(op) \
    case GB_FC64_code:   GB_RED_WORKER (op, _fc64  , GxB_FC64_t)
#endif

if (typecode != GB_BOOL_code)
{ 

    //--------------------------------------------------------------------------
    // non-boolean case
    //--------------------------------------------------------------------------

    switch (opcode)
    {

        case GB_MIN_binop_code   : 

            switch (typecode)
            {
                GB_CASE_INT8   (_min)
                GB_CASE_INT16  (_min)
                GB_CASE_INT32  (_min)
                GB_CASE_INT64  (_min)
                GB_CASE_UINT8  (_min)
                GB_CASE_UINT16 (_min)
                GB_CASE_UINT32 (_min)
                GB_CASE_UINT64 (_min)
                GB_CASE_FP32   (_min)
                GB_CASE_FP64   (_min)
                default: ;
            }
            break ;

        case GB_MAX_binop_code   : 

            switch (typecode)
            {
                GB_CASE_INT8   (_max)
                GB_CASE_INT16  (_max)
                GB_CASE_INT32  (_max)
                GB_CASE_INT64  (_max)
                GB_CASE_UINT8  (_max)
                GB_CASE_UINT16 (_max)
                GB_CASE_UINT32 (_max)
                GB_CASE_UINT64 (_max)
                GB_CASE_FP32   (_max)
                GB_CASE_FP64   (_max)
                default: ;
            }
            break ;

        case GB_PLUS_binop_code  : 

            switch (typecode)
            {
                GB_CASE_INT8   (_plus)
                GB_CASE_INT16  (_plus)
                GB_CASE_INT32  (_plus)
                GB_CASE_INT64  (_plus)
                GB_CASE_UINT8  (_plus)
                GB_CASE_UINT16 (_plus)
                GB_CASE_UINT32 (_plus)
                GB_CASE_UINT64 (_plus)
                GB_CASE_FP32   (_plus)
                GB_CASE_FP64   (_plus)
                GB_CASE_FC32   (_plus)
                GB_CASE_FC64   (_plus)
                default: ;
            }
            break ;

        case GB_TIMES_binop_code : 

            switch (typecode)
            {
                GB_CASE_INT8   (_times)
                GB_CASE_INT16  (_times)
                GB_CASE_INT32  (_times)
                GB_CASE_INT64  (_times)
                GB_CASE_UINT8  (_times)
                GB_CASE_UINT16 (_times)
                GB_CASE_UINT32 (_times)
                GB_CASE_UINT64 (_times)
                GB_CASE_FP32   (_times)
                GB_CASE_FP64   (_times)
                GB_CASE_FC32   (_times)
                GB_CASE_FC64   (_times)
                default: ;
            }
            break ;

        case GB_ANY_binop_code : 

            switch (typecode)
            {
                GB_CASE_INT8   (_any)
                GB_CASE_INT16  (_any)
                GB_CASE_INT32  (_any)
                GB_CASE_INT64  (_any)
                GB_CASE_UINT8  (_any)
                GB_CASE_UINT16 (_any)
                GB_CASE_UINT32 (_any)
                GB_CASE_UINT64 (_any)
                GB_CASE_FP32   (_any)
                GB_CASE_FP64   (_any)
                GB_CASE_FC32   (_any)
                GB_CASE_FC64   (_any)
                default: ;
            }
            break ;

        default: ;
    }

}
else
{ 

    //--------------------------------------------------------------------------
    // boolean case: rename the opcode as needed
    //--------------------------------------------------------------------------

    #ifndef GxB_NO_BOOL
    switch (GB_boolean_rename (opcode))
    {
        case GB_LOR_binop_code    : GB_RED_WORKER (_lor,    _bool, bool)
        case GB_LAND_binop_code   : GB_RED_WORKER (_land,   _bool, bool)
        case GB_LXOR_binop_code   : GB_RED_WORKER (_lxor,   _bool, bool)
        case GB_EQ_binop_code     : GB_RED_WORKER (_eq,     _bool, bool)
        case GB_ANY_binop_code    : GB_RED_WORKER (_any,    _bool, bool)
        default: ;
    }
    #endif
}

#undef GB_CASE_INT8
#undef GB_CASE_INT16
#undef GB_CASE_INT32
#undef GB_CASE_INT64
#undef GB_CASE_UINT8
#undef GB_CASE_UINT16
#undef GB_CASE_UINT32
#undef GB_CASE_UINT64
#undef GB_CASE_FP32
#undef GB_CASE_FP64
#undef GB_CASE_FC32
#undef GB_CASE_FC64

