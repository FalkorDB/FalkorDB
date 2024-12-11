//------------------------------------------------------------------------------
// GB_AxB_compare_factory.c: switch factory for C=A*B with comparator ops
//------------------------------------------------------------------------------

// SuiteSparse:GraphBLAS, Timothy A. Davis, (c) 2017-2023, All Rights Reserved.
// SPDX-License-Identifier: Apache-2.0

//------------------------------------------------------------------------------

// A template file #include'd in GB_AxB_factory.c, which calls 50 or 55
// semirings, with 5 monoids (lor, land, eq, lxor, any) and 10 or 11 types (the
// 10 real, non-boolean times, plus boolean).

// The multiply operator is a comparator: EQ, NE, GT, LT, GE, LE.
// z=f(x,y): x and x are either boolean or non-boolean.  z is boolean.

// Since z is boolean, the only monoids available are OR, AND, XOR, EQ, and
// ANY.  All the other four (max==plus==or, min==times==and) are redundant.
// Those opcodes have been renamed, and handled by the OR and AND workers
// defined here.

// There is one special case to consider.  For boolean x, y, and z, the
// function z=NE(x,y) is the same as z=XOR(x,y).  If z is boolean, the multiply
// operator NE has already been renamed XOR by GB_AxB_semiring_builtin, and
// thus NE will never use the boolean case, below.  Thus it is removed with the
// #ifndef GB_NO_BOOLEAN, resulting in 50 semirings for the NE muliply
// operator.

#if defined (GxB_NO_BOOL)
#define GB_CASE_BOOL(op)
#else
#define GB_CASE_BOOL(op) \
    case GB_BOOL_code:   GB_AxB_WORKER (op, GB_MNAME, _bool  )
#endif

#if defined (GxB_NO_INT8)
#define GB_CASE_INT8(op)
#else
#define GB_CASE_INT8(op) \
    case GB_INT8_code:   GB_AxB_WORKER (op, GB_MNAME, _int8  )
#endif

#if defined (GxB_NO_INT16)
#define GB_CASE_INT16(op)
#else
#define GB_CASE_INT16(op) \
    case GB_INT16_code:  GB_AxB_WORKER (op, GB_MNAME, _int16 )
#endif

#if defined (GxB_NO_INT32)
#define GB_CASE_INT32(op)
#else
#define GB_CASE_INT32(op) \
    case GB_INT32_code:  GB_AxB_WORKER (op, GB_MNAME, _int32 )
#endif

#if defined (GxB_NO_INT64)
#define GB_CASE_INT64(op)
#else
#define GB_CASE_INT64(op) \
    case GB_INT64_code:  GB_AxB_WORKER (op, GB_MNAME, _int64 )
#endif

#if defined (GxB_NO_UINT8)
#define GB_CASE_UINT8(op)
#else
#define GB_CASE_UINT8(op) \
    case GB_UINT8_code:  GB_AxB_WORKER (op, GB_MNAME, _uint8 )
#endif

#if defined (GxB_NO_UINT16)
#define GB_CASE_UINT16(op)
#else
#define GB_CASE_UINT16(op) \
    case GB_UINT16_code: GB_AxB_WORKER (op, GB_MNAME, _uint16)
#endif

#if defined (GxB_NO_UINT32)
#define GB_CASE_UINT32(op)
#else
#define GB_CASE_UINT32(op) \
    case GB_UINT32_code: GB_AxB_WORKER (op, GB_MNAME, _uint32)
#endif

#if defined (GxB_NO_UINT64)
#define GB_CASE_UINT64(op)
#else
#define GB_CASE_UINT64(op) \
    case GB_UINT64_code: GB_AxB_WORKER (op, GB_MNAME, _uint64)
#endif

#if defined (GxB_NO_FP32)
#define GB_CASE_FP32(op)
#else
#define GB_CASE_FP32(op) \
    case GB_FP32_code:   GB_AxB_WORKER (op, GB_MNAME, _fp32  )
#endif

#if defined (GxB_NO_FP64)
#define GB_CASE_FP64(op)
#else
#define GB_CASE_FP64(op) \
    case GB_FP64_code:   GB_AxB_WORKER (op, GB_MNAME, _fp64  )
#endif

ASSERT (zcode == GB_BOOL_code) ;
{

    // C = A*B where C is boolean, but A and B are non-boolean.
    // The result of the compare(A,B) operation is boolean.
    // There are 4 monoids available: OR, AND, XOR, EQ

    switch (add_binop_code)
    {

        case GB_LOR_binop_code     : 

            switch (xcode)
            {
                #ifndef GB_NO_BOOLEAN
                GB_CASE_BOOL   (_lor)
                #endif
                GB_CASE_INT8   (_lor)
                GB_CASE_INT16  (_lor)
                GB_CASE_INT32  (_lor)
                GB_CASE_INT64  (_lor)
                GB_CASE_UINT8  (_lor)
                GB_CASE_UINT16 (_lor)
                GB_CASE_UINT32 (_lor)
                GB_CASE_UINT64 (_lor)
                GB_CASE_FP32   (_lor)
                GB_CASE_FP64   (_lor)
                default: ;
            }
            break ;

        case GB_LAND_binop_code    : 

            switch (xcode)
            {
                // 10 real, non-boolean types, plus boolean
                #ifndef GB_NO_BOOLEAN
                GB_CASE_BOOL   (_land)
                #endif
                GB_CASE_INT8   (_land)
                GB_CASE_INT16  (_land)
                GB_CASE_INT32  (_land)
                GB_CASE_INT64  (_land)
                GB_CASE_UINT8  (_land)
                GB_CASE_UINT16 (_land)
                GB_CASE_UINT32 (_land)
                GB_CASE_UINT64 (_land)
                GB_CASE_FP32   (_land)
                GB_CASE_FP64   (_land)
                default: ;
            }
            break ;

        case GB_LXOR_binop_code    : 

            switch (xcode)
            {
                #ifndef GB_NO_BOOLEAN
                GB_CASE_BOOL   (_lxor)
                #endif
                GB_CASE_INT8   (_lxor)
                GB_CASE_INT16  (_lxor)
                GB_CASE_INT32  (_lxor)
                GB_CASE_INT64  (_lxor)
                GB_CASE_UINT8  (_lxor)
                GB_CASE_UINT16 (_lxor)
                GB_CASE_UINT32 (_lxor)
                GB_CASE_UINT64 (_lxor)
                GB_CASE_FP32   (_lxor)
                GB_CASE_FP64   (_lxor)
                default: ;
            }
            break ;

        case GB_EQ_binop_code    : 

            switch (xcode)
            {
                #ifndef GB_NO_BOOLEAN
                GB_CASE_BOOL   (_eq)
                #endif
                GB_CASE_INT8   (_eq)
                GB_CASE_INT16  (_eq)
                GB_CASE_INT32  (_eq)
                GB_CASE_INT64  (_eq)
                GB_CASE_UINT8  (_eq)
                GB_CASE_UINT16 (_eq)
                GB_CASE_UINT32 (_eq)
                GB_CASE_UINT64 (_eq)
                GB_CASE_FP32   (_eq)
                GB_CASE_FP64   (_eq)
                default: ;
            }
            break ;

        #ifndef GB_NO_ANY_MONOID
        case GB_ANY_binop_code    : 

            switch (xcode)
            {
                #ifndef GB_NO_BOOLEAN
                GB_CASE_BOOL   (_any)
                #endif
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
                default: ;
            }
            break ;
        #endif

        default: ;
    }
}

#undef GB_NO_BOOLEAN
#undef GB_MNAME

#undef GB_CASE_BOOL
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

