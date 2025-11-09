//------------------------------------------------------------------------------
// GB_AxB_type_factory.c: switch factory for C=A*B
//------------------------------------------------------------------------------

// SuiteSparse:GraphBLAS, Timothy A. Davis, (c) 2017-2025, All Rights Reserved.
// SPDX-License-Identifier: Apache-2.0

//------------------------------------------------------------------------------

// A template file #include'd in GB_AxB_factory.c, which calls up to 61
// semirings.  Not all multiplicative operators and types are used with every
// monoid.  The 2 complex types appear only in the times, plus, and any
// monoids, for a subset of the multiply operators.

//  min monoid:     10 real, non-boolean types
//  max monoid:     10 real, non-boolean types
//  times monoid:   10 real, non-boolean types (+2 if complex)
//  plus monoid:    10 real, non-boolean types (+2 if complex)
//  any monoid:     10 real, non-boolean types (+2 if complex)
//  boolean:        5 monoids: lor, land, eq, lxor, any

// GB_NO_BOOLEAN is defined for multiply operators in the #include'ing file
// (min, max, plus, minus, rminus, times, div, rdiv, is*) since those multiply
// operators are redundant and have been renamed.  For these, the boolean
// monoids are not needed.

// GB_NO_MIN_MAX_ANY_TIMES_MONOIDS is defined for the PAIR, LOR, LAND, LXOR
// multiply operators; these are valid semirings, but not useful.  The
// corresponding semirings (such as GxB_TIMES_LOR_FP32) still exist, but are
// done using the generic methods, not via fast methods controlled by this case
// statement.  For the PAIR operator, these semirings are all done by the
// single ANY_PAIR iso semiring, since C is always iso in that case.

// the additive operator is a monoid, where all types of x,y,z are the same
ASSERT (zcode == xcode) ;
ASSERT (zcode == ycode) ;
ASSERT (mult_binop_code != GB_ANY_binop_code) ;

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

#if defined (GxB_NO_FC32)
#define GB_CASE_FC32(op)
#else
#define GB_CASE_FC32(op) \
    case GB_FC32_code:   GB_AxB_WORKER (op, GB_MNAME, _fc32  )
#endif

#if defined (GxB_NO_FC64)
#define GB_CASE_FC64(op)
#else
#define GB_CASE_FC64(op) \
    case GB_FC64_code:   GB_AxB_WORKER (op, GB_MNAME, _fc64  )
#endif

if (xcode != GB_BOOL_code)
{ 
    switch (add_binop_code)
    {

        // disable the MIN, MAX, ANY, and TIMES monoids for some multops
        #ifndef GB_NO_MIN_MAX_ANY_TIMES_MONOIDS

        case GB_MIN_binop_code : 

            switch (xcode)
            {
                // 10 real, non-boolean types
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

        case GB_MAX_binop_code : 

            switch (xcode)
            {
                // 10 real, non-boolean types
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

        case GB_TIMES_binop_code : 

            switch (xcode)
            {
                // 10 real, non-boolean types, plus 2 complex
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
                #if defined ( GB_COMPLEX )
                GB_CASE_FC32   (_times)
                GB_CASE_FC64   (_times)
                #endif
                default: ;
            }
            break ;

        #ifndef GB_NO_ANY_MONOID
        case GB_ANY_binop_code : 

            switch (xcode)
            {
                // 10 real, non-boolean types, plus 2 complex
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
                #if defined ( GB_COMPLEX )
                // the ANY monoid is non-atomic for complex types
                GB_CASE_FC32   (_any)
                GB_CASE_FC64   (_any)
                #endif
                default: ;
            }
            break ;
        #endif
        #endif

        case GB_PLUS_binop_code : 

            switch (xcode)
            {
                // 10 real, non-boolean types, plus 2 complex
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
                #if defined ( GB_COMPLEX )
                // only the PLUS monoid is atomic for complex types
                GB_CASE_FC32   (_plus)
                GB_CASE_FC64   (_plus)
                #endif
                default: ;
            }
            break ;

        default: ;
    }
}

#ifndef GB_NO_BOOLEAN
else
{ 
        switch (add_binop_code)
        {
            #if !defined (GxB_NO_BOOL)
            // 5 boolean monoids
            #ifndef GB_MULT_IS_PAIR_OPERATOR
            // all these semirings are replaced with the ANY_PAIR iso semiring
            case GB_LOR_binop_code  : GB_AxB_WORKER (_lor , GB_MNAME, _bool)
            case GB_LAND_binop_code : GB_AxB_WORKER (_land, GB_MNAME, _bool)
            case GB_EQ_binop_code   : GB_AxB_WORKER (_eq  , GB_MNAME, _bool)
            #ifndef GB_NO_ANY_MONOID
            case GB_ANY_binop_code  : GB_AxB_WORKER (_any , GB_MNAME, _bool)
            #endif
            #endif
            case GB_LXOR_binop_code : GB_AxB_WORKER (_lxor, GB_MNAME, _bool)
            #endif
            default: ;
        }
}
#endif

#undef GB_NO_BOOLEAN
#undef GB_MNAME
#undef GB_COMPLEX
#undef GB_NO_MIN_MAX_ANY_TIMES_MONOIDS
#undef GB_MULT_IS_PAIR_OPERATOR

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

