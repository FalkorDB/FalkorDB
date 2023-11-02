//------------------------------------------------------------------------------
// GB_AxB_type_factory.c: switch factory for C=A*B
//------------------------------------------------------------------------------

// SuiteSparse:GraphBLAS, Timothy A. Davis, (c) 2017-2023, All Rights Reserved.
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

if (xcode != GB_BOOL_code)
{
    switch (add_binop_code)
    {

        // disable the MIN, MAX, ANY, and TIMES monoids for some multops
        #ifndef GB_NO_MIN_MAX_ANY_TIMES_MONOIDS

        case GB_MIN_binop_code:

            switch (xcode)
            {
                // 10 real, non-boolean types
                case GB_INT8_code   : GB_AxB_WORKER (_min, GB_MNAME, _int8  )
                case GB_INT16_code  : GB_AxB_WORKER (_min, GB_MNAME, _int16 )
                case GB_INT32_code  : GB_AxB_WORKER (_min, GB_MNAME, _int32 )
                case GB_INT64_code  : GB_AxB_WORKER (_min, GB_MNAME, _int64 )
                case GB_UINT8_code  : GB_AxB_WORKER (_min, GB_MNAME, _uint8 )
                case GB_UINT16_code : GB_AxB_WORKER (_min, GB_MNAME, _uint16)
                case GB_UINT32_code : GB_AxB_WORKER (_min, GB_MNAME, _uint32)
                case GB_UINT64_code : GB_AxB_WORKER (_min, GB_MNAME, _uint64)
                case GB_FP32_code   : GB_AxB_WORKER (_min, GB_MNAME, _fp32  )
                case GB_FP64_code   : GB_AxB_WORKER (_min, GB_MNAME, _fp64  )
                default: ;
            }
            break ;

        case GB_MAX_binop_code:

            switch (xcode)
            {
                // 10 real, non-boolean types
                case GB_INT8_code   : GB_AxB_WORKER (_max, GB_MNAME, _int8  )
                case GB_INT16_code  : GB_AxB_WORKER (_max, GB_MNAME, _int16 )
                case GB_INT32_code  : GB_AxB_WORKER (_max, GB_MNAME, _int32 )
                case GB_INT64_code  : GB_AxB_WORKER (_max, GB_MNAME, _int64 )
                case GB_UINT8_code  : GB_AxB_WORKER (_max, GB_MNAME, _uint8 )
                case GB_UINT16_code : GB_AxB_WORKER (_max, GB_MNAME, _uint16)
                case GB_UINT32_code : GB_AxB_WORKER (_max, GB_MNAME, _uint32)
                case GB_UINT64_code : GB_AxB_WORKER (_max, GB_MNAME, _uint64)
                case GB_FP32_code   : GB_AxB_WORKER (_max, GB_MNAME, _fp32  )
                case GB_FP64_code   : GB_AxB_WORKER (_max, GB_MNAME, _fp64  )
                default: ;
            }
            break ;

        case GB_TIMES_binop_code:

            switch (xcode)
            {
                // 10 real, non-boolean types, plus 2 complex
                case GB_INT8_code   : GB_AxB_WORKER (_times, GB_MNAME, _int8  )
                case GB_INT16_code  : GB_AxB_WORKER (_times, GB_MNAME, _int16 )
                case GB_INT32_code  : GB_AxB_WORKER (_times, GB_MNAME, _int32 )
                case GB_INT64_code  : GB_AxB_WORKER (_times, GB_MNAME, _int64 )
                case GB_UINT8_code  : GB_AxB_WORKER (_times, GB_MNAME, _uint8 )
                case GB_UINT16_code : GB_AxB_WORKER (_times, GB_MNAME, _uint16)
                case GB_UINT32_code : GB_AxB_WORKER (_times, GB_MNAME, _uint32)
                case GB_UINT64_code : GB_AxB_WORKER (_times, GB_MNAME, _uint64)
                case GB_FP32_code   : GB_AxB_WORKER (_times, GB_MNAME, _fp32  )
                case GB_FP64_code   : GB_AxB_WORKER (_times, GB_MNAME, _fp64  )
                #if defined ( GB_COMPLEX )
                case GB_FC32_code   : GB_AxB_WORKER (_times, GB_MNAME, _fc32  )
                case GB_FC64_code   : GB_AxB_WORKER (_times, GB_MNAME, _fc64  )
                #endif
                default: ;
            }
            break ;

        #ifndef GB_NO_ANY_MONOID
        case GB_ANY_binop_code:

            switch (xcode)
            {
                // 10 real, non-boolean types, plus 2 complex
                case GB_INT8_code   : GB_AxB_WORKER (_any, GB_MNAME, _int8  )
                case GB_INT16_code  : GB_AxB_WORKER (_any, GB_MNAME, _int16 )
                case GB_INT32_code  : GB_AxB_WORKER (_any, GB_MNAME, _int32 )
                case GB_INT64_code  : GB_AxB_WORKER (_any, GB_MNAME, _int64 )
                case GB_UINT8_code  : GB_AxB_WORKER (_any, GB_MNAME, _uint8 )
                case GB_UINT16_code : GB_AxB_WORKER (_any, GB_MNAME, _uint16)
                case GB_UINT32_code : GB_AxB_WORKER (_any, GB_MNAME, _uint32)
                case GB_UINT64_code : GB_AxB_WORKER (_any, GB_MNAME, _uint64)
                case GB_FP32_code   : GB_AxB_WORKER (_any, GB_MNAME, _fp32  )
                case GB_FP64_code   : GB_AxB_WORKER (_any, GB_MNAME, _fp64  )
                #if defined ( GB_COMPLEX )
                // the ANY monoid is non-atomic for complex types
                case GB_FC32_code   : GB_AxB_WORKER (_any, GB_MNAME, _fc32  )
                case GB_FC64_code   : GB_AxB_WORKER (_any, GB_MNAME, _fc64  )
                #endif
                default: ;
            }
            break ;
        #endif
        #endif

        case GB_PLUS_binop_code:

            switch (xcode)
            {
                // 10 real, non-boolean types, plus 2 complex
                case GB_INT8_code   : GB_AxB_WORKER (_plus, GB_MNAME, _int8  )
                case GB_INT16_code  : GB_AxB_WORKER (_plus, GB_MNAME, _int16 )
                case GB_INT32_code  : GB_AxB_WORKER (_plus, GB_MNAME, _int32 )
                case GB_INT64_code  : GB_AxB_WORKER (_plus, GB_MNAME, _int64 )
                case GB_UINT8_code  : GB_AxB_WORKER (_plus, GB_MNAME, _uint8 )
                case GB_UINT16_code : GB_AxB_WORKER (_plus, GB_MNAME, _uint16)
                case GB_UINT32_code : GB_AxB_WORKER (_plus, GB_MNAME, _uint32)
                case GB_UINT64_code : GB_AxB_WORKER (_plus, GB_MNAME, _uint64)
                case GB_FP32_code   : GB_AxB_WORKER (_plus, GB_MNAME, _fp32  )
                case GB_FP64_code   : GB_AxB_WORKER (_plus, GB_MNAME, _fp64  )
                #if defined ( GB_COMPLEX )
                // only the PLUS monoid is atomic for complex types
                case GB_FC32_code   : GB_AxB_WORKER (_plus, GB_MNAME, _fc32  )
                case GB_FC64_code   : GB_AxB_WORKER (_plus, GB_MNAME, _fc64  )
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
            default: ;
        }
}
#endif

#undef GB_NO_BOOLEAN
#undef GB_MNAME
#undef GB_COMPLEX
#undef GB_NO_MIN_MAX_ANY_TIMES_MONOIDS
#undef GB_MULT_IS_PAIR_OPERATOR

