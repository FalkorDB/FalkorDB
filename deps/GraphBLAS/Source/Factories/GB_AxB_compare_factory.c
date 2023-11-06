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
                case GB_BOOL_code   : GB_AxB_WORKER (_lor, GB_MNAME, _bool  )
                #endif
                case GB_INT8_code   : GB_AxB_WORKER (_lor, GB_MNAME, _int8  )
                case GB_INT16_code  : GB_AxB_WORKER (_lor, GB_MNAME, _int16 )
                case GB_INT32_code  : GB_AxB_WORKER (_lor, GB_MNAME, _int32 )
                case GB_INT64_code  : GB_AxB_WORKER (_lor, GB_MNAME, _int64 )
                case GB_UINT8_code  : GB_AxB_WORKER (_lor, GB_MNAME, _uint8 )
                case GB_UINT16_code : GB_AxB_WORKER (_lor, GB_MNAME, _uint16)
                case GB_UINT32_code : GB_AxB_WORKER (_lor, GB_MNAME, _uint32)
                case GB_UINT64_code : GB_AxB_WORKER (_lor, GB_MNAME, _uint64)
                case GB_FP32_code   : GB_AxB_WORKER (_lor, GB_MNAME, _fp32  )
                case GB_FP64_code   : GB_AxB_WORKER (_lor, GB_MNAME, _fp64  )
                default: ;
            }
            break ;

        case GB_LAND_binop_code    :

            switch (xcode)
            {
                // 10 real, non-boolean types, plus boolean
                #ifndef GB_NO_BOOLEAN
                case GB_BOOL_code   : GB_AxB_WORKER (_land, GB_MNAME, _bool  )
                #endif
                case GB_INT8_code   : GB_AxB_WORKER (_land, GB_MNAME, _int8  )
                case GB_INT16_code  : GB_AxB_WORKER (_land, GB_MNAME, _int16 )
                case GB_INT32_code  : GB_AxB_WORKER (_land, GB_MNAME, _int32 )
                case GB_INT64_code  : GB_AxB_WORKER (_land, GB_MNAME, _int64 )
                case GB_UINT8_code  : GB_AxB_WORKER (_land, GB_MNAME, _uint8 )
                case GB_UINT16_code : GB_AxB_WORKER (_land, GB_MNAME, _uint16)
                case GB_UINT32_code : GB_AxB_WORKER (_land, GB_MNAME, _uint32)
                case GB_UINT64_code : GB_AxB_WORKER (_land, GB_MNAME, _uint64)
                case GB_FP32_code   : GB_AxB_WORKER (_land, GB_MNAME, _fp32  )
                case GB_FP64_code   : GB_AxB_WORKER (_land, GB_MNAME, _fp64  )
                default: ;
            }
            break ;

        case GB_LXOR_binop_code    :

            switch (xcode)
            {
                #ifndef GB_NO_BOOLEAN
                case GB_BOOL_code   : GB_AxB_WORKER (_lxor, GB_MNAME, _bool  )
                #endif
                case GB_INT8_code   : GB_AxB_WORKER (_lxor, GB_MNAME, _int8  )
                case GB_INT16_code  : GB_AxB_WORKER (_lxor, GB_MNAME, _int16 )
                case GB_INT32_code  : GB_AxB_WORKER (_lxor, GB_MNAME, _int32 )
                case GB_INT64_code  : GB_AxB_WORKER (_lxor, GB_MNAME, _int64 )
                case GB_UINT8_code  : GB_AxB_WORKER (_lxor, GB_MNAME, _uint8 )
                case GB_UINT16_code : GB_AxB_WORKER (_lxor, GB_MNAME, _uint16)
                case GB_UINT32_code : GB_AxB_WORKER (_lxor, GB_MNAME, _uint32)
                case GB_UINT64_code : GB_AxB_WORKER (_lxor, GB_MNAME, _uint64)
                case GB_FP32_code   : GB_AxB_WORKER (_lxor, GB_MNAME, _fp32  )
                case GB_FP64_code   : GB_AxB_WORKER (_lxor, GB_MNAME, _fp64  )
                default: ;
            }
            break ;

        case GB_EQ_binop_code    :

            switch (xcode)
            {
                #ifndef GB_NO_BOOLEAN
                case GB_BOOL_code   : GB_AxB_WORKER (_eq, GB_MNAME, _bool  )
                #endif
                case GB_INT8_code   : GB_AxB_WORKER (_eq, GB_MNAME, _int8  )
                case GB_INT16_code  : GB_AxB_WORKER (_eq, GB_MNAME, _int16 )
                case GB_INT32_code  : GB_AxB_WORKER (_eq, GB_MNAME, _int32 )
                case GB_INT64_code  : GB_AxB_WORKER (_eq, GB_MNAME, _int64 )
                case GB_UINT8_code  : GB_AxB_WORKER (_eq, GB_MNAME, _uint8 )
                case GB_UINT16_code : GB_AxB_WORKER (_eq, GB_MNAME, _uint16)
                case GB_UINT32_code : GB_AxB_WORKER (_eq, GB_MNAME, _uint32)
                case GB_UINT64_code : GB_AxB_WORKER (_eq, GB_MNAME, _uint64)
                case GB_FP32_code   : GB_AxB_WORKER (_eq, GB_MNAME, _fp32  )
                case GB_FP64_code   : GB_AxB_WORKER (_eq, GB_MNAME, _fp64  )
                default: ;
            }
            break ;

        #ifndef GB_NO_ANY_MONOID
        case GB_ANY_binop_code    :

            switch (xcode)
            {
                #ifndef GB_NO_BOOLEAN
                case GB_BOOL_code   : GB_AxB_WORKER (_any, GB_MNAME, _bool  )
                #endif
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
                default: ;
            }
            break ;
        #endif

        default: ;
    }
}

#undef GB_NO_BOOLEAN
#undef GB_MNAME

