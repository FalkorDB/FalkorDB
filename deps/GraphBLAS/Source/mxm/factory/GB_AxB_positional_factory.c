//------------------------------------------------------------------------------
// GB_AxB_positional_factory.c: switch factory for C=A*B for positional ops
//------------------------------------------------------------------------------

// SuiteSparse:GraphBLAS, Timothy A. Davis, (c) 2017-2025, All Rights Reserved.
// SPDX-License-Identifier: Apache-2.0

//------------------------------------------------------------------------------

// the additive operator is a monoid, where all types of x,y,z are int64_t or
// int32_t

ASSERT (xcode == zcode) ;
ASSERT (ycode == zcode) ;
ASSERT (GB_IS_BUILTIN_BINOP_CODE_POSITIONAL (mult_binop_code)) ;

{
    if (zcode == GB_INT32_code)
    {
        #if !defined (GxB_NO_INT32)
        switch (add_binop_code)
        {
            case GB_MIN_binop_code   : GB_AxB_WORKER (_min,   GB_MNAME, _int32)
            case GB_MAX_binop_code   : GB_AxB_WORKER (_max,   GB_MNAME, _int32)
            case GB_TIMES_binop_code : GB_AxB_WORKER (_times, GB_MNAME, _int32)
            case GB_PLUS_binop_code  : GB_AxB_WORKER (_plus,  GB_MNAME, _int32)
            #ifndef GB_NO_ANY_MONOID
            case GB_ANY_binop_code   : GB_AxB_WORKER (_any,   GB_MNAME, _int32)
            #endif
            default: ;
        }
        #endif
    }
    else // zcode == GB_INT64_code
    {
        #if !defined (GxB_NO_INT64)
        ASSERT (zcode == GB_INT64_code) ;
        switch (add_binop_code)
        {
            case GB_MIN_binop_code   : GB_AxB_WORKER (_min,   GB_MNAME, _int64)
            case GB_MAX_binop_code   : GB_AxB_WORKER (_max,   GB_MNAME, _int64)
            case GB_TIMES_binop_code : GB_AxB_WORKER (_times, GB_MNAME, _int64)
            case GB_PLUS_binop_code  : GB_AxB_WORKER (_plus,  GB_MNAME, _int64)
            #ifndef GB_NO_ANY_MONOID
            case GB_ANY_binop_code   : GB_AxB_WORKER (_any,   GB_MNAME, _int64)
            #endif
            default: ;
        }
        #endif
    }
}

#undef GB_MNAME

