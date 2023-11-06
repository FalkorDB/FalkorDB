//------------------------------------------------------------------------------
// GB_bld_factory.c: switch factory for builder
//------------------------------------------------------------------------------

// SuiteSparse:GraphBLAS, Timothy A. Davis, (c) 2017-2023, All Rights Reserved.
// SPDX-License-Identifier: Apache-2.0

//------------------------------------------------------------------------------

// This is a generic body of code for creating hard-coded versions of code for
// 87 combinations of operators and built-in types:

//  20:  min, max: 10 non-boolean real types
//  24:  plus, times:  12 non-boolean types
//  4:   lor, land, eq (same as lxnor), lxor for boolean
//  13:  any: for all 13 types
//  26:  first, second: for all 13 types

// FUTUTE: add band, bor, bxor, bxnor for uint [8,16,32,64].
// Or, just use the JIT.

if (tcode != GB_BOOL_code)
{

    switch (opcode)
    {

        //----------------------------------------------------------------------
        // non-boolean, associative operators
        //----------------------------------------------------------------------

        case GB_MIN_binop_code   :

            switch (tcode)
            {
                case GB_INT8_code   : GB_BLD_WORKER (_min, _int8,   int8_t  )
                case GB_INT16_code  : GB_BLD_WORKER (_min, _int16,  int16_t )
                case GB_INT32_code  : GB_BLD_WORKER (_min, _int32,  int32_t )
                case GB_INT64_code  : GB_BLD_WORKER (_min, _int64,  int64_t )
                case GB_UINT8_code  : GB_BLD_WORKER (_min, _uint8,  uint8_t )
                case GB_UINT16_code : GB_BLD_WORKER (_min, _uint16, uint16_t)
                case GB_UINT32_code : GB_BLD_WORKER (_min, _uint32, uint32_t)
                case GB_UINT64_code : GB_BLD_WORKER (_min, _uint64, uint64_t)
                case GB_FP32_code   : GB_BLD_WORKER (_min, _fp32,   float   )
                case GB_FP64_code   : GB_BLD_WORKER (_min, _fp64,   double  )
                default: ;
            }
            break ;

        case GB_MAX_binop_code   :

            switch (tcode)
            {
                case GB_INT8_code   : GB_BLD_WORKER (_max, _int8,   int8_t  )
                case GB_INT16_code  : GB_BLD_WORKER (_max, _int16,  int16_t )
                case GB_INT32_code  : GB_BLD_WORKER (_max, _int32,  int32_t )
                case GB_INT64_code  : GB_BLD_WORKER (_max, _int64,  int64_t )
                case GB_UINT8_code  : GB_BLD_WORKER (_max, _uint8,  uint8_t )
                case GB_UINT16_code : GB_BLD_WORKER (_max, _uint16, uint16_t)
                case GB_UINT32_code : GB_BLD_WORKER (_max, _uint32, uint32_t)
                case GB_UINT64_code : GB_BLD_WORKER (_max, _uint64, uint64_t)
                case GB_FP32_code   : GB_BLD_WORKER (_max, _fp32,   float   )
                case GB_FP64_code   : GB_BLD_WORKER (_max, _fp64,   double  )
                default: ;
            }
            break ;

        case GB_PLUS_binop_code  :

            switch (tcode)
            {
                case GB_INT8_code   : GB_BLD_WORKER (_plus, _int8,   int8_t  )
                case GB_INT16_code  : GB_BLD_WORKER (_plus, _int16,  int16_t )
                case GB_INT32_code  : GB_BLD_WORKER (_plus, _int32,  int32_t )
                case GB_INT64_code  : GB_BLD_WORKER (_plus, _int64,  int64_t )
                case GB_UINT8_code  : GB_BLD_WORKER (_plus, _uint8,  uint8_t )
                case GB_UINT16_code : GB_BLD_WORKER (_plus, _uint16, uint16_t)
                case GB_UINT32_code : GB_BLD_WORKER (_plus, _uint32, uint32_t)
                case GB_UINT64_code : GB_BLD_WORKER (_plus, _uint64, uint64_t)
                case GB_FP32_code   : GB_BLD_WORKER (_plus, _fp32,   float   )
                case GB_FP64_code   : GB_BLD_WORKER (_plus, _fp64,   double  )
                case GB_FC32_code   : GB_BLD_WORKER (_plus, _fc32,   GxB_FC32_t)
                case GB_FC64_code   : GB_BLD_WORKER (_plus, _fc64,   GxB_FC64_t)
                default: ;
            }
            break ;

        case GB_TIMES_binop_code :

            switch (tcode)
            {
                case GB_INT8_code   : GB_BLD_WORKER (_times, _int8,   int8_t  )
                case GB_INT16_code  : GB_BLD_WORKER (_times, _int16,  int16_t )
                case GB_INT32_code  : GB_BLD_WORKER (_times, _int32,  int32_t )
                case GB_INT64_code  : GB_BLD_WORKER (_times, _int64,  int64_t )
                case GB_UINT8_code  : GB_BLD_WORKER (_times, _uint8,  uint8_t )
                case GB_UINT16_code : GB_BLD_WORKER (_times, _uint16, uint16_t)
                case GB_UINT32_code : GB_BLD_WORKER (_times, _uint32, uint32_t)
                case GB_UINT64_code : GB_BLD_WORKER (_times, _uint64, uint64_t)
                case GB_FP32_code   : GB_BLD_WORKER (_times, _fp32,   float   )
                case GB_FP64_code   : GB_BLD_WORKER (_times, _fp64,   double  )
                case GB_FC32_code   : GB_BLD_WORKER (_times, _fc32, GxB_FC32_t)
                case GB_FC64_code   : GB_BLD_WORKER (_times, _fc64, GxB_FC64_t)
                default: ;
            }
            break ;

        case GB_ANY_binop_code :

            switch (tcode)
            {
                case GB_INT8_code   : GB_BLD_WORKER (_any, _int8,   int8_t  )
                case GB_INT16_code  : GB_BLD_WORKER (_any, _int16,  int16_t )
                case GB_INT32_code  : GB_BLD_WORKER (_any, _int32,  int32_t )
                case GB_INT64_code  : GB_BLD_WORKER (_any, _int64,  int64_t )
                case GB_UINT8_code  : GB_BLD_WORKER (_any, _uint8,  uint8_t )
                case GB_UINT16_code : GB_BLD_WORKER (_any, _uint16, uint16_t)
                case GB_UINT32_code : GB_BLD_WORKER (_any, _uint32, uint32_t)
                case GB_UINT64_code : GB_BLD_WORKER (_any, _uint64, uint64_t)
                case GB_FP32_code   : GB_BLD_WORKER (_any, _fp32,   float   )
                case GB_FP64_code   : GB_BLD_WORKER (_any, _fp64,   double  )
                case GB_FC32_code   : GB_BLD_WORKER (_any, _fc32, GxB_FC32_t)
                case GB_FC64_code   : GB_BLD_WORKER (_any, _fc64, GxB_FC64_t)
                default: ;
            }
            break ;

        //----------------------------------------------------------------------
        // FIRST and SECOND
        //----------------------------------------------------------------------

        case GB_FIRST_binop_code :

            switch (tcode)
            {
                case GB_INT8_code   : GB_BLD_WORKER (_first, _int8,   int8_t  )
                case GB_INT16_code  : GB_BLD_WORKER (_first, _int16,  int16_t )
                case GB_INT32_code  : GB_BLD_WORKER (_first, _int32,  int32_t )
                case GB_INT64_code  : GB_BLD_WORKER (_first, _int64,  int64_t )
                case GB_UINT8_code  : GB_BLD_WORKER (_first, _uint8,  uint8_t )
                case GB_UINT16_code : GB_BLD_WORKER (_first, _uint16, uint16_t)
                case GB_UINT32_code : GB_BLD_WORKER (_first, _uint32, uint32_t)
                case GB_UINT64_code : GB_BLD_WORKER (_first, _uint64, uint64_t)
                case GB_FP32_code   : GB_BLD_WORKER (_first, _fp32,   float   )
                case GB_FP64_code   : GB_BLD_WORKER (_first, _fp64,   double  )
                case GB_FC32_code   : GB_BLD_WORKER (_first, _fc32, GxB_FC32_t)
                case GB_FC64_code   : GB_BLD_WORKER (_first, _fc64, GxB_FC64_t)
                default: ;
            }
            break ;

        case GB_SECOND_binop_code :

            switch (tcode)
            {
                case GB_INT8_code   : GB_BLD_WORKER (_second, _int8,   int8_t  )
                case GB_INT16_code  : GB_BLD_WORKER (_second, _int16,  int16_t )
                case GB_INT32_code  : GB_BLD_WORKER (_second, _int32,  int32_t )
                case GB_INT64_code  : GB_BLD_WORKER (_second, _int64,  int64_t )
                case GB_UINT8_code  : GB_BLD_WORKER (_second, _uint8,  uint8_t )
                case GB_UINT16_code : GB_BLD_WORKER (_second, _uint16, uint16_t)
                case GB_UINT32_code : GB_BLD_WORKER (_second, _uint32, uint32_t)
                case GB_UINT64_code : GB_BLD_WORKER (_second, _uint64, uint64_t)
                case GB_FP32_code   : GB_BLD_WORKER (_second, _fp32,   float   )
                case GB_FP64_code   : GB_BLD_WORKER (_second, _fp64,   double  )
                case GB_FC32_code   : GB_BLD_WORKER (_second, _fc32, GxB_FC32_t)
                case GB_FC64_code   : GB_BLD_WORKER (_second, _fc64, GxB_FC64_t)
                default: ;
            }
            break ;

        default: ;
    }
}
else
{

    //--------------------------------------------------------------------------
    // boolean operators
    //--------------------------------------------------------------------------

    switch (opcode)
    {
        case GB_LOR_binop_code    : GB_BLD_WORKER (_lor,    _bool, bool)
        case GB_LAND_binop_code   : GB_BLD_WORKER (_land,   _bool, bool)
        case GB_LXOR_binop_code   : GB_BLD_WORKER (_lxor,   _bool, bool)
        case GB_EQ_binop_code     : GB_BLD_WORKER (_eq,     _bool, bool)
        case GB_ANY_binop_code    : GB_BLD_WORKER (_any,    _bool, bool)
        case GB_FIRST_binop_code  : GB_BLD_WORKER (_first,  _bool, bool)
        case GB_SECOND_binop_code : GB_BLD_WORKER (_second, _bool, bool)
        default: ;
    }
}

