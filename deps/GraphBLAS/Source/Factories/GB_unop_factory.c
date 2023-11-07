//------------------------------------------------------------------------------
// GB_unop_factory.c:  switch factory for unary operators and 2 types
//------------------------------------------------------------------------------

// SuiteSparse:GraphBLAS, Timothy A. Davis, (c) 2017-2023, All Rights Reserved.
// SPDX-License-Identifier: Apache-2.0

//------------------------------------------------------------------------------

// Switch factory for applying a non-positional unary operator.  This file is
// #include'd into GB_apply_op.c and GB_transpose_op.c, which must define the
// GrB_UnaryOp op and the GrB_Type Atype.  This factory does not handle
// GrB_BinaryOp or GrB_IndexUnaryOp.

// If the op is user-defined, or if the combinations of z and x type are not
// handled by the built-in operator, then this switch factory falls through
// with no action taken.

{
    // switch factory for two types, controlled by code1 and code2
    GB_Type_code code1 = op->ztype->code ;      // defines ztype
    GB_Type_code code2 = Atype->code ;          // defines the type of A
    GB_Opcode opcode = op->opcode ;

    ASSERT (code1 <= GB_UDT_code) ;
    ASSERT (code2 <= GB_UDT_code) ;
    ASSERT (opcode != GB_ONE_unop_code) ; // C is iso and the factory isn't used

    if (opcode == GB_IDENTITY_unop_code)
    { 

        //----------------------------------------------------------------------
        // z = (ztype) x, with arbitrary typecasting
        //----------------------------------------------------------------------

        // the identity operator is only used with typecasting via this switch
        // factory, so code1 is never equal to code2.

        ASSERT (code1 != code2)
        #define GB_OPNAME _identity
        #define GB_EXCLUDE_SAME_TYPES
        #include "GB_twotype_factory.c"

    }
    else if ((code1 == GB_FP32_code && code2 == GB_FC32_code) ||
             (code1 == GB_FP64_code && code2 == GB_FC64_code))
    {

        //----------------------------------------------------------------------
        // z = f (x) where z is real and x is complex (same base type)
        //----------------------------------------------------------------------

        switch (opcode)
        {

            case GB_ABS_unop_code :     // z = abs (x), for x complex

                switch (code2)
                {
                    case GB_FC32_code : GB_WORKER (_abs, _fp32, float , _fc32, GxB_FC32_t)
                    case GB_FC64_code : GB_WORKER (_abs, _fp64, double, _fc64, GxB_FC64_t)
                    default: ;
                }
                break ;

            case GB_CREAL_unop_code :   // z = creal (x)

                switch (code2)
                {
                    case GB_FC32_code : GB_WORKER (_creal, _fp32, float , _fc32, GxB_FC32_t)
                    case GB_FC64_code : GB_WORKER (_creal, _fp64, double, _fc64, GxB_FC64_t)
                    default: ;
                }
                break ;

            case GB_CIMAG_unop_code :   // z = cimag (x)

                switch (code2)
                {
                    case GB_FC32_code : GB_WORKER (_cimag, _fp32, float , _fc32, GxB_FC32_t)
                    case GB_FC64_code : GB_WORKER (_cimag, _fp64, double, _fc64, GxB_FC64_t)
                    default: ;
                }
                break ;

            case GB_CARG_unop_code :    // z = carg (x)

                switch (code2)
                {
                    case GB_FC32_code : GB_WORKER (_carg, _fp32, float , _fc32, GxB_FC32_t)
                    case GB_FC64_code : GB_WORKER (_carg, _fp64, double, _fc64, GxB_FC64_t)
                    default: ;
                }
                break ;

            default: ;
        }

    }
    else if (code1 == GB_BOOL_code && (code2 >= GB_FP32_code && code2 <= GB_FC64_code))
    {

        //----------------------------------------------------------------------
        // z = f (x) where z is boolean and x is floating-point
        //----------------------------------------------------------------------

        switch (opcode)
        {

            case GB_ISINF_unop_code :   // z = isinf (x)

                switch (code2)
                {
                    case GB_FP32_code   : GB_WORKER (_isinf, _bool, bool, _fp32, float     )
                    case GB_FP64_code   : GB_WORKER (_isinf, _bool, bool, _fp64, double    )
                    case GB_FC32_code   : GB_WORKER (_isinf, _bool, bool, _fc32, GxB_FC32_t)
                    case GB_FC64_code   : GB_WORKER (_isinf, _bool, bool, _fc64, GxB_FC64_t)
                    default: ;
                }
                break ;

            case GB_ISNAN_unop_code :   // z = isnan (x)

                switch (code2)
                {
                    case GB_FP32_code   : GB_WORKER (_isnan, _bool, bool, _fp32, float     )
                    case GB_FP64_code   : GB_WORKER (_isnan, _bool, bool, _fp64, double    )
                    case GB_FC32_code   : GB_WORKER (_isnan, _bool, bool, _fc32, GxB_FC32_t)
                    case GB_FC64_code   : GB_WORKER (_isnan, _bool, bool, _fc64, GxB_FC64_t)
                    default: ;
                }
                break ;

            case GB_ISFINITE_unop_code :// z = isfinite (x)

                switch (code2)
                {
                    case GB_FP32_code   : GB_WORKER (_isfinite, _bool, bool, _fp32, float     )
                    case GB_FP64_code   : GB_WORKER (_isfinite, _bool, bool, _fp64, double    )
                    case GB_FC32_code   : GB_WORKER (_isfinite, _bool, bool, _fc32, GxB_FC32_t)
                    case GB_FC64_code   : GB_WORKER (_isfinite, _bool, bool, _fc64, GxB_FC64_t)
                    default: ;
                }
                break ;

            default: ;

        }

    }
    else if (code1 == code2)
    {

        //----------------------------------------------------------------------
        // z = f (x) with no typecasting
        //----------------------------------------------------------------------

        switch (opcode)
        {

            case GB_AINV_unop_code :      // z = -x, all 13 types

                switch (code1)
                {
                    case GB_BOOL_code   : GB_WORKER (_ainv, _bool  , bool      , _bool  , bool      )
                    case GB_INT8_code   : GB_WORKER (_ainv, _int8  , int8_t    , _int8  , int8_t    )
                    case GB_INT16_code  : GB_WORKER (_ainv, _int16 , int16_t   , _int16 , int16_t   )
                    case GB_INT32_code  : GB_WORKER (_ainv, _int32 , int32_t   , _int32 , int32_t   )
                    case GB_INT64_code  : GB_WORKER (_ainv, _int64 , int64_t   , _int64 , int64_t   )
                    case GB_UINT8_code  : GB_WORKER (_ainv, _uint8 , uint8_t   , _uint8 , uint8_t   )
                    case GB_UINT16_code : GB_WORKER (_ainv, _uint16, uint16_t  , _uint16, uint16_t  )
                    case GB_UINT32_code : GB_WORKER (_ainv, _uint32, uint32_t  , _uint32, uint32_t  )
                    case GB_UINT64_code : GB_WORKER (_ainv, _uint64, uint64_t  , _uint64, uint64_t  )
                    case GB_FP32_code   : GB_WORKER (_ainv, _fp32  , float     , _fp32  , float     )
                    case GB_FP64_code   : GB_WORKER (_ainv, _fp64  , double    , _fp64  , double    )
                    case GB_FC32_code   : GB_WORKER (_ainv, _fc32  , GxB_FC32_t, _fc32  , GxB_FC32_t)
                    case GB_FC64_code   : GB_WORKER (_ainv, _fc64  , GxB_FC64_t, _fc64  , GxB_FC64_t)
                    default: ;
                }
                break ;

            case GB_MINV_unop_code :      // z = 1/x, all 13 types

                switch (code1)
                {
                    case GB_BOOL_code   : GB_WORKER (_minv, _bool  , bool      , _bool  , bool      )
                    case GB_INT8_code   : GB_WORKER (_minv, _int8  , int8_t    , _int8  , int8_t    )
                    case GB_INT16_code  : GB_WORKER (_minv, _int16 , int16_t   , _int16 , int16_t   )
                    case GB_INT32_code  : GB_WORKER (_minv, _int32 , int32_t   , _int32 , int32_t   )
                    case GB_INT64_code  : GB_WORKER (_minv, _int64 , int64_t   , _int64 , int64_t   )
                    case GB_UINT8_code  : GB_WORKER (_minv, _uint8 , uint8_t   , _uint8 , uint8_t   )
                    case GB_UINT16_code : GB_WORKER (_minv, _uint16, uint16_t  , _uint16, uint16_t  )
                    case GB_UINT32_code : GB_WORKER (_minv, _uint32, uint32_t  , _uint32, uint32_t  )
                    case GB_UINT64_code : GB_WORKER (_minv, _uint64, uint64_t  , _uint64, uint64_t  )
                    case GB_FP32_code   : GB_WORKER (_minv, _fp32  , float     , _fp32  , float     )
                    case GB_FP64_code   : GB_WORKER (_minv, _fp64  , double    , _fp64  , double    )
                    case GB_FC32_code   : GB_WORKER (_minv, _fc32  , GxB_FC32_t, _fc32  , GxB_FC32_t)
                    case GB_FC64_code   : GB_WORKER (_minv, _fc64  , GxB_FC64_t, _fc64  , GxB_FC64_t)
                    default: ;
                }
                break ;

            case GB_ABS_unop_code :       // z = abs (x), for all but complex

                switch (code1)
                {
                    case GB_BOOL_code   : GB_WORKER (_abs, _bool  , bool      , _bool  , bool    )
                    case GB_INT8_code   : GB_WORKER (_abs, _int8  , int8_t    , _int8  , int8_t  )
                    case GB_INT16_code  : GB_WORKER (_abs, _int16 , int16_t   , _int16 , int16_t )
                    case GB_INT32_code  : GB_WORKER (_abs, _int32 , int32_t   , _int32 , int32_t )
                    case GB_INT64_code  : GB_WORKER (_abs, _int64 , int64_t   , _int64 , int64_t )
                    case GB_UINT8_code  : GB_WORKER (_abs, _uint8 , uint8_t   , _uint8 , uint8_t )
                    case GB_UINT16_code : GB_WORKER (_abs, _uint16, uint16_t  , _uint16, uint16_t)
                    case GB_UINT32_code : GB_WORKER (_abs, _uint32, uint32_t  , _uint32, uint32_t)
                    case GB_UINT64_code : GB_WORKER (_abs, _uint64, uint64_t  , _uint64, uint64_t)
                    case GB_FP32_code   : GB_WORKER (_abs, _fp32  , float     , _fp32  , float   )
                    case GB_FP64_code   : GB_WORKER (_abs, _fp64  , double    , _fp64  , double  )
                    default: ;
                }
                break ;

            case GB_LNOT_unop_code :      // z = ! (x != 0), no complex case

                switch (code1)
                {
                    case GB_BOOL_code   : GB_WORKER (_lnot, _bool  , bool    , _bool  , bool    )
                    case GB_INT8_code   : GB_WORKER (_lnot, _int8  , int8_t  , _int8  , int8_t  )
                    case GB_INT16_code  : GB_WORKER (_lnot, _int16 , int16_t , _int16 , int16_t )
                    case GB_INT32_code  : GB_WORKER (_lnot, _int32 , int32_t , _int32 , int32_t )
                    case GB_INT64_code  : GB_WORKER (_lnot, _int64 , int64_t , _int64 , int64_t )
                    case GB_UINT8_code  : GB_WORKER (_lnot, _uint8 , uint8_t , _uint8 , uint8_t )
                    case GB_UINT16_code : GB_WORKER (_lnot, _uint16, uint16_t, _uint16, uint16_t)
                    case GB_UINT32_code : GB_WORKER (_lnot, _uint32, uint32_t, _uint32, uint32_t)
                    case GB_UINT64_code : GB_WORKER (_lnot, _uint64, uint64_t, _uint64, uint64_t)
                    case GB_FP32_code   : GB_WORKER (_lnot, _fp32  , float   , _fp32  , float   )
                    case GB_FP64_code   : GB_WORKER (_lnot, _fp64  , double  , _fp64  , double  )
                    default: ;
                }
                break ;

            case GB_BNOT_unop_code :    // z = ~x (bitwise complement), integers only

                switch (code1)
                {
                    case GB_INT8_code   : GB_WORKER (_bnot, _int8  , int8_t  , _int8  , int8_t  )
                    case GB_INT16_code  : GB_WORKER (_bnot, _int16 , int16_t , _int16 , int16_t )
                    case GB_INT32_code  : GB_WORKER (_bnot, _int32 , int32_t , _int32 , int32_t )
                    case GB_INT64_code  : GB_WORKER (_bnot, _int64 , int64_t , _int64 , int64_t )
                    case GB_UINT8_code  : GB_WORKER (_bnot, _uint8 , uint8_t , _uint8 , uint8_t )
                    case GB_UINT16_code : GB_WORKER (_bnot, _uint16, uint16_t, _uint16, uint16_t)
                    case GB_UINT32_code : GB_WORKER (_bnot, _uint32, uint32_t, _uint32, uint32_t)
                    case GB_UINT64_code : GB_WORKER (_bnot, _uint64, uint64_t, _uint64, uint64_t)
                    default: ;
                }
                break ;

            case GB_SQRT_unop_code :    // z = sqrt (x)

                switch (code1)
                {
                    case GB_FP32_code   : GB_WORKER (_sqrt, _fp32, float     , _fp32, float     )
                    case GB_FP64_code   : GB_WORKER (_sqrt, _fp64, double    , _fp64, double    )
                    case GB_FC32_code   : GB_WORKER (_sqrt, _fc32, GxB_FC32_t, _fc32, GxB_FC32_t)
                    case GB_FC64_code   : GB_WORKER (_sqrt, _fc64, GxB_FC64_t, _fc64, GxB_FC64_t)
                    default: ;
                }
                break ;

            case GB_LOG_unop_code :     // z = log (x)

                switch (code1)
                {
                    case GB_FP32_code   : GB_WORKER (_log, _fp32, float     , _fp32, float     )
                    case GB_FP64_code   : GB_WORKER (_log, _fp64, double    , _fp64, double    )
                    case GB_FC32_code   : GB_WORKER (_log, _fc32, GxB_FC32_t, _fc32, GxB_FC32_t)
                    case GB_FC64_code   : GB_WORKER (_log, _fc64, GxB_FC64_t, _fc64, GxB_FC64_t)
                    default: ;
                }
                break ;


            case GB_EXP_unop_code :     // z = exp (x)

                switch (code1)
                {
                    case GB_FP32_code   : GB_WORKER (_exp, _fp32, float     , _fp32, float     )
                    case GB_FP64_code   : GB_WORKER (_exp, _fp64, double    , _fp64, double    )
                    case GB_FC32_code   : GB_WORKER (_exp, _fc32, GxB_FC32_t, _fc32, GxB_FC32_t)
                    case GB_FC64_code   : GB_WORKER (_exp, _fc64, GxB_FC64_t, _fc64, GxB_FC64_t)
                    default: ;
                }
                break ;


            case GB_SIN_unop_code :     // z = sin (x)

                switch (code1)
                {
                    case GB_FP32_code   : GB_WORKER (_sin, _fp32, float     , _fp32, float     )
                    case GB_FP64_code   : GB_WORKER (_sin, _fp64, double    , _fp64, double    )
                    case GB_FC32_code   : GB_WORKER (_sin, _fc32, GxB_FC32_t, _fc32, GxB_FC32_t)
                    case GB_FC64_code   : GB_WORKER (_sin, _fc64, GxB_FC64_t, _fc64, GxB_FC64_t)
                    default: ;
                }
                break ;

            case GB_COS_unop_code :     // z = cos (x)

                switch (code1)
                {
                    case GB_FP32_code   : GB_WORKER (_cos, _fp32, float     , _fp32, float     )
                    case GB_FP64_code   : GB_WORKER (_cos, _fp64, double    , _fp64, double    )
                    case GB_FC32_code   : GB_WORKER (_cos, _fc32, GxB_FC32_t, _fc32, GxB_FC32_t)
                    case GB_FC64_code   : GB_WORKER (_cos, _fc64, GxB_FC64_t, _fc64, GxB_FC64_t)
                    default: ;
                }
                break ;

            case GB_TAN_unop_code :     // z = tan (x)

                switch (code1)
                {
                    case GB_FP32_code   : GB_WORKER (_tan, _fp32, float     , _fp32, float     )
                    case GB_FP64_code   : GB_WORKER (_tan, _fp64, double    , _fp64, double    )
                    case GB_FC32_code   : GB_WORKER (_tan, _fc32, GxB_FC32_t, _fc32, GxB_FC32_t)
                    case GB_FC64_code   : GB_WORKER (_tan, _fc64, GxB_FC64_t, _fc64, GxB_FC64_t)
                    default: ;
                }
                break ;


            case GB_ASIN_unop_code :    // z = asin (x)

                switch (code1)
                {
                    case GB_FP32_code   : GB_WORKER (_asin, _fp32, float     , _fp32, float     )
                    case GB_FP64_code   : GB_WORKER (_asin, _fp64, double    , _fp64, double    )
                    case GB_FC32_code   : GB_WORKER (_asin, _fc32, GxB_FC32_t, _fc32, GxB_FC32_t)
                    case GB_FC64_code   : GB_WORKER (_asin, _fc64, GxB_FC64_t, _fc64, GxB_FC64_t)
                    default: ;
                }
                break ;

            case GB_ACOS_unop_code :    // z = acos (x)

                switch (code1)
                {
                    case GB_FP32_code   : GB_WORKER (_acos, _fp32, float     , _fp32, float     )
                    case GB_FP64_code   : GB_WORKER (_acos, _fp64, double    , _fp64, double    )
                    case GB_FC32_code   : GB_WORKER (_acos, _fc32, GxB_FC32_t, _fc32, GxB_FC32_t)
                    case GB_FC64_code   : GB_WORKER (_acos, _fc64, GxB_FC64_t, _fc64, GxB_FC64_t)
                    default: ;
                }
                break ;

            case GB_ATAN_unop_code :    // z = atan (x)

                switch (code1)
                {
                    case GB_FP32_code   : GB_WORKER (_atan, _fp32, float     , _fp32, float     )
                    case GB_FP64_code   : GB_WORKER (_atan, _fp64, double    , _fp64, double    )
                    case GB_FC32_code   : GB_WORKER (_atan, _fc32, GxB_FC32_t, _fc32, GxB_FC32_t)
                    case GB_FC64_code   : GB_WORKER (_atan, _fc64, GxB_FC64_t, _fc64, GxB_FC64_t)
                    default: ;
                }
                break ;


            case GB_SINH_unop_code :    // z = sinh (x)

                switch (code1)
                {
                    case GB_FP32_code   : GB_WORKER (_sinh, _fp32, float     , _fp32, float     )
                    case GB_FP64_code   : GB_WORKER (_sinh, _fp64, double    , _fp64, double    )
                    case GB_FC32_code   : GB_WORKER (_sinh, _fc32, GxB_FC32_t, _fc32, GxB_FC32_t)
                    case GB_FC64_code   : GB_WORKER (_sinh, _fc64, GxB_FC64_t, _fc64, GxB_FC64_t)
                    default: ;
                }
                break ;

            case GB_COSH_unop_code :    // z = cosh (x)

                switch (code1)
                {
                    case GB_FP32_code   : GB_WORKER (_cosh, _fp32, float     , _fp32, float     )
                    case GB_FP64_code   : GB_WORKER (_cosh, _fp64, double    , _fp64, double    )
                    case GB_FC32_code   : GB_WORKER (_cosh, _fc32, GxB_FC32_t, _fc32, GxB_FC32_t)
                    case GB_FC64_code   : GB_WORKER (_cosh, _fc64, GxB_FC64_t, _fc64, GxB_FC64_t)
                    default: ;
                }
                break ;

            case GB_TANH_unop_code :    // z = tanh (x)

                switch (code1)
                {
                    case GB_FP32_code   : GB_WORKER (_tanh, _fp32, float     , _fp32, float     )
                    case GB_FP64_code   : GB_WORKER (_tanh, _fp64, double    , _fp64, double    )
                    case GB_FC32_code   : GB_WORKER (_tanh, _fc32, GxB_FC32_t, _fc32, GxB_FC32_t)
                    case GB_FC64_code   : GB_WORKER (_tanh, _fc64, GxB_FC64_t, _fc64, GxB_FC64_t)
                    default: ;
                }
                break ;


            case GB_ASINH_unop_code :   // z = asinh (x)

                switch (code1)
                {
                    case GB_FP32_code   : GB_WORKER (_asinh, _fp32, float     , _fp32, float     )
                    case GB_FP64_code   : GB_WORKER (_asinh, _fp64, double    , _fp64, double    )
                    case GB_FC32_code   : GB_WORKER (_asinh, _fc32, GxB_FC32_t, _fc32, GxB_FC32_t)
                    case GB_FC64_code   : GB_WORKER (_asinh, _fc64, GxB_FC64_t, _fc64, GxB_FC64_t)
                    default: ;
                }
                break ;

            case GB_ACOSH_unop_code :   // z = acosh (x)

                switch (code1)
                {
                    case GB_FP32_code   : GB_WORKER (_acosh, _fp32, float     , _fp32, float     )
                    case GB_FP64_code   : GB_WORKER (_acosh, _fp64, double    , _fp64, double    )
                    case GB_FC32_code   : GB_WORKER (_acosh, _fc32, GxB_FC32_t, _fc32, GxB_FC32_t)
                    case GB_FC64_code   : GB_WORKER (_acosh, _fc64, GxB_FC64_t, _fc64, GxB_FC64_t)
                    default: ;
                }
                break ;

            case GB_ATANH_unop_code :   // z = atanh (x)

                switch (code1)
                {
                    case GB_FP32_code   : GB_WORKER (_atanh, _fp32, float     , _fp32, float     )
                    case GB_FP64_code   : GB_WORKER (_atanh, _fp64, double    , _fp64, double    )
                    case GB_FC32_code   : GB_WORKER (_atanh, _fc32, GxB_FC32_t, _fc32, GxB_FC32_t)
                    case GB_FC64_code   : GB_WORKER (_atanh, _fc64, GxB_FC64_t, _fc64, GxB_FC64_t)
                    default: ;
                }
                break ;

            case GB_SIGNUM_unop_code :  // z = signum (x)

                switch (code1)
                {
                    case GB_FP32_code   : GB_WORKER (_signum, _fp32, float     , _fp32, float     )
                    case GB_FP64_code   : GB_WORKER (_signum, _fp64, double    , _fp64, double    )
                    case GB_FC32_code   : GB_WORKER (_signum, _fc32, GxB_FC32_t, _fc32, GxB_FC32_t)
                    case GB_FC64_code   : GB_WORKER (_signum, _fc64, GxB_FC64_t, _fc64, GxB_FC64_t)
                    default: ;
                }
                break ;

            case GB_CEIL_unop_code :    // z = ceil (x)

                switch (code1)
                {
                    case GB_FP32_code   : GB_WORKER (_ceil, _fp32, float     , _fp32, float     )
                    case GB_FP64_code   : GB_WORKER (_ceil, _fp64, double    , _fp64, double    )
                    case GB_FC32_code   : GB_WORKER (_ceil, _fc32, GxB_FC32_t, _fc32, GxB_FC32_t)
                    case GB_FC64_code   : GB_WORKER (_ceil, _fc64, GxB_FC64_t, _fc64, GxB_FC64_t)
                    default: ;
                }
                break ;

            case GB_FLOOR_unop_code :   // z = floor (x)

                switch (code1)
                {
                    case GB_FP32_code   : GB_WORKER (_floor, _fp32, float     , _fp32, float     )
                    case GB_FP64_code   : GB_WORKER (_floor, _fp64, double    , _fp64, double    )
                    case GB_FC32_code   : GB_WORKER (_floor, _fc32, GxB_FC32_t, _fc32, GxB_FC32_t)
                    case GB_FC64_code   : GB_WORKER (_floor, _fc64, GxB_FC64_t, _fc64, GxB_FC64_t)
                    default: ;
                }
                break ;

            case GB_ROUND_unop_code :   // z = round (x)

                switch (code1)
                {
                    case GB_FP32_code   : GB_WORKER (_round, _fp32, float     , _fp32, float     )
                    case GB_FP64_code   : GB_WORKER (_round, _fp64, double    , _fp64, double    )
                    case GB_FC32_code   : GB_WORKER (_round, _fc32, GxB_FC32_t, _fc32, GxB_FC32_t)
                    case GB_FC64_code   : GB_WORKER (_round, _fc64, GxB_FC64_t, _fc64, GxB_FC64_t)
                    default: ;
                }
                break ;

            case GB_TRUNC_unop_code :   // z = trunc (x)

                switch (code1)
                {
                    case GB_FP32_code   : GB_WORKER (_trunc, _fp32, float     , _fp32, float     )
                    case GB_FP64_code   : GB_WORKER (_trunc, _fp64, double    , _fp64, double    )
                    case GB_FC32_code   : GB_WORKER (_trunc, _fc32, GxB_FC32_t, _fc32, GxB_FC32_t)
                    case GB_FC64_code   : GB_WORKER (_trunc, _fc64, GxB_FC64_t, _fc64, GxB_FC64_t)
                    default: ;
                }
                break ;


            case GB_EXP2_unop_code :    // z = exp2 (x)

                switch (code1)
                {
                    case GB_FP32_code   : GB_WORKER (_exp2, _fp32, float     , _fp32, float     )
                    case GB_FP64_code   : GB_WORKER (_exp2, _fp64, double    , _fp64, double    )
                    case GB_FC32_code   : GB_WORKER (_exp2, _fc32, GxB_FC32_t, _fc32, GxB_FC32_t)
                    case GB_FC64_code   : GB_WORKER (_exp2, _fc64, GxB_FC64_t, _fc64, GxB_FC64_t)
                    default: ;
                }
                break ;

            case GB_EXPM1_unop_code :   // z = expm1 (x)

                switch (code1)
                {
                    case GB_FP32_code   : GB_WORKER (_expm1, _fp32, float     , _fp32, float     )
                    case GB_FP64_code   : GB_WORKER (_expm1, _fp64, double    , _fp64, double    )
                    case GB_FC32_code   : GB_WORKER (_expm1, _fc32, GxB_FC32_t, _fc32, GxB_FC32_t)
                    case GB_FC64_code   : GB_WORKER (_expm1, _fc64, GxB_FC64_t, _fc64, GxB_FC64_t)
                    default: ;
                }
                break ;

            case GB_LOG10_unop_code :   // z = log10 (x)

                switch (code1)
                {
                    case GB_FP32_code   : GB_WORKER (_log10, _fp32, float     , _fp32, float     )
                    case GB_FP64_code   : GB_WORKER (_log10, _fp64, double    , _fp64, double    )
                    case GB_FC32_code   : GB_WORKER (_log10, _fc32, GxB_FC32_t, _fc32, GxB_FC32_t)
                    case GB_FC64_code   : GB_WORKER (_log10, _fc64, GxB_FC64_t, _fc64, GxB_FC64_t)
                    default: ;
                }
                break ;

            case GB_LOG1P_unop_code :   // z = log1P (x)

                switch (code1)
                {
                    case GB_FP32_code   : GB_WORKER (_log1p, _fp32, float     , _fp32, float     )
                    case GB_FP64_code   : GB_WORKER (_log1p, _fp64, double    , _fp64, double    )
                    case GB_FC32_code   : GB_WORKER (_log1p, _fc32, GxB_FC32_t, _fc32, GxB_FC32_t)
                    case GB_FC64_code   : GB_WORKER (_log1p, _fc64, GxB_FC64_t, _fc64, GxB_FC64_t)
                    default: ;
                }
                break ;

            case GB_LOG2_unop_code :    // z = log2 (x)

                switch (code1)
                {
                    case GB_FP32_code   : GB_WORKER (_log2, _fp32, float     , _fp32, float     )
                    case GB_FP64_code   : GB_WORKER (_log2, _fp64, double    , _fp64, double    )
                    case GB_FC32_code   : GB_WORKER (_log2, _fc32, GxB_FC32_t, _fc32, GxB_FC32_t)
                    case GB_FC64_code   : GB_WORKER (_log2, _fc64, GxB_FC64_t, _fc64, GxB_FC64_t)
                    default: ;
                }
                break ;

            case GB_LGAMMA_unop_code :  // z = lgamma (x)

                switch (code1)
                {
                    case GB_FP32_code   : GB_WORKER (_lgamma, _fp32, float     , _fp32, float )
                    case GB_FP64_code   : GB_WORKER (_lgamma, _fp64, double    , _fp64, double)
                    default: ;
                }
                break ;

            case GB_TGAMMA_unop_code :  // z = tgamma (x)

                switch (code1)
                {
                    case GB_FP32_code   : GB_WORKER (_tgamma, _fp32, float     , _fp32, float )
                    case GB_FP64_code   : GB_WORKER (_tgamma, _fp64, double    , _fp64, double)
                    default: ;
                }
                break ;

            case GB_ERF_unop_code :     // z = erf (x)

                switch (code1)
                {
                    case GB_FP32_code   : GB_WORKER (_erf, _fp32, float     , _fp32, float )
                    case GB_FP64_code   : GB_WORKER (_erf, _fp64, double    , _fp64, double)
                    default: ;
                }
                break ;

            case GB_ERFC_unop_code :    // z = erfc (x)

                switch (code1)
                {
                    case GB_FP32_code   : GB_WORKER (_erfc, _fp32, float     , _fp32, float )
                    case GB_FP64_code   : GB_WORKER (_erfc, _fp64, double    , _fp64, double)
                    default: ;
                }
                break ;

            case GB_CBRT_unop_code :    // z = cbrt (x)

                switch (code1)
                {
                    case GB_FP32_code   : GB_WORKER (_cbrt, _fp32, float     , _fp32, float )
                    case GB_FP64_code   : GB_WORKER (_cbrt, _fp64, double    , _fp64, double)
                    default: ;
                }
                break ;

            case GB_FREXPX_unop_code :  // z = frexpx (x), mantissa from C11 frexp

                switch (code1)
                {
                    case GB_FP32_code   : GB_WORKER (_frexpx, _fp32, float     , _fp32, float )
                    case GB_FP64_code   : GB_WORKER (_frexpx, _fp64, double    , _fp64, double)
                    default: ;
                }
                break ;

            case GB_FREXPE_unop_code :  // z = frexpe (x), exponent from C11 frexp

                switch (code1)
                {
                    case GB_FP32_code   : GB_WORKER (_frexpe, _fp32, float     , _fp32, float )
                    case GB_FP64_code   : GB_WORKER (_frexpe, _fp64, double    , _fp64, double)
                    default: ;
                }
                break ;

            case GB_CONJ_unop_code :    // z = conj (x)

                switch (code1)
                {
                    case GB_FC32_code   : GB_WORKER (_conj, _fc32, GxB_FC32_t, _fc32, GxB_FC32_t)
                    case GB_FC64_code   : GB_WORKER (_conj, _fc64, GxB_FC64_t, _fc64, GxB_FC64_t)
                    default: ;
                }
                break ;

            default: ;
        }
    }
}

