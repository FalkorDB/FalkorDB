//------------------------------------------------------------------------------
// GB_unop_factory.c:  switch factory for unary operators and 2 types
//------------------------------------------------------------------------------

// SuiteSparse:GraphBLAS, Timothy A. Davis, (c) 2017-2025, All Rights Reserved.
// SPDX-License-Identifier: Apache-2.0

//------------------------------------------------------------------------------

// Switch factory for applying a non-positional unary operator.  This file is
// #include'd into GB_apply_op.c and GB_transpose_op.c, which must define the
// GrB_UnaryOp op and the GrB_Type Atype.  This factory does not handle
// GrB_BinaryOp or GrB_IndexUnaryOp.

// If the op is user-defined, or if the combinations of z and x type are not
// handled by the built-in operator, then this switch factory falls through
// with no action taken.

#if defined (GxB_NO_BOOL)
#define GB_CASE_BOOL(op,zname,ztype)
#else
#define GB_CASE_BOOL(op,zname,ztype) \
    case GB_BOOL_code:   GB_WORKER (op, zname, ztype, _bool  , bool    )
#endif

#if defined (GxB_NO_INT8)
#define GB_CASE_INT8(op,zname,ztype)
#else
#define GB_CASE_INT8(op,zname,ztype) \
    case GB_INT8_code:   GB_WORKER (op, zname, ztype, _int8  , int8_t  )
#endif

#if defined (GxB_NO_INT16)
#define GB_CASE_INT16(op,zname,ztype)
#else
#define GB_CASE_INT16(op,zname,ztype) \
    case GB_INT16_code:  GB_WORKER (op, zname, ztype, _int16 , int16_t )
#endif

#if defined (GxB_NO_INT32)
#define GB_CASE_INT32(op,zname,ztype)
#else
#define GB_CASE_INT32(op,zname,ztype) \
    case GB_INT32_code:  GB_WORKER (op, zname, ztype, _int32 , int32_t )
#endif

#if defined (GxB_NO_INT64)
#define GB_CASE_INT64(op,zname,ztype)
#else
#define GB_CASE_INT64(op,zname,ztype) \
    case GB_INT64_code:  GB_WORKER (op, zname, ztype, _int64 , int64_t )
#endif

#if defined (GxB_NO_UINT8)
#define GB_CASE_UINT8(op,zname,ztype)
#else
#define GB_CASE_UINT8(op,zname,ztype) \
    case GB_UINT8_code:  GB_WORKER (op, zname, ztype, _uint8 , uint8_t )
#endif

#if defined (GxB_NO_UINT16)
#define GB_CASE_UINT16(op,zname,ztype)
#else
#define GB_CASE_UINT16(op,zname,ztype) \
    case GB_UINT16_code: GB_WORKER (op, zname, ztype, _uint16, uint16_t)
#endif

#if defined (GxB_NO_UINT32)
#define GB_CASE_UINT32(op,zname,ztype)
#else
#define GB_CASE_UINT32(op,zname,ztype) \
    case GB_UINT32_code: GB_WORKER (op, zname, ztype, _uint32, uint32_t)
#endif

#if defined (GxB_NO_UINT64)
#define GB_CASE_UINT64(op,zname,ztype)
#else
#define GB_CASE_UINT64(op,zname,ztype) \
    case GB_UINT64_code: GB_WORKER (op, zname, ztype, _uint64, uint64_t)
#endif

#if defined (GxB_NO_FP32)
#define GB_CASE_FP32(op,zname,ztype)
#else
#define GB_CASE_FP32(op,zname,ztype) \
    case GB_FP32_code:   GB_WORKER (op, zname, ztype, _fp32  , float   )
#endif

#if defined (GxB_NO_FP64)
#define GB_CASE_FP64(op,zname,ztype)
#else
#define GB_CASE_FP64(op,zname,ztype) \
    case GB_FP64_code:   GB_WORKER (op, zname, ztype, _fp64  , double  )
#endif

#if defined (GxB_NO_FC32)
#define GB_CASE_FC32(op,zname,ztype)
#else
#define GB_CASE_FC32(op,zname,ztype) \
    case GB_FC32_code:   GB_WORKER (op, zname, ztype, _fc32  , GxB_FC32_t)
#endif

#if defined (GxB_NO_FC64)
#define GB_CASE_FC64(op,zname,ztype)
#else
#define GB_CASE_FC64(op,zname,ztype) \
    case GB_FC64_code:   GB_WORKER (op, zname, ztype, _fc64  , GxB_FC64_t)
#endif

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
        #include "apply/factory/GB_twotype_factory.c"

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
                    GB_CASE_FC32 (_abs, _fp32, float )
                    GB_CASE_FC64 (_abs, _fp64, double)
                    default: ;
                }
                break ;

            case GB_CREAL_unop_code :   // z = creal (x)

                switch (code2)
                {
                    GB_CASE_FC32 (_creal, _fp32, float )
                    GB_CASE_FC64 (_creal, _fp64, double)
                    default: ;
                }
                break ;

            case GB_CIMAG_unop_code :   // z = cimag (x)

                switch (code2)
                {
                    GB_CASE_FC32 (_cimag, _fp32, float )
                    GB_CASE_FC64 (_cimag, _fp64, double)
                    default: ;
                }
                break ;

            case GB_CARG_unop_code :    // z = carg (x)

                switch (code2)
                {
                    GB_CASE_FC32 (_carg, _fp32, float )
                    GB_CASE_FC64 (_carg, _fp64, double)
                    default: ;
                }
                break ;

            default: ;
        }

    }
    else if (code1 == GB_BOOL_code && 
            (code2 >= GB_FP32_code && code2 <= GB_FC64_code))
    { 

        //----------------------------------------------------------------------
        // z = f (x) where z is boolean and x is floating-point
        //----------------------------------------------------------------------

        switch (opcode)
        {

            case GB_ISINF_unop_code :   // z = isinf (x)

                switch (code2)
                {
                    GB_CASE_FP32 (_isinf, _bool, bool)
                    GB_CASE_FP64 (_isinf, _bool, bool)
                    GB_CASE_FC32 (_isinf, _bool, bool)
                    GB_CASE_FC64 (_isinf, _bool, bool)
                    default: ;
                }
                break ;

            case GB_ISNAN_unop_code :   // z = isnan (x)

                switch (code2)
                {
                    GB_CASE_FP32 (_isnan, _bool, bool)
                    GB_CASE_FP64 (_isnan, _bool, bool)
                    GB_CASE_FC32 (_isnan, _bool, bool)
                    GB_CASE_FC64 (_isnan, _bool, bool)
                    default: ;
                }
                break ;

            case GB_ISFINITE_unop_code :// z = isfinite (x)

                switch (code2)
                {
                    GB_CASE_FP32 (_isfinite, _bool, bool)
                    GB_CASE_FP64 (_isfinite, _bool, bool)
                    GB_CASE_FC32 (_isfinite, _bool, bool)
                    GB_CASE_FC64 (_isfinite, _bool, bool)
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

            case GB_AINV_unop_code :      // z = -x, all signed types

                switch (code1)
                {
                    GB_CASE_INT8   (_ainv, _int8  , int8_t    )
                    GB_CASE_INT16  (_ainv, _int16 , int16_t   )
                    GB_CASE_INT32  (_ainv, _int32 , int32_t   )
                    GB_CASE_INT64  (_ainv, _int64 , int64_t   )
                    GB_CASE_FP32   (_ainv, _fp32  , float     )
                    GB_CASE_FP64   (_ainv, _fp64  , double    )
                    GB_CASE_FC32   (_ainv, _fc32  , GxB_FC32_t)
                    GB_CASE_FC64   (_ainv, _fc64  , GxB_FC64_t)
                    default: ;
                }
                break ;

            case GB_MINV_unop_code :      // z = 1/x, just floating-point types

                switch (code1)
                {
                    GB_CASE_FP32   (_minv, _fp32  , float     )
                    GB_CASE_FP64   (_minv, _fp64  , double    )
                    GB_CASE_FC32   (_minv, _fc32  , GxB_FC32_t)
                    GB_CASE_FC64   (_minv, _fc64  , GxB_FC64_t)
                    default: ;
                }
                break ;

            case GB_ABS_unop_code :       // z = abs (x), signed, not cmplx

                switch (code1)
                {
                    GB_CASE_INT8   (_abs, _int8  , int8_t  )
                    GB_CASE_INT16  (_abs, _int16 , int16_t )
                    GB_CASE_INT32  (_abs, _int32 , int32_t )
                    GB_CASE_INT64  (_abs, _int64 , int64_t )
                    GB_CASE_FP32   (_abs, _fp32  , float   )
                    GB_CASE_FP64   (_abs, _fp64  , double  )
                    default: ;
                }
                break ;

            case GB_LNOT_unop_code :      // z = !x, boolean only

                switch (code1)
                {
                    GB_CASE_BOOL   (_lnot, _bool  , bool    )
                    default: ;
                }
                break ;

            case GB_BNOT_unop_code :    // z = ~x (bitwise compl), integers only

                switch (code1)
                {
                    GB_CASE_UINT8  (_bnot, _uint8 , uint8_t )
                    GB_CASE_UINT16 (_bnot, _uint16, uint16_t)
                    GB_CASE_UINT32 (_bnot, _uint32, uint32_t)
                    GB_CASE_UINT64 (_bnot, _uint64, uint64_t)
                    default: ;
                }
                break ;

            case GB_SQRT_unop_code :    // z = sqrt (x)

                switch (code1)
                {
                    GB_CASE_FP32 (_sqrt, _fp32, float     )
                    GB_CASE_FP64 (_sqrt, _fp64, double    )
                    GB_CASE_FC32 (_sqrt, _fc32, GxB_FC32_t)
                    GB_CASE_FC64 (_sqrt, _fc64, GxB_FC64_t)
                    default: ;
                }
                break ;

            case GB_LOG_unop_code :     // z = log (x)

                switch (code1)
                {
                    GB_CASE_FP32 (_log, _fp32, float     )
                    GB_CASE_FP64 (_log, _fp64, double    )
                    GB_CASE_FC32 (_log, _fc32, GxB_FC32_t)
                    GB_CASE_FC64 (_log, _fc64, GxB_FC64_t)
                    default: ;
                }
                break ;


            case GB_EXP_unop_code :     // z = exp (x)

                switch (code1)
                {
                    GB_CASE_FP32 (_exp, _fp32, float     )
                    GB_CASE_FP64 (_exp, _fp64, double    )
                    GB_CASE_FC32 (_exp, _fc32, GxB_FC32_t)
                    GB_CASE_FC64 (_exp, _fc64, GxB_FC64_t)
                    default: ;
                }
                break ;


            case GB_SIN_unop_code :     // z = sin (x)

                switch (code1)
                {
                    GB_CASE_FP32 (_sin, _fp32, float     )
                    GB_CASE_FP64 (_sin, _fp64, double    )
                    GB_CASE_FC32 (_sin, _fc32, GxB_FC32_t)
                    GB_CASE_FC64 (_sin, _fc64, GxB_FC64_t)
                    default: ;
                }
                break ;

            case GB_COS_unop_code :     // z = cos (x)

                switch (code1)
                {
                    GB_CASE_FP32 (_cos, _fp32, float     )
                    GB_CASE_FP64 (_cos, _fp64, double    )
                    GB_CASE_FC32 (_cos, _fc32, GxB_FC32_t)
                    GB_CASE_FC64 (_cos, _fc64, GxB_FC64_t)
                    default: ;
                }
                break ;

            case GB_TAN_unop_code :     // z = tan (x)

                switch (code1)
                {
                    GB_CASE_FP32 (_tan, _fp32, float     )
                    GB_CASE_FP64 (_tan, _fp64, double    )
                    GB_CASE_FC32 (_tan, _fc32, GxB_FC32_t)
                    GB_CASE_FC64 (_tan, _fc64, GxB_FC64_t)
                    default: ;
                }
                break ;


            case GB_ASIN_unop_code :    // z = asin (x)

                switch (code1)
                {
                    GB_CASE_FP32 (_asin, _fp32, float     )
                    GB_CASE_FP64 (_asin, _fp64, double    )
                    GB_CASE_FC32 (_asin, _fc32, GxB_FC32_t)
                    GB_CASE_FC64 (_asin, _fc64, GxB_FC64_t)
                    default: ;
                }
                break ;

            case GB_ACOS_unop_code :    // z = acos (x)

                switch (code1)
                {
                    GB_CASE_FP32 (_acos, _fp32, float     )
                    GB_CASE_FP64 (_acos, _fp64, double    )
                    GB_CASE_FC32 (_acos, _fc32, GxB_FC32_t)
                    GB_CASE_FC64 (_acos, _fc64, GxB_FC64_t)
                    default: ;
                }
                break ;

            case GB_ATAN_unop_code :    // z = atan (x)

                switch (code1)
                {
                    GB_CASE_FP32 (_atan, _fp32, float     )
                    GB_CASE_FP64 (_atan, _fp64, double    )
                    GB_CASE_FC32 (_atan, _fc32, GxB_FC32_t)
                    GB_CASE_FC64 (_atan, _fc64, GxB_FC64_t)
                    default: ;
                }
                break ;


            case GB_SINH_unop_code :    // z = sinh (x)

                switch (code1)
                {
                    GB_CASE_FP32 (_sinh, _fp32, float     )
                    GB_CASE_FP64 (_sinh, _fp64, double    )
                    GB_CASE_FC32 (_sinh, _fc32, GxB_FC32_t)
                    GB_CASE_FC64 (_sinh, _fc64, GxB_FC64_t)
                    default: ;
                }
                break ;

            case GB_COSH_unop_code :    // z = cosh (x)

                switch (code1)
                {
                    GB_CASE_FP32 (_cosh, _fp32, float     )
                    GB_CASE_FP64 (_cosh, _fp64, double    )
                    GB_CASE_FC32 (_cosh, _fc32, GxB_FC32_t)
                    GB_CASE_FC64 (_cosh, _fc64, GxB_FC64_t)
                    default: ;
                }
                break ;

            case GB_TANH_unop_code :    // z = tanh (x)

                switch (code1)
                {
                    GB_CASE_FP32 (_tanh, _fp32, float     )
                    GB_CASE_FP64 (_tanh, _fp64, double    )
                    GB_CASE_FC32 (_tanh, _fc32, GxB_FC32_t)
                    GB_CASE_FC64 (_tanh, _fc64, GxB_FC64_t)
                    default: ;
                }
                break ;


            case GB_ASINH_unop_code :   // z = asinh (x)

                switch (code1)
                {
                    GB_CASE_FP32 (_asinh, _fp32, float     )
                    GB_CASE_FP64 (_asinh, _fp64, double    )
                    GB_CASE_FC32 (_asinh, _fc32, GxB_FC32_t)
                    GB_CASE_FC64 (_asinh, _fc64, GxB_FC64_t)
                    default: ;
                }
                break ;

            case GB_ACOSH_unop_code :   // z = acosh (x)

                switch (code1)
                {
                    GB_CASE_FP32 (_acosh, _fp32, float     )
                    GB_CASE_FP64 (_acosh, _fp64, double    )
                    GB_CASE_FC32 (_acosh, _fc32, GxB_FC32_t)
                    GB_CASE_FC64 (_acosh, _fc64, GxB_FC64_t)
                    default: ;
                }
                break ;

            case GB_ATANH_unop_code :   // z = atanh (x)

                switch (code1)
                {
                    GB_CASE_FP32 (_atanh, _fp32, float     )
                    GB_CASE_FP64 (_atanh, _fp64, double    )
                    GB_CASE_FC32 (_atanh, _fc32, GxB_FC32_t)
                    GB_CASE_FC64 (_atanh, _fc64, GxB_FC64_t)
                    default: ;
                }
                break ;

            case GB_SIGNUM_unop_code :  // z = signum (x)

                switch (code1)
                {
                    GB_CASE_FP32 (_signum, _fp32, float     )
                    GB_CASE_FP64 (_signum, _fp64, double    )
                    GB_CASE_FC32 (_signum, _fc32, GxB_FC32_t)
                    GB_CASE_FC64 (_signum, _fc64, GxB_FC64_t)
                    default: ;
                }
                break ;

            case GB_CEIL_unop_code :    // z = ceil (x)

                switch (code1)
                {
                    GB_CASE_FP32 (_ceil, _fp32, float     )
                    GB_CASE_FP64 (_ceil, _fp64, double    )
                    GB_CASE_FC32 (_ceil, _fc32, GxB_FC32_t)
                    GB_CASE_FC64 (_ceil, _fc64, GxB_FC64_t)
                    default: ;
                }
                break ;

            case GB_FLOOR_unop_code :   // z = floor (x)

                switch (code1)
                {
                    GB_CASE_FP32 (_floor, _fp32, float     )
                    GB_CASE_FP64 (_floor, _fp64, double    )
                    GB_CASE_FC32 (_floor, _fc32, GxB_FC32_t)
                    GB_CASE_FC64 (_floor, _fc64, GxB_FC64_t)
                    default: ;
                }
                break ;

            case GB_ROUND_unop_code :   // z = round (x)

                switch (code1)
                {
                    GB_CASE_FP32 (_round, _fp32, float     )
                    GB_CASE_FP64 (_round, _fp64, double    )
                    GB_CASE_FC32 (_round, _fc32, GxB_FC32_t)
                    GB_CASE_FC64 (_round, _fc64, GxB_FC64_t)
                    default: ;
                }
                break ;

            case GB_TRUNC_unop_code :   // z = trunc (x)

                switch (code1)
                {
                    GB_CASE_FP32 (_trunc, _fp32, float     )
                    GB_CASE_FP64 (_trunc, _fp64, double    )
                    GB_CASE_FC32 (_trunc, _fc32, GxB_FC32_t)
                    GB_CASE_FC64 (_trunc, _fc64, GxB_FC64_t)
                    default: ;
                }
                break ;


            case GB_EXP2_unop_code :    // z = exp2 (x)

                switch (code1)
                {
                    GB_CASE_FP32 (_exp2, _fp32, float     )
                    GB_CASE_FP64 (_exp2, _fp64, double    )
                    GB_CASE_FC32 (_exp2, _fc32, GxB_FC32_t)
                    GB_CASE_FC64 (_exp2, _fc64, GxB_FC64_t)
                    default: ;
                }
                break ;

            case GB_EXPM1_unop_code :   // z = expm1 (x)

                switch (code1)
                {
                    GB_CASE_FP32 (_expm1, _fp32, float     )
                    GB_CASE_FP64 (_expm1, _fp64, double    )
                    GB_CASE_FC32 (_expm1, _fc32, GxB_FC32_t)
                    GB_CASE_FC64 (_expm1, _fc64, GxB_FC64_t)
                    default: ;
                }
                break ;

            case GB_LOG10_unop_code :   // z = log10 (x)

                switch (code1)
                {
                    GB_CASE_FP32 (_log10, _fp32, float     )
                    GB_CASE_FP64 (_log10, _fp64, double    )
                    GB_CASE_FC32 (_log10, _fc32, GxB_FC32_t)
                    GB_CASE_FC64 (_log10, _fc64, GxB_FC64_t)
                    default: ;
                }
                break ;

            case GB_LOG1P_unop_code :   // z = log1P (x)

                switch (code1)
                {
                    GB_CASE_FP32 (_log1p, _fp32, float     )
                    GB_CASE_FP64 (_log1p, _fp64, double    )
                    GB_CASE_FC32 (_log1p, _fc32, GxB_FC32_t)
                    GB_CASE_FC64 (_log1p, _fc64, GxB_FC64_t)
                    default: ;
                }
                break ;

            case GB_LOG2_unop_code :    // z = log2 (x)

                switch (code1)
                {
                    GB_CASE_FP32 (_log2, _fp32, float     )
                    GB_CASE_FP64 (_log2, _fp64, double    )
                    GB_CASE_FC32 (_log2, _fc32, GxB_FC32_t)
                    GB_CASE_FC64 (_log2, _fc64, GxB_FC64_t)
                    default: ;
                }
                break ;

            case GB_LGAMMA_unop_code :  // z = lgamma (x)

                switch (code1)
                {
                    GB_CASE_FP32 (_lgamma, _fp32, float )
                    GB_CASE_FP64 (_lgamma, _fp64, double)
                    default: ;
                }
                break ;

            case GB_TGAMMA_unop_code :  // z = tgamma (x)

                switch (code1)
                {
                    GB_CASE_FP32 (_tgamma, _fp32, float )
                    GB_CASE_FP64 (_tgamma, _fp64, double)
                    default: ;
                }
                break ;

            case GB_ERF_unop_code :     // z = erf (x)

                switch (code1)
                {
                    GB_CASE_FP32 (_erf, _fp32, float )
                    GB_CASE_FP64 (_erf, _fp64, double)
                    default: ;
                }
                break ;

            case GB_ERFC_unop_code :    // z = erfc (x)

                switch (code1)
                {
                    GB_CASE_FP32 (_erfc, _fp32, float )
                    GB_CASE_FP64 (_erfc, _fp64, double)
                    default: ;
                }
                break ;

            case GB_CBRT_unop_code :    // z = cbrt (x)

                switch (code1)
                {
                    GB_CASE_FP32 (_cbrt, _fp32, float )
                    GB_CASE_FP64 (_cbrt, _fp64, double)
                    default: ;
                }
                break ;

            case GB_FREXPX_unop_code :  // z = frexpx (x), mantissa of frexp

                switch (code1)
                {
                    GB_CASE_FP32 (_frexpx, _fp32, float )
                    GB_CASE_FP64 (_frexpx, _fp64, double)
                    default: ;
                }
                break ;

            case GB_FREXPE_unop_code :  // z = frexpe (x), exponent of frexp

                switch (code1)
                {
                    GB_CASE_FP32 (_frexpe, _fp32, float )
                    GB_CASE_FP64 (_frexpe, _fp64, double)
                    default: ;
                }
                break ;

            case GB_CONJ_unop_code :    // z = conj (x)

                switch (code1)
                {
                    GB_CASE_FC32 (_conj, _fc32, GxB_FC32_t)
                    GB_CASE_FC64 (_conj, _fc64, GxB_FC64_t)
                    default: ;
                }
                break ;

            default: ;
        }
    }
}

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
#undef GB_CASE_FC32
#undef GB_CASE_FC64

