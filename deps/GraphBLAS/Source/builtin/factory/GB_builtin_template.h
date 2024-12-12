//------------------------------------------------------------------------------
// GB_builtin_template.h: define the unary and binary functions and operators
//------------------------------------------------------------------------------

// SuiteSparse:GraphBLAS, Timothy A. Davis, (c) 2017-2023, All Rights Reserved.
// SPDX-License-Identifier: Apache-2.0

//------------------------------------------------------------------------------

// This file is #include'd many times in GB_builtin.h to define the unary and
// binary functions.

#define GB_UNOP_STRUCT(op,xtype) \
    GB_GLOBAL struct GB_UnaryOp_opaque GB_OPAQUE (GB_EVAL3 (op, _, xtype))

#define GB_BINOP_STRUCT(op,xtype) \
    GB_GLOBAL struct GB_BinaryOp_opaque GB_OPAQUE (GB_EVAL3 (op, _, xtype))

#define GB_IDXOP_STRUCT(op,xtype) \
    GB_GLOBAL struct GB_IndexUnaryOp_opaque \
        GB_OPAQUE (GB_EVAL3 (op, _, xtype))

//------------------------------------------------------------------------------
// z = one (x)
//------------------------------------------------------------------------------

GB_UNOP_STRUCT (ONE,GB_XTYPE) ;
inline void GB_FUNC (ONE) (GB_TYPE *z, const GB_TYPE *x)
{
    #if defined ( GB_FLOAT_COMPLEX )
    (*z) = GxB_CMPLXF (1,0) ;
    #elif defined ( GB_DOUBLE_COMPLEX )
    (*z) = GxB_CMPLX (1,0) ;
    #else
    (*z) = ((GB_TYPE) 1) ;
    #endif
}

//------------------------------------------------------------------------------
// z = identity (x)
//------------------------------------------------------------------------------

GB_UNOP_STRUCT (IDENTITY, GB_XTYPE) ;
inline void GB_FUNC (IDENTITY) (GB_TYPE *z, const GB_TYPE *x)
{
    (*z) = (*x) ;
}

//------------------------------------------------------------------------------
// z = ainv (x)
//------------------------------------------------------------------------------

GB_UNOP_STRUCT (AINV, GB_XTYPE) ;
inline void GB_FUNC (AINV) (GB_TYPE *z, const GB_TYPE *x)
{
    #if defined ( GB_FLOAT_COMPLEX )
        (*z) = GB_FC32_ainv (*x) ;
    #elif defined ( GB_DOUBLE_COMPLEX )
        (*z) = GB_FC64_ainv (*x) ;
    #elif defined ( GB_BOOLEAN )
        (*z) = (*x) ;
    #else
        // integer (signed or unsigned).  unsigned int remains unsigned.
        (*z) = -(*x) ;
    #endif
}

//------------------------------------------------------------------------------
// z = minv (x)
//------------------------------------------------------------------------------

GB_UNOP_STRUCT (MINV, GB_XTYPE) ;
inline void GB_FUNC (MINV) (GB_TYPE *z, const GB_TYPE *x)
{
    #if defined ( GB_BOOLEAN )
        (*z) = true ;
    #elif defined ( GB_SIGNED_INT )
        #if ( GB_X_NBITS == 8)
            (*z) = GB_idiv_int8 (1, (*x)) ;
        #elif ( GB_X_NBITS == 16)
            (*z) = GB_idiv_int16 (1, (*x)) ;
        #elif ( GB_X_NBITS == 32)
            (*z) = GB_idiv_int32 (1, (*x)) ;
        #elif ( GB_X_NBITS == 64)
            (*z) = GB_idiv_int64 (1, (*x)) ;
        #endif
    #elif defined ( GB_UNSIGNED_INT )
        #if ( GB_X_NBITS == 8)
            (*z) = GB_idiv_uint8 (1, (*x)) ;
        #elif ( GB_X_NBITS == 16)
            (*z) = GB_idiv_uint16 (1, (*x)) ;
        #elif ( GB_X_NBITS == 32)
            (*z) = GB_idiv_uint32 (1, (*x)) ;
        #elif ( GB_X_NBITS == 64)
            (*z) = GB_idiv_uint64 (1, (*x)) ;
        #endif
    #elif defined ( GB_FLOAT )
        (*z) = 1 / (*x) ;
    #elif defined ( GB_DOUBLE )
        (*z) = 1 / (*x) ;
    #elif defined ( GB_FLOAT_COMPLEX )
        (*z) = GB_FC32_div (GxB_CMPLXF (1,0), *x) ;
    #elif defined ( GB_DOUBLE_COMPLEX )
        (*z) = GB_FC64_div (GxB_CMPLX  (1,0), *x) ;
    #endif
}

//------------------------------------------------------------------------------
// z = abs (x)
//------------------------------------------------------------------------------

GB_UNOP_STRUCT (ABS, GB_XTYPE) ;

#if defined ( GB_REAL )

    // GrB_ABS_* for non-complex types
    inline void GB_FUNC (ABS) (GB_TYPE *z, const GB_TYPE *x)
    {
        #if defined ( GB_BOOLEAN )
            (*z) = (*x) ;
        #elif defined ( GB_SIGNED_INT )
            (*z) = GB_IABS ((*x)) ;
        #elif defined ( GB_UNSIGNED_INT )
            (*z) = (*x) ;
        #elif defined ( GB_FLOAT )
            (*z) = fabsf (*x) ;
        #elif defined ( GB_DOUBLE )
            (*z) = fabs (*x) ;
        #endif
    }

#else

    // GxB_ABS_FC* for complex types
    #if defined ( GB_FLOAT_COMPLEX )
        inline void GB_FUNC (ABS) (float *z, const GB_TYPE *x)
        {
            (*z) = GB_cabsf (*x) ;
        }
    #else
        inline void GB_FUNC (ABS) (double *z, const GB_TYPE *x)
        {
            (*z) = GB_cabs (*x) ;
        }
    #endif

#endif

//------------------------------------------------------------------------------
// z = lnot (x), for real types only
//------------------------------------------------------------------------------

#if defined ( GB_REAL )

    GB_UNOP_STRUCT (LNOT, GB_XTYPE) ;
    inline void GB_FUNC (LNOT) (GB_TYPE *z, const GB_TYPE *x)
    {
        #if defined ( GB_BOOLEAN )
            (*z) = ! (*x) ;
        #else
            (*z) = ! ((*x) != 0) ;
        #endif
    }

#endif

//------------------------------------------------------------------------------
// z = bnot (x), bitwise complement, for integer types only
//------------------------------------------------------------------------------

#if defined ( GB_SIGNED_INT ) || defined ( GB_UNSIGNED_INT )

    GB_UNOP_STRUCT (BNOT, GB_XTYPE) ;
    inline void GB_FUNC (BNOT) (GB_TYPE *z, const GB_TYPE *x)
    {
        (*z) = ~ (*x) ;
    }

#endif

//------------------------------------------------------------------------------
// z = frexpx (x) and z = frexpe (x)
//------------------------------------------------------------------------------

#if defined ( GB_FLOAT )

    GB_UNOP_STRUCT (FREXPX, GB_XTYPE) ;
    inline void GB_FUNC (FREXPX) (float *z, const float *x)
    {
        (*z) = GB_frexpxf (*x) ;
    }

    GB_UNOP_STRUCT (FREXPE, GB_XTYPE) ;
    inline void GB_FUNC (FREXPE) (float *z, const float *x)
    {
        (*z) = GB_frexpef (*x) ;
    }

#elif defined ( GB_DOUBLE )

    GB_UNOP_STRUCT (FREXPX, GB_XTYPE) ;
    inline void GB_FUNC (FREXPX) (double *z, const double *x)
    {
        (*z) = GB_frexpx (*x) ;
    }

    GB_UNOP_STRUCT (FREXPE, GB_XTYPE) ;
    inline void GB_FUNC (FREXPE) (double *z, const double *x)
    {
        (*z) = GB_frexpe (*x) ;
    }

#endif

//------------------------------------------------------------------------------
// unary operators for floating-point types
//------------------------------------------------------------------------------

// For these operators, the input and output types are the same.

#undef  GB_UNOP_DEFINE
#define GB_UNOP_DEFINE(op,func)                                 \
    GB_UNOP_STRUCT (op, GB_XTYPE) ;                             \
    inline void GB_FUNC (op) (GB_TYPE *z, const GB_TYPE *x)     \
    {                                                           \
        (*z) = func (*x) ;                                      \
    }

#if defined ( GB_FLOAT )

    //--------------------------------------------------------------------------
    // float
    //--------------------------------------------------------------------------

    GB_UNOP_DEFINE (SQRT  , sqrtf   )
    GB_UNOP_DEFINE (LOG   , logf    )
    GB_UNOP_DEFINE (EXP   , expf    )

    GB_UNOP_DEFINE (SIN   , sinf    )
    GB_UNOP_DEFINE (COS   , cosf    )
    GB_UNOP_DEFINE (TAN   , tanf    )

    GB_UNOP_DEFINE (ASIN  , asinf   )
    GB_UNOP_DEFINE (ACOS  , acosf   )
    GB_UNOP_DEFINE (ATAN  , atanf   )

    GB_UNOP_DEFINE (SINH  , sinhf   )
    GB_UNOP_DEFINE (COSH  , coshf   )
    GB_UNOP_DEFINE (TANH  , tanhf   )

    GB_UNOP_DEFINE (ASINH , asinhf  )
    GB_UNOP_DEFINE (ACOSH , acoshf  )
    GB_UNOP_DEFINE (ATANH , atanhf  )

    GB_UNOP_DEFINE (SIGNUM, GB_signumf )
    GB_UNOP_DEFINE (CEIL  , ceilf   )
    GB_UNOP_DEFINE (FLOOR , floorf  )
    GB_UNOP_DEFINE (ROUND , roundf  )
    GB_UNOP_DEFINE (TRUNC , truncf  )

    GB_UNOP_DEFINE (EXP2  , exp2f   )
    GB_UNOP_DEFINE (EXPM1 , expm1f  )
    GB_UNOP_DEFINE (LOG10 , log10f  )
    GB_UNOP_DEFINE (LOG1P , log1pf  )
    GB_UNOP_DEFINE (LOG2  , log2f   )

    // real only
    GB_UNOP_DEFINE (LGAMMA, lgammaf )
    GB_UNOP_DEFINE (TGAMMA, tgammaf )
    GB_UNOP_DEFINE (ERF   , erff    )
    GB_UNOP_DEFINE (ERFC  , erfcf   )
    GB_UNOP_DEFINE (CBRT  , cbrtf   )

#elif defined ( GB_DOUBLE )

    //--------------------------------------------------------------------------
    // double
    //--------------------------------------------------------------------------

    GB_UNOP_DEFINE (SQRT  , sqrt    )
    GB_UNOP_DEFINE (LOG   , log     )
    GB_UNOP_DEFINE (EXP   , exp     )

    GB_UNOP_DEFINE (SIN   , sin     )
    GB_UNOP_DEFINE (COS   , cos     )
    GB_UNOP_DEFINE (TAN   , tan     )

    GB_UNOP_DEFINE (ASIN  , asin    )
    GB_UNOP_DEFINE (ACOS  , acos    )
    GB_UNOP_DEFINE (ATAN  , atan    )

    GB_UNOP_DEFINE (SINH  , sinh    )
    GB_UNOP_DEFINE (COSH  , cosh    )
    GB_UNOP_DEFINE (TANH  , tanh    )

    GB_UNOP_DEFINE (ASINH , asinh   )
    GB_UNOP_DEFINE (ACOSH , acosh   )
    GB_UNOP_DEFINE (ATANH , atanh   )

    GB_UNOP_DEFINE (SIGNUM, GB_signum )
    GB_UNOP_DEFINE (CEIL  , ceil    )
    GB_UNOP_DEFINE (FLOOR , floor   )
    GB_UNOP_DEFINE (ROUND , round   )
    GB_UNOP_DEFINE (TRUNC , trunc   )

    GB_UNOP_DEFINE (EXP2  , exp2    )
    GB_UNOP_DEFINE (EXPM1 , expm1   )
    GB_UNOP_DEFINE (LOG10 , log10   )
    GB_UNOP_DEFINE (LOG1P , log1p   )
    GB_UNOP_DEFINE (LOG2  , log2    )

    // real only
    GB_UNOP_DEFINE (LGAMMA, lgamma )
    GB_UNOP_DEFINE (TGAMMA, tgamma )
    GB_UNOP_DEFINE (ERF   , erf    )
    GB_UNOP_DEFINE (ERFC  , erfc   )
    GB_UNOP_DEFINE (CBRT  , cbrt   )

#elif defined ( GB_FLOAT_COMPLEX )

    //--------------------------------------------------------------------------
    // float complex
    //--------------------------------------------------------------------------

    GB_UNOP_DEFINE (SQRT  , GB_csqrtf   )
    GB_UNOP_DEFINE (LOG   , GB_clogf    )
    GB_UNOP_DEFINE (EXP   , GB_cexpf    )

    GB_UNOP_DEFINE (SIN   , GB_csinf    )
    GB_UNOP_DEFINE (COS   , GB_ccosf    )
    GB_UNOP_DEFINE (TAN   , GB_ctanf    )

    GB_UNOP_DEFINE (ASIN  , GB_casinf   )
    GB_UNOP_DEFINE (ACOS  , GB_cacosf   )
    GB_UNOP_DEFINE (ATAN  , GB_catanf   )

    GB_UNOP_DEFINE (SINH  , GB_csinhf   )
    GB_UNOP_DEFINE (COSH  , GB_ccoshf   )
    GB_UNOP_DEFINE (TANH  , GB_ctanhf   )

    GB_UNOP_DEFINE (ASINH , GB_casinhf  )
    GB_UNOP_DEFINE (ACOSH , GB_cacoshf  )
    GB_UNOP_DEFINE (ATANH , GB_catanhf  )

    GB_UNOP_DEFINE (SIGNUM, GB_csignumf )
    GB_UNOP_DEFINE (CEIL  , GB_cceilf   )
    GB_UNOP_DEFINE (FLOOR , GB_cfloorf  )
    GB_UNOP_DEFINE (ROUND , GB_croundf  )
    GB_UNOP_DEFINE (TRUNC , GB_ctruncf  )

    GB_UNOP_DEFINE (EXP2  , GB_cexp2f   )
    GB_UNOP_DEFINE (EXPM1 , GB_cexpm1f  )
    GB_UNOP_DEFINE (LOG10 , GB_clog10f  )
    GB_UNOP_DEFINE (LOG1P , GB_clog1pf  )
    GB_UNOP_DEFINE (LOG2  , GB_clog2f   )

    GB_UNOP_DEFINE (CONJ  , GB_conjf    )

#elif defined ( GB_DOUBLE_COMPLEX )

    //--------------------------------------------------------------------------
    // double complex
    //--------------------------------------------------------------------------

    GB_UNOP_DEFINE (SQRT  , GB_csqrt    )
    GB_UNOP_DEFINE (LOG   , GB_clog     )
    GB_UNOP_DEFINE (EXP   , GB_cexp     )

    GB_UNOP_DEFINE (SIN   , GB_csin     )
    GB_UNOP_DEFINE (COS   , GB_ccos     )
    GB_UNOP_DEFINE (TAN   , GB_ctan     )

    GB_UNOP_DEFINE (ASIN  , GB_casin    )
    GB_UNOP_DEFINE (ACOS  , GB_cacos    )
    GB_UNOP_DEFINE (ATAN  , GB_catan    )

    GB_UNOP_DEFINE (SINH  , GB_csinh    )
    GB_UNOP_DEFINE (COSH  , GB_ccosh    )
    GB_UNOP_DEFINE (TANH  , GB_ctanh    )

    GB_UNOP_DEFINE (ASINH , GB_casinh   )
    GB_UNOP_DEFINE (ACOSH , GB_cacosh   )
    GB_UNOP_DEFINE (ATANH , GB_catanh   )

    GB_UNOP_DEFINE (SIGNUM, GB_csignum  )
    GB_UNOP_DEFINE (CEIL  , GB_cceil    )
    GB_UNOP_DEFINE (FLOOR , GB_cfloor   )
    GB_UNOP_DEFINE (ROUND , GB_cround   )
    GB_UNOP_DEFINE (TRUNC , GB_ctrunc   )

    GB_UNOP_DEFINE (EXP2  , GB_cexp2    )
    GB_UNOP_DEFINE (EXPM1 , GB_cexpm1   )
    GB_UNOP_DEFINE (LOG10 , GB_clog10   )
    GB_UNOP_DEFINE (LOG1P , GB_clog1p   )
    GB_UNOP_DEFINE (LOG2  , GB_clog2    )

    GB_UNOP_DEFINE (CONJ  , GB_conj     )

#endif

//------------------------------------------------------------------------------
// unary operators z=f(x) where z and x have different types
//------------------------------------------------------------------------------

// x is float, double, float complex, or double complex

#undef  GB_UNOP_DEFINE
#define GB_UNOP_DEFINE(op,expression,z_t,x_t)               \
    GB_UNOP_STRUCT(op, GB_XTYPE) ;                          \
    inline void GB_FUNC (op) (z_t *z, const x_t *x)         \
    {                                                       \
        (*z) = expression ;                                 \
    }

#if defined ( GB_FLOAT )

    GB_UNOP_DEFINE (ISINF    , (isinf (*x))    , bool, float)
    GB_UNOP_DEFINE (ISNAN    , (isnan (*x))    , bool, float)
    GB_UNOP_DEFINE (ISFINITE , (isfinite (*x)) , bool, float)

#elif defined ( GB_DOUBLE )

    GB_UNOP_DEFINE (ISINF    , (isinf (*x))    , bool, double)
    GB_UNOP_DEFINE (ISNAN    , (isnan (*x))    , bool, double)
    GB_UNOP_DEFINE (ISFINITE , (isfinite (*x)) , bool, double)

#elif defined ( GB_FLOAT_COMPLEX )

    GB_UNOP_DEFINE (ISINF    , GB_cisinff (*x)   , bool, GxB_FC32_t)
    GB_UNOP_DEFINE (ISNAN    , GB_cisnanf (*x)   , bool, GxB_FC32_t)
    GB_UNOP_DEFINE (ISFINITE , GB_cisfinitef (*x), bool, GxB_FC32_t)

    // complex only
    GB_UNOP_DEFINE (CREAL , GB_crealf (*x), float, GxB_FC32_t)
    GB_UNOP_DEFINE (CIMAG , GB_cimagf (*x), float, GxB_FC32_t)
    GB_UNOP_DEFINE (CARG  , GB_cargf  (*x), float, GxB_FC32_t)

#elif defined ( GB_DOUBLE_COMPLEX )

    GB_UNOP_DEFINE (ISINF    , GB_cisinf (*x)    , bool, GxB_FC64_t)
    GB_UNOP_DEFINE (ISNAN    , GB_cisnan  (*x)   , bool, GxB_FC64_t)
    GB_UNOP_DEFINE (ISFINITE , GB_cisfinite (*x) , bool, GxB_FC64_t)

    // complex only
    GB_UNOP_DEFINE (CREAL , GB_creal (*x), double, GxB_FC64_t)
    GB_UNOP_DEFINE (CIMAG , GB_cimag (*x), double, GxB_FC64_t)
    GB_UNOP_DEFINE (CARG  , GB_carg  (*x), double, GxB_FC64_t)

#endif

//------------------------------------------------------------------------------
// binary functions z=f(x,y) where x,y,z have the same type, for all types
//------------------------------------------------------------------------------

// first, second, pair, any, plus, minus, rminus, times, div, rdiv, pow

#define GB_Z_X_Y_ARGS GB_TYPE *z, const GB_TYPE *x, const GB_TYPE *y

GB_BINOP_STRUCT (FIRST, GB_XTYPE) ;
inline void GB_FUNC (FIRST) (GB_Z_X_Y_ARGS)
{
    (*z) = (*x) ;
}

GB_BINOP_STRUCT (SECOND, GB_XTYPE) ;
inline void GB_FUNC (SECOND) (GB_Z_X_Y_ARGS)
{
    (*z) = (*y) ;
}

GB_BINOP_STRUCT (PAIR, GB_XTYPE) ;
inline void GB_FUNC (PAIR) (GB_Z_X_Y_ARGS)
{
    #if defined ( GB_FLOAT_COMPLEX )
        (*z) = GxB_CMPLXF (1, 0) ;
    #elif defined ( GB_DOUBLE_COMPLEX )
        (*z) = GxB_CMPLX (1, 0) ;
    #else
        (*z) = 1 ;
    #endif
}

GB_BINOP_STRUCT (ANY, GB_XTYPE) ;
inline void GB_FUNC (ANY) (GB_Z_X_Y_ARGS)      // same as SECOND
{
    (*z) = (*y) ; 
}

GB_BINOP_STRUCT (PLUS, GB_XTYPE) ;
inline void GB_FUNC (PLUS) (GB_Z_X_Y_ARGS)
{
    #if defined ( GB_FLOAT_COMPLEX )
        (*z) = GB_FC32_add (*x,*y) ;
    #elif defined ( GB_DOUBLE_COMPLEX )
        (*z) = GB_FC64_add (*x,*y) ;
    #else
        (*z) = (*x) + (*y) ;
    #endif
}

GB_BINOP_STRUCT (MINUS, GB_XTYPE) ;
inline void GB_FUNC (MINUS) (GB_Z_X_Y_ARGS)
{
    #if defined ( GB_FLOAT_COMPLEX )
        (*z) = GB_FC32_minus (*x,*y) ;
    #elif defined ( GB_DOUBLE_COMPLEX )
        (*z) = GB_FC64_minus (*x,*y) ;
    #else
        (*z) = (*x) - (*y) ;
    #endif
}

GB_BINOP_STRUCT (RMINUS, GB_XTYPE) ;
inline void GB_FUNC (RMINUS) (GB_Z_X_Y_ARGS)
{
    #if defined ( GB_FLOAT_COMPLEX )
        (*z) = GB_FC32_minus (*y,*x) ;
    #elif defined ( GB_DOUBLE_COMPLEX )
        (*z) = GB_FC64_minus (*y,*x) ;
    #else
        (*z) = (*y) - (*x) ;
    #endif
}

GB_BINOP_STRUCT (TIMES, GB_XTYPE) ;
inline void GB_FUNC (TIMES) (GB_Z_X_Y_ARGS)
{
    #if defined ( GB_FLOAT_COMPLEX )
        (*z) = GB_FC32_mul (*x,*y) ;
    #elif defined ( GB_DOUBLE_COMPLEX )
        (*z) = GB_FC64_mul (*x,*y) ;
    #else
        (*z) = (*x) * (*y) ;
    #endif
}

GB_BINOP_STRUCT (DIV, GB_XTYPE) ;
inline void GB_FUNC (DIV) (GB_Z_X_Y_ARGS)
{
    #if defined ( GB_BOOLEAN )
        // boolean div (== first)
        (*z) = (*x) ;
    #elif defined ( GB_SIGNED_INT )
        #if ( GB_X_NBITS == 8)
            (*z) = GB_idiv_int8 ((*x), (*y)) ;
        #elif ( GB_X_NBITS == 16)
            (*z) = GB_idiv_int16 ((*x), (*y)) ;
        #elif ( GB_X_NBITS == 32)
            (*z) = GB_idiv_int32 ((*x), (*y)) ;
        #elif ( GB_X_NBITS == 64)
            (*z) = GB_idiv_int64 ((*x), (*y)) ;
        #endif
    #elif defined ( GB_UNSIGNED_INT )
        #if ( GB_X_NBITS == 8)
            (*z) = GB_idiv_uint8 ((*x), (*y)) ;
        #elif ( GB_X_NBITS == 16)
            (*z) = GB_idiv_uint16 ((*x), (*y)) ;
        #elif ( GB_X_NBITS == 32)
            (*z) = GB_idiv_uint32 ((*x), (*y)) ;
        #elif ( GB_X_NBITS == 64)
            (*z) = GB_idiv_uint64 ((*x), (*y)) ;
        #endif
    #elif defined ( GB_FLOAT ) || defined ( GB_DOUBLE )
        (*z) = (*x) / (*y) ;
    #elif defined ( GB_FLOAT_COMPLEX )
        (*z) = GB_FC32_div (*x, *y) ;
    #elif defined ( GB_DOUBLE_COMPLEX )
        (*z) = GB_FC64_div (*x, *y) ;
    #endif
}

GB_BINOP_STRUCT (RDIV, GB_XTYPE) ;
inline void GB_FUNC (RDIV) (GB_Z_X_Y_ARGS)
{
    #if defined ( GB_BOOLEAN )
        // boolean rdiv (== second)
        (*z) = (*y) ;
    #elif defined ( GB_SIGNED_INT )
        #if ( GB_X_NBITS == 8)
            (*z) = GB_idiv_int8 ((*y), (*x)) ;
        #elif ( GB_X_NBITS == 16)
            (*z) = GB_idiv_int16 ((*y), (*x)) ;
        #elif ( GB_X_NBITS == 32)
            (*z) = GB_idiv_int32 ((*y), (*x)) ;
        #elif ( GB_X_NBITS == 64)
            (*z) = GB_idiv_int64 ((*y), (*x)) ;
        #endif
    #elif defined ( GB_UNSIGNED_INT )
        #if ( GB_X_NBITS == 8)
            (*z) = GB_idiv_uint8 ((*y), (*x)) ;
        #elif ( GB_X_NBITS == 16)
            (*z) = GB_idiv_uint16 ((*y), (*x)) ;
        #elif ( GB_X_NBITS == 32)
            (*z) = GB_idiv_uint32 ((*y), (*x)) ;
        #elif ( GB_X_NBITS == 64)
            (*z) = GB_idiv_uint64 ((*y), (*x)) ;
        #endif
    #elif defined ( GB_FLOAT ) || defined ( GB_DOUBLE )
        (*z) = (*y) / (*x) ;
    #elif defined ( GB_FLOAT_COMPLEX )
        (*z) = GB_FC32_div (*y, *x) ;
    #elif defined ( GB_DOUBLE_COMPLEX )
        (*z) = GB_FC64_div (*y, *x) ;
    #endif
}

// z = pow (x,y)
GB_BINOP_STRUCT (POW, GB_XTYPE) ;
inline void GB_FUNC (POW) (GB_Z_X_Y_ARGS)
{
    #if defined ( GB_BOOLEAN )
        (*z) = (*x) || (!(*y)) ;
    #elif defined ( GB_SIGNED_INT )
        #if ( GB_X_NBITS == 8)
            (*z) = GB_pow_int8 ((*x), (*y)) ;
        #elif ( GB_X_NBITS == 16)
            (*z) = GB_pow_int16 ((*x), (*y)) ;
        #elif ( GB_X_NBITS == 32)
            (*z) = GB_pow_int32 ((*x), (*y)) ;
        #elif ( GB_X_NBITS == 64)
            (*z) = GB_pow_int64 ((*x), (*y)) ;
        #endif
    #elif defined ( GB_UNSIGNED_INT )
        #if ( GB_X_NBITS == 8)
            (*z) = GB_pow_uint8 ((*x), (*y)) ;
        #elif ( GB_X_NBITS == 16)
            (*z) = GB_pow_uint16 ((*x), (*y)) ;
        #elif ( GB_X_NBITS == 32)
            (*z) = GB_pow_uint32 ((*x), (*y)) ;
        #elif ( GB_X_NBITS == 64)
            (*z) = GB_pow_uint64 ((*x), (*y)) ;
        #endif
    #elif defined ( GB_FLOAT )
        (*z) = GB_powf ((*x), (*y)) ;
    #elif defined ( GB_DOUBLE )
        (*z) = GB_pow ((*x), (*y)) ;
    #elif defined ( GB_FLOAT_COMPLEX )
        (*z) = GB_FC32_pow ((*x), (*y)) ;
    #elif defined ( GB_DOUBLE_COMPLEX )
        (*z) = GB_FC64_pow ((*x), (*y)) ;
    #endif
}

//------------------------------------------------------------------------------
// binary operators for real types only
//------------------------------------------------------------------------------

// min and max: real only, not complex
#if defined ( GB_REAL )

    GB_BINOP_STRUCT (MIN, GB_XTYPE) ;
    inline void GB_FUNC (MIN) (GB_Z_X_Y_ARGS)
    {
        #if defined ( GB_FLOAT )
            (*z) = fminf ((*x), (*y)) ;
        #elif defined ( GB_DOUBLE )
            (*z) = fmin ((*x), (*y)) ;
        #else
            (*z) = GB_IMIN ((*x), (*y)) ;
        #endif
    }

    GB_BINOP_STRUCT (MAX, GB_XTYPE) ;
    inline void GB_FUNC (MAX) (GB_Z_X_Y_ARGS)
    {
        #if defined ( GB_FLOAT )
            (*z) = fmaxf ((*x), (*y)) ;
        #elif defined ( GB_DOUBLE )
            (*z) = fmax ((*x), (*y)) ;
        #else
            (*z) = GB_IMAX ((*x), (*y)) ;
        #endif
    }

#endif

//------------------------------------------------------------------------------
// binary operators for integer types only
//------------------------------------------------------------------------------

#if defined ( GB_SIGNED_INT ) || defined ( GB_UNSIGNED_INT )

    GB_BINOP_STRUCT (BOR, GB_XTYPE) ;
    inline void GB_FUNC (BOR  ) (GB_Z_X_Y_ARGS) { (*z) = (*x) | (*y) ; }

    GB_BINOP_STRUCT (BAND, GB_XTYPE) ;
    inline void GB_FUNC (BAND ) (GB_Z_X_Y_ARGS) { (*z) = (*x) & (*y) ; }

    GB_BINOP_STRUCT (BXOR, GB_XTYPE) ;
    inline void GB_FUNC (BXOR ) (GB_Z_X_Y_ARGS) { (*z) = (*x) ^ (*y) ; }

    GB_BINOP_STRUCT (BXNOR, GB_XTYPE) ;
    inline void GB_FUNC (BXNOR) (GB_Z_X_Y_ARGS) { (*z) = ~((*x) ^ (*y)) ; }

    GB_BINOP_STRUCT (BGET, GB_XTYPE) ;
    inline void GB_FUNC (BGET) (GB_Z_X_Y_ARGS)
    {
        // bitget (x,y) returns a single bit from x, as 0 or 1, whose position
        // is given by y.  y = 1 is the least significant bit, and y =
        // GB_X_NBITS (64 for uint64, for example) is the most significant bit.
        // If y is outside this range, the result is zero.

        #if defined ( GB_SIGNED_INT )

            #if ( GB_X_NBITS == 8)
                (*z) = GB_bitget_int8 ((*x), (*y)) ;
            #elif ( GB_X_NBITS == 16)
                (*z) = GB_bitget_int16 ((*x), (*y)) ;
            #elif ( GB_X_NBITS == 32)
                (*z) = GB_bitget_int32 ((*x), (*y)) ;
            #elif ( GB_X_NBITS == 64)
                (*z) = GB_bitget_int64 ((*x), (*y)) ;
            #endif

        #elif defined ( GB_UNSIGNED_INT )

            #if ( GB_X_NBITS == 8)
                (*z) = GB_bitget_uint8 ((*x), (*y)) ;
            #elif ( GB_X_NBITS == 16)
                (*z) = GB_bitget_uint16 ((*x), (*y)) ;
            #elif ( GB_X_NBITS == 32)
                (*z) = GB_bitget_uint32 ((*x), (*y)) ;
            #elif ( GB_X_NBITS == 64)
                (*z) = GB_bitget_uint64 ((*x), (*y)) ;
            #endif

        #endif
    }

    GB_BINOP_STRUCT (BSET, GB_XTYPE) ;
    inline void GB_FUNC (BSET) (GB_Z_X_Y_ARGS)
    {
        // bitset (x,y) returns x modified by setting a bit from x to 1, whose
        // position is given by y.  If y is in the range 1 to GB_X_NBITS, then
        // y gives the position of the bit to set.  If y is outside the range 1
        // to GB_X_NBITS, then z = x is returned, unmodified.

        #if defined ( GB_SIGNED_INT )

            #if ( GB_X_NBITS == 8)
                (*z) = GB_bitset_int8  ((*x), (*y)) ;
            #elif ( GB_X_NBITS == 16)
                (*z) = GB_bitset_int16 ((*x), (*y)) ;
            #elif ( GB_X_NBITS == 32)
                (*z) = GB_bitset_int32 ((*x), (*y)) ;
            #elif ( GB_X_NBITS == 64)
                (*z) = GB_bitset_int64 ((*x), (*y)) ;
            #endif

        #elif defined ( GB_UNSIGNED_INT )

            #if ( GB_X_NBITS == 8)
                (*z) = GB_bitset_uint8  ((*x), (*y)) ;
            #elif ( GB_X_NBITS == 16)
                (*z) = GB_bitset_uint16 ((*x), (*y)) ;
            #elif ( GB_X_NBITS == 32)
                (*z) = GB_bitset_uint32 ((*x), (*y)) ;
            #elif ( GB_X_NBITS == 64)
                (*z) = GB_bitset_uint64 ((*x), (*y)) ;
            #endif

        #endif
    }

    GB_BINOP_STRUCT (BCLR, GB_XTYPE) ;
    inline void GB_FUNC (BCLR) (GB_Z_X_Y_ARGS)
    {
        // bitclr (x,y) returns x modified by setting a bit from x to 0, whose
        // position is given by y.  If y is in the range 1 to GB_X_NBITS, then
        // y gives the position of the bit to clear.  If y is outside the range
        // 1 to GB_X_NBITS, then z = x is returned, unmodified.

        #if defined ( GB_SIGNED_INT )

            #if ( GB_X_NBITS == 8)
                (*z) = GB_bitclr_int8  ((*x), (*y)) ;
            #elif ( GB_X_NBITS == 16)
                (*z) = GB_bitclr_int16 ((*x), (*y)) ;
            #elif ( GB_X_NBITS == 32)
                (*z) = GB_bitclr_int32 ((*x), (*y)) ;
            #elif ( GB_X_NBITS == 64)
                (*z) = GB_bitclr_int64 ((*x), (*y)) ;
            #endif

        #elif defined ( GB_UNSIGNED_INT )

            #if ( GB_X_NBITS == 8)
                (*z) = GB_bitclr_uint8  ((*x), (*y)) ;
            #elif ( GB_X_NBITS == 16)
                (*z) = GB_bitclr_uint16 ((*x), (*y)) ;
            #elif ( GB_X_NBITS == 32)
                (*z) = GB_bitclr_uint32 ((*x), (*y)) ;
            #elif ( GB_X_NBITS == 64)
                (*z) = GB_bitclr_uint64 ((*x), (*y)) ;
            #endif

        #endif
    }


    // z = bitshift (x,y)
    GB_BINOP_STRUCT (BSHIFT, GB_XTYPE) ;
    inline void GB_FUNC (BSHIFT) (GB_TYPE *z, const GB_TYPE *x, const int8_t *y)
    {
        // bitshift (x,k) shifts x to the left by k bits if k > 0, and the
        // right by -k bits if k < 0.

        #if defined ( GB_SIGNED_INT )

            #if ( GB_X_NBITS == 8)
                (*z) = GB_bitshift_int8 ((*x), (*y)) ;
            #elif ( GB_X_NBITS == 16)
                (*z) = GB_bitshift_int16 ((*x), (*y)) ;
            #elif ( GB_X_NBITS == 32)
                (*z) = GB_bitshift_int32 ((*x), (*y)) ;
            #elif ( GB_X_NBITS == 64)
                (*z) = GB_bitshift_int64 ((*x), (*y)) ;
            #endif

        #elif defined ( GB_UNSIGNED_INT )

            #if ( GB_X_NBITS == 8)
                (*z) = GB_bitshift_uint8 ((*x), (*y)) ;
            #elif ( GB_X_NBITS == 16)
                (*z) = GB_bitshift_uint16 ((*x), (*y)) ;
            #elif ( GB_X_NBITS == 32)
                (*z) = GB_bitshift_uint32 ((*x), (*y)) ;
            #elif ( GB_X_NBITS == 64)
                (*z) = GB_bitshift_uint64 ((*x), (*y)) ;
            #endif

        #endif
    }

#endif

//------------------------------------------------------------------------------
// binary operators for real floating-point inputs only
//------------------------------------------------------------------------------

#if defined ( GB_FLOAT )

    inline void GB_FUNC (ATAN2) (GB_Z_X_Y_ARGS) { (*z) = atan2f ((*x),(*y)) ; }
    inline void GB_FUNC (HYPOT) (GB_Z_X_Y_ARGS) { (*z) = hypotf ((*x),(*y)) ; }
    inline void GB_FUNC (FMOD)  (GB_Z_X_Y_ARGS) { (*z) = fmodf  ((*x),(*y)) ; }

    inline void GB_FUNC (REMAINDER) (GB_Z_X_Y_ARGS)
    {
        (*z) = remainderf ((*x),(*y)) ;
    }
    inline void GB_FUNC (COPYSIGN) (GB_Z_X_Y_ARGS)
    {
        (*z) = copysignf ((*x),(*y)) ;
    }
    inline void GB_FUNC (LDEXP) (GB_Z_X_Y_ARGS)
    {
        (*z) = ldexpf ((*x), (int) truncf (*y)) ;
    }
    inline void GB_FUNC (CMPLX) (GxB_FC32_t *z, const float *x, const float *y)
    {
        #if defined ( __cplusplus ) || defined (GxB_HAVE_COMPLEX_MSVC) || defined (CMPLX)
        (*z) = GxB_CMPLXF ((*x),(*y)) ;
        #else
        ((float *) z) [0] = (*x) ;
        ((float *) z) [1] = (*y) ;
        #endif
    }

#elif defined ( GB_DOUBLE )

    inline void GB_FUNC (ATAN2) (GB_Z_X_Y_ARGS) { (*z) = atan2 ((*x),(*y)) ; }
    inline void GB_FUNC (HYPOT) (GB_Z_X_Y_ARGS) { (*z) = hypot ((*x),(*y)) ; }
    inline void GB_FUNC (FMOD)  (GB_Z_X_Y_ARGS) { (*z) = fmod  ((*x),(*y)) ; }

    inline void GB_FUNC (REMAINDER) (GB_Z_X_Y_ARGS)
    {
        (*z) = remainder ((*x),(*y)) ;
    }
    inline void GB_FUNC (COPYSIGN) (GB_Z_X_Y_ARGS)
    {
        (*z) = copysign ((*x),(*y)) ;
    }
    inline void GB_FUNC (LDEXP) (GB_Z_X_Y_ARGS)
    {
        (*z) = ldexp ((*x), (int) trunc (*y)) ;
    }
    inline void GB_FUNC (CMPLX) (GxB_FC64_t *z,
        const double *x, const double *y)
    {
        #if defined ( __cplusplus ) || defined (GxB_HAVE_COMPLEX_MSVC) || defined (CMPLX)
        (*z) = GxB_CMPLX ((*x),(*y)) ;
        #else
        ((double *) z) [0] = (*x) ;
        ((double *) z) [1] = (*y) ;
        #endif
    }

#endif

#if defined (GB_FLOAT) || defined (GB_DOUBLE)

    GB_BINOP_STRUCT (ATAN2, GB_XTYPE) ;
    GB_BINOP_STRUCT (HYPOT, GB_XTYPE) ;
    GB_BINOP_STRUCT (FMOD, GB_XTYPE) ;
    GB_BINOP_STRUCT (REMAINDER, GB_XTYPE) ;
    GB_BINOP_STRUCT (COPYSIGN, GB_XTYPE) ;
    GB_BINOP_STRUCT (LDEXP, GB_XTYPE) ;
    GB_BINOP_STRUCT (CMPLX, GB_XTYPE) ;

#endif

//------------------------------------------------------------------------------
// 6 binary comparators z=f(x,y), where x,y,z have the same type
//------------------------------------------------------------------------------

// iseq and isne: all 13 types, including complex types.
// isgt, islt, isge, isle: 11 real types only.

GB_BINOP_STRUCT (ISEQ, GB_XTYPE) ;
inline void GB_FUNC (ISEQ) (GB_Z_X_Y_ARGS)
{
    #if defined ( GB_FLOAT_COMPLEX )
    (*z) = GB_FC32_iseq (*x, *y) ;
    #elif defined ( GB_DOUBLE_COMPLEX )
    (*z) = GB_FC64_iseq (*x, *y) ;
    #else
    (*z) = (GB_TYPE) ((*x) == (*y)) ;
    #endif
}

GB_BINOP_STRUCT (ISNE, GB_XTYPE) ;
inline void GB_FUNC (ISNE) (GB_Z_X_Y_ARGS)
{
    #if defined ( GB_FLOAT_COMPLEX )
    (*z) = GB_FC32_isne (*x, *y) ;
    #elif defined ( GB_DOUBLE_COMPLEX )
    (*z) = GB_FC64_isne (*x, *y) ;
    #else
    (*z) = (GB_TYPE) ((*x) != (*y)) ;
    #endif
}

#if defined ( GB_REAL )

    GB_BINOP_STRUCT (ISGT, GB_XTYPE) ;
    inline void GB_FUNC (ISGT) (GB_Z_X_Y_ARGS)
    {
        (*z) = (GB_TYPE) ((*x) >  (*y)) ;
    }

    GB_BINOP_STRUCT (ISLT, GB_XTYPE) ;
    inline void GB_FUNC (ISLT) (GB_Z_X_Y_ARGS)
    {
        (*z) = (GB_TYPE) ((*x) <  (*y)) ;
    }

    GB_BINOP_STRUCT (ISGE, GB_XTYPE) ;
    inline void GB_FUNC (ISGE) (GB_Z_X_Y_ARGS)
    {
        (*z) = (GB_TYPE) ((*x) >= (*y)) ;
    }

    GB_BINOP_STRUCT (ISLE, GB_XTYPE) ;
    inline void GB_FUNC (ISLE) (GB_Z_X_Y_ARGS)
    {
        (*z) = (GB_TYPE) ((*x) <= (*y)) ;
    }

#endif

//------------------------------------------------------------------------------
// 3 boolean binary functions z=f(x,y), all x,y,z the same type, real types only
//------------------------------------------------------------------------------

#if defined ( GB_REAL )

    #if defined ( GB_BOOLEAN )

        inline void GB_FUNC (LOR)  (GB_Z_X_Y_ARGS) { (*z) = ((*x) || (*y)) ; }
        inline void GB_FUNC (LAND) (GB_Z_X_Y_ARGS) { (*z) = ((*x) && (*y)) ; }
        inline void GB_FUNC (LXOR) (GB_Z_X_Y_ARGS) { (*z) = ((*x) != (*y)) ; }

    #else

        // The inputs are of type T but are then implicitly converted to boolean
        // The output z is of type T, either 1 or 0 in that type.
        inline void GB_FUNC (LOR)  (GB_Z_X_Y_ARGS)
        {
            (*z) = (GB_TYPE) (((*x) != 0) || ((*y) != 0)) ;
        }

        inline void GB_FUNC (LAND) (GB_Z_X_Y_ARGS) 
        {
            (*z) = (GB_TYPE) (((*x) != 0) && ((*y) != 0)) ;
        }

        inline void GB_FUNC (LXOR) (GB_Z_X_Y_ARGS)
        {
            (*z) = (GB_TYPE) (((*x) != 0) != ((*y) != 0)) ;
        }

    #endif

    GB_BINOP_STRUCT (LOR, GB_XTYPE) ;
    GB_BINOP_STRUCT (LAND, GB_XTYPE) ;
    GB_BINOP_STRUCT (LXOR, GB_XTYPE) ;

#endif

#undef GB_Z_X_Y_ARGS

//------------------------------------------------------------------------------
// 6 binary functions z=f(x,y), returning bool
//------------------------------------------------------------------------------

// eq, ne: for all 13 types
// gt, lt, ge, le: for 11 real types, not complex

#define GB_Zbool_X_Y_ARGS bool *z, const GB_TYPE *x, const GB_TYPE *y

GB_BINOP_STRUCT (EQ, GB_XTYPE) ;
inline void GB_FUNC (EQ) (GB_Zbool_X_Y_ARGS)
{
    #if defined ( GB_FLOAT_COMPLEX )
    (*z) = GB_FC32_eq (*x, *y) ;
    #elif defined ( GB_DOUBLE_COMPLEX )
    (*z) = GB_FC64_eq (*x, *y) ;
    #else
    (*z) = ((*x) == (*y)) ;
    #endif
}

GB_BINOP_STRUCT (NE, GB_XTYPE) ;
inline void GB_FUNC (NE) (GB_Zbool_X_Y_ARGS)
{
    #if defined ( GB_FLOAT_COMPLEX )
    (*z) = GB_FC32_ne (*x, *y) ;
    #elif defined ( GB_DOUBLE_COMPLEX )
    (*z) = GB_FC64_ne (*x, *y) ;
    #else
    (*z) = ((*x) != (*y)) ;
    #endif
}

#if !defined ( GB_COMPLEX )

    GB_BINOP_STRUCT (GT, GB_XTYPE) ;
    inline void GB_FUNC (GT) (GB_Zbool_X_Y_ARGS) { (*z) = ((*x) >  (*y)) ; }

    GB_BINOP_STRUCT (LT, GB_XTYPE) ;
    inline void GB_FUNC (LT) (GB_Zbool_X_Y_ARGS) { (*z) = ((*x) <  (*y)) ; }

    GB_BINOP_STRUCT (GE, GB_XTYPE) ;
    inline void GB_FUNC (GE) (GB_Zbool_X_Y_ARGS) { (*z) = ((*x) >= (*y)) ; }

    GB_BINOP_STRUCT (LE, GB_XTYPE) ;
    inline void GB_FUNC (LE) (GB_Zbool_X_Y_ARGS) { (*z) = ((*x) <= (*y)) ; }

#endif

#undef GB_Zbool_X_Y_ARGS

//------------------------------------------------------------------------------
// index_unary functions
//------------------------------------------------------------------------------

#if defined ( GB_SIGNED_INDEX )

    //--------------------------------------------------------------------------
    // z = f (x, i, j, y) where z and y are both int32 or int64
    //--------------------------------------------------------------------------

    GB_IDXOP_STRUCT (ROWINDEX, GB_XTYPE) ;
    inline void GB_FUNC (ROWINDEX) (GB_TYPE *z, const void *unused,
        GrB_Index i, GrB_Index j_unused, const GB_TYPE *y)
    {
        (*z) = (GB_TYPE) (((int64_t) i) + (*y)) ;
    }
    GB_IDXOP_STRUCT (COLINDEX, GB_XTYPE) ;
    inline void GB_FUNC (COLINDEX) (GB_TYPE *z, const void *unused,
        GrB_Index i_unused, GrB_Index j, const GB_TYPE *y)
    {
        (*z) = (GB_TYPE) (((int64_t) j) + (*y)) ;
    }
    GB_IDXOP_STRUCT (DIAGINDEX, GB_XTYPE) ;
    inline void GB_FUNC (DIAGINDEX) (GB_TYPE *z, const void *unused,
        GrB_Index i, GrB_Index j, const GB_TYPE *y)
    {
        (*z) = (GB_TYPE) (((int64_t) j) - (((int64_t) i) + (*y))) ;
    }
    GB_IDXOP_STRUCT (FLIPDIAGINDEX, GB_XTYPE) ;
    inline void GB_FUNC (FLIPDIAGINDEX) (GB_TYPE *z, const void *unused,
        GrB_Index i, GrB_Index j, const GB_TYPE *y)
    {
        (*z) = (GB_TYPE) (((int64_t) i) - (((int64_t) j) + (*y))) ;
    }

#endif

#if defined ( GB_SIGNED_INDEX64 )

    //--------------------------------------------------------------------------
    // z = f (x, i, j, y) where z is bool, y is type int64
    //--------------------------------------------------------------------------

    GB_IDXOP_STRUCT (TRIL, GB_XTYPE) ;
    inline void GB_FUNC (TRIL) (bool *z, const void *unused,
        GrB_Index i, GrB_Index j, const GB_TYPE *y)
    {
        (*z) = (((int64_t) j) <= (((int64_t) i) + (*y))) ;
    }

    GB_IDXOP_STRUCT (TRIU, GB_XTYPE) ;
    inline void GB_FUNC (TRIU) (bool *z, const void *unused,
        GrB_Index i, GrB_Index j, const GB_TYPE *y)
    {
        (*z) = (((int64_t) j) >= (((int64_t) i) + (*y))) ;
    }

    GB_IDXOP_STRUCT (DIAG, GB_XTYPE) ;
    inline void GB_FUNC (DIAG) (bool *z, const void *unused,
        GrB_Index i, GrB_Index j, const GB_TYPE *y)
    {
        (*z) = (((int64_t) j) == (((int64_t) i) + (*y))) ;
    }

    GB_IDXOP_STRUCT (OFFDIAG, GB_XTYPE) ;
    inline void GB_FUNC (OFFDIAG) (bool *z, const void *unused,
        GrB_Index i, GrB_Index j, const GB_TYPE *y)
    {
        (*z) = (((int64_t) j) != (((int64_t) i) + (*y))) ;
    }

    GB_IDXOP_STRUCT (COLLE, GB_XTYPE) ;
    inline void GB_FUNC (COLLE) (bool *z, const void *unused,
        GrB_Index i_unused, GrB_Index j, const GB_TYPE *y)
    {
        (*z) = (((int64_t) j) <= (*y)) ;
    }

    GB_IDXOP_STRUCT (COLGT, GB_XTYPE) ;
    inline void GB_FUNC (COLGT) (bool *z, const void *unused,
        GrB_Index i_unused, GrB_Index j, const GB_TYPE *y)
    {
        (*z) = (((int64_t) j) > (*y)) ;
    }

    GB_IDXOP_STRUCT (ROWLE, GB_XTYPE) ;
    inline void GB_FUNC (ROWLE) (bool *z, const void *unused,
        GrB_Index i, GrB_Index j_unused, const GB_TYPE *y)
    {
        (*z) = (((int64_t) i) <= (*y)) ;
    }

    GB_IDXOP_STRUCT (ROWGT, GB_XTYPE) ;
    inline void GB_FUNC (ROWGT) (bool *z, const void *unused,
        GrB_Index i, GrB_Index j_unused, const GB_TYPE *y)
    {
        (*z) = (((int64_t) i) > (*y)) ;
    }

#endif

    //--------------------------------------------------------------------------
    // z = f (x, i, j, y) where z is bool, y is any built-in type
    //--------------------------------------------------------------------------

    GB_IDXOP_STRUCT (VALUEEQ, GB_XTYPE) ;
    inline void GB_FUNC (VALUEEQ) (bool *z, const GB_TYPE *x,
        GrB_Index i_unused, GrB_Index j_unused, const GB_TYPE *y)
    {
        #if defined ( GB_FLOAT_COMPLEX )
        (*z) = GB_FC32_eq (*x, *y) ;
        #elif defined ( GB_DOUBLE_COMPLEX )
        (*z) = GB_FC64_eq (*x, *y) ;
        #else
        (*z) = ((*x) == (*y)) ;
        #endif
    }

    GB_IDXOP_STRUCT (VALUENE, GB_XTYPE) ;
    inline void GB_FUNC (VALUENE) (bool *z, const GB_TYPE *x,
        GrB_Index i_unused, GrB_Index j_unused, const GB_TYPE *y)
    {
        #if defined ( GB_FLOAT_COMPLEX )
        (*z) = GB_FC32_ne (*x, *y) ;
        #elif defined ( GB_DOUBLE_COMPLEX )
        (*z) = GB_FC64_ne (*x, *y) ;
        #else
        (*z) = ((*x) != (*y)) ;
        #endif
    }

#if defined ( GB_REAL )

    //--------------------------------------------------------------------------
    // z = f (x, i, j, y) where z is bool, y is any real built-in type
    //--------------------------------------------------------------------------

    GB_IDXOP_STRUCT (VALUELT, GB_XTYPE) ;
    inline void GB_FUNC (VALUELT) (bool *z, const GB_TYPE *x,
        GrB_Index i_unused, GrB_Index j_unused, const GB_TYPE *y)
    {
        (*z) = ((*x) < (*y)) ;
    }

    GB_IDXOP_STRUCT (VALUELE, GB_XTYPE) ;
    inline void GB_FUNC (VALUELE) (bool *z, const GB_TYPE *x,
        GrB_Index i_unused, GrB_Index j_unused, const GB_TYPE *y)
    {
        (*z) = ((*x) <= (*y)) ;
    }

    GB_IDXOP_STRUCT (VALUEGT, GB_XTYPE) ;
    inline void GB_FUNC (VALUEGT) (bool *z, const GB_TYPE *x,
        GrB_Index i_unused, GrB_Index j_unused, const GB_TYPE *y)
    {
        (*z) = ((*x) > (*y)) ;
    }

    GB_IDXOP_STRUCT (VALUEGE, GB_XTYPE) ;
    inline void GB_FUNC (VALUEGE) (bool *z, const GB_TYPE *x,
        GrB_Index i_unused, GrB_Index j_unused, const GB_TYPE *y)
    {
        (*z) = ((*x) >= (*y)) ;
    }

#endif

//------------------------------------------------------------------------------
// builtin index binary operators
//------------------------------------------------------------------------------

#if defined ( GB_SIGNED_INDEX )

    GB_BINOP_STRUCT (FIRSTI, GB_XTYPE) ;
    inline void GB_FUNC (FIRSTI) (GB_TYPE *z,
        const void *x, GrB_Index ix, GrB_Index jx,
        const void *y, GrB_Index iy, GrB_Index jy,
        const void *theta)
    {
        (*z) = (GB_TYPE) ix ;
    }

    GB_BINOP_STRUCT (FIRSTI1, GB_XTYPE) ;
    inline void GB_FUNC (FIRSTI1) (GB_TYPE *z,
        const void *x, GrB_Index ix, GrB_Index jx,
        const void *y, GrB_Index iy, GrB_Index jy,
        const void *theta)
    {
        (*z) = ((GB_TYPE) ix) + 1 ;
    }

    GB_BINOP_STRUCT (FIRSTJ, GB_XTYPE) ;
    inline void GB_FUNC (FIRSTJ) (GB_TYPE *z,
        const void *x, GrB_Index ix, GrB_Index jx,
        const void *y, GrB_Index iy, GrB_Index jy,
        const void *theta)
    {
        (*z) = (GB_TYPE) jx ;
    }

    GB_BINOP_STRUCT (FIRSTJ1, GB_XTYPE) ;
    inline void GB_FUNC (FIRSTJ1) (GB_TYPE *z,
        const void *x, GrB_Index ix, GrB_Index jx,
        const void *y, GrB_Index iy, GrB_Index jy,
        const void *theta)
    {
        (*z) = ((GB_TYPE) jx) + 1 ;
    }

    GB_BINOP_STRUCT (SECONDI, GB_XTYPE) ;
    inline void GB_FUNC (SECONDI) (GB_TYPE *z,
        const void *x, GrB_Index ix, GrB_Index jx,
        const void *y, GrB_Index iy, GrB_Index jy,
        const void *theta)
    {
        (*z) = (GB_TYPE) iy ;
    }

    GB_BINOP_STRUCT (SECONDI1, GB_XTYPE) ;
    inline void GB_FUNC (SECONDI1) (GB_TYPE *z,
        const void *x, GrB_Index ix, GrB_Index jx,
        const void *y, GrB_Index iy, GrB_Index jy,
        const void *theta)
    {
        (*z) = ((GB_TYPE) iy) + 1 ;
    }

    GB_BINOP_STRUCT (SECONDJ, GB_XTYPE) ;
    inline void GB_FUNC (SECONDJ) (GB_TYPE *z,
        const void *x, GrB_Index ix, GrB_Index jx,
        const void *y, GrB_Index iy, GrB_Index jy,
        const void *theta)
    {
        (*z) = (GB_TYPE) jy ;
    }

    GB_BINOP_STRUCT (SECONDJ1, GB_XTYPE) ;
    inline void GB_FUNC (SECONDJ1) (GB_TYPE *z,
        const void *x, GrB_Index ix, GrB_Index jx,
        const void *y, GrB_Index iy, GrB_Index jy,
        const void *theta)
    {
        (*z) = ((GB_TYPE) jy) + 1 ;
    }

#endif

//------------------------------------------------------------------------------
// clear macros for next use of this file
//------------------------------------------------------------------------------

#undef GB_TYPE
#undef GB_XTYPE
#undef GB_UNOP_DEFINE
#undef GB_BOOLEAN
#undef GB_FLOATING_POINT
#undef GB_UNSIGNED_INT
#undef GB_SIGNED_INT
#undef GB_SIGNED_INDEX
#undef GB_SIGNED_INDEX64
#undef GB_X_NBITS
#undef GB_REAL
#undef GB_DOUBLE
#undef GB_FLOAT
#undef GB_DOUBLE_COMPLEX
#undef GB_FLOAT_COMPLEX
#undef GB_COMPLEX
#undef GB_UNOP_STRUCT
#undef GB_BINOP_STRUCT

