//------------------------------------------------------------------------------
// GB_ops_template.h: define the unary and binary functions and operators
//------------------------------------------------------------------------------

// SuiteSparse:GraphBLAS, Timothy A. Davis, (c) 2017-2023, All Rights Reserved.
// SPDX-License-Identifier: Apache-2.0

//------------------------------------------------------------------------------

// This file is #include'd many times in GB_ops.h to define the unary and
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
        #if ( GB_BITS == 8)
            (*z) = GB_idiv_int8 (1, (*x)) ;
        #elif ( GB_BITS == 16)
            (*z) = GB_idiv_int16 (1, (*x)) ;
        #elif ( GB_BITS == 32)
            (*z) = GB_idiv_int32 (1, (*x)) ;
        #elif ( GB_BITS == 64)
            (*z) = GB_idiv_int64 (1, (*x)) ;
        #endif
    #elif defined ( GB_UNSIGNED_INT )
        #if ( GB_BITS == 8)
            (*z) = GB_idiv_uint8 (1, (*x)) ;
        #elif ( GB_BITS == 16)
            (*z) = GB_idiv_uint16 (1, (*x)) ;
        #elif ( GB_BITS == 32)
            (*z) = GB_idiv_uint32 (1, (*x)) ;
        #elif ( GB_BITS == 64)
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

#undef  GB_OP
#define GB_OP(op,func)                                          \
    GB_UNOP_STRUCT (op, GB_XTYPE) ;                             \
    inline void GB_FUNC (op) (GB_TYPE *z, const GB_TYPE *x)     \
    {                                                           \
        (*z) = func (*x) ;                                      \
    }

#if defined ( GB_FLOAT )

    //--------------------------------------------------------------------------
    // float
    //--------------------------------------------------------------------------

    GB_OP (SQRT  , sqrtf   )
    GB_OP (LOG   , logf    )
    GB_OP (EXP   , expf    )

    GB_OP (SIN   , sinf    )
    GB_OP (COS   , cosf    )
    GB_OP (TAN   , tanf    )

    GB_OP (ASIN  , asinf   )
    GB_OP (ACOS  , acosf   )
    GB_OP (ATAN  , atanf   )

    GB_OP (SINH  , sinhf   )
    GB_OP (COSH  , coshf   )
    GB_OP (TANH  , tanhf   )

    GB_OP (ASINH , asinhf  )
    GB_OP (ACOSH , acoshf  )
    GB_OP (ATANH , atanhf  )

    GB_OP (SIGNUM, GB_signumf )
    GB_OP (CEIL  , ceilf   )
    GB_OP (FLOOR , floorf  )
    GB_OP (ROUND , roundf  )
    GB_OP (TRUNC , truncf  )

    GB_OP (EXP2  , exp2f   )
    GB_OP (EXPM1 , expm1f  )
    GB_OP (LOG10 , log10f  )
    GB_OP (LOG1P , log1pf  )
    GB_OP (LOG2  , log2f   )

    // real only
    GB_OP (LGAMMA, lgammaf )
    GB_OP (TGAMMA, tgammaf )
    GB_OP (ERF   , erff    )
    GB_OP (ERFC  , erfcf   )
    GB_OP (CBRT  , cbrtf   )

#elif defined ( GB_DOUBLE )

    //--------------------------------------------------------------------------
    // double
    //--------------------------------------------------------------------------

    GB_OP (SQRT  , sqrt    )
    GB_OP (LOG   , log     )
    GB_OP (EXP   , exp     )

    GB_OP (SIN   , sin     )
    GB_OP (COS   , cos     )
    GB_OP (TAN   , tan     )

    GB_OP (ASIN  , asin    )
    GB_OP (ACOS  , acos    )
    GB_OP (ATAN  , atan    )

    GB_OP (SINH  , sinh    )
    GB_OP (COSH  , cosh    )
    GB_OP (TANH  , tanh    )

    GB_OP (ASINH , asinh   )
    GB_OP (ACOSH , acosh   )
    GB_OP (ATANH , atanh   )

    GB_OP (SIGNUM, GB_signum )
    GB_OP (CEIL  , ceil    )
    GB_OP (FLOOR , floor   )
    GB_OP (ROUND , round   )
    GB_OP (TRUNC , trunc   )

    GB_OP (EXP2  , exp2    )
    GB_OP (EXPM1 , expm1   )
    GB_OP (LOG10 , log10   )
    GB_OP (LOG1P , log1p   )
    GB_OP (LOG2  , log2    )

    // real only
    GB_OP (LGAMMA, lgamma )
    GB_OP (TGAMMA, tgamma )
    GB_OP (ERF   , erf    )
    GB_OP (ERFC  , erfc   )
    GB_OP (CBRT  , cbrt   )

#elif defined ( GB_FLOAT_COMPLEX )

    //--------------------------------------------------------------------------
    // float complex
    //--------------------------------------------------------------------------

    GB_OP (SQRT  , GB_csqrtf   )
    GB_OP (LOG   , GB_clogf    )
    GB_OP (EXP   , GB_cexpf    )

    GB_OP (SIN   , GB_csinf    )
    GB_OP (COS   , GB_ccosf    )
    GB_OP (TAN   , GB_ctanf    )

    GB_OP (ASIN  , GB_casinf   )
    GB_OP (ACOS  , GB_cacosf   )
    GB_OP (ATAN  , GB_catanf   )

    GB_OP (SINH  , GB_csinhf   )
    GB_OP (COSH  , GB_ccoshf   )
    GB_OP (TANH  , GB_ctanhf   )

    GB_OP (ASINH , GB_casinhf  )
    GB_OP (ACOSH , GB_cacoshf  )
    GB_OP (ATANH , GB_catanhf  )

    GB_OP (SIGNUM, GB_csignumf )
    GB_OP (CEIL  , GB_cceilf   )
    GB_OP (FLOOR , GB_cfloorf  )
    GB_OP (ROUND , GB_croundf  )
    GB_OP (TRUNC , GB_ctruncf  )

    GB_OP (EXP2  , GB_cexp2f   )
    GB_OP (EXPM1 , GB_cexpm1f  )
    GB_OP (LOG10 , GB_clog10f  )
    GB_OP (LOG1P , GB_clog1pf  )
    GB_OP (LOG2  , GB_clog2f   )

    GB_OP (CONJ  , GB_conjf    )

#elif defined ( GB_DOUBLE_COMPLEX )

    //--------------------------------------------------------------------------
    // double complex
    //--------------------------------------------------------------------------

    GB_OP (SQRT  , GB_csqrt    )
    GB_OP (LOG   , GB_clog     )
    GB_OP (EXP   , GB_cexp     )

    GB_OP (SIN   , GB_csin     )
    GB_OP (COS   , GB_ccos     )
    GB_OP (TAN   , GB_ctan     )

    GB_OP (ASIN  , GB_casin    )
    GB_OP (ACOS  , GB_cacos    )
    GB_OP (ATAN  , GB_catan    )

    GB_OP (SINH  , GB_csinh    )
    GB_OP (COSH  , GB_ccosh    )
    GB_OP (TANH  , GB_ctanh    )

    GB_OP (ASINH , GB_casinh   )
    GB_OP (ACOSH , GB_cacosh   )
    GB_OP (ATANH , GB_catanh   )

    GB_OP (SIGNUM, GB_csignum  )
    GB_OP (CEIL  , GB_cceil    )
    GB_OP (FLOOR , GB_cfloor   )
    GB_OP (ROUND , GB_cround   )
    GB_OP (TRUNC , GB_ctrunc   )

    GB_OP (EXP2  , GB_cexp2    )
    GB_OP (EXPM1 , GB_cexpm1   )
    GB_OP (LOG10 , GB_clog10   )
    GB_OP (LOG1P , GB_clog1p   )
    GB_OP (LOG2  , GB_clog2    )

    GB_OP (CONJ  , GB_conj     )

#endif

//------------------------------------------------------------------------------
// unary operators z=f(x) where z and x have different types
//------------------------------------------------------------------------------

// x is float, double, float complex, or double complex

#undef  GB_OP
#define GB_OP(op,expression,z_t,x_t)                        \
    GB_UNOP_STRUCT(op, GB_XTYPE) ;                          \
    inline void GB_FUNC (op) (z_t *z, const x_t *x)         \
    {                                                       \
        (*z) = expression ;                                 \
    }

#if defined ( GB_FLOAT )

    GB_OP (ISINF    , (isinf (*x))    , bool, float)
    GB_OP (ISNAN    , (isnan (*x))    , bool, float)
    GB_OP (ISFINITE , (isfinite (*x)) , bool, float)

#elif defined ( GB_DOUBLE )

    GB_OP (ISINF    , (isinf (*x))    , bool, double)
    GB_OP (ISNAN    , (isnan (*x))    , bool, double)
    GB_OP (ISFINITE , (isfinite (*x)) , bool, double)

#elif defined ( GB_FLOAT_COMPLEX )

    GB_OP (ISINF    , GB_cisinff (*x)   , bool, GxB_FC32_t)
    GB_OP (ISNAN    , GB_cisnanf (*x)   , bool, GxB_FC32_t)
    GB_OP (ISFINITE , GB_cisfinitef (*x), bool, GxB_FC32_t)

    // complex only
    GB_OP (CREAL , GB_crealf (*x), float, GxB_FC32_t)
    GB_OP (CIMAG , GB_cimagf (*x), float, GxB_FC32_t)
    GB_OP (CARG  , GB_cargf  (*x), float, GxB_FC32_t)

#elif defined ( GB_DOUBLE_COMPLEX )

    GB_OP (ISINF    , GB_cisinf (*x)    , bool, GxB_FC64_t)
    GB_OP (ISNAN    , GB_cisnan  (*x)   , bool, GxB_FC64_t)
    GB_OP (ISFINITE , GB_cisfinite (*x) , bool, GxB_FC64_t)

    // complex only
    GB_OP (CREAL , GB_creal (*x), double, GxB_FC64_t)
    GB_OP (CIMAG , GB_cimag (*x), double, GxB_FC64_t)
    GB_OP (CARG  , GB_carg  (*x), double, GxB_FC64_t)

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
        #if ( GB_BITS == 8)
            (*z) = GB_idiv_int8 ((*x), (*y)) ;
        #elif ( GB_BITS == 16)
            (*z) = GB_idiv_int16 ((*x), (*y)) ;
        #elif ( GB_BITS == 32)
            (*z) = GB_idiv_int32 ((*x), (*y)) ;
        #elif ( GB_BITS == 64)
            (*z) = GB_idiv_int64 ((*x), (*y)) ;
        #endif
    #elif defined ( GB_UNSIGNED_INT )
        #if ( GB_BITS == 8)
            (*z) = GB_idiv_uint8 ((*x), (*y)) ;
        #elif ( GB_BITS == 16)
            (*z) = GB_idiv_uint16 ((*x), (*y)) ;
        #elif ( GB_BITS == 32)
            (*z) = GB_idiv_uint32 ((*x), (*y)) ;
        #elif ( GB_BITS == 64)
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
        #if ( GB_BITS == 8)
            (*z) = GB_idiv_int8 ((*y), (*x)) ;
        #elif ( GB_BITS == 16)
            (*z) = GB_idiv_int16 ((*y), (*x)) ;
        #elif ( GB_BITS == 32)
            (*z) = GB_idiv_int32 ((*y), (*x)) ;
        #elif ( GB_BITS == 64)
            (*z) = GB_idiv_int64 ((*y), (*x)) ;
        #endif
    #elif defined ( GB_UNSIGNED_INT )
        #if ( GB_BITS == 8)
            (*z) = GB_idiv_uint8 ((*y), (*x)) ;
        #elif ( GB_BITS == 16)
            (*z) = GB_idiv_uint16 ((*y), (*x)) ;
        #elif ( GB_BITS == 32)
            (*z) = GB_idiv_uint32 ((*y), (*x)) ;
        #elif ( GB_BITS == 64)
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
        #if ( GB_BITS == 8)
            (*z) = GB_pow_int8 ((*x), (*y)) ;
        #elif ( GB_BITS == 16)
            (*z) = GB_pow_int16 ((*x), (*y)) ;
        #elif ( GB_BITS == 32)
            (*z) = GB_pow_int32 ((*x), (*y)) ;
        #elif ( GB_BITS == 64)
            (*z) = GB_pow_int64 ((*x), (*y)) ;
        #endif
    #elif defined ( GB_UNSIGNED_INT )
        #if ( GB_BITS == 8)
            (*z) = GB_pow_uint8 ((*x), (*y)) ;
        #elif ( GB_BITS == 16)
            (*z) = GB_pow_uint16 ((*x), (*y)) ;
        #elif ( GB_BITS == 32)
            (*z) = GB_pow_uint32 ((*x), (*y)) ;
        #elif ( GB_BITS == 64)
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
        // is given by y.  y = 1 is the least significant bit, and y = GB_BITS
        // (64 for uint64) is the most significant bit. If y is outside this
        // range, the result is zero.

        #if defined ( GB_SIGNED_INT )

            #if ( GB_BITS == 8)
                (*z) = GB_bitget_int8 ((*x), (*y)) ;
            #elif ( GB_BITS == 16)
                (*z) = GB_bitget_int16 ((*x), (*y)) ;
            #elif ( GB_BITS == 32)
                (*z) = GB_bitget_int32 ((*x), (*y)) ;
            #elif ( GB_BITS == 64)
                (*z) = GB_bitget_int64 ((*x), (*y)) ;
            #endif

        #elif defined ( GB_UNSIGNED_INT )

            #if ( GB_BITS == 8)
                (*z) = GB_bitget_uint8 ((*x), (*y)) ;
            #elif ( GB_BITS == 16)
                (*z) = GB_bitget_uint16 ((*x), (*y)) ;
            #elif ( GB_BITS == 32)
                (*z) = GB_bitget_uint32 ((*x), (*y)) ;
            #elif ( GB_BITS == 64)
                (*z) = GB_bitget_uint64 ((*x), (*y)) ;
            #endif

        #endif
    }

    GB_BINOP_STRUCT (BSET, GB_XTYPE) ;
    inline void GB_FUNC (BSET) (GB_Z_X_Y_ARGS)
    {
        // bitset (x,y) returns x modified by setting a bit from x to 1, whose
        // position is given by y.  If y is in the range 1 to GB_BITS, then y
        // gives the position of the bit to set.  If y is outside the range 1
        // to GB_BITS, then z = x is returned, unmodified.

        #if defined ( GB_SIGNED_INT )

            #if ( GB_BITS == 8)
                (*z) = GB_bitset_int8  ((*x), (*y)) ;
            #elif ( GB_BITS == 16)
                (*z) = GB_bitset_int16 ((*x), (*y)) ;
            #elif ( GB_BITS == 32)
                (*z) = GB_bitset_int32 ((*x), (*y)) ;
            #elif ( GB_BITS == 64)
                (*z) = GB_bitset_int64 ((*x), (*y)) ;
            #endif

        #elif defined ( GB_UNSIGNED_INT )

            #if ( GB_BITS == 8)
                (*z) = GB_bitset_uint8  ((*x), (*y)) ;
            #elif ( GB_BITS == 16)
                (*z) = GB_bitset_uint16 ((*x), (*y)) ;
            #elif ( GB_BITS == 32)
                (*z) = GB_bitset_uint32 ((*x), (*y)) ;
            #elif ( GB_BITS == 64)
                (*z) = GB_bitset_uint64 ((*x), (*y)) ;
            #endif

        #endif
    }

    GB_BINOP_STRUCT (BCLR, GB_XTYPE) ;
    inline void GB_FUNC (BCLR) (GB_Z_X_Y_ARGS)
    {
        // bitclr (x,y) returns x modified by setting a bit from x to 0, whose
        // position is given by y.  If y is in the range 1 to GB_BITS, then y
        // gives the position of the bit to clear.  If y is outside the range 1
        // to GB_BITS, then z = x is returned, unmodified.

        #if defined ( GB_SIGNED_INT )

            #if ( GB_BITS == 8)
                (*z) = GB_bitclr_int8  ((*x), (*y)) ;
            #elif ( GB_BITS == 16)
                (*z) = GB_bitclr_int16 ((*x), (*y)) ;
            #elif ( GB_BITS == 32)
                (*z) = GB_bitclr_int32 ((*x), (*y)) ;
            #elif ( GB_BITS == 64)
                (*z) = GB_bitclr_int64 ((*x), (*y)) ;
            #endif

        #elif defined ( GB_UNSIGNED_INT )

            #if ( GB_BITS == 8)
                (*z) = GB_bitclr_uint8  ((*x), (*y)) ;
            #elif ( GB_BITS == 16)
                (*z) = GB_bitclr_uint16 ((*x), (*y)) ;
            #elif ( GB_BITS == 32)
                (*z) = GB_bitclr_uint32 ((*x), (*y)) ;
            #elif ( GB_BITS == 64)
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

            #if ( GB_BITS == 8)
                (*z) = GB_bitshift_int8 ((*x), (*y)) ;
            #elif ( GB_BITS == 16)
                (*z) = GB_bitshift_int16 ((*x), (*y)) ;
            #elif ( GB_BITS == 32)
                (*z) = GB_bitshift_int32 ((*x), (*y)) ;
            #elif ( GB_BITS == 64)
                (*z) = GB_bitshift_int64 ((*x), (*y)) ;
            #endif

        #elif defined ( GB_UNSIGNED_INT )

            #if ( GB_BITS == 8)
                (*z) = GB_bitshift_uint8 ((*x), (*y)) ;
            #elif ( GB_BITS == 16)
                (*z) = GB_bitshift_uint16 ((*x), (*y)) ;
            #elif ( GB_BITS == 32)
                (*z) = GB_bitshift_uint32 ((*x), (*y)) ;
            #elif ( GB_BITS == 64)
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
        #if defined ( __cplusplus ) || GB_COMPILER_MSC || defined (CMPLX)
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
        #if defined ( __cplusplus ) || GB_COMPILER_MSC || defined (CMPLX)
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
// clear macros for next use of this file
//------------------------------------------------------------------------------

#undef GB_TYPE
#undef GB_XTYPE
#undef GB_OP
#undef GB_BOOLEAN
#undef GB_FLOATING_POINT
#undef GB_UNSIGNED_INT
#undef GB_SIGNED_INT
#undef GB_SIGNED_INDEX
#undef GB_SIGNED_INDEX64
#undef GB_BITS
#undef GB_REAL
#undef GB_DOUBLE
#undef GB_FLOAT
#undef GB_DOUBLE_COMPLEX
#undef GB_FLOAT_COMPLEX
#undef GB_COMPLEX
#undef GB_UNOP_STRUCT
#undef GB_BINOP_STRUCT

