//------------------------------------------------------------------------------
// GB_ops_template.c: built-in unary and binary functions and operators
//------------------------------------------------------------------------------

// SuiteSparse:GraphBLAS, Timothy A. Davis, (c) 2017-2023, All Rights Reserved.
// SPDX-License-Identifier: Apache-2.0

//------------------------------------------------------------------------------

// This file is #include'd many times in GB_ops.c to define the built-in unary
// and binary operators.  In that file, GB_TYPE is a built-in C type (bool,
// int8_t, uint64_t, double, etc) for the inputs x and y, and GB_XTYPE is
// corresponding GraphBLAS type, without the prefix (BOOL, INT8, etc).

//------------------------------------------------------------------------------
// unary functions z=f(x) where z and x have the same type
//------------------------------------------------------------------------------

GXB_OP1 (ONE, "one") ;

#if defined ( GB_COMPLEX )

    // complex types
    GXB_OP1 (IDENTITY , "identity" ) ;
    GXB_OP1 (AINV     , "ainv"     ) ;
    GXB_OP1 (MINV     , "minv"     ) ;

#else

    // real types
    GRB_OP1 (IDENTITY , "identity" ) ;
    GRB_OP1 (AINV     , "ainv"     ) ;
    GRB_OP1 (MINV     , "minv"     ) ;

    // z=abs(x), z and x have the same type
    GRB_OP1 (ABS      , "abs"      ) ;

    // GxB_ABS_* is now GrB_ABS_*, and GxB_ABS_* is historical.
    // The new name is preferred.  The old name will be kept for historical
    // compatibility.
    GXB_OP1_RENAME (ABS) ;

    // LNOT is only defined for real types, not complex
    GXB_OP1 (LNOT     , "not"      ) ;

#endif

#if defined ( GB_FLOATING_POINT )

    GXB_OP1 (SQRT     , "sqrt"     ) ;
    GXB_OP1 (LOG      , "log"      ) ;
    GXB_OP1 (EXP      , "exp"      ) ;

    GXB_OP1 (SIN      , "sin"      ) ;
    GXB_OP1 (COS      , "cos"      ) ;
    GXB_OP1 (TAN      , "tan"      ) ;

    GXB_OP1 (ASIN     , "asin"     ) ;
    GXB_OP1 (ACOS     , "acos"     ) ;
    GXB_OP1 (ATAN     , "atan"     ) ;

    GXB_OP1 (SINH     , "sinh"     ) ;
    GXB_OP1 (COSH     , "cosh"     ) ;
    GXB_OP1 (TANH     , "tanh"     ) ;

    GXB_OP1 (ASINH    , "asinh"    ) ;
    GXB_OP1 (ACOSH    , "acosh"    ) ;
    GXB_OP1 (ATANH    , "atanh"    ) ;

    GXB_OP1 (SIGNUM   , "signum"   ) ;
    GXB_OP1 (CEIL     , "ceil"     ) ;
    GXB_OP1 (FLOOR    , "floor"    ) ;
    GXB_OP1 (ROUND    , "round"    ) ;
    GXB_OP1 (TRUNC    , "trunc"    ) ;

    GXB_OP1 (EXP2     , "exp2"     ) ;
    GXB_OP1 (EXPM1    , "expm1"    ) ;
    GXB_OP1 (LOG10    , "log10"    ) ;
    GXB_OP1 (LOG1P    , "log1p"    ) ;
    GXB_OP1 (LOG2     , "log2"     ) ;

    #if defined ( GB_COMPLEX )
    // complex only
    GXB_OP1 (CONJ     , "conj"     ) ;
    #else
    // real only
    GXB_OP1 (LGAMMA   , "lgamma"   ) ;
    GXB_OP1 (TGAMMA   , "tgamma"   ) ;
    GXB_OP1 (ERF      , "erf"      ) ;
    GXB_OP1 (ERFC     , "erfc"     ) ;
    GXB_OP1 (CBRT     , "cbrt"     ) ;
    GXB_OP1 (FREXPX   , "frexpx"   ) ;
    GXB_OP1 (FREXPE   , "frexpe"   ) ;
    #endif

#endif

#if defined ( GB_SIGNED_INT ) || defined ( GB_UNSIGNED_INT )

    // bitwise complement
    GRB_OP1 (BNOT     , "bnot"     ) ;

#endif

//------------------------------------------------------------------------------
// unary functions z=f(x) where z and x can have different types
//------------------------------------------------------------------------------

#if defined ( GB_FLOAT )

    // z = f(x) where x is float, and z is bool
    GXB_OP1z (ISINF     , "isinf"     , bool   , BOOL ) ;
    GXB_OP1z (ISNAN     , "isnan"     , bool   , BOOL ) ;
    GXB_OP1z (ISFINITE  , "isfinite"  , bool   , BOOL ) ;

#elif defined ( GB_DOUBLE )

    // z = f(x) where x is double, and z is bool
    GXB_OP1z (ISINF     , "isinf"     , bool   , BOOL ) ;
    GXB_OP1z (ISNAN     , "isnan"     , bool   , BOOL ) ;
    GXB_OP1z (ISFINITE  , "isfinite"  , bool   , BOOL ) ;

#elif defined ( GB_FLOAT_COMPLEX )

    // z = f(x) where x is float complex, and the type of z is listed below:
    GXB_OP1z (ABS       , "abs"       , float  , FP32) ;
    GXB_OP1z (ISINF     , "isinf"     , bool   , BOOL) ;
    GXB_OP1z (ISNAN     , "isnan"     , bool   , BOOL) ;
    GXB_OP1z (ISFINITE  , "isfinite"  , bool   , BOOL ) ;

    GXB_OP1z (CREAL     , "creal"     , float  , FP32) ;
    GXB_OP1z (CIMAG     , "cimag"     , float  , FP32) ;
    GXB_OP1z (CARG      , "carg"      , float  , FP32) ;

#elif defined ( GB_DOUBLE_COMPLEX )

    // z = f(x) where x is double complex, and the type of z is listed below:
    GXB_OP1z (ABS       , "abs"       , double , FP64) ;
    GXB_OP1z (ISINF     , "isinf"     , bool   , BOOL) ;
    GXB_OP1z (ISNAN     , "isnan"     , bool   , BOOL) ;
    GXB_OP1z (ISFINITE  , "isfinite"  , bool   , BOOL ) ;

    GXB_OP1z (CREAL     , "creal"     , double , FP64) ;
    GXB_OP1z (CIMAG     , "cimag"     , double , FP64) ;
    GXB_OP1z (CARG      , "carg"      , double , FP64) ;

#endif

//------------------------------------------------------------------------------
// binary functions z=f(x,y) where z, x, and y all have the same type
//------------------------------------------------------------------------------

GXB_OP2 (RMINUS , "rminus") ;
GXB_OP2 (RDIV   , "rdiv"  ) ;
GXB_OP2 (PAIR   , "pair"  ) ;
GXB_OP2 (ANY    , "any"   ) ;
GXB_OP2 (ISEQ   , "iseq"  ) ;
GXB_OP2 (ISNE   , "isne"  ) ;
GXB_OP2 (POW    , "pow"   ) ;

#if defined ( GB_COMPLEX ) 

    // complex types
    GXB_OP2 (FIRST  , "first" ) ;
    GXB_OP2 (SECOND , "second") ;
    GXB_OP2 (PLUS   , "plus"  ) ;
    GXB_OP2 (MINUS  , "minus" ) ;
    GXB_OP2 (TIMES  , "times" ) ;
    GXB_OP2 (DIV    , "div"   ) ;

#else

    // real types
    GRB_OP2 (FIRST  , "first" ) ;
    GRB_OP2 (SECOND , "second") ;
    GRB_OP2 (PLUS   , "plus"  ) ;
    GRB_OP2 (MINUS  , "minus" ) ;
    GRB_OP2 (TIMES  , "times" ) ;
    GRB_OP2 (DIV    , "div"   ) ;

    GRB_OP2 (MIN    , "min" ) ;
    GRB_OP2 (MAX    , "max" ) ;

    GXB_OP2 (LOR    , "or"  ) ;
    GXB_OP2 (LAND   , "and" ) ;
    GXB_OP2 (LXOR   , "xor" ) ;

    GXB_OP2 (ISGT   , "isgt") ;
    GXB_OP2 (ISLT   , "islt") ;
    GXB_OP2 (ISGE   , "isge") ;
    GXB_OP2 (ISLE   , "isle") ;

#endif

#if defined (GB_FLOAT) || defined (GB_DOUBLE)

    // these operators are only defined for float and double
    GXB_OP2 (ATAN2    , "atan2"    ) ;
    GXB_OP2 (HYPOT    , "hypot"    ) ;
    GXB_OP2 (FMOD     , "fmod"     ) ;
    GXB_OP2 (REMAINDER, "remainder") ;
    GXB_OP2 (COPYSIGN , "copysign" ) ;
    GXB_OP2 (LDEXP    , "ldexp"    ) ;

#endif

#if defined ( GB_SIGNED_INT ) || defined ( GB_UNSIGNED_INT )

    // bitwise binary operators
    GRB_OP2 (BOR      , "bitor"   ) ;
    GRB_OP2 (BAND     , "bitand"  ) ;
    GRB_OP2 (BXOR     , "bitxor"  ) ;
    GRB_OP2 (BXNOR    , "bitxnor" ) ;

    GXB_OP2 (BGET     , "bitget"   ) ;
    GXB_OP2 (BSET     , "bitset"   ) ;
    GXB_OP2 (BCLR     , "bitclear" ) ;

    GXB_OP2shift (BSHIFT, "bitshift") ;

#endif

//------------------------------------------------------------------------------
// binary functions z=f(x,y) where z, x, and y can have different types
//------------------------------------------------------------------------------

#if defined ( GB_FLOAT )

    // z = cmplx(x,y) where z is float complex, x and y are float
    GXB_OP2z (CMPLX, "cmplx", GxB_FC32_t, FC32) ;

#endif

#if defined ( GB_DOUBLE )

    // z = cmplx(x,y) where z is double complex, x and y are double
    GXB_OP2z (CMPLX, "cmplx", GxB_FC64_t, FC64) ;

#endif

#if defined ( GB_COMPLEX )

    // complex types
    GXB_OP2z (EQ, "eq", bool, BOOL) ;
    GXB_OP2z (NE, "ne", bool, BOOL) ;

#else

    // real types
    GRB_OP2z (EQ, "eq", bool, BOOL) ;
    GRB_OP2z (NE, "ne", bool, BOOL) ;
    GRB_OP2z (GT, "gt", bool, BOOL) ;
    GRB_OP2z (LT, "lt", bool, BOOL) ;
    GRB_OP2z (LE, "le", bool, BOOL) ;
    GRB_OP2z (GE, "ge", bool, BOOL) ;

#endif

//------------------------------------------------------------------------------
// index_unary functions z=f(x,i,j,y)
//------------------------------------------------------------------------------

#if defined ( GB_SIGNED_INDEX )

    // z = f (x, i, j, y) where z and y have type int32 or int64
    GRB_IDXOP_POSITIONAL (ROWINDEX,  "rowindex" ) ;
    GRB_IDXOP_POSITIONAL (COLINDEX,  "colindex" ) ;
    GRB_IDXOP_POSITIONAL (DIAGINDEX, "diagindex") ;
    GXB_IDXOP_POSITIONAL (FLIPDIAGINDEX, "flipdiagindex") ;

#endif

#if defined ( GB_SIGNED_INDEX64 )

    // z = f (x, i, j, y) where z is bool; y has type int64 only
    GRB_IDXOP_POSITIONAL_BOOL (TRIL,    "tril" ) ;
    GRB_IDXOP_POSITIONAL_BOOL (TRIU,    "triu" ) ;
    GRB_IDXOP_POSITIONAL_BOOL (DIAG,    "diag" ) ;
    GRB_IDXOP_POSITIONAL_BOOL (OFFDIAG, "offdiag" ) ;
    GRB_IDXOP_POSITIONAL_BOOL (COLLE,   "colle" ) ;
    GRB_IDXOP_POSITIONAL_BOOL (COLGT,   "colgt" ) ;
    GRB_IDXOP_POSITIONAL_BOOL (ROWLE,   "rowle" ) ;
    GRB_IDXOP_POSITIONAL_BOOL (ROWGT,   "rowgt" ) ;

#endif

#if defined ( GB_COMPLEX )

    // z = f (x, i, j, y) where z is bool; y is complex
    GXB_IDXOP_VALUE (VALUEEQ, "valueeq") ;
    GXB_IDXOP_VALUE (VALUENE, "valuene") ;

#else

    // z = f (x, i, j, y) where z is bool; y is real
    GRB_IDXOP_VALUE (VALUEEQ, "valueeq") ;
    GRB_IDXOP_VALUE (VALUENE, "valuene") ;
    GRB_IDXOP_VALUE (VALUELT, "valuelt") ;
    GRB_IDXOP_VALUE (VALUELE, "valuele") ;
    GRB_IDXOP_VALUE (VALUEGT, "valuegt") ;
    GRB_IDXOP_VALUE (VALUEGE, "valuege") ;

#endif

//------------------------------------------------------------------------------
// clear macros for next use of this file
//------------------------------------------------------------------------------

#undef GB_TYPE
#undef GB_XTYPE
#undef GB_FLOATING_POINT
#undef GB_COMPLEX
#undef GB_FLOAT
#undef GB_DOUBLE
#undef GB_FLOAT_COMPLEX
#undef GB_DOUBLE_COMPLEX
#undef GB_SIGNED_INT
#undef GB_UNSIGNED_INT
#undef GB_SIGNED_INDEX
#undef GB_SIGNED_INDEX64

