//------------------------------------------------------------------------------
// GB_enumify_unop: convert unary or idxunary opcode and xcode into a enum
//------------------------------------------------------------------------------

// SuiteSparse:GraphBLAS, Timothy A. Davis, (c) 2017-2023, All Rights Reserved.
// SPDX-License-Identifier: Apache-2.0

//------------------------------------------------------------------------------

// Handles GrB_UnaryOp (ecodes 0 to 160) and GrB_IndexUnaryOp (231 to 254).
//
// A GrB_UnaryOp is a function of the form z = f(x), but the C signature for
// user-defined functions is:
//
//      void f (void *z, const void *x)
//
// A GrB_IndexUnaryOp has the form z = f (x, i, j, y) where y is the scalar
// thunk value.  The C signature for the user-defined function is:
//
//      void f (void *z, const void *x, uint64_t i, uint64_t j, const void *y)

#include "GB.h"
#include "GB_stringify.h"

void GB_enumify_unop    // enumify a GrB_UnaryOp or GrB_IndexUnaryOp
(
    // output:
    int *ecode,         // enumerated operator, range 0 to 254
    bool *depends_on_x, // true if the op depends on x
    bool *depends_on_i, // true if the op depends on i
    bool *depends_on_j, // true if the op depends on j
    bool *depends_on_y, // true if the op depends on y
    // input:
    bool flipij,        // if true, then the i and j indices are flipped
    GB_Opcode opcode,   // opcode of GraphBLAS operator to convert into a macro
    GB_Type_code xcode  // op->xtype->code of the operator
)
{ 

    int e = 0 ;
    bool i_dep = false ;
    bool j_dep = false ;

    switch (opcode)
    {

        //----------------------------------------------------------------------
        // user-defined GrB_UnaryOp
        //----------------------------------------------------------------------

        case GB_USER_unop_code      : 

            e = 0 ; break ;                 // user-defined GrB_UnaryOp

        //----------------------------------------------------------------------
        // primary unary operators x=f(x)
        //----------------------------------------------------------------------

        case GB_ONE_unop_code       : 
            e = 1 ; break ;                 // z = 1

        case GB_IDENTITY_unop_code  :       // z = x
        case GB_NOP_code            :       // none (treat as identity op)

            e = 2 ; break ;                 // z = x

        case GB_AINV_unop_code      :       // z = -x

            switch (xcode)
            {
                case GB_BOOL_code   : e =   2 ; break ; // z = x
                case GB_FC32_code   : e =   3 ; break ; // z = GB_FC32_ainv (x)
                case GB_FC64_code   : e =   4 ; break ; // z = GB_FC64_ainv (x)
                default             : e =   5 ; break ; // z = -x
            }
            break ;

        case GB_ABS_unop_code       :       // z = abs(x) ; z real if x complex

            switch (xcode)
            {
                case GB_INT8_code   : 
                case GB_INT16_code  : 
                case GB_INT32_code  : 
                case GB_INT64_code  : e =   6 ; break ; // z = GB_IABS (x)
                case GB_FP32_code   : e =   7 ; break ; // z = fabsf (x)
                case GB_FP64_code   : e =   8 ; break ; // z = fabs (x)
                case GB_FC32_code   : e =   9 ; break ; // z = GB_cabsf (x)
                case GB_FC64_code   : e =  10 ; break ; // z = GB_cabs (x)
                default             : e =   2 ; break ; // z = x (uint, bool)
            }
            break ;

        case GB_MINV_unop_code      :       // z = 1/x

            switch (xcode)
            {
                default:
                case GB_BOOL_code   : e =   1 ; break ; // z = 1 (minv for bool)
                case GB_INT8_code   : e =  11 ; break ; // z = GJ_idiv_* (1, x)
                case GB_INT16_code  : e =  12 ; break ; // z = GJ_idiv_* (1, x)
                case GB_INT32_code  : e =  13 ; break ; // z = GJ_idiv_* (1, x)
                case GB_INT64_code  : e =  14 ; break ; // z = GJ_idiv_* (1, x)
                case GB_UINT8_code  : e =  15 ; break ; // z = GJ_idiv_* (1, x)
                case GB_UINT16_code : e =  16 ; break ; // z = GJ_idiv_* (1, x)
                case GB_UINT32_code : e =  17 ; break ; // z = GJ_idiv_* (1, x)
                case GB_UINT64_code : e =  18 ; break ; // z = GJ_idiv_* (1, x)
                case GB_FP32_code   : e =  19 ; break ; // z = (1.0F)/x
                case GB_FP64_code   : e =  20 ; break ; // z = 1./x
                case GB_FC32_code   : e =  21 ; break ; // z = GJ_FC32_div (1,x)
                case GB_FC64_code   : e =  22 ; break ; // z = GJ_FC64_div (1,x)
            }
            break ;

        case GB_LNOT_unop_code      :       // z = !x (logical negation)

            switch (xcode)
            {
                case GB_BOOL_code   : e =  23 ; break ; // z = !x
                default             : e =  24 ; break ; // z = !(x != 0)
            }
            break ;

        case GB_BNOT_unop_code      :       // z = ~x (bitwise complement)

            e = 25 ; break ;                // z = ~(x)

        //----------------------------------------------------------------------
        // unary operators for floating-point types (real and complex)
        //----------------------------------------------------------------------

        case GB_SQRT_unop_code      :       // z = sqrt (x)

            switch (xcode)
            {
                default:
                case GB_FP32_code   : e =  26 ; break ; // z = sqrtf (x)
                case GB_FP64_code   : e =  27 ; break ; // z = sqrt (x)
                case GB_FC32_code   : e =  28 ; break ; // z = GB_csqrtf (x)
                case GB_FC64_code   : e =  29 ; break ; // z = GB_csqrt (x)
            }
            break ;

        case GB_LOG_unop_code       :       // z = log (x)

            switch (xcode)
            {
                default:
                case GB_FP32_code   : e =  30 ; break ; // z = logf (x)
                case GB_FP64_code   : e =  31 ; break ; // z = log (x)
                case GB_FC32_code   : e =  32 ; break ; // z = GB_clogf (x)
                case GB_FC64_code   : e =  33 ; break ; // z = GB_clog (x)
            }
            break ;

        case GB_EXP_unop_code       :       // z = exp (x)

            switch (xcode)
            {
                default:
                case GB_FP32_code   : e =  34 ; break ; // z = expf (x)
                case GB_FP64_code   : e =  35 ; break ; // z = exp (x)
                case GB_FC32_code   : e =  36 ; break ; // z = GB_cexpf (x)
                case GB_FC64_code   : e =  37 ; break ; // z = GB_cexp (x)
            }
            break ;

        case GB_SIN_unop_code       :       // z = sin (x)

            switch (xcode)
            {
                default:
                case GB_FP32_code   : e =  38 ; break ; // z = sinf (x)
                case GB_FP64_code   : e =  39 ; break ; // z = sin (x)
                case GB_FC32_code   : e =  40 ; break ; // z = GB_csinf (x)
                case GB_FC64_code   : e =  41 ; break ; // z = GB_csin (x)
            }
            break ;

        case GB_COS_unop_code       :       // z = cos (x)

            switch (xcode)
            {
                default:
                case GB_FP32_code   : e =  42 ; break ; // z = cosf (x)
                case GB_FP64_code   : e =  43 ; break ; // z = cos (x)
                case GB_FC32_code   : e =  44 ; break ; // z = GB_cosf (x)
                case GB_FC64_code   : e =  45 ; break ; // z = GB_cos (x)
            }
            break ;

        case GB_TAN_unop_code       :       // z = tan (x)

            switch (xcode)
            {
                default:
                case GB_FP32_code   : e =  46 ; break ; // z = tanf (x)
                case GB_FP64_code   : e =  47 ; break ; // z = tan (x)
                case GB_FC32_code   : e =  48 ; break ; // z = GB_tanf (x)
                case GB_FC64_code   : e =  49 ; break ; // z = GB_tan (x)
            }
            break ;

        case GB_ASIN_unop_code      :       // z = asin (x)

            switch (xcode)
            {
                default:
                case GB_FP32_code   : e =  50 ; break ; // z = asinf (x)
                case GB_FP64_code   : e =  51 ; break ; // z = asin (x)
                case GB_FC32_code   : e =  52 ; break ; // z = GB_asinf (x)
                case GB_FC64_code   : e =  53 ; break ; // z = GB_asin (x)
            }
            break ;

        case GB_ACOS_unop_code      :       // z = acos (x)

            switch (xcode)
            {
                default:
                case GB_FP32_code   : e =  54 ; break ; // z = acosf (x)
                case GB_FP64_code   : e =  55 ; break ; // z = acos (x)
                case GB_FC32_code   : e =  56 ; break ; // z = GB_acosf (x)
                case GB_FC64_code   : e =  57 ; break ; // z = GB_acos (x)
            }
            break ;

        case GB_ATAN_unop_code      :       // z = atan (x)

            switch (xcode)
            {
                default:
                case GB_FP32_code   : e =  58 ; break ; // z = atanf (x)
                case GB_FP64_code   : e =  59 ; break ; // z = atan (x)
                case GB_FC32_code   : e =  60 ; break ; // z = GB_atanf (x)
                case GB_FC64_code   : e =  61 ; break ; // z = GB_atan (x)
            }
            break ;

        case GB_SINH_unop_code      :       // z = sinh (x)

            switch (xcode)
            {
                default:
                case GB_FP32_code   : e =  62 ; break ; // z = sinhf (x)
                case GB_FP64_code   : e =  63 ; break ; // z = sinh (x)
                case GB_FC32_code   : e =  64 ; break ; // z = GB_sinhf (x)
                case GB_FC64_code   : e =  65 ; break ; // z = GB_sinh (x)
            }
            break ;

        case GB_COSH_unop_code      :       // z = cosh (x)

            switch (xcode)
            {
                default:
                case GB_FP32_code   : e =  66 ; break ; // z = coshf (x)
                case GB_FP64_code   : e =  67 ; break ; // z = cosh (x)
                case GB_FC32_code   : e =  68 ; break ; // z = GB_coshf (x)
                case GB_FC64_code   : e =  69 ; break ; // z = GB_cosh (x)
            }
            break ;

        case GB_TANH_unop_code      :       // z = tanh (x)

            switch (xcode)
            {
                default:
                case GB_FP32_code   : e =  70 ; break ; // z = tanhf (x)
                case GB_FP64_code   : e =  71 ; break ; // z = tanh (x)
                case GB_FC32_code   : e =  72 ; break ; // z = GB_tanhf (x)
                case GB_FC64_code   : e =  73 ; break ; // z = GB_tanh (x)
            }
            break ;

        case GB_ASINH_unop_code     :       // z = asinh (x)

            switch (xcode)
            {
                default:
                case GB_FP32_code   : e =  74 ; break ; // z = asinhf (x)
                case GB_FP64_code   : e =  75 ; break ; // z = asinh (x)
                case GB_FC32_code   : e =  76 ; break ; // z = GB_asinhf (x)
                case GB_FC64_code   : e =  77 ; break ; // z = GB_asinh (x)
            }
            break ;

        case GB_ACOSH_unop_code     :       // z = acosh (x)

            switch (xcode)
            {
                default:
                case GB_FP32_code   : e =  78 ; break ; // z = acoshf (x)
                case GB_FP64_code   : e =  79 ; break ; // z = acosh (x)
                case GB_FC32_code   : e =  80 ; break ; // z = GB_acoshf (x)
                case GB_FC64_code   : e =  81 ; break ; // z = GB_acosh (x)
            }
            break ;

        case GB_ATANH_unop_code     :       // z = atanh (x)

            switch (xcode)
            {
                default:
                case GB_FP32_code   : e =  82 ; break ; // z = atanhf (x)
                case GB_FP64_code   : e =  83 ; break ; // z = atanh (x)
                case GB_FC32_code   : e =  84 ; break ; // z = GB_atanhf (x)
                case GB_FC64_code   : e =  85 ; break ; // z = GB_atanh (x)
            }
            break ;

        case GB_SIGNUM_unop_code    :       // z = signum (x)

            switch (xcode)
            {
                default:
                case GB_FP32_code   : e =  86 ; break ; // z = GJ_signumf (x)
                case GB_FP64_code   : e =  87 ; break ; // z = GJ_signum (x)
                case GB_FC32_code   : e =  88 ; break ; // z = GJ_csignumf (x)
                case GB_FC64_code   : e =  89 ; break ; // z = GJ_csignum (x)
            }
            break ;

        case GB_CEIL_unop_code      :       // z = ceil (x)

            switch (xcode)
            {
                default:
                case GB_FP32_code   : e =  90 ; break ; // z = ceilf (x)
                case GB_FP64_code   : e =  91 ; break ; // z = ceil (x)
                case GB_FC32_code   : e =  92 ; break ; // z = GJ_cceilf (x)
                case GB_FC64_code   : e =  93 ; break ; // z = GJ_cceil (x)
            }
            break ;

        case GB_FLOOR_unop_code     :       // z = floor (x)

            switch (xcode)
            {
                default:
                case GB_FP32_code   : e =  94 ; break ; // z = floorf (x)
                case GB_FP64_code   : e =  95 ; break ; // z = floor (x)
                case GB_FC32_code   : e =  96 ; break ; // z = GJ_cfloorf (x)
                case GB_FC64_code   : e =  97 ; break ; // z = GJ_cfloor (x)
            }
            break ;

        case GB_ROUND_unop_code     :       // z = round (x)

            switch (xcode)
            {
                default:
                case GB_FP32_code   : e =  98 ; break ; // z = roundf (x)
                case GB_FP64_code   : e =  99 ; break ; // z = round (x)
                case GB_FC32_code   : e = 100 ; break ; // z = GJ_croundf (x)
                case GB_FC64_code   : e = 101 ; break ; // z = GJ_cround (x)
            }
            break ;

        case GB_TRUNC_unop_code     :       // z = trunc (x)

            switch (xcode)
            {
                default:
                case GB_FP32_code   : e = 102 ; break ; // z = truncf (x)
                case GB_FP64_code   : e = 103 ; break ; // z = trunc (x)
                case GB_FC32_code   : e = 104 ; break ; // z = GJ_ctruncf (x)
                case GB_FC64_code   : e = 105 ; break ; // z = GJ_ctrunc (x)
            }
            break ;

        case GB_EXP2_unop_code      :       // z = exp2 (x)

            switch (xcode)
            {
                default:
                case GB_FP32_code   : e = 106 ; break ; // z = exp2f (x)
                case GB_FP64_code   : e = 107 ; break ; // z = exp2 (x)
                case GB_FC32_code   : e = 108 ; break ; // z = GJ_cexp2f (x)
                case GB_FC64_code   : e = 109 ; break ; // z = GJ_cexp2 (x)
            }
            break ;

        case GB_EXPM1_unop_code     :       // z = expm1 (x)

            switch (xcode)
            {
                default:
                case GB_FP32_code   : e = 110 ; break ; // z = expm1f (x)
                case GB_FP64_code   : e = 111 ; break ; // z = expm1 (x)
                case GB_FC32_code   : e = 112 ; break ; // z = GJ_cexpm1f (x)
                case GB_FC64_code   : e = 113 ; break ; // z = GJ_cexpm1 (x)
            }
            break ;

        case GB_LOG10_unop_code     :       // z = log10 (x)

            switch (xcode)
            {
                default:
                case GB_FP32_code   : e = 114 ; break ; // z = log10f (x)
                case GB_FP64_code   : e = 115 ; break ; // z = log10 (x)
                case GB_FC32_code   : e = 116 ; break ; // z = GJ_clog10f (x)
                case GB_FC64_code   : e = 117 ; break ; // z = GJ_clog10 (x)
            }
            break ;

        case GB_LOG1P_unop_code     :       // z = log1P (x)

            switch (xcode)
            {
                default:
                case GB_FP32_code   : e = 118 ; break ; // z = log1pf (x)
                case GB_FP64_code   : e = 119 ; break ; // z = log1p (x)
                case GB_FC32_code   : e = 120 ; break ; // z = GJ_clog1pf (x)
                case GB_FC64_code   : e = 121 ; break ; // z = GJ_clog1p (x)
            }
            break ;

        case GB_LOG2_unop_code      :       // z = log2 (x)

            switch (xcode)
            {
                default:
                case GB_FP32_code   : e = 122 ; break ; // z = log2f (x)
                case GB_FP64_code   : e = 123 ; break ; // z = log2 (x)
                case GB_FC32_code   : e = 124 ; break ; // z = GJ_clog2f (x)
                case GB_FC64_code   : e = 125 ; break ; // z = GJ_clog2 (x)
            }
            break ;

        //----------------------------------------------------------------------
        // unary operators for real floating-point types
        //----------------------------------------------------------------------

        case GB_LGAMMA_unop_code    :       // z = lgamma (x)

            switch (xcode)
            {
                default:
                case GB_FP32_code   : e = 126 ; break ; // z = lgammaf (x)
                case GB_FP64_code   : e = 127 ; break ; // z = lgamma (x)
            }
            break ;

        case GB_TGAMMA_unop_code    :       // z = tgamma (x)

            switch (xcode)
            {
                default:
                case GB_FP32_code   : e = 128 ; break ; // z = tgammaf (x)
                case GB_FP64_code   : e = 129 ; break ; // z = tgamma (x)
            }
            break ;

        case GB_ERF_unop_code       :       // z = erf (x)

            switch (xcode)
            {
                default:
                case GB_FP32_code   : e = 130 ; break ; // z = erff (x)
                case GB_FP64_code   : e = 131 ; break ; // z = erf (x)
            }
            break ;

        case GB_ERFC_unop_code      :       // z = erfc (x)

            switch (xcode)
            {
                default:
                case GB_FP32_code   : e = 132 ; break ; // z = erfcf (x)
                case GB_FP64_code   : e = 133 ; break ; // z = erfc (x)
            }
            break ;

        case GB_CBRT_unop_code      :       // z = cbrt (x)

            switch (xcode)
            {
                default:
                case GB_FP32_code   : e = 134 ; break ; // z = cbrtf (x)
                case GB_FP64_code   : e = 135 ; break ; // z = cbrt (x)
            }
            break ;

        case GB_FREXPX_unop_code    :       // z = frexpx (x), mantissa frexp

            switch (xcode)
            {
                default:
                case GB_FP32_code   : e = 136 ; break ; // z = GJ_frexpxf (x)
                case GB_FP64_code   : e = 137 ; break ; // z = GJ_frexpx (x)
            }
            break ;

        case GB_FREXPE_unop_code    :       // z = frexpe (x), exponent frexp

            switch (xcode)
            {
                default:
                case GB_FP32_code   : e = 138 ; break ; // z = GJ_frexpef (x)
                case GB_FP64_code   : e = 139 ; break ; // z = GJ_frexpe (x)
            }
            break ;

        //----------------------------------------------------------------------
        // unary operators for complex types only
        //----------------------------------------------------------------------

        case GB_CONJ_unop_code      :       // z = GB_conj (x)

            switch (xcode)
            {
                default:
                case GB_FC32_code   : e = 140 ; break ; // z = GB_conjf (x)
                case GB_FC64_code   : e = 141 ; break ; // z = GB_conj (x)
            }
            break ;

        //----------------------------------------------------------------------
        // unary operators where z is real and x is complex
        //----------------------------------------------------------------------

        case GB_CREAL_unop_code     :       // z = creal (x)

            switch (xcode)
            {
                default:
                case GB_FC32_code   : e = 142 ; break ; // z = GB_crealf (x)
                case GB_FC64_code   : e = 143 ; break ; // z = GB_creal (x)
            }
            break ;

        case GB_CIMAG_unop_code     :       // z = cimag (x)

            switch (xcode)
            {
                default:
                case GB_FC32_code   : e = 144 ; break ; // z = GB_cimagf (x)
                case GB_FC64_code   : e = 145 ; break ; // z = GB_cimag (x)
            }
            break ;

        case GB_CARG_unop_code      :       // z = carg (x)

            switch (xcode)
            {
                default:
                case GB_FC32_code   : e = 146 ; break ; // z = GB_cargf (x)
                case GB_FC64_code   : e = 147 ; break ; // z = GB_carg (x)
            }
            break ;

        //----------------------------------------------------------------------
        // unary operators where z is bool and x is any floating-point type
        //----------------------------------------------------------------------

        case GB_ISINF_unop_code     :       // z = isinf (x)

            switch (xcode)
            {
                default:
                case GB_FP32_code   : 
                case GB_FP64_code   : e = 148 ; break ; // z = isinf (x)
                case GB_FC32_code   : e = 149 ; break ; // z = GJ_cisinff (x)
                case GB_FC64_code   : e = 150 ; break ; // z = GJ_cisinf (x)
            }
            break ;

        case GB_ISNAN_unop_code     :       // z = isnan (x)

            switch (xcode)
            {
                default:
                case GB_FP32_code   : 
                case GB_FP64_code   : e = 151 ; break ; // z = isnan (x)
                case GB_FC32_code   : e = 152 ; break ; // z = GJ_cisnanf (x)
                case GB_FC64_code   : e = 153 ; break ; // z = GJ_cisnan (x)
            }
            break ;

        case GB_ISFINITE_unop_code  :       // z = isfinite (x)

            switch (xcode)
            {
                default:
                case GB_FP32_code   : 
                case GB_FP64_code   : e = 154 ; break ; // z = isfinite (x)
                case GB_FC32_code   : e = 155 ; break ; // z = GJ_cisfinitef (x)
                case GB_FC64_code   : e = 156 ; break ; // z = GJ_cisfinite (x)
            }
            break ;

        //----------------------------------------------------------------------
        // positional unary operators: z is int32 or int64, x is ignored
        //----------------------------------------------------------------------

        case GB_POSITIONI_unop_code     :   // z = position_i(A(i,j)) == i
            i_dep = true ;
            e = 157 ; break ;               // z = i

        case GB_POSITIONI1_unop_code    :   // z = position_i1(A(i,j)) == i+1
            i_dep = true ;
            e = 158 ; break ;               // z = i+1

        case GB_POSITIONJ_unop_code     :   // z = position_j(A(i,j)) == j
            j_dep = true ;
            e = 159 ; break ;               // z = j

        case GB_POSITIONJ1_unop_code    :   // z = position_j1(A(i,j)) == j+1
            j_dep = true ;
            e = 160 ; break ;               // z = j+1

        //----------------------------------------------------------------------
        // built-in GrB_IndexUnaryOps that do not depend on x
        //----------------------------------------------------------------------

        // x is the matrix entry A(i,j), y is the thunk value

        case GB_NONZOMBIE_idxunop_code  :   // z = (i >= 0) ;
            i_dep = true ;
            e = 231 ; break ;

        // Result is INT32 or INT64, depending on i and y:
        case GB_ROWINDEX_idxunop_code   :   // z = (i+y)
            i_dep = true ;
            e = 232 ; break ;

        // Result is BOOL, depending on i and y:
        case GB_ROWLE_idxunop_code      :   // z = (i <= y)
            i_dep = true ;
            e = 233 ; break ;

        case GB_ROWGT_idxunop_code      :   // z = (i > y)
            i_dep = true ;
            e = 234 ; break ;

        // Result is INT32 or INT64, depending on j and y:
        case GB_COLINDEX_idxunop_code   :   // z = (j+y)
            j_dep = true ;
            e = 235 ; break ;

        // Result is BOOL, depending on j and y:
        case GB_COLLE_idxunop_code      :   // z = (j <= y)
            j_dep = true ;
            e = 236 ; break ;

        case GB_COLGT_idxunop_code      :   // z = (j > y)
            j_dep = true ;
            e = 237 ; break ;

        // Result is INT32 or INT64, depending on i, j, and y:
        case GB_DIAGINDEX_idxunop_code  :   // z = (j-(i+y))
            i_dep = true ;
            j_dep = true ;
            e = 238 ; break ;

        case GB_FLIPDIAGINDEX_idxunop_code : // z = (i-(j+y))
            i_dep = true ;
            j_dep = true ;
            e = 239 ; break ;

        // Result is BOOL, depending on i, j, and y:
        case GB_TRIL_idxunop_code       :   // z = (j <= (i+y))
            i_dep = true ;
            j_dep = true ;
            e = 240 ; break ;

        case GB_TRIU_idxunop_code       :   // z = (j >= (i+y))
            i_dep = true ;
            j_dep = true ;
            e = 241 ; break ;

        case GB_DIAG_idxunop_code       :   // z = (j == (i+y))
            i_dep = true ;
            j_dep = true ;
            e = 242 ; break ;

        case GB_OFFDIAG_idxunop_code    :   // z = (j != (i+y))
            i_dep = true ;
            j_dep = true ;
            e = 243 ; break ;

        //----------------------------------------------------------------------
        // built-in GrB_IndexUnaryOps that depend on x
        //----------------------------------------------------------------------

        // Result is BOOL, depending on the value x and y:
        case GB_VALUENE_idxunop_code    :   // z = (x != y)

            switch (xcode)
            {
                case GB_FC32_code   : e = 244 ; break ; // GB_FC32_ne (x,y)
                case GB_FC64_code   : e = 245 ; break ; // GB_FC64_ne (x,y)
                default             : e = 246 ; break ; // z = (x != y)
            }
            break ;

        case GB_VALUEEQ_idxunop_code    :   // z = (x == y)

            switch (xcode)
            {
                case GB_FC32_code   : e = 247 ; break ; // GB_FC32_eq (x,y)
                case GB_FC64_code   : e = 248 ; break ; // GB_FC64_eq (x,y)
                default             : e = 249 ; break ; // z = (x == y)
            }
            break ;

        case GB_VALUEGT_idxunop_code    :   // z = (x > y)
            e = 250 ; break ;
        case GB_VALUEGE_idxunop_code    :   // z = (x >= y)
            e = 251 ; break ;
        case GB_VALUELT_idxunop_code    :   // z = (x < y)
            e = 252 ; break ;
        case GB_VALUELE_idxunop_code    :   // z = (x <= y)
            e = 253 ; break ;

        //----------------------------------------------------------------------
        // user-defined GrB_IndexUnaryOp
        //----------------------------------------------------------------------

        case GB_USER_idxunop_code       : 
            i_dep = true ;
            j_dep = true ;
            e = 254 ; break ;               // user-defined GrB_IndexUnaryOp

        default:;
    }

    //--------------------------------------------------------------------------
    // determine dependencies
    //--------------------------------------------------------------------------

    // all IDX ops depend on y, except for NONZOMBIE
    (*depends_on_y) = (e >= 232) ;

    // many operators depend on x:
    (*depends_on_x) = (e == 0)      // user unaryop
        || (e >= 2 && e <= 156)     // all unaryops except 1 and positional
        || (e >= 244) ;             // VALUE ops and user idxop

    // operators that depend on i and/or j are affected by flipij:
    (*depends_on_i) = (flipij) ? j_dep : i_dep ;
    (*depends_on_j) = (flipij) ? i_dep : j_dep ;

    (*ecode) = e ;
}

