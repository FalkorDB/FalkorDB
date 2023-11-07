//------------------------------------------------------------------------------
// GB_macrofy_unop: construct the macro and defn for a unary operator
//------------------------------------------------------------------------------

// SuiteSparse:GraphBLAS, Timothy A. Davis, (c) 2017-2023, All Rights Reserved.
// SPDX-License-Identifier: Apache-2.0

//------------------------------------------------------------------------------

// Create a macro for a unary operator, of the form
//
//      #define GB_UNARYOP(z,x,i,j,y) z = f (x,i,j,y)
//
// if flipij is true:
//
//      #define GB_UNARYOP(z,x,j,i,y) z = f (x,i,j,y)

#include "GB.h"
#include "GB_stringify.h"
#include <ctype.h>

void GB_macrofy_unop
(
    FILE *fp,
    // input:
    const char *macro_name,
    bool flipij,                // if true: op is f(z,x,j,i,y) with ij flipped
    int ecode,
    GB_Operator op              // GrB_UnaryOp or GrB_IndexUnaryOp
)
{

    const char *f = "" ;
    const char *ij = (flipij) ? "j,i" : "i,j" ;

    if (ecode == 0)
    {

        //----------------------------------------------------------------------
        // user-defined GrB_UnaryOp
        //----------------------------------------------------------------------

        ASSERT (op != NULL) ;
        GB_macrofy_defn (fp, 3, op->name, op->defn) ;
        fprintf (fp, "#define %s(z,x,%s,y)  %s (&(z), &(x))\n", macro_name,
            ij, op->name) ;

    }
    else if (ecode == 254)
    {

        //----------------------------------------------------------------------
        // user-defined GrB_IndexUnaryOp
        //----------------------------------------------------------------------

        ASSERT (op != NULL) ;
        GB_macrofy_defn (fp, 3, op->name, op->defn) ;
        fprintf (fp, "#define %s(z,x,%s,y) %s (&(z), &(x), i, j, &(y))\n",
            macro_name, ij, op->name) ;

    }
    else
    { 

        //----------------------------------------------------------------------
        // built-in operator
        //----------------------------------------------------------------------

        switch (ecode)
        {

            //------------------------------------------------------------------
            // primary unary operators x=f(x)
            //------------------------------------------------------------------

            case   1 : f = "z = 1" ;                            break ;

            case   2 : f = "z = x" ;                            break ;

            case   3 : f = "z = GB_FC32_ainv (x)" ;             break ;
            case   4 : f = "z = GB_FC64_ainv (x)" ;             break ;
            case   5 : f = "z = -(x)" ;                         break ;

            case   6 : f = "z = (((x) >= 0) ? (x) : (-(x)))" ;  break ;
            case   7 : f = "z = fabsf (x)" ;                    break ;
            case   8 : f = "z = fabs (x)" ;                     break ;
            case   9 : f = "z = GB_cabsf (x)" ;                 break ;
            case  10 : f = "z = GB_cabs (x)" ;                  break ;

            case  11 : f = "z = GJ_idiv_int8 (1, x)" ;
                GB_macrofy_defn (fp, 0, "GJ_idiv_int8", GJ_idiv_int8_DEFN) ;
                break ;

            case  12 : f = "z = GJ_idiv_int16 (1, x)" ;
                GB_macrofy_defn (fp, 0, "GJ_idiv_int16", GJ_idiv_int16_DEFN) ;
                break ;

            case  13 : f = "z = GJ_idiv_int32 (1, x)" ;
                GB_macrofy_defn (fp, 0, "GJ_idiv_int32", GJ_idiv_int32_DEFN) ;
                break ;

            case  14 : f = "z = GJ_idiv_int64 (1, x)" ;
                GB_macrofy_defn (fp, 0, "GJ_idiv_int64", GJ_idiv_int64_DEFN) ;
                break ;

            case  15 : f = "z = GJ_idiv_uint8 (1, x)" ;
                GB_macrofy_defn (fp, 0, "GJ_idiv_uint8", GJ_idiv_uint8_DEFN) ;
                break ;

            case  16 : f = "z = GJ_idiv_uint16 (1, x)" ;
                GB_macrofy_defn (fp, 0, "GJ_idiv_uint16", GJ_idiv_uint16_DEFN) ;
                break ;

            case  17 : f = "z = GJ_idiv_uint32 (1, x)" ;
                GB_macrofy_defn (fp, 0, "GJ_idiv_uint32", GJ_idiv_uint32_DEFN) ;
                break ;

            case  18 : f = "z = GJ_idiv_uint64 (1, x)" ;
                GB_macrofy_defn (fp, 0, "GJ_idiv_uint64", GJ_idiv_uint64_DEFN) ;
                break ;

            case  19 : f = "z = (1.0F)/(x)" ;               break ;
            case  20 : f = "z = 1./(x)" ;                   break ;

            case  21 : f = "z = GJ_FC32_div (GxB_CMPLXF (1,0), x)" ;
                GB_macrofy_defn (fp, 0, "GJ_FC64_div", GJ_FC64_div_DEFN) ;
                GB_macrofy_defn (fp, 0, "GJ_FC32_div", GJ_FC32_div_DEFN) ;
                break ;

            case  22 : f = "z = GJ_FC64_div (GxB_CMPLX  (1,0), x)" ;
                GB_macrofy_defn (fp, 0, "GJ_FC64_div", GJ_FC64_div_DEFN) ;
                break ;

            case  23 : f = "z = !(x)" ;                     break ;
            case  24 : f = "z = !(x != 0)" ;                break ;

            case  25 : f = "z = ~(x)" ;                     break ;

            //------------------------------------------------------------------
            // unary operators for floating-point types (real and complex)" ;
            //------------------------------------------------------------------

            case  26 : f = "z = sqrtf (x)" ;        break ;
            case  27 : f = "z = sqrt (x)" ;         break ;
            case  28 : f = "z = GB_csqrtf (x)" ;    break ;
            case  29 : f = "z = GB_csqrt (x)" ;     break ;

            case  30 : f = "z = logf (x)" ;         break ;
            case  31 : f = "z = log (x)" ;          break ;
            case  32 : f = "z = GB_clogf (x)" ;     break ;
            case  33 : f = "z = GB_clog (x)" ;      break ;

            case  34 : f = "z = expf (x)" ;         break ;
            case  35 : f = "z = exp (x)" ;          break ;
            case  36 : f = "z = GB_cexpf (x)" ;     break ;
            case  37 : f = "z = GB_cexp (x)" ;      break ;

            case  38 : f = "z = sinf (x)" ;         break ;
            case  39 : f = "z = sin (x)" ;          break ;
            case  40 : f = "z = GB_csinf (x)" ;     break ;
            case  41 : f = "z = GB_csin (x)" ;      break ;

            case  42 : f = "z = cosf (x)" ;         break ;
            case  43 : f = "z = cos (x)" ;          break ;
            case  44 : f = "z = GB_ccosf (x)" ;     break ;
            case  45 : f = "z = GB_ccos (x)" ;      break ;

            case  46 : f = "z = tanf (x)" ;         break ;
            case  47 : f = "z = tan (x)" ;          break ;
            case  48 : f = "z = GB_ctanf (x)" ;     break ;
            case  49 : f = "z = GB_ctan (x)" ;      break ;

            case  50 : f = "z = asinf (x)" ;        break ;
            case  51 : f = "z = asin (x)" ;         break ;
            case  52 : f = "z = GB_casinf (x)" ;    break ;
            case  53 : f = "z = GB_casin (x)" ;     break ;

            case  54 : f = "z = acosf (x)" ;        break ;
            case  55 : f = "z = acos (x)" ;         break ;
            case  56 : f = "z = GB_cacosf (x)" ;    break ;
            case  57 : f = "z = GB_cacos (x)" ;     break ;

            case  58 : f = "z = atanf (x)" ;        break ;
            case  59 : f = "z = atan (x)" ;         break ;
            case  60 : f = "z = GB_catanf (x)" ;    break ;
            case  61 : f = "z = GB_catan (x)" ;     break ;

            case  62 : f = "z = sinhf (x)" ;        break ;
            case  63 : f = "z = sinh (x)" ;         break ;
            case  64 : f = "z = GB_csinhf (x)" ;    break ;
            case  65 : f = "z = GB_csinh (x)" ;     break ;

            case  66 : f = "z = coshf (x)" ;        break ;
            case  67 : f = "z = cosh (x)" ;         break ;
            case  68 : f = "z = GB_ccoshf (x)" ;    break ;
            case  69 : f = "z = GB_ccosh (x)" ;     break ;

            case  70 : f = "z = tanhf (x)" ;        break ;
            case  71 : f = "z = tanh (x)" ;         break ;
            case  72 : f = "z = GB_ctanhf (x)" ;    break ;
            case  73 : f = "z = GB_ctanh (x)" ;     break ;

            case  74 : f = "z = asinhf (x)" ;       break ;
            case  75 : f = "z = asinh (x)" ;        break ;
            case  76 : f = "z = GB_casinhf (x)" ;   break ;
            case  77 : f = "z = GB_casinh (x)" ;    break ;

            case  78 : f = "z = acoshf (x)" ;       break ;
            case  79 : f = "z = acosh (x)" ;        break ;
            case  80 : f = "z = GB_cacoshf (x)" ;   break ;
            case  81 : f = "z = GB_cacosh (x)" ;    break ;

            case  82 : f = "z = atanhf (x)" ;       break ;
            case  83 : f = "z = atanh (x)" ;        break ;
            case  84 : f = "z = GB_catanhf (x)" ;   break ;
            case  85 : f = "z = GB_catanh (x)" ;    break ;

            case  86 : f = "z = GJ_signumf (x)" ;
                GB_macrofy_defn (fp, 0, "GJ_signumf", GJ_signumf_DEFN) ;
                break ;

            case  87 : f = "z = GJ_signum (x)" ;
                GB_macrofy_defn (fp, 0, "GJ_signum", GJ_signum_DEFN) ;
                break ;

            case  88 : f = "z = GJ_csignumf (x)" ;
                GB_macrofy_defn (fp, 0, "GJ_csignumf", GJ_csignumf_DEFN) ;
                break ;

            case  89 : f = "z = GJ_csignum (x)" ;
                GB_macrofy_defn (fp, 0, "GJ_csignum", GJ_csignum_DEFN) ;
                break ;

            case  90 : f = "z = ceilf (x)" ;        break ;
            case  91 : f = "z = ceil (x)" ;         break ;

            case  92 : f = "z = GJ_cceilf (x)" ;
                GB_macrofy_defn (fp, 0, "GJ_cceilf", GJ_cceilf_DEFN) ;
                break ;

            case  93 : f = "z = GJ_cceil (x)" ;
                GB_macrofy_defn (fp, 0, "GJ_cceil", GJ_cceil_DEFN) ;
                break ;

            case  94 : f = "z = floorf (x)" ;       break ;
            case  95 : f = "z = floor (x)" ;        break ;

            case  96 : f = "z = GJ_cfloorf (x)" ;
                GB_macrofy_defn (fp, 0, "GJ_cfloorf", GJ_cfloorf_DEFN) ;
                break ;

            case  97 : f = "z = GJ_cfloor (x)" ;
                GB_macrofy_defn (fp, 0, "GJ_cfloor", GJ_cfloor_DEFN) ;
                break ;

            case  98 : f = "z = roundf (x)" ;       break ;
            case  99 : f = "z = round (x)" ;        break ;

            case 100 : f = "z = GJ_croundf (x)" ;
                GB_macrofy_defn (fp, 0, "GJ_croundf", GJ_croundf_DEFN) ;
                break ;

            case 101 : f = "z = GJ_cround (x)" ;
                GB_macrofy_defn (fp, 0, "GJ_cround", GJ_cround_DEFN) ;
                break ;

            case 102 : f = "z = truncf (x)" ;       break ;
            case 103 : f = "z = trunc (x)" ;        break ;

            case 104 : f = "z = GJ_ctruncf (x)" ;
                GB_macrofy_defn (fp, 0, "GJ_ctruncf", GJ_ctruncf_DEFN) ;
                break ;

            case 105 : f = "z = GJ_ctrunc (x)" ;
                GB_macrofy_defn (fp, 0, "GJ_ctrunc", GJ_ctrunc_DEFN) ;
                break ;

            case 106 : f = "z = exp2f (x)" ;        break ;
            case 107 : f = "z = exp2 (x)" ;         break ;

            case 108 : f = "z = GJ_cexp2f (x)" ;
                GB_macrofy_defn (fp, 0, "GJ_powf", GJ_powf_DEFN) ;
                GB_macrofy_defn (fp, 0, "GJ_FC32_pow", GJ_FC32_pow_DEFN) ;
                GB_macrofy_defn (fp, 0, "GJ_cexp2f", GJ_cexp2f_DEFN) ;
                break ;

            case 109 : f = "z = GJ_cexp2 (x)" ;
                GB_macrofy_defn (fp, 0, "GJ_pow", GJ_pow_DEFN) ;
                GB_macrofy_defn (fp, 0, "GJ_FC64_pow", GJ_FC64_pow_DEFN) ;
                GB_macrofy_defn (fp, 0, "GJ_cexp2", GJ_cexp2_DEFN) ;
                break ;

            case 110 : f = "z = expm1f (x)" ;       break ;
            case 111 : f = "z = expm1 (x)" ;        break ;

            case 112 : f = "z = GJ_cexpm1f (x)" ;
                GB_macrofy_defn (fp, 0, "GJ_cexpm1", GJ_cexpm1_DEFN) ;
                GB_macrofy_defn (fp, 0, "GJ_cexpm1f", GJ_cexpm1f_DEFN) ;
                break ;

            case 113 : f = "z = GJ_cexpm1 (x)" ;
                GB_macrofy_defn (fp, 0, "GJ_cexpm1", GJ_cexpm1_DEFN) ;
                break ;

            case 114 : f = "z = log10f (x)" ;       break ;
            case 115 : f = "z = log10 (x)" ;        break ;

            case 116 : f = "z = GJ_clog10f (x)" ;
                GB_macrofy_defn (fp, 0, "GJ_FC64_div", GJ_FC64_div_DEFN) ;
                GB_macrofy_defn (fp, 0, "GJ_FC32_div", GJ_FC32_div_DEFN) ;
                GB_macrofy_defn (fp, 0, "GJ_clog10f", GJ_clog10f_DEFN) ;
                break ;

            case 117 : f = "z = GJ_clog10 (x)" ;
                GB_macrofy_defn (fp, 0, "GJ_FC64_div", GJ_FC64_div_DEFN) ;
                GB_macrofy_defn (fp, 0, "GJ_clog10", GJ_clog10_DEFN) ;
                break ;

            case 118 : f = "z = log1pf (x)" ;       break ;
            case 119 : f = "z = log1p (x)" ;        break ;

            case 120 : f = "z = GJ_clog1pf (x)" ;
                GB_macrofy_defn (fp, 0, "GJ_clog1p", GJ_clog1p_DEFN) ;
                GB_macrofy_defn (fp, 0, "GJ_clog1pf", GJ_clog1pf_DEFN) ;
                break ;

            case 121 : f = "z = GJ_clog1p (x)" ;
                GB_macrofy_defn (fp, 0, "GJ_clog1p", GJ_clog1p_DEFN) ;
                break ;

            case 122 : f = "z = log2f (x)" ;        break ;
            case 123 : f = "z = log2 (x)" ;         break ;

            case 124 : f = "z = GJ_clog2f (x)" ;
                GB_macrofy_defn (fp, 0, "GJ_FC64_div", GJ_FC64_div_DEFN) ;
                GB_macrofy_defn (fp, 0, "GJ_FC32_div", GJ_FC32_div_DEFN) ;
                GB_macrofy_defn (fp, 0, "GJ_clog2f", GJ_clog2f_DEFN) ;
                break ;

            case 125 : f = "z = GJ_clog2 (x)" ;
                GB_macrofy_defn (fp, 0, "GJ_FC64_div", GJ_FC64_div_DEFN) ;
                GB_macrofy_defn (fp, 0, "GJ_clog2", GJ_clog2_DEFN) ;
                break ;

            //------------------------------------------------------------------
            // unary operators for real floating-point types
            //------------------------------------------------------------------

            case 126 : f = "z = lgammaf (x)" ;      break ;
            case 127 : f = "z = lgamma (x)" ;       break ;

            case 128 : f = "z = tgammaf (x)" ;      break ;
            case 129 : f = "z = tgamma (x)" ;       break ;

            case 130 : f = "z = erff (x)" ;         break ;
            case 131 : f = "z = erf (x)" ;          break ;

            case 132 : f = "z = erfcf (x)" ;        break ;
            case 133 : f = "z = erfc (x)" ;         break ;

            case 134 : f = "z = cbrtf (x)" ;        break ;
            case 135 : f = "z = cbrt (x)" ;         break ;

            case 136 : f = "z = GJ_frexpxf (x)" ;
                GB_macrofy_defn (fp, 0, "GJ_frexpxf", GJ_frexpxf_DEFN) ;
                break ;

            case 137 : f = "z = GJ_frexpx (x)" ;
                GB_macrofy_defn (fp, 0, "GJ_frexpx", GJ_frexpx_DEFN) ;
                break ;

            case 138 : f = "z = GJ_frexpef (x)" ;
                GB_macrofy_defn (fp, 0, "GJ_frexpef", GJ_frexpef_DEFN) ;
                break ;

            case 139 : f = "z = GJ_frexpe (x)" ;
                GB_macrofy_defn (fp, 0, "GJ_frexpe", GJ_frexpe_DEFN) ;
                break ;

            //------------------------------------------------------------------
            // unary operators for complex types only
            //------------------------------------------------------------------

            case 140 : f = "z = GB_conjf (x)" ;     break ;
            case 141 : f = "z = GB_conj (x)" ;      break ;

            //------------------------------------------------------------------
            // unary operators where z is real and x is complex
            //------------------------------------------------------------------

            case 142 : f = "z = GB_crealf (x)" ;    break ;
            case 143 : f = "z = GB_creal (x)" ;     break ;
            case 144 : f = "z = GB_cimagf (x)" ;    break ;
            case 145 : f = "z = GB_cimag (x)" ;     break ;
            case 146 : f = "z = GB_cargf (x)" ;     break ;
            case 147 : f = "z = GB_carg (x)" ;      break ;

            //------------------------------------------------------------------
            // unary operators where z is bool and x is any floating-point type
            //------------------------------------------------------------------

            case 148 : f = "z = isinf (x)" ;            break ;

            case 149 : f = "z = GJ_cisinff (x)" ;
                GB_macrofy_defn (fp, 0, "GJ_cisinff", GJ_cisinff_DEFN) ;
                break ;

            case 150 : f = "z = GJ_cisinf (x)" ;
                GB_macrofy_defn (fp, 0, "GJ_cisinf", GJ_cisinf_DEFN) ;
                break ;

            case 151 : f = "z = isnan (x)" ;            break ;

            case 152 : f = "z = GJ_cisnanf (x)" ;
                GB_macrofy_defn (fp, 0, "GJ_cisnanf", GJ_cisnanf_DEFN) ;
                break ;

            case 153 : f = "z = GJ_cisnan (x)" ;
                GB_macrofy_defn (fp, 0, "GJ_cisnan", GJ_cisnan_DEFN) ;
                break ;

            case 154 : f = "z = isfinite (x)" ;         break ;

            case 155 : f = "z = GJ_cisfinitef (x)" ;
                GB_macrofy_defn (fp, 0, "GJ_cisfinitef", GJ_cisfinitef_DEFN) ;
                break ;

            case 156 : f = "z = GJ_cisfinite (x)" ;
                GB_macrofy_defn (fp, 0, "GJ_cisfinite", GJ_cisfinite_DEFN) ;
                break ;

            //------------------------------------------------------------------
            // positional unary operators: z is int32 or int64, x is ignored
            //------------------------------------------------------------------

            case 157 : f = "z = (i)" ;                  break ;
            case 158 : f = "z = (i) + 1" ;              break ;
            case 159 : f = "z = (j)" ;                  break ;
            case 160 : f = "z = (j) + 1" ;              break ;

            //------------------------------------------------------------------
            // IndexUnaryOps
            //------------------------------------------------------------------

            case 231 : f = "z = (i >= 0)" ;             break ;

            case 232 : f = "z = ((i) + (y))" ;          break ;
            case 233 : f = "z = ((i) <= (y))" ;         break ;
            case 234 : f = "z = ((i) > (y))" ;          break ;

            case 235 : f = "z = ((j) + (y))" ;          break ;
            case 236 : f = "z = ((j) <= (y))" ;         break ;
            case 237 : f = "z = ((j) > (y))" ;          break ;

            case 238 : f = "z = ((j) - ((i) + (y)))" ;  break ;
            case 239 : f = "z = ((i) - ((j) + (y)))" ;  break ;
            case 240 : f = "z = ((j) <= ((i) + (y)))" ; break ;
            case 241 : f = "z = ((j) >= ((i) + (y)))" ; break ;
            case 242 : f = "z = ((j) == ((i) + (y)))" ; break ;
            case 243 : f = "z = ((j) != ((i) + (y)))" ; break ;

            case 244 : f = "z = GB_FC32_ne (x,y)" ;     break ;
            case 245 : f = "z = GB_FC64_ne (x,y)" ;     break ;
            case 246 : f = "z = ((x) != (y))" ;         break ;

            case 247 : f = "z = GB_FC32_eq (x,y)" ;     break ;
            case 248 : f = "z = GB_FC64_eq (x,y)" ;     break ;
            case 249 : f = "z = ((x) == (y))" ;         break ;

            case 250 : f = "z = ((x) > (y))" ;          break ;
            case 251 : f = "z = ((x) >= (y))" ;         break ;
            case 252 : f = "z = ((x) < (y))" ;          break ;
            case 253 : f = "z = ((x) <= (y))" ;         break ;

            default: ;
        }

        //----------------------------------------------------------------------
        // create the macro
        //----------------------------------------------------------------------

        fprintf (fp, "#define %s(z,x,%s,y) %s\n", macro_name, ij, f) ;
    }
}

