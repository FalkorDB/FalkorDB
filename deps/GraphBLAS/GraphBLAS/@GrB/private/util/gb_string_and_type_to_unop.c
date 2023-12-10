//------------------------------------------------------------------------------
// gb_string_and_type_to_unop: get a GraphBLAS operator from a string and type
//------------------------------------------------------------------------------

// SuiteSparse:GraphBLAS, Timothy A. Davis, (c) 2017-2023, All Rights Reserved.
// SPDX-License-Identifier: Apache-2.0

//------------------------------------------------------------------------------

#include "gb_interface.h"

// op_name: a built-in string defining the operator name.

// For all 13 types:
//      identity, ainv, minv, one, abs
//      (for complex, abs returns a real result)

// For all 11 real types: (result is same type as input)
//      lnot

// For 4 floating-point types (real & complex)x(single & double)
//      (result is same type as input):
//      sqrt, log, exp,
//      sin, cos, tan, acos, asin, atan,
//      sinh, cosh, tanh, acosh, asinh, atanh,
//      signum, ceil, floor, round, trunc, exp2, expm1, log10, log1p, log2

// for complex only:
//      creal, cimag, carg  (result is real)
//      conj (result is complex)

// For all 4 floating-point types (result is logical)
//      isinf, isnan, isfinite

// For single and double:
//      lgamma, tgamma, erf, erfc, frexpx, frexpe (result same type as input)

// for integer types only:
//      bitcmp

// for int32 and int64:
//      positioni0, positioni1
//      positionj0, positionj1

// The following equivalent synonyms are available:
//  identity    +       uplus
//  ainv        -       uminus  negate
//  lnot        ~       not
//  one         1
//  creal       real
//  cimag       imag
//  carg        angle
//  lgamma      gammaln
//  tgamma      gamma
//  exp2        pow2
//  i           i1  positioni   positioni1 (since default is 1-based)
//  i0          positioni0
//  j           j1  positionj   positionj1 (since default is 1-based)
//  j0          positionj0

GrB_UnaryOp gb_string_and_type_to_unop  // return op from string and type
(
    const char *op_name,        // name of the operator, as a string
    const GrB_Type type,        // type of the input to the operator
    const bool type_not_given   // true if no type present in the string
)
{

    if (MATCH (op_name, "identity") || MATCH (op_name, "+") ||
        MATCH (op_name, "uplus"))
    { 

        if (type == GrB_BOOL  ) return (GrB_IDENTITY_BOOL  ) ;
        if (type == GrB_INT8  ) return (GrB_IDENTITY_INT8  ) ;
        if (type == GrB_INT16 ) return (GrB_IDENTITY_INT16 ) ;
        if (type == GrB_INT32 ) return (GrB_IDENTITY_INT32 ) ;
        if (type == GrB_INT64 ) return (GrB_IDENTITY_INT64 ) ;
        if (type == GrB_UINT8 ) return (GrB_IDENTITY_UINT8 ) ;
        if (type == GrB_UINT16) return (GrB_IDENTITY_UINT16) ;
        if (type == GrB_UINT32) return (GrB_IDENTITY_UINT32) ;
        if (type == GrB_UINT64) return (GrB_IDENTITY_UINT64) ;
        if (type == GrB_FP32  ) return (GrB_IDENTITY_FP32  ) ;
        if (type == GrB_FP64  ) return (GrB_IDENTITY_FP64  ) ;
        if (type == GxB_FC32  ) return (GxB_IDENTITY_FC32  ) ;
        if (type == GxB_FC64  ) return (GxB_IDENTITY_FC64  ) ;

    }
    else if (MATCH (op_name, "ainv"  ) || MATCH (op_name, "-") ||
             MATCH (op_name, "negate") || MATCH (op_name, "uminus"))
    { 

        if (type == GrB_BOOL  ) return (GrB_AINV_BOOL  ) ;
        if (type == GrB_INT8  ) return (GrB_AINV_INT8  ) ;
        if (type == GrB_INT16 ) return (GrB_AINV_INT16 ) ;
        if (type == GrB_INT32 ) return (GrB_AINV_INT32 ) ;
        if (type == GrB_INT64 ) return (GrB_AINV_INT64 ) ;
        if (type == GrB_UINT8 ) return (GrB_AINV_UINT8 ) ;
        if (type == GrB_UINT16) return (GrB_AINV_UINT16) ;
        if (type == GrB_UINT32) return (GrB_AINV_UINT32) ;
        if (type == GrB_UINT64) return (GrB_AINV_UINT64) ;
        if (type == GrB_FP32  ) return (GrB_AINV_FP32  ) ;
        if (type == GrB_FP64  ) return (GrB_AINV_FP64  ) ;
        if (type == GxB_FC32  ) return (GxB_AINV_FC32  ) ;
        if (type == GxB_FC64  ) return (GxB_AINV_FC64  ) ;

    }
    else if (MATCH (op_name, "minv"))
    { 

        if (type == GrB_BOOL  ) return (GrB_MINV_BOOL  ) ;
        if (type == GrB_INT8  ) return (GrB_MINV_INT8  ) ;
        if (type == GrB_INT16 ) return (GrB_MINV_INT16 ) ;
        if (type == GrB_INT32 ) return (GrB_MINV_INT32 ) ;
        if (type == GrB_INT64 ) return (GrB_MINV_INT64 ) ;
        if (type == GrB_UINT8 ) return (GrB_MINV_UINT8 ) ;
        if (type == GrB_UINT16) return (GrB_MINV_UINT16) ;
        if (type == GrB_UINT32) return (GrB_MINV_UINT32) ;
        if (type == GrB_UINT64) return (GrB_MINV_UINT64) ;
        if (type == GrB_FP32  ) return (GrB_MINV_FP32  ) ;
        if (type == GrB_FP64  ) return (GrB_MINV_FP64  ) ;
        if (type == GxB_FC32  ) return (GxB_MINV_FC32  ) ;
        if (type == GxB_FC64  ) return (GxB_MINV_FC64  ) ;

    }
    else if (MATCH (op_name, "lnot") || MATCH (op_name, "~") ||
             MATCH (op_name, "not"))
    { 

        if (type == GrB_BOOL  ) return (GxB_LNOT_BOOL  ) ;  // == GrB_LNOT
        if (type == GrB_INT8  ) return (GxB_LNOT_INT8  ) ;
        if (type == GrB_INT16 ) return (GxB_LNOT_INT16 ) ;
        if (type == GrB_INT32 ) return (GxB_LNOT_INT32 ) ;
        if (type == GrB_INT64 ) return (GxB_LNOT_INT64 ) ;
        if (type == GrB_UINT8 ) return (GxB_LNOT_UINT8 ) ;
        if (type == GrB_UINT16) return (GxB_LNOT_UINT16) ;
        if (type == GrB_UINT32) return (GxB_LNOT_UINT32) ;
        if (type == GrB_UINT64) return (GxB_LNOT_UINT64) ;
        if (type == GrB_FP32  ) return (GxB_LNOT_FP32  ) ;
        if (type == GrB_FP64  ) return (GxB_LNOT_FP64  ) ;

    }
    else if (MATCH (op_name, "one") || MATCH (op_name, "1"))
    { 

        if (type == GrB_BOOL  ) return (GxB_ONE_BOOL  ) ;
        if (type == GrB_INT8  ) return (GxB_ONE_INT8  ) ;
        if (type == GrB_INT16 ) return (GxB_ONE_INT16 ) ;
        if (type == GrB_INT32 ) return (GxB_ONE_INT32 ) ;
        if (type == GrB_INT64 ) return (GxB_ONE_INT64 ) ;
        if (type == GrB_UINT8 ) return (GxB_ONE_UINT8 ) ;
        if (type == GrB_UINT16) return (GxB_ONE_UINT16) ;
        if (type == GrB_UINT32) return (GxB_ONE_UINT32) ;
        if (type == GrB_UINT64) return (GxB_ONE_UINT64) ;
        if (type == GrB_FP32  ) return (GxB_ONE_FP32  ) ;
        if (type == GrB_FP64  ) return (GxB_ONE_FP64  ) ;
        if (type == GxB_FC32  ) return (GxB_ONE_FC32  ) ;
        if (type == GxB_FC64  ) return (GxB_ONE_FC64  ) ;

    }
    else if (MATCH (op_name, "abs"))
    { 

        if (type == GrB_BOOL  ) return (GrB_ABS_BOOL  ) ;
        if (type == GrB_INT8  ) return (GrB_ABS_INT8  ) ;
        if (type == GrB_INT16 ) return (GrB_ABS_INT16 ) ;
        if (type == GrB_INT32 ) return (GrB_ABS_INT32 ) ;
        if (type == GrB_INT64 ) return (GrB_ABS_INT64 ) ;
        if (type == GrB_UINT8 ) return (GrB_ABS_UINT8 ) ;
        if (type == GrB_UINT16) return (GrB_ABS_UINT16) ;
        if (type == GrB_UINT32) return (GrB_ABS_UINT32) ;
        if (type == GrB_UINT64) return (GrB_ABS_UINT64) ;
        if (type == GrB_FP32  ) return (GrB_ABS_FP32  ) ;
        if (type == GrB_FP64  ) return (GrB_ABS_FP64  ) ;
        if (type == GxB_FC32  ) return (GxB_ABS_FC32  ) ;
        if (type == GxB_FC64  ) return (GxB_ABS_FC64  ) ;

    }
    else if (MATCH (op_name, "sqrt"))
    { 

        if (type == GrB_FP32  ) return (GxB_SQRT_FP32) ;
        if (type == GrB_FP64  ) return (GxB_SQRT_FP64) ;
        if (type == GxB_FC32  ) return (GxB_SQRT_FC32) ;
        if (type == GxB_FC64  ) return (GxB_SQRT_FC64) ;

    }
    else if (MATCH (op_name, "log"))
    { 

        if (type == GrB_FP32  ) return (GxB_LOG_FP32) ;
        if (type == GrB_FP64  ) return (GxB_LOG_FP64) ;
        if (type == GxB_FC32  ) return (GxB_LOG_FC32) ;
        if (type == GxB_FC64  ) return (GxB_LOG_FC64) ;

    }
    else if (MATCH (op_name, "exp"))
    { 

        if (type == GrB_FP32  ) return (GxB_EXP_FP32) ;
        if (type == GrB_FP64  ) return (GxB_EXP_FP64) ;
        if (type == GxB_FC32  ) return (GxB_EXP_FC32) ;
        if (type == GxB_FC64  ) return (GxB_EXP_FC64) ;

    }
    else if (MATCH (op_name, "sin"))
    { 

        if (type == GrB_FP32  ) return (GxB_SIN_FP32) ;
        if (type == GrB_FP64  ) return (GxB_SIN_FP64) ;
        if (type == GxB_FC32  ) return (GxB_SIN_FC32) ;
        if (type == GxB_FC64  ) return (GxB_SIN_FC64) ;

    }
    else if (MATCH (op_name, "cos"))
    { 

        if (type == GrB_FP32  ) return (GxB_COS_FP32) ;
        if (type == GrB_FP64  ) return (GxB_COS_FP64) ;
        if (type == GxB_FC32  ) return (GxB_COS_FC32) ;
        if (type == GxB_FC64  ) return (GxB_COS_FC64) ;

    }
    else if (MATCH (op_name, "tan"))
    { 

        if (type == GrB_FP32  ) return (GxB_TAN_FP32) ;
        if (type == GrB_FP64  ) return (GxB_TAN_FP64) ;
        if (type == GxB_FC32  ) return (GxB_TAN_FC32) ;
        if (type == GxB_FC64  ) return (GxB_TAN_FC64) ;

    }
    else if (MATCH (op_name, "asin"))
    { 

        if (type == GrB_FP32  ) return (GxB_ASIN_FP32) ;
        if (type == GrB_FP64  ) return (GxB_ASIN_FP64) ;
        if (type == GxB_FC32  ) return (GxB_ASIN_FC32) ;
        if (type == GxB_FC64  ) return (GxB_ASIN_FC64) ;

    }
    else if (MATCH (op_name, "acos"))
    { 

        if (type == GrB_FP32  ) return (GxB_ACOS_FP32) ;
        if (type == GrB_FP64  ) return (GxB_ACOS_FP64) ;
        if (type == GxB_FC32  ) return (GxB_ACOS_FC32) ;
        if (type == GxB_FC64  ) return (GxB_ACOS_FC64) ;

    }
    else if (MATCH (op_name, "atan"))
    { 

        if (type == GrB_FP32  ) return (GxB_ATAN_FP32) ;
        if (type == GrB_FP64  ) return (GxB_ATAN_FP64) ;
        if (type == GxB_FC32  ) return (GxB_ATAN_FC32) ;
        if (type == GxB_FC64  ) return (GxB_ATAN_FC64) ;

    }
    else if (MATCH (op_name, "sinh"))
    { 

        if (type == GrB_FP32  ) return (GxB_SINH_FP32) ;
        if (type == GrB_FP64  ) return (GxB_SINH_FP64) ;
        if (type == GxB_FC32  ) return (GxB_SINH_FC32) ;
        if (type == GxB_FC64  ) return (GxB_SINH_FC64) ;

    }
    else if (MATCH (op_name, "cosh"))
    { 

        if (type == GrB_FP32  ) return (GxB_COSH_FP32) ;
        if (type == GrB_FP64  ) return (GxB_COSH_FP64) ;
        if (type == GxB_FC32  ) return (GxB_COSH_FC32) ;
        if (type == GxB_FC64  ) return (GxB_COSH_FC64) ;

    }
    else if (MATCH (op_name, "tanh"))
    { 

        if (type == GrB_FP32  ) return (GxB_TANH_FP32) ;
        if (type == GrB_FP64  ) return (GxB_TANH_FP64) ;
        if (type == GxB_FC32  ) return (GxB_TANH_FC32) ;
        if (type == GxB_FC64  ) return (GxB_TANH_FC64) ;

    }
    else if (MATCH (op_name, "asinh"))
    { 

        if (type == GrB_FP32  ) return (GxB_ASINH_FP32) ;
        if (type == GrB_FP64  ) return (GxB_ASINH_FP64) ;
        if (type == GxB_FC32  ) return (GxB_ASINH_FC32) ;
        if (type == GxB_FC64  ) return (GxB_ASINH_FC64) ;

    }
    else if (MATCH (op_name, "acosh"))
    { 

        if (type == GrB_FP32  ) return (GxB_ACOSH_FP32) ;
        if (type == GrB_FP64  ) return (GxB_ACOSH_FP64) ;
        if (type == GxB_FC32  ) return (GxB_ACOSH_FC32) ;
        if (type == GxB_FC64  ) return (GxB_ACOSH_FC64) ;

    }
    else if (MATCH (op_name, "atanh"))
    { 

        if (type == GrB_FP32  ) return (GxB_ATANH_FP32) ;
        if (type == GrB_FP64  ) return (GxB_ATANH_FP64) ;
        if (type == GxB_FC32  ) return (GxB_ATANH_FC32) ;
        if (type == GxB_FC64  ) return (GxB_ATANH_FC64) ;

    }
    else if (MATCH (op_name, "sign") ||  MATCH (op_name, "signum"))
    { 

        if (type == GrB_FP32  ) return (GxB_SIGNUM_FP32) ;
        if (type == GrB_FP64  ) return (GxB_SIGNUM_FP64) ;
        if (type == GxB_FC32  ) return (GxB_SIGNUM_FC32) ;
        if (type == GxB_FC64  ) return (GxB_SIGNUM_FC64) ;

    }
    else if (MATCH (op_name, "ceil"))
    { 

        if (type == GrB_FP32  ) return (GxB_CEIL_FP32) ;
        if (type == GrB_FP64  ) return (GxB_CEIL_FP64) ;
        if (type == GxB_FC32  ) return (GxB_CEIL_FC32) ;
        if (type == GxB_FC64  ) return (GxB_CEIL_FC64) ;

    }
    else if (MATCH (op_name, "floor"))
    { 

        if (type == GrB_FP32  ) return (GxB_FLOOR_FP32) ;
        if (type == GrB_FP64  ) return (GxB_FLOOR_FP64) ;
        if (type == GxB_FC32  ) return (GxB_FLOOR_FC32) ;
        if (type == GxB_FC64  ) return (GxB_FLOOR_FC64) ;

    }
    else if (MATCH (op_name, "round"))
    { 

        if (type == GrB_FP32  ) return (GxB_ROUND_FP32) ;
        if (type == GrB_FP64  ) return (GxB_ROUND_FP64) ;
        if (type == GxB_FC32  ) return (GxB_ROUND_FC32) ;
        if (type == GxB_FC64  ) return (GxB_ROUND_FC64) ;

    }
    else if (MATCH (op_name, "trunc") || MATCH (op_name, "fix"))
    { 

        if (type == GrB_FP32  ) return (GxB_TRUNC_FP32) ;
        if (type == GrB_FP64  ) return (GxB_TRUNC_FP64) ;
        if (type == GxB_FC32  ) return (GxB_TRUNC_FC32) ;
        if (type == GxB_FC64  ) return (GxB_TRUNC_FC64) ;

    }
    else if (MATCH (op_name, "exp2") || MATCH (op_name, "pow2"))
    { 

        if (type == GrB_FP32  ) return (GxB_EXP2_FP32) ;
        if (type == GrB_FP64  ) return (GxB_EXP2_FP64) ;
        if (type == GxB_FC32  ) return (GxB_EXP2_FC32) ;
        if (type == GxB_FC64  ) return (GxB_EXP2_FC64) ;

    }
    else if (MATCH (op_name, "expm1"))
    { 

        if (type == GrB_FP32  ) return (GxB_EXPM1_FP32) ;
        if (type == GrB_FP64  ) return (GxB_EXPM1_FP64) ;
        if (type == GxB_FC32  ) return (GxB_EXPM1_FC32) ;
        if (type == GxB_FC64  ) return (GxB_EXPM1_FC64) ;

    }
    else if (MATCH (op_name, "log10"))
    { 

        if (type == GrB_FP32  ) return (GxB_LOG10_FP32) ;
        if (type == GrB_FP64  ) return (GxB_LOG10_FP64) ;
        if (type == GxB_FC32  ) return (GxB_LOG10_FC32) ;
        if (type == GxB_FC64  ) return (GxB_LOG10_FC64) ;

    }
    else if (MATCH (op_name, "log1p"))
    { 

        if (type == GrB_FP32  ) return (GxB_LOG1P_FP32) ;
        if (type == GrB_FP64  ) return (GxB_LOG1P_FP64) ;
        if (type == GxB_FC32  ) return (GxB_LOG1P_FC32) ;
        if (type == GxB_FC64  ) return (GxB_LOG1P_FC64) ;

    }
    else if (MATCH (op_name, "log2"))
    { 

        if (type == GrB_FP32  ) return (GxB_LOG2_FP32) ;
        if (type == GrB_FP64  ) return (GxB_LOG2_FP64) ;
        if (type == GxB_FC32  ) return (GxB_LOG2_FC32) ;
        if (type == GxB_FC64  ) return (GxB_LOG2_FC64) ;

    }
    else if (MATCH (op_name, "lgamma") || MATCH (op_name, "gammaln"))
    { 

        if (type == GrB_FP32  ) return (GxB_LGAMMA_FP32) ;
        if (type == GrB_FP64  ) return (GxB_LGAMMA_FP64) ;

    }
    else if (MATCH (op_name, "tgamma") || MATCH (op_name, "gamma"))
    { 

        if (type == GrB_FP32  ) return (GxB_TGAMMA_FP32) ;
        if (type == GrB_FP64  ) return (GxB_TGAMMA_FP64) ;

    }
    else if (MATCH (op_name, "erf"))
    { 

        if (type == GrB_FP32  ) return (GxB_ERF_FP32) ;
        if (type == GrB_FP64  ) return (GxB_ERF_FP64) ;

    }
    else if (MATCH (op_name, "erfc"))
    { 

        if (type == GrB_FP32  ) return (GxB_ERFC_FP32) ;
        if (type == GrB_FP64  ) return (GxB_ERFC_FP64) ;

    }
    else if (MATCH (op_name, "cbrt"))
    { 

        if (type == GrB_FP32  ) return (GxB_CBRT_FP32) ;
        if (type == GrB_FP64  ) return (GxB_CBRT_FP64) ;

    }
    else if (MATCH (op_name, "conj"))
    { 

        if (type == GxB_FC32  ) return (GxB_CONJ_FC32) ;
        if (type == GxB_FC64  ) return (GxB_CONJ_FC64) ;

    }
    else if (MATCH (op_name, "creal") || MATCH (op_name, "real"))
    { 

        if (type == GxB_FC32  ) return (GxB_CREAL_FC32) ;
        if (type == GxB_FC64  ) return (GxB_CREAL_FC64) ;

    }
    else if (MATCH (op_name, "cimag") || MATCH (op_name, "imag"))
    { 

        if (type == GxB_FC32  ) return (GxB_CIMAG_FC32) ;
        if (type == GxB_FC64  ) return (GxB_CIMAG_FC64) ;

    }
    else if (MATCH (op_name, "carg") || MATCH (op_name, "angle"))
    { 

        if (type == GxB_FC32  ) return (GxB_CARG_FC32) ;
        if (type == GxB_FC64  ) return (GxB_CARG_FC64) ;

    }
    else if (MATCH (op_name, "isinf"))
    { 

        if (type == GrB_FP32  ) return (GxB_ISINF_FP32) ;
        if (type == GrB_FP64  ) return (GxB_ISINF_FP64) ;
        if (type == GxB_FC32  ) return (GxB_ISINF_FC32) ;
        if (type == GxB_FC64  ) return (GxB_ISINF_FC64) ;

    }
    else if (MATCH (op_name, "isnan"))
    { 

        if (type == GrB_FP32  ) return (GxB_ISNAN_FP32) ;
        if (type == GrB_FP64  ) return (GxB_ISNAN_FP64) ;
        if (type == GxB_FC32  ) return (GxB_ISNAN_FC32) ;
        if (type == GxB_FC64  ) return (GxB_ISNAN_FC64) ;

    }
    else if (MATCH (op_name, "isfinite"))
    { 

        if (type == GrB_FP32  ) return (GxB_ISFINITE_FP32) ;
        if (type == GrB_FP64  ) return (GxB_ISFINITE_FP64) ;
        if (type == GxB_FC32  ) return (GxB_ISFINITE_FC32) ;
        if (type == GxB_FC64  ) return (GxB_ISFINITE_FC64) ;

    }
    else if (MATCH (op_name, "frexpx"))
    { 

        if (type == GrB_FP32  ) return (GxB_FREXPX_FP32) ;
        if (type == GrB_FP64  ) return (GxB_FREXPX_FP64) ;

    }
    else if (MATCH (op_name, "frexpe"))
    { 

        if (type == GrB_FP32  ) return (GxB_FREXPE_FP32) ;
        if (type == GrB_FP64  ) return (GxB_FREXPE_FP64) ;

    }
    else if (MATCH (op_name, "bitcmp") || MATCH (op_name, "bitnot"))
    { 

        if (type == GrB_INT8  ) return (GrB_BNOT_INT8  ) ;
        if (type == GrB_INT16 ) return (GrB_BNOT_INT16 ) ;
        if (type == GrB_INT32 ) return (GrB_BNOT_INT32 ) ;
        if (type == GrB_INT64 ) return (GrB_BNOT_INT64 ) ;
        if (type == GrB_UINT8 ) return (GrB_BNOT_UINT8 ) ;
        if (type == GrB_UINT16) return (GrB_BNOT_UINT16) ;
        if (type == GrB_UINT32) return (GrB_BNOT_UINT32) ;
        if (type == GrB_UINT64) return (GrB_BNOT_UINT64) ;

    }
    else if (MATCH (op_name, "positioni0") || MATCH (op_name, "i0"))
    { 

        if (type == GrB_INT32) return (GxB_POSITIONI_INT32) ;
        if (type == GrB_INT64
        ||  type_not_given   ) return (GxB_POSITIONI_INT64) ;

    }
    else if (MATCH (op_name, "positioni1") || MATCH (op_name, "i1") ||
             MATCH (op_name, "positioni" ) || MATCH (op_name, "i"))
    { 

        if (type == GrB_INT32) return (GxB_POSITIONI1_INT32) ;
        if (type == GrB_INT64
        ||  type_not_given   ) return (GxB_POSITIONI1_INT64) ;

    }
    else if (MATCH (op_name, "positionj0") || MATCH (op_name, "j0"))
    { 

        if (type == GrB_INT32) return (GxB_POSITIONJ_INT32) ;
        if (type == GrB_INT64
        ||  type_not_given   ) return (GxB_POSITIONJ_INT64) ;

    }
    else if (MATCH (op_name, "positionj1") || MATCH (op_name, "j1") ||
             MATCH (op_name, "positionj" ) || MATCH (op_name, "j"))
    { 

        if (type == GrB_INT32) return (GxB_POSITIONJ1_INT32) ;
        if (type == GrB_INT64
        ||  type_not_given   ) return (GxB_POSITIONJ1_INT64) ;

    }

    //--------------------------------------------------------------------------
    // unknown type or operator
    //--------------------------------------------------------------------------

    // the type can be NULL for positional operators, but no others

    CHECK_ERROR (type == NULL, "unknown type") ;
    ERROR2 ("unknown unary operator", op_name) ;
    return (NULL) ;
}

