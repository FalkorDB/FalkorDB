//------------------------------------------------------------------------------
// GB_op_name_get: get the user_name of any operator
//------------------------------------------------------------------------------

// SuiteSparse:GraphBLAS, Timothy A. Davis, (c) 2017-2023, All Rights Reserved.
// SPDX-License-Identifier: Apache-2.0

//------------------------------------------------------------------------------

// Three operators are left unnamed:  identity_udt, 1st_udt, and 2nd_udt.
// These are created by GB_unop_identity, GB_reduce_to_vector, and
// GB_binop_second, and do not exist outside GraphBLAS.  The user application
// cannot pass them to GrB_get.

#include "get_set/GB_get_set.h"

const char *GB_op_name_get (GB_Operator op)
{

    GB_Opcode opcode = op->opcode ;
    GB_Type_code xcode = (op->xtype == NULL) ? 0 : op->xtype->code ;
    GB_Type_code zcode = (op->ztype == NULL) ? 0 : op->ztype->code ;

    switch (opcode)
    {

        case GB_NOP_code : return ("GxB_IGNORE_DUP") ;

        //----------------------------------------------------------------------
        // unary operators
        //----------------------------------------------------------------------

        case GB_ONE_unop_code        :  // z = 1

            switch (xcode)
            {
                case GB_BOOL_code    : return ("GxB_ONE_BOOL"  ) ;
                case GB_INT8_code    : return ("GxB_ONE_INT8"  ) ;
                case GB_INT16_code   : return ("GxB_ONE_INT16" ) ;
                case GB_INT32_code   : return ("GxB_ONE_INT32" ) ;
                case GB_INT64_code   : return ("GxB_ONE_INT64" ) ;
                case GB_UINT8_code   : return ("GxB_ONE_UINT8" ) ;
                case GB_UINT16_code  : return ("GxB_ONE_UINT16") ;
                case GB_UINT32_code  : return ("GxB_ONE_UINT32") ;
                case GB_UINT64_code  : return ("GxB_ONE_UINT64") ;
                case GB_FP32_code    : return ("GxB_ONE_FP32"  ) ;
                case GB_FP64_code    : return ("GxB_ONE_FP64"  ) ;
                case GB_FC32_code    : return ("GxB_ONE_FC32"  ) ;
                case GB_FC64_code    : return ("GxB_ONE_FC64"  ) ;
                default :;
            }
            break ;

        case GB_IDENTITY_unop_code   :  // z = x

            switch (xcode)
            {
                case GB_BOOL_code    : return ("GrB_IDENTITY_BOOL"  ) ;
                case GB_INT8_code    : return ("GrB_IDENTITY_INT8"  ) ;
                case GB_INT16_code   : return ("GrB_IDENTITY_INT16" ) ;
                case GB_INT32_code   : return ("GrB_IDENTITY_INT32" ) ;
                case GB_INT64_code   : return ("GrB_IDENTITY_INT64" ) ;
                case GB_UINT8_code   : return ("GrB_IDENTITY_UINT8" ) ;
                case GB_UINT16_code  : return ("GrB_IDENTITY_UINT16") ;
                case GB_UINT32_code  : return ("GrB_IDENTITY_UINT32") ;
                case GB_UINT64_code  : return ("GrB_IDENTITY_UINT64") ;
                case GB_FP32_code    : return ("GrB_IDENTITY_FP32"  ) ;
                case GB_FP64_code    : return ("GrB_IDENTITY_FP64"  ) ;
                case GB_FC32_code    : return ("GxB_IDENTITY_FC32"  ) ;
                case GB_FC64_code    : return ("GxB_IDENTITY_FC64"  ) ;
                // see GB_unop_identity:
//              case GB_UDT_code     :
//                  return ("identity_udt") ;
                default :;
            }
            break ;

        case GB_AINV_unop_code       :  // z = -x

            switch (xcode)
            {
                case GB_BOOL_code    : return ("GrB_AINV_BOOL"  ) ;
                case GB_INT8_code    : return ("GrB_AINV_INT8"  ) ;
                case GB_INT16_code   : return ("GrB_AINV_INT16" ) ;
                case GB_INT32_code   : return ("GrB_AINV_INT32" ) ;
                case GB_INT64_code   : return ("GrB_AINV_INT64" ) ;
                case GB_UINT8_code   : return ("GrB_AINV_UINT8" ) ;
                case GB_UINT16_code  : return ("GrB_AINV_UINT16") ;
                case GB_UINT32_code  : return ("GrB_AINV_UINT32") ;
                case GB_UINT64_code  : return ("GrB_AINV_UINT64") ;
                case GB_FP32_code    : return ("GrB_AINV_FP32"  ) ;
                case GB_FP64_code    : return ("GrB_AINV_FP64"  ) ;
                case GB_FC32_code    : return ("GxB_AINV_FC32"  ) ;
                case GB_FC64_code    : return ("GxB_AINV_FC64"  ) ;
                default :;
            }
            break ;

        case GB_ABS_unop_code        :  // z = abs(x)

            switch (xcode)
            {
                case GB_BOOL_code    : return ("GrB_ABS_BOOL"  ) ;
                case GB_INT8_code    : return ("GrB_ABS_INT8"  ) ;
                case GB_INT16_code   : return ("GrB_ABS_INT16" ) ;
                case GB_INT32_code   : return ("GrB_ABS_INT32" ) ;
                case GB_INT64_code   : return ("GrB_ABS_INT64" ) ;
                case GB_UINT8_code   : return ("GrB_ABS_UINT8" ) ;
                case GB_UINT16_code  : return ("GrB_ABS_UINT16") ;
                case GB_UINT32_code  : return ("GrB_ABS_UINT32") ;
                case GB_UINT64_code  : return ("GrB_ABS_UINT64") ;
                case GB_FP32_code    : return ("GrB_ABS_FP32"  ) ;
                case GB_FP64_code    : return ("GrB_ABS_FP64"  ) ;
                case GB_FC32_code    : return ("GxB_ABS_FC32"  ) ;
                case GB_FC64_code    : return ("GxB_ABS_FC64"  ) ;
                default :;
            }
            break ;

        case GB_MINV_unop_code       :  // z = 1/x

            switch (xcode)
            {
                case GB_BOOL_code    : return ("GrB_MINV_BOOL"  ) ;
                case GB_INT8_code    : return ("GrB_MINV_INT8"  ) ;
                case GB_INT16_code   : return ("GrB_MINV_INT16" ) ;
                case GB_INT32_code   : return ("GrB_MINV_INT32" ) ;
                case GB_INT64_code   : return ("GrB_MINV_INT64" ) ;
                case GB_UINT8_code   : return ("GrB_MINV_UINT8" ) ;
                case GB_UINT16_code  : return ("GrB_MINV_UINT16") ;
                case GB_UINT32_code  : return ("GrB_MINV_UINT32") ;
                case GB_UINT64_code  : return ("GrB_MINV_UINT64") ;
                case GB_FP32_code    : return ("GrB_MINV_FP32"  ) ;
                case GB_FP64_code    : return ("GrB_MINV_FP64"  ) ;
                case GB_FC32_code    : return ("GxB_MINV_FC32"  ) ;
                case GB_FC64_code    : return ("GxB_MINV_FC64"  ) ;
                default :;
            }
            break ;

        case GB_LNOT_unop_code       :  // z = !x

            switch (xcode)
            {
                case GB_BOOL_code    : return ("GrB_LNOT"       ) ;
                case GB_INT8_code    : return ("GxB_LNOT_INT8"  ) ;
                case GB_INT16_code   : return ("GxB_LNOT_INT16" ) ;
                case GB_INT32_code   : return ("GxB_LNOT_INT32" ) ;
                case GB_INT64_code   : return ("GxB_LNOT_INT64" ) ;
                case GB_UINT8_code   : return ("GxB_LNOT_UINT8" ) ;
                case GB_UINT16_code  : return ("GxB_LNOT_UINT16") ;
                case GB_UINT32_code  : return ("GxB_LNOT_UINT32") ;
                case GB_UINT64_code  : return ("GxB_LNOT_UINT64") ;
                case GB_FP32_code    : return ("GxB_LNOT_FP32"  ) ;
                case GB_FP64_code    : return ("GxB_LNOT_FP64"  ) ;
                default :;
            }
            break ;

        case GB_BNOT_unop_code       :  // z = ~x

            switch (xcode)
            {
                case GB_INT8_code    : return ("GrB_BNOT_INT8"  ) ;
                case GB_INT16_code   : return ("GrB_BNOT_INT16" ) ;
                case GB_INT32_code   : return ("GrB_BNOT_INT32" ) ;
                case GB_INT64_code   : return ("GrB_BNOT_INT64" ) ;
                case GB_UINT8_code   : return ("GrB_BNOT_UINT8" ) ;
                case GB_UINT16_code  : return ("GrB_BNOT_UINT16") ;
                case GB_UINT32_code  : return ("GrB_BNOT_UINT32") ;
                case GB_UINT64_code  : return ("GrB_BNOT_UINT64") ;
                default :;
            }
            break ;

        case GB_SQRT_unop_code       :  // z = sqrt (x)

            switch (xcode)
            {
                case GB_FP32_code    : return ("GxB_SQRT_FP32") ;
                case GB_FP64_code    : return ("GxB_SQRT_FP64") ;
                case GB_FC32_code    : return ("GxB_SQRT_FC32") ;
                case GB_FC64_code    : return ("GxB_SQRT_FC64") ;
                default :;
            }
            break ;

        case GB_LOG_unop_code        :  // z = log (x)

            switch (xcode)
            {
                case GB_FP32_code    : return ("GxB_LOG_FP32") ;
                case GB_FP64_code    : return ("GxB_LOG_FP64") ;
                case GB_FC32_code    : return ("GxB_LOG_FC32") ;
                case GB_FC64_code    : return ("GxB_LOG_FC64") ;
                default :;
            }
            break ;

        case GB_EXP_unop_code        :  // z = exp (x)

            switch (xcode)
            {
                case GB_FP32_code    : return ("GxB_EXP_FP32") ;
                case GB_FP64_code    : return ("GxB_EXP_FP64") ;
                case GB_FC32_code    : return ("GxB_EXP_FC32") ;
                case GB_FC64_code    : return ("GxB_EXP_FC64") ;
                default :;
            }
            break ;

        case GB_SIN_unop_code        :  // z = sin (x)

            switch (xcode)
            {
                case GB_FP32_code    : return ("GxB_SIN_FP32") ;
                case GB_FP64_code    : return ("GxB_SIN_FP64") ;
                case GB_FC32_code    : return ("GxB_SIN_FC32") ;
                case GB_FC64_code    : return ("GxB_SIN_FC64") ;
                default :;
            }
            break ;

        case GB_COS_unop_code        :  // z = cos (x)

            switch (xcode)
            {
                case GB_FP32_code    : return ("GxB_COS_FP32") ;
                case GB_FP64_code    : return ("GxB_COS_FP64") ;
                case GB_FC32_code    : return ("GxB_COS_FC32") ;
                case GB_FC64_code    : return ("GxB_COS_FC64") ;
                default :;
            }
            break ;

        case GB_TAN_unop_code        :  // z = tan (x)

            switch (xcode)
            {
                case GB_FP32_code    : return ("GxB_TAN_FP32") ;
                case GB_FP64_code    : return ("GxB_TAN_FP64") ;
                case GB_FC32_code    : return ("GxB_TAN_FC32") ;
                case GB_FC64_code    : return ("GxB_TAN_FC64") ;
                default :;
            }
            break ;

        case GB_ASIN_unop_code       :  // z = asin (x)

            switch (xcode)
            {
                case GB_FP32_code    : return ("GxB_ASIN_FP32") ;
                case GB_FP64_code    : return ("GxB_ASIN_FP64") ;
                case GB_FC32_code    : return ("GxB_ASIN_FC32") ;
                case GB_FC64_code    : return ("GxB_ASIN_FC64") ;
                default :;
            }
            break ;

        case GB_ACOS_unop_code       :  // z = acos (x)

            switch (xcode)
            {
                case GB_FP32_code    : return ("GxB_ACOS_FP32") ;
                case GB_FP64_code    : return ("GxB_ACOS_FP64") ;
                case GB_FC32_code    : return ("GxB_ACOS_FC32") ;
                case GB_FC64_code    : return ("GxB_ACOS_FC64") ;
                default :;
            }
            break ;

        case GB_ATAN_unop_code       :  // z = atan (x)

            switch (xcode)
            {
                case GB_FP32_code    : return ("GxB_ATAN_FP32") ;
                case GB_FP64_code    : return ("GxB_ATAN_FP64") ;
                case GB_FC32_code    : return ("GxB_ATAN_FC32") ;
                case GB_FC64_code    : return ("GxB_ATAN_FC64") ;
                default :;
            }
            break ;

        case GB_SINH_unop_code       :  // z = sinh (x)

            switch (xcode)
            {
                case GB_FP32_code    : return ("GxB_SINH_FP32") ;
                case GB_FP64_code    : return ("GxB_SINH_FP64") ;
                case GB_FC32_code    : return ("GxB_SINH_FC32") ;
                case GB_FC64_code    : return ("GxB_SINH_FC64") ;
                default :;
            }
            break ;

        case GB_COSH_unop_code       :  // z = cosh (x)

            switch (xcode)
            {
                case GB_FP32_code    : return ("GxB_COSH_FP32") ;
                case GB_FP64_code    : return ("GxB_COSH_FP64") ;
                case GB_FC32_code    : return ("GxB_COSH_FC32") ;
                case GB_FC64_code    : return ("GxB_COSH_FC64") ;
                default :;
            }
            break ;

        case GB_TANH_unop_code       :  // z = tanh (x)

            switch (xcode)
            {
                case GB_FP32_code    : return ("GxB_TANH_FP32") ;
                case GB_FP64_code    : return ("GxB_TANH_FP64") ;
                case GB_FC32_code    : return ("GxB_TANH_FC32") ;
                case GB_FC64_code    : return ("GxB_TANH_FC64") ;
                default :;
            }
            break ;

        case GB_ASINH_unop_code      :  // z = asinh (x)

            switch (xcode)
            {
                case GB_FP32_code    : return ("GxB_ASINH_FP32") ;
                case GB_FP64_code    : return ("GxB_ASINH_FP64") ;
                case GB_FC32_code    : return ("GxB_ASINH_FC32") ;
                case GB_FC64_code    : return ("GxB_ASINH_FC64") ;
                default :;
            }
            break ;

        case GB_ACOSH_unop_code      :  // z = acosh (x)

            switch (xcode)
            {
                case GB_FP32_code    : return ("GxB_ACOSH_FP32") ;
                case GB_FP64_code    : return ("GxB_ACOSH_FP64") ;
                case GB_FC32_code    : return ("GxB_ACOSH_FC32") ;
                case GB_FC64_code    : return ("GxB_ACOSH_FC64") ;
                default :;
            }
            break ;

        case GB_ATANH_unop_code      :  // z = atanh (x)

            switch (xcode)
            {
                case GB_FP32_code    : return ("GxB_ATANH_FP32") ;
                case GB_FP64_code    : return ("GxB_ATANH_FP64") ;
                case GB_FC32_code    : return ("GxB_ATANH_FC32") ;
                case GB_FC64_code    : return ("GxB_ATANH_FC64") ;
                default :;
            }
            break ;

        case GB_SIGNUM_unop_code     :  // z = signum (x)

            switch (xcode)
            {
                case GB_FP32_code    : return ("GxB_SIGNUM_FP32") ;
                case GB_FP64_code    : return ("GxB_SIGNUM_FP64") ;
                case GB_FC32_code    : return ("GxB_SIGNUM_FC32") ;
                case GB_FC64_code    : return ("GxB_SIGNUM_FC64") ;
                default :;
            }
            break ;

        case GB_CEIL_unop_code       :  // z = ceil (x)

            switch (xcode)
            {
                case GB_FP32_code    : return ("GxB_CEIL_FP32") ;
                case GB_FP64_code    : return ("GxB_CEIL_FP64") ;
                case GB_FC32_code    : return ("GxB_CEIL_FC32") ;
                case GB_FC64_code    : return ("GxB_CEIL_FC64") ;
                default :;
            }
            break ;

        case GB_FLOOR_unop_code      :  // z = floor (x)

            switch (xcode)
            {
                case GB_FP32_code    : return ("GxB_FLOOR_FP32") ;
                case GB_FP64_code    : return ("GxB_FLOOR_FP64") ;
                case GB_FC32_code    : return ("GxB_FLOOR_FC32") ;
                case GB_FC64_code    : return ("GxB_FLOOR_FC64") ;
                default :;
            }
            break ;

        case GB_ROUND_unop_code      :  // z = round (x)

            switch (xcode)
            {
                case GB_FP32_code    : return ("GxB_ROUND_FP32") ;
                case GB_FP64_code    : return ("GxB_ROUND_FP64") ;
                case GB_FC32_code    : return ("GxB_ROUND_FC32") ;
                case GB_FC64_code    : return ("GxB_ROUND_FC64") ;
                default :;
            }
            break ;

        case GB_TRUNC_unop_code      :  // z = trunc (x)

            switch (xcode)
            {
                case GB_FP32_code    : return ("GxB_TRUNC_FP32") ;
                case GB_FP64_code    : return ("GxB_TRUNC_FP64") ;
                case GB_FC32_code    : return ("GxB_TRUNC_FC32") ;
                case GB_FC64_code    : return ("GxB_TRUNC_FC64") ;
                default :;
            }
            break ;

        case GB_EXP2_unop_code       :  // z = exp2 (x)

            switch (xcode)
            {
                case GB_FP32_code    : return ("GxB_EXP2_FP32") ;
                case GB_FP64_code    : return ("GxB_EXP2_FP64") ;
                case GB_FC32_code    : return ("GxB_EXP2_FC32") ;
                case GB_FC64_code    : return ("GxB_EXP2_FC64") ;
                default :;
            }
            break ;

        case GB_EXPM1_unop_code      :  // z = expm1 (x)

            switch (xcode)
            {
                case GB_FP32_code    : return ("GxB_EXPM1_FP32") ;
                case GB_FP64_code    : return ("GxB_EXPM1_FP64") ;
                case GB_FC32_code    : return ("GxB_EXPM1_FC32") ;
                case GB_FC64_code    : return ("GxB_EXPM1_FC64") ;
                default :;
            }
            break ;

        case GB_LOG10_unop_code      :  // z = log10 (x)

            switch (xcode)
            {
                case GB_FP32_code    : return ("GxB_LOG10_FP32") ;
                case GB_FP64_code    : return ("GxB_LOG10_FP64") ;
                case GB_FC32_code    : return ("GxB_LOG10_FC32") ;
                case GB_FC64_code    : return ("GxB_LOG10_FC64") ;
                default :;
            }
            break ;

        case GB_LOG1P_unop_code      :  // z = log1P (x)

            switch (xcode)
            {
                case GB_FP32_code    : return ("GxB_LOG1P_FP32") ;
                case GB_FP64_code    : return ("GxB_LOG1P_FP64") ;
                case GB_FC32_code    : return ("GxB_LOG1P_FC32") ;
                case GB_FC64_code    : return ("GxB_LOG1P_FC64") ;
                default :;
            }
            break ;

        case GB_LOG2_unop_code       :  // z = log2 (x)

            switch (xcode)
            {
                case GB_FP32_code    : return ("GxB_LOG2_FP32") ;
                case GB_FP64_code    : return ("GxB_LOG2_FP64") ;
                case GB_FC32_code    : return ("GxB_LOG2_FC32") ;
                case GB_FC64_code    : return ("GxB_LOG2_FC64") ;
                default :;
            }
            break ;

        case GB_LGAMMA_unop_code     :  // z = lgamma (x)

            switch (xcode)
            {
                case GB_FP32_code    : return ("GxB_LGAMMA_FP32") ;
                case GB_FP64_code    : return ("GxB_LGAMMA_FP64") ;
                default :;
            }
            break ;

        case GB_TGAMMA_unop_code     :  // z = tgamma (x)

            switch (xcode)
            {
                case GB_FP32_code    : return ("GxB_TGAMMA_FP32") ;
                case GB_FP64_code    : return ("GxB_TGAMMA_FP64") ;
                default :;
            }
            break ;

        case GB_ERF_unop_code        :  // z = erf (x)

            switch (xcode)
            {
                case GB_FP32_code    : return ("GxB_ERF_FP32") ;
                case GB_FP64_code    : return ("GxB_ERF_FP64") ;
                default :;
            }
            break ;

        case GB_ERFC_unop_code       :  // z = erfc (x)

            switch (xcode)
            {
                case GB_FP32_code    : return ("GxB_ERFC_FP32") ;
                case GB_FP64_code    : return ("GxB_ERFC_FP64") ;
                default :;
            }
            break ;

        case GB_CBRT_unop_code       :  // z = cbrt (x)

            switch (xcode)
            {
                case GB_FP32_code    : return ("GxB_CBRT_FP32") ;
                case GB_FP64_code    : return ("GxB_CBRT_FP64") ;
                default :;
            }
            break ;

        case GB_FREXPX_unop_code     :  // z = frexpx (x)

            switch (xcode)
            {
                case GB_FP32_code    : return ("GxB_FREXPX_FP32") ;
                case GB_FP64_code    : return ("GxB_FREXPX_FP64") ;
                default :;
            }
            break ;

        case GB_FREXPE_unop_code     :  // z = frexpe (x)

            switch (xcode)
            {
                case GB_FP32_code    : return ("GxB_FREXPE_FP32") ;
                case GB_FP64_code    : return ("GxB_FREXPE_FP64") ;
                default :;
            }
            break ;

        case GB_CONJ_unop_code       :  // z = conj (x)

            switch (xcode)
            {
                case GB_FC32_code    : return ("GxB_CONJ_FC32") ;
                case GB_FC64_code    : return ("GxB_CONJ_FC64") ;
                default :;
            }
            break ;

        case GB_CREAL_unop_code      :  // z = creal (x)

            switch (xcode)
            {
                case GB_FC32_code    : return ("GxB_CREAL_FC32") ;
                case GB_FC64_code    : return ("GxB_CREAL_FC64") ;
                default :;
            }
            break ;

        case GB_CIMAG_unop_code      :  // z = cimag (x)

            switch (xcode)
            {
                case GB_FC32_code    : return ("GxB_CIMAG_FC32") ;
                case GB_FC64_code    : return ("GxB_CIMAG_FC64") ;
                default :;
            }
            break ;

        case GB_CARG_unop_code       :  // z = carg (x)

            switch (xcode)
            {
                case GB_FC32_code    : return ("GxB_CARG_FC32") ;
                case GB_FC64_code    : return ("GxB_CARG_FC64") ;
                default :;
            }
            break ;

        case GB_ISINF_unop_code      :  // z = isinf (x)

            switch (xcode)
            {
                case GB_FP32_code    : return ("GxB_ISINF_FP32") ;
                case GB_FP64_code    : return ("GxB_ISINF_FP64") ;
                case GB_FC32_code    : return ("GxB_ISINF_FC32") ;
                case GB_FC64_code    : return ("GxB_ISINF_FC64") ;
                default :;
            }
            break ;

        case GB_ISNAN_unop_code      :  // z = isnan (x)

            switch (xcode)
            {
                case GB_FP32_code    : return ("GxB_ISNAN_FP32") ;
                case GB_FP64_code    : return ("GxB_ISNAN_FP64") ;
                case GB_FC32_code    : return ("GxB_ISNAN_FC32") ;
                case GB_FC64_code    : return ("GxB_ISNAN_FC64") ;
                default :;
            }
            break ;

        case GB_ISFINITE_unop_code   :  // z = isfinite (x)

            switch (xcode)
            {
                case GB_FP32_code    : return ("GxB_ISFINITE_FP32") ;
                case GB_FP64_code    : return ("GxB_ISFINITE_FP64") ;
                case GB_FC32_code    : return ("GxB_ISFINITE_FC32") ;
                case GB_FC64_code    : return ("GxB_ISFINITE_FC64") ;
                default :;
            }
            break ;

        case GB_POSITIONI_unop_code  :  // z = position_i(A(i,j)) == i

            switch (zcode)
            {
                case GB_INT32_code   : return ("GxB_POSITIONI_INT32") ;
                case GB_INT64_code   : return ("GxB_POSITIONI_INT64") ;
                default :;
            }
            break ;

        case GB_POSITIONI1_unop_code :  // z = position_i1(A(i,j)) == i+1

            switch (zcode)
            {
                case GB_INT32_code   : return ("GxB_POSITIONI1_INT32") ;
                case GB_INT64_code   : return ("GxB_POSITIONI1_INT64") ;
                default :;
            }
            break ;

        case GB_POSITIONJ_unop_code  :  // z = position_j(A(i,j)) == j

            switch (zcode)
            {
                case GB_INT32_code   : return ("GxB_POSITIONJ_INT32") ;
                case GB_INT64_code   : return ("GxB_POSITIONJ_INT64") ;
                default :;
            }
            break ;

        case GB_POSITIONJ1_unop_code :  // z = position_j1(A(i,j)) == j+1

            switch (zcode)
            {
                case GB_INT32_code   : return ("GxB_POSITIONJ1_INT32") ;
                case GB_INT64_code   : return ("GxB_POSITIONJ1_INT64") ;
                default :;
            }
            break ;

        //----------------------------------------------------------------------
        // index_unary operators
        //----------------------------------------------------------------------

        case GB_ROWINDEX_idxunop_code      : // (i+thunk): row index - thunk

            switch (zcode)
            {
                case GB_INT32_code   : return ("GrB_ROWINDEX_INT32") ;
                case GB_INT64_code   : return ("GrB_ROWINDEX_INT64") ;
                default :;
            }
            break ;

        case GB_COLINDEX_idxunop_code      : // (j+thunk): col index - thunk

            switch (zcode)
            {
                case GB_INT32_code   : return ("GrB_COLINDEX_INT32") ;
                case GB_INT64_code   : return ("GrB_COLINDEX_INT64") ;
                default :;
            }
            break ;

        case GB_DIAGINDEX_idxunop_code     : // (j-(i+thunk)): diag index+thunk

            switch (zcode)
            {
                case GB_INT32_code   : return ("GrB_DIAGINDEX_INT32") ;
                case GB_INT64_code   : return ("GrB_DIAGINDEX_INT64") ;
                default :;
            }
            break ;

        case GB_FLIPDIAGINDEX_idxunop_code : // (i-(j+thunk)), internal use only

            switch (zcode)
            {
                case GB_INT32_code   : return ("GxB_FLIPDIAGINDEX_INT32") ;
                case GB_INT64_code   : return ("GxB_FLIPDIAGINDEX_INT64") ;
                default :;
            }
            break ;

        case GB_TRIL_idxunop_code      : return ("GrB_TRIL"   ) ;
        case GB_TRIU_idxunop_code      : return ("GrB_TRIU"   ) ;
        case GB_DIAG_idxunop_code      : return ("GrB_DIAG"   ) ;
        case GB_OFFDIAG_idxunop_code   : return ("GrB_OFFDIAG") ;
        case GB_COLLE_idxunop_code     : return ("GrB_COLLE"  ) ;
        case GB_COLGT_idxunop_code     : return ("GrB_COLGT"  ) ;
        case GB_ROWLE_idxunop_code     : return ("GrB_ROWLE"  ) ;
        case GB_ROWGT_idxunop_code     : return ("GrB_ROWGT"  ) ;

        case GB_NONZOMBIE_idxunop_code : return ("GxB_NONZOMBIE") ;

        case GB_VALUENE_idxunop_code   :   // (aij != thunk)

            switch (xcode)
            {
                case GB_BOOL_code    : return ("GrB_VALUENE_BOOL"  ) ;
                case GB_INT8_code    : return ("GrB_VALUENE_INT8"  ) ;
                case GB_INT16_code   : return ("GrB_VALUENE_INT16" ) ;
                case GB_INT32_code   : return ("GrB_VALUENE_INT32" ) ;
                case GB_INT64_code   : return ("GrB_VALUENE_INT64" ) ;
                case GB_UINT8_code   : return ("GrB_VALUENE_UINT8" ) ;
                case GB_UINT16_code  : return ("GrB_VALUENE_UINT16") ;
                case GB_UINT32_code  : return ("GrB_VALUENE_UINT32") ;
                case GB_UINT64_code  : return ("GrB_VALUENE_UINT64") ;
                case GB_FP32_code    : return ("GrB_VALUENE_FP32"  ) ;
                case GB_FP64_code    : return ("GrB_VALUENE_FP64"  ) ;
                case GB_FC32_code    : return ("GxB_VALUENE_FC32"  ) ;
                case GB_FC64_code    : return ("GxB_VALUENE_FC64"  ) ;
                default :;
            }
            break ;

        case GB_VALUEEQ_idxunop_code   :   // (aij == thunk)

            switch (xcode)
            {
                case GB_BOOL_code    : return ("GrB_VALUEEQ_BOOL"  ) ;
                case GB_INT8_code    : return ("GrB_VALUEEQ_INT8"  ) ;
                case GB_INT16_code   : return ("GrB_VALUEEQ_INT16" ) ;
                case GB_INT32_code   : return ("GrB_VALUEEQ_INT32" ) ;
                case GB_INT64_code   : return ("GrB_VALUEEQ_INT64" ) ;
                case GB_UINT8_code   : return ("GrB_VALUEEQ_UINT8" ) ;
                case GB_UINT16_code  : return ("GrB_VALUEEQ_UINT16") ;
                case GB_UINT32_code  : return ("GrB_VALUEEQ_UINT32") ;
                case GB_UINT64_code  : return ("GrB_VALUEEQ_UINT64") ;
                case GB_FP32_code    : return ("GrB_VALUEEQ_FP32"  ) ;
                case GB_FP64_code    : return ("GrB_VALUEEQ_FP64"  ) ;
                case GB_FC32_code    : return ("GxB_VALUEEQ_FC32"  ) ;
                case GB_FC64_code    : return ("GxB_VALUEEQ_FC64"  ) ;
                default :;
            }
            break ;

        case GB_VALUEGT_idxunop_code   :   // (aij > thunk)

            switch (xcode)
            {
                case GB_BOOL_code    : return ("GrB_VALUEGT_BOOL"  ) ;
                case GB_INT8_code    : return ("GrB_VALUEGT_INT8"  ) ;
                case GB_INT16_code   : return ("GrB_VALUEGT_INT16" ) ;
                case GB_INT32_code   : return ("GrB_VALUEGT_INT32" ) ;
                case GB_INT64_code   : return ("GrB_VALUEGT_INT64" ) ;
                case GB_UINT8_code   : return ("GrB_VALUEGT_UINT8" ) ;
                case GB_UINT16_code  : return ("GrB_VALUEGT_UINT16") ;
                case GB_UINT32_code  : return ("GrB_VALUEGT_UINT32") ;
                case GB_UINT64_code  : return ("GrB_VALUEGT_UINT64") ;
                case GB_FP32_code    : return ("GrB_VALUEGT_FP32"  ) ;
                case GB_FP64_code    : return ("GrB_VALUEGT_FP64"  ) ;
                default :;
            }
            break ;

        case GB_VALUEGE_idxunop_code   :   // (aij >= thunk)

            switch (xcode)
            {
                case GB_BOOL_code    : return ("GrB_VALUEGE_BOOL"  ) ;
                case GB_INT8_code    : return ("GrB_VALUEGE_INT8"  ) ;
                case GB_INT16_code   : return ("GrB_VALUEGE_INT16" ) ;
                case GB_INT32_code   : return ("GrB_VALUEGE_INT32" ) ;
                case GB_INT64_code   : return ("GrB_VALUEGE_INT64" ) ;
                case GB_UINT8_code   : return ("GrB_VALUEGE_UINT8" ) ;
                case GB_UINT16_code  : return ("GrB_VALUEGE_UINT16") ;
                case GB_UINT32_code  : return ("GrB_VALUEGE_UINT32") ;
                case GB_UINT64_code  : return ("GrB_VALUEGE_UINT64") ;
                case GB_FP32_code    : return ("GrB_VALUEGE_FP32"  ) ;
                case GB_FP64_code    : return ("GrB_VALUEGE_FP64"  ) ;
                default :;
            }
            break ;

        case GB_VALUELT_idxunop_code   :   // (aij < thunk)

            switch (xcode)
            {
                case GB_BOOL_code    : return ("GrB_VALUELT_BOOL"  ) ;
                case GB_INT8_code    : return ("GrB_VALUELT_INT8"  ) ;
                case GB_INT16_code   : return ("GrB_VALUELT_INT16" ) ;
                case GB_INT32_code   : return ("GrB_VALUELT_INT32" ) ;
                case GB_INT64_code   : return ("GrB_VALUELT_INT64" ) ;
                case GB_UINT8_code   : return ("GrB_VALUELT_UINT8" ) ;
                case GB_UINT16_code  : return ("GrB_VALUELT_UINT16") ;
                case GB_UINT32_code  : return ("GrB_VALUELT_UINT32") ;
                case GB_UINT64_code  : return ("GrB_VALUELT_UINT64") ;
                case GB_FP32_code    : return ("GrB_VALUELT_FP32"  ) ;
                case GB_FP64_code    : return ("GrB_VALUELT_FP64"  ) ;
                default :;
            }
            break ;

        case GB_VALUELE_idxunop_code   :   // (aij <= thunk)

            switch (xcode)
            {
                case GB_BOOL_code    : return ("GrB_VALUELE_BOOL"  ) ;
                case GB_INT8_code    : return ("GrB_VALUELE_INT8"  ) ;
                case GB_INT16_code   : return ("GrB_VALUELE_INT16" ) ;
                case GB_INT32_code   : return ("GrB_VALUELE_INT32" ) ;
                case GB_INT64_code   : return ("GrB_VALUELE_INT64" ) ;
                case GB_UINT8_code   : return ("GrB_VALUELE_UINT8" ) ;
                case GB_UINT16_code  : return ("GrB_VALUELE_UINT16") ;
                case GB_UINT32_code  : return ("GrB_VALUELE_UINT32") ;
                case GB_UINT64_code  : return ("GrB_VALUELE_UINT64") ;
                case GB_FP32_code    : return ("GrB_VALUELE_FP32"  ) ;
                case GB_FP64_code    : return ("GrB_VALUELE_FP64"  ) ;
                default :;
            }
            break ;

        //----------------------------------------------------------------------
        // binary operators
        //----------------------------------------------------------------------

        case GB_FIRST_binop_code     :   // z = x

            switch (xcode)
            {
                case GB_BOOL_code    : return ("GrB_FIRST_BOOL"  ) ;
                case GB_INT8_code    : return ("GrB_FIRST_INT8"  ) ;
                case GB_INT16_code   : return ("GrB_FIRST_INT16" ) ;
                case GB_INT32_code   : return ("GrB_FIRST_INT32" ) ;
                case GB_INT64_code   : return ("GrB_FIRST_INT64" ) ;
                case GB_UINT8_code   : return ("GrB_FIRST_UINT8" ) ;
                case GB_UINT16_code  : return ("GrB_FIRST_UINT16") ;
                case GB_UINT32_code  : return ("GrB_FIRST_UINT32") ;
                case GB_UINT64_code  : return ("GrB_FIRST_UINT64") ;
                case GB_FP32_code    : return ("GrB_FIRST_FP32"  ) ;
                case GB_FP64_code    : return ("GrB_FIRST_FP64"  ) ;
                case GB_FC32_code    : return ("GxB_FIRST_FC32"  ) ;
                case GB_FC64_code    : return ("GxB_FIRST_FC64"  ) ;
                // see GB_reduce_to_vector:
//              case GB_UDT_code     :
//                  return ("1st_udt") ;
                default :;
            }
            break ;

        case GB_SECOND_binop_code    :   // z = y

            switch (xcode)
            {
                case GB_BOOL_code    : return ("GrB_SECOND_BOOL"  ) ;
                case GB_INT8_code    : return ("GrB_SECOND_INT8"  ) ;
                case GB_INT16_code   : return ("GrB_SECOND_INT16" ) ;
                case GB_INT32_code   : return ("GrB_SECOND_INT32" ) ;
                case GB_INT64_code   : return ("GrB_SECOND_INT64" ) ;
                case GB_UINT8_code   : return ("GrB_SECOND_UINT8" ) ;
                case GB_UINT16_code  : return ("GrB_SECOND_UINT16") ;
                case GB_UINT32_code  : return ("GrB_SECOND_UINT32") ;
                case GB_UINT64_code  : return ("GrB_SECOND_UINT64") ;
                case GB_FP32_code    : return ("GrB_SECOND_FP32"  ) ;
                case GB_FP64_code    : return ("GrB_SECOND_FP64"  ) ;
                case GB_FC32_code    : return ("GxB_SECOND_FC32"  ) ;
                case GB_FC64_code    : return ("GxB_SECOND_FC64"  ) ;
                // see GB_binop_second:
//              case GB_UDT_code     :
//                  return ("2nd_udt") ;
                default :;
            }
            break ;

        case GB_ANY_binop_code       :   // z = x or y

            switch (xcode)
            {
                case GB_BOOL_code    : return ("GxB_ANY_BOOL"  ) ;
                case GB_INT8_code    : return ("GxB_ANY_INT8"  ) ;
                case GB_INT16_code   : return ("GxB_ANY_INT16" ) ;
                case GB_INT32_code   : return ("GxB_ANY_INT32" ) ;
                case GB_INT64_code   : return ("GxB_ANY_INT64" ) ;
                case GB_UINT8_code   : return ("GxB_ANY_UINT8" ) ;
                case GB_UINT16_code  : return ("GxB_ANY_UINT16") ;
                case GB_UINT32_code  : return ("GxB_ANY_UINT32") ;
                case GB_UINT64_code  : return ("GxB_ANY_UINT64") ;
                case GB_FP32_code    : return ("GxB_ANY_FP32"  ) ;
                case GB_FP64_code    : return ("GxB_ANY_FP64"  ) ;
                case GB_FC32_code    : return ("GxB_ANY_FC32"  ) ;
                case GB_FC64_code    : return ("GxB_ANY_FC64"  ) ;
                default :;
            }
            break ;

        case GB_PAIR_binop_code      :   // z = 1

            switch (xcode)
            {
                case GB_BOOL_code    : return ("GrB_ONEB_BOOL"  ) ;
                case GB_INT8_code    : return ("GrB_ONEB_INT8"  ) ;
                case GB_INT16_code   : return ("GrB_ONEB_INT16" ) ;
                case GB_INT32_code   : return ("GrB_ONEB_INT32" ) ;
                case GB_INT64_code   : return ("GrB_ONEB_INT64" ) ;
                case GB_UINT8_code   : return ("GrB_ONEB_UINT8" ) ;
                case GB_UINT16_code  : return ("GrB_ONEB_UINT16") ;
                case GB_UINT32_code  : return ("GrB_ONEB_UINT32") ;
                case GB_UINT64_code  : return ("GrB_ONEB_UINT64") ;
                case GB_FP32_code    : return ("GrB_ONEB_FP32"  ) ;
                case GB_FP64_code    : return ("GrB_ONEB_FP64"  ) ;
                case GB_FC32_code    : return ("GxB_ONEB_FC32"  ) ;
                case GB_FC64_code    : return ("GxB_ONEB_FC64"  ) ;
                default :;
            }
            break ;

        case GB_MIN_binop_code       :   // z = min(x,y)

            switch (xcode)
            {
                case GB_BOOL_code    : return ("GrB_MIN_BOOL"  ) ;
                case GB_INT8_code    : return ("GrB_MIN_INT8"  ) ;
                case GB_INT16_code   : return ("GrB_MIN_INT16" ) ;
                case GB_INT32_code   : return ("GrB_MIN_INT32" ) ;
                case GB_INT64_code   : return ("GrB_MIN_INT64" ) ;
                case GB_UINT8_code   : return ("GrB_MIN_UINT8" ) ;
                case GB_UINT16_code  : return ("GrB_MIN_UINT16") ;
                case GB_UINT32_code  : return ("GrB_MIN_UINT32") ;
                case GB_UINT64_code  : return ("GrB_MIN_UINT64") ;
                case GB_FP32_code    : return ("GrB_MIN_FP32"  ) ;
                case GB_FP64_code    : return ("GrB_MIN_FP64"  ) ;
                default :;
            }
            break ;

        case GB_MAX_binop_code       :   // z = max(x,y)

            switch (xcode)
            {
                case GB_BOOL_code    : return ("GrB_MAX_BOOL"  ) ;
                case GB_INT8_code    : return ("GrB_MAX_INT8"  ) ;
                case GB_INT16_code   : return ("GrB_MAX_INT16" ) ;
                case GB_INT32_code   : return ("GrB_MAX_INT32" ) ;
                case GB_INT64_code   : return ("GrB_MAX_INT64" ) ;
                case GB_UINT8_code   : return ("GrB_MAX_UINT8" ) ;
                case GB_UINT16_code  : return ("GrB_MAX_UINT16") ;
                case GB_UINT32_code  : return ("GrB_MAX_UINT32") ;
                case GB_UINT64_code  : return ("GrB_MAX_UINT64") ;
                case GB_FP32_code    : return ("GrB_MAX_FP32"  ) ;
                case GB_FP64_code    : return ("GrB_MAX_FP64"  ) ;
                default :;
            }
            break ;

        case GB_PLUS_binop_code      :   // z = x + y

            switch (xcode)
            {
                case GB_BOOL_code    : return ("GrB_PLUS_BOOL"  ) ;
                case GB_INT8_code    : return ("GrB_PLUS_INT8"  ) ;
                case GB_INT16_code   : return ("GrB_PLUS_INT16" ) ;
                case GB_INT32_code   : return ("GrB_PLUS_INT32" ) ;
                case GB_INT64_code   : return ("GrB_PLUS_INT64" ) ;
                case GB_UINT8_code   : return ("GrB_PLUS_UINT8" ) ;
                case GB_UINT16_code  : return ("GrB_PLUS_UINT16") ;
                case GB_UINT32_code  : return ("GrB_PLUS_UINT32") ;
                case GB_UINT64_code  : return ("GrB_PLUS_UINT64") ;
                case GB_FP32_code    : return ("GrB_PLUS_FP32"  ) ;
                case GB_FP64_code    : return ("GrB_PLUS_FP64"  ) ;
                case GB_FC32_code    : return ("GxB_PLUS_FC32"  ) ;
                case GB_FC64_code    : return ("GxB_PLUS_FC64"  ) ;
                default :;
            }
            break ;

        case GB_MINUS_binop_code     :   // z = x - y

            switch (xcode)
            {
                case GB_BOOL_code    : return ("GrB_MINUS_BOOL"  ) ;
                case GB_INT8_code    : return ("GrB_MINUS_INT8"  ) ;
                case GB_INT16_code   : return ("GrB_MINUS_INT16" ) ;
                case GB_INT32_code   : return ("GrB_MINUS_INT32" ) ;
                case GB_INT64_code   : return ("GrB_MINUS_INT64" ) ;
                case GB_UINT8_code   : return ("GrB_MINUS_UINT8" ) ;
                case GB_UINT16_code  : return ("GrB_MINUS_UINT16") ;
                case GB_UINT32_code  : return ("GrB_MINUS_UINT32") ;
                case GB_UINT64_code  : return ("GrB_MINUS_UINT64") ;
                case GB_FP32_code    : return ("GrB_MINUS_FP32"  ) ;
                case GB_FP64_code    : return ("GrB_MINUS_FP64"  ) ;
                case GB_FC32_code    : return ("GxB_MINUS_FC32"  ) ;
                case GB_FC64_code    : return ("GxB_MINUS_FC64"  ) ;
                default :;
            }
            break ;

        case GB_RMINUS_binop_code    :   // z = y - x

            switch (xcode)
            {
                case GB_BOOL_code    : return ("GxB_RMINUS_BOOL"  ) ;
                case GB_INT8_code    : return ("GxB_RMINUS_INT8"  ) ;
                case GB_INT16_code   : return ("GxB_RMINUS_INT16" ) ;
                case GB_INT32_code   : return ("GxB_RMINUS_INT32" ) ;
                case GB_INT64_code   : return ("GxB_RMINUS_INT64" ) ;
                case GB_UINT8_code   : return ("GxB_RMINUS_UINT8" ) ;
                case GB_UINT16_code  : return ("GxB_RMINUS_UINT16") ;
                case GB_UINT32_code  : return ("GxB_RMINUS_UINT32") ;
                case GB_UINT64_code  : return ("GxB_RMINUS_UINT64") ;
                case GB_FP32_code    : return ("GxB_RMINUS_FP32"  ) ;
                case GB_FP64_code    : return ("GxB_RMINUS_FP64"  ) ;
                case GB_FC32_code    : return ("GxB_RMINUS_FC32"  ) ;
                case GB_FC64_code    : return ("GxB_RMINUS_FC64"  ) ;
                default :;
            }
            break ;

        case GB_TIMES_binop_code     :   // z = x * y

            switch (xcode)
            {
                case GB_BOOL_code    : return ("GrB_TIMES_BOOL"  ) ;
                case GB_INT8_code    : return ("GrB_TIMES_INT8"  ) ;
                case GB_INT16_code   : return ("GrB_TIMES_INT16" ) ;
                case GB_INT32_code   : return ("GrB_TIMES_INT32" ) ;
                case GB_INT64_code   : return ("GrB_TIMES_INT64" ) ;
                case GB_UINT8_code   : return ("GrB_TIMES_UINT8" ) ;
                case GB_UINT16_code  : return ("GrB_TIMES_UINT16") ;
                case GB_UINT32_code  : return ("GrB_TIMES_UINT32") ;
                case GB_UINT64_code  : return ("GrB_TIMES_UINT64") ;
                case GB_FP32_code    : return ("GrB_TIMES_FP32"  ) ;
                case GB_FP64_code    : return ("GrB_TIMES_FP64"  ) ;
                case GB_FC32_code    : return ("GxB_TIMES_FC32"  ) ;
                case GB_FC64_code    : return ("GxB_TIMES_FC64"  ) ;
                default :;
            }
            break ;

        case GB_DIV_binop_code       :   // z = x / y

            switch (xcode)
            {
                case GB_BOOL_code    : return ("GrB_DIV_BOOL"  ) ;
                case GB_INT8_code    : return ("GrB_DIV_INT8"  ) ;
                case GB_INT16_code   : return ("GrB_DIV_INT16" ) ;
                case GB_INT32_code   : return ("GrB_DIV_INT32" ) ;
                case GB_INT64_code   : return ("GrB_DIV_INT64" ) ;
                case GB_UINT8_code   : return ("GrB_DIV_UINT8" ) ;
                case GB_UINT16_code  : return ("GrB_DIV_UINT16") ;
                case GB_UINT32_code  : return ("GrB_DIV_UINT32") ;
                case GB_UINT64_code  : return ("GrB_DIV_UINT64") ;
                case GB_FP32_code    : return ("GrB_DIV_FP32"  ) ;
                case GB_FP64_code    : return ("GrB_DIV_FP64"  ) ;
                case GB_FC32_code    : return ("GxB_DIV_FC32"  ) ;
                case GB_FC64_code    : return ("GxB_DIV_FC64"  ) ;
                default :;
            }
            break ;

        case GB_RDIV_binop_code      :   // z = y / x

            switch (xcode)
            {
                case GB_BOOL_code    : return ("GxB_RDIV_BOOL"  ) ;
                case GB_INT8_code    : return ("GxB_RDIV_INT8"  ) ;
                case GB_INT16_code   : return ("GxB_RDIV_INT16" ) ;
                case GB_INT32_code   : return ("GxB_RDIV_INT32" ) ;
                case GB_INT64_code   : return ("GxB_RDIV_INT64" ) ;
                case GB_UINT8_code   : return ("GxB_RDIV_UINT8" ) ;
                case GB_UINT16_code  : return ("GxB_RDIV_UINT16") ;
                case GB_UINT32_code  : return ("GxB_RDIV_UINT32") ;
                case GB_UINT64_code  : return ("GxB_RDIV_UINT64") ;
                case GB_FP32_code    : return ("GxB_RDIV_FP32"  ) ;
                case GB_FP64_code    : return ("GxB_RDIV_FP64"  ) ;
                case GB_FC32_code    : return ("GxB_RDIV_FC32"  ) ;
                case GB_FC64_code    : return ("GxB_RDIV_FC64"  ) ;
                default :;
            }
            break ;

        case GB_POW_binop_code       :   // z = pow (x,y)

            switch (xcode)
            {
                case GB_BOOL_code    : return ("GxB_POW_BOOL"  ) ;
                case GB_INT8_code    : return ("GxB_POW_INT8"  ) ;
                case GB_INT16_code   : return ("GxB_POW_INT16" ) ;
                case GB_INT32_code   : return ("GxB_POW_INT32" ) ;
                case GB_INT64_code   : return ("GxB_POW_INT64" ) ;
                case GB_UINT8_code   : return ("GxB_POW_UINT8" ) ;
                case GB_UINT16_code  : return ("GxB_POW_UINT16") ;
                case GB_UINT32_code  : return ("GxB_POW_UINT32") ;
                case GB_UINT64_code  : return ("GxB_POW_UINT64") ;
                case GB_FP32_code    : return ("GxB_POW_FP32"  ) ;
                case GB_FP64_code    : return ("GxB_POW_FP64"  ) ;
                case GB_FC32_code    : return ("GxB_POW_FC32"  ) ;
                case GB_FC64_code    : return ("GxB_POW_FC64"  ) ;
                default :;
            }
            break ;

        case GB_ISEQ_binop_code      :   // z = (x == y)

            switch (xcode)
            {
                case GB_BOOL_code    : return ("GxB_ISEQ_BOOL"  ) ;
                case GB_INT8_code    : return ("GxB_ISEQ_INT8"  ) ;
                case GB_INT16_code   : return ("GxB_ISEQ_INT16" ) ;
                case GB_INT32_code   : return ("GxB_ISEQ_INT32" ) ;
                case GB_INT64_code   : return ("GxB_ISEQ_INT64" ) ;
                case GB_UINT8_code   : return ("GxB_ISEQ_UINT8" ) ;
                case GB_UINT16_code  : return ("GxB_ISEQ_UINT16") ;
                case GB_UINT32_code  : return ("GxB_ISEQ_UINT32") ;
                case GB_UINT64_code  : return ("GxB_ISEQ_UINT64") ;
                case GB_FP32_code    : return ("GxB_ISEQ_FP32"  ) ;
                case GB_FP64_code    : return ("GxB_ISEQ_FP64"  ) ;
                case GB_FC32_code    : return ("GxB_ISEQ_FC32"  ) ;
                case GB_FC64_code    : return ("GxB_ISEQ_FC64"  ) ;
                default :;
            }
            break ;

        case GB_ISNE_binop_code      :   // z = (x != y)

            switch (xcode)
            {
                case GB_BOOL_code    : return ("GxB_ISNE_BOOL"  ) ;
                case GB_INT8_code    : return ("GxB_ISNE_INT8"  ) ;
                case GB_INT16_code   : return ("GxB_ISNE_INT16" ) ;
                case GB_INT32_code   : return ("GxB_ISNE_INT32" ) ;
                case GB_INT64_code   : return ("GxB_ISNE_INT64" ) ;
                case GB_UINT8_code   : return ("GxB_ISNE_UINT8" ) ;
                case GB_UINT16_code  : return ("GxB_ISNE_UINT16") ;
                case GB_UINT32_code  : return ("GxB_ISNE_UINT32") ;
                case GB_UINT64_code  : return ("GxB_ISNE_UINT64") ;
                case GB_FP32_code    : return ("GxB_ISNE_FP32"  ) ;
                case GB_FP64_code    : return ("GxB_ISNE_FP64"  ) ;
                case GB_FC32_code    : return ("GxB_ISNE_FC32"  ) ;
                case GB_FC64_code    : return ("GxB_ISNE_FC64"  ) ;
                default :;
            }
            break ;

        case GB_ISGT_binop_code      :   // z = (x >  y)

            switch (xcode)
            {
                case GB_BOOL_code    : return ("GxB_ISGT_BOOL"  ) ;
                case GB_INT8_code    : return ("GxB_ISGT_INT8"  ) ;
                case GB_INT16_code   : return ("GxB_ISGT_INT16" ) ;
                case GB_INT32_code   : return ("GxB_ISGT_INT32" ) ;
                case GB_INT64_code   : return ("GxB_ISGT_INT64" ) ;
                case GB_UINT8_code   : return ("GxB_ISGT_UINT8" ) ;
                case GB_UINT16_code  : return ("GxB_ISGT_UINT16") ;
                case GB_UINT32_code  : return ("GxB_ISGT_UINT32") ;
                case GB_UINT64_code  : return ("GxB_ISGT_UINT64") ;
                case GB_FP32_code    : return ("GxB_ISGT_FP32"  ) ;
                case GB_FP64_code    : return ("GxB_ISGT_FP64"  ) ;
                default :;
            }
            break ;

        case GB_ISLT_binop_code      :   // z = (x <  y)

            switch (xcode)
            {
                case GB_BOOL_code    : return ("GxB_ISLT_BOOL"  ) ;
                case GB_INT8_code    : return ("GxB_ISLT_INT8"  ) ;
                case GB_INT16_code   : return ("GxB_ISLT_INT16" ) ;
                case GB_INT32_code   : return ("GxB_ISLT_INT32" ) ;
                case GB_INT64_code   : return ("GxB_ISLT_INT64" ) ;
                case GB_UINT8_code   : return ("GxB_ISLT_UINT8" ) ;
                case GB_UINT16_code  : return ("GxB_ISLT_UINT16") ;
                case GB_UINT32_code  : return ("GxB_ISLT_UINT32") ;
                case GB_UINT64_code  : return ("GxB_ISLT_UINT64") ;
                case GB_FP32_code    : return ("GxB_ISLT_FP32"  ) ;
                case GB_FP64_code    : return ("GxB_ISLT_FP64"  ) ;
                default :;
            }
            break ;

        case GB_ISGE_binop_code      :   // z = (x >= y)

            switch (xcode)
            {
                case GB_BOOL_code    : return ("GxB_ISGE_BOOL"  ) ;
                case GB_INT8_code    : return ("GxB_ISGE_INT8"  ) ;
                case GB_INT16_code   : return ("GxB_ISGE_INT16" ) ;
                case GB_INT32_code   : return ("GxB_ISGE_INT32" ) ;
                case GB_INT64_code   : return ("GxB_ISGE_INT64" ) ;
                case GB_UINT8_code   : return ("GxB_ISGE_UINT8" ) ;
                case GB_UINT16_code  : return ("GxB_ISGE_UINT16") ;
                case GB_UINT32_code  : return ("GxB_ISGE_UINT32") ;
                case GB_UINT64_code  : return ("GxB_ISGE_UINT64") ;
                case GB_FP32_code    : return ("GxB_ISGE_FP32"  ) ;
                case GB_FP64_code    : return ("GxB_ISGE_FP64"  ) ;
                default :;
            }
            break ;

        case GB_ISLE_binop_code      :   // z = (x <= y)

            switch (xcode)
            {
                case GB_BOOL_code    : return ("GxB_ISLE_BOOL"  ) ;
                case GB_INT8_code    : return ("GxB_ISLE_INT8"  ) ;
                case GB_INT16_code   : return ("GxB_ISLE_INT16" ) ;
                case GB_INT32_code   : return ("GxB_ISLE_INT32" ) ;
                case GB_INT64_code   : return ("GxB_ISLE_INT64" ) ;
                case GB_UINT8_code   : return ("GxB_ISLE_UINT8" ) ;
                case GB_UINT16_code  : return ("GxB_ISLE_UINT16") ;
                case GB_UINT32_code  : return ("GxB_ISLE_UINT32") ;
                case GB_UINT64_code  : return ("GxB_ISLE_UINT64") ;
                case GB_FP32_code    : return ("GxB_ISLE_FP32"  ) ;
                case GB_FP64_code    : return ("GxB_ISLE_FP64"  ) ;
                default :;
            }
            break ;

        case GB_LOR_binop_code       :   // z = (x != 0) || (y != 0)

            switch (xcode)
            {
                case GB_BOOL_code    : return ("GrB_LOR"       ) ;
                case GB_INT8_code    : return ("GxB_LOR_INT8"  ) ;
                case GB_INT16_code   : return ("GxB_LOR_INT16" ) ;
                case GB_INT32_code   : return ("GxB_LOR_INT32" ) ;
                case GB_INT64_code   : return ("GxB_LOR_INT64" ) ;
                case GB_UINT8_code   : return ("GxB_LOR_UINT8" ) ;
                case GB_UINT16_code  : return ("GxB_LOR_UINT16") ;
                case GB_UINT32_code  : return ("GxB_LOR_UINT32") ;
                case GB_UINT64_code  : return ("GxB_LOR_UINT64") ;
                case GB_FP32_code    : return ("GxB_LOR_FP32"  ) ;
                case GB_FP64_code    : return ("GxB_LOR_FP64"  ) ;
                default :;
            }
            break ;

        case GB_LAND_binop_code      :   // z = (x != 0) && (y != 0)

            switch (xcode)
            {
                case GB_BOOL_code    : return ("GrB_LAND"       ) ;
                case GB_INT8_code    : return ("GxB_LAND_INT8"  ) ;
                case GB_INT16_code   : return ("GxB_LAND_INT16" ) ;
                case GB_INT32_code   : return ("GxB_LAND_INT32" ) ;
                case GB_INT64_code   : return ("GxB_LAND_INT64" ) ;
                case GB_UINT8_code   : return ("GxB_LAND_UINT8" ) ;
                case GB_UINT16_code  : return ("GxB_LAND_UINT16") ;
                case GB_UINT32_code  : return ("GxB_LAND_UINT32") ;
                case GB_UINT64_code  : return ("GxB_LAND_UINT64") ;
                case GB_FP32_code    : return ("GxB_LAND_FP32"  ) ;
                case GB_FP64_code    : return ("GxB_LAND_FP64"  ) ;
                default :;
            }
            break ;

        case GB_LXOR_binop_code      :   // z = (x != 0) != (y != 0)

            switch (xcode)
            {
                case GB_BOOL_code    : return ("GrB_LXOR"       ) ;
                case GB_INT8_code    : return ("GxB_LXOR_INT8"  ) ;
                case GB_INT16_code   : return ("GxB_LXOR_INT16" ) ;
                case GB_INT32_code   : return ("GxB_LXOR_INT32" ) ;
                case GB_INT64_code   : return ("GxB_LXOR_INT64" ) ;
                case GB_UINT8_code   : return ("GxB_LXOR_UINT8" ) ;
                case GB_UINT16_code  : return ("GxB_LXOR_UINT16") ;
                case GB_UINT32_code  : return ("GxB_LXOR_UINT32") ;
                case GB_UINT64_code  : return ("GxB_LXOR_UINT64") ;
                case GB_FP32_code    : return ("GxB_LXOR_FP32"  ) ;
                case GB_FP64_code    : return ("GxB_LXOR_FP64"  ) ;
                default :;
            }
            break ;

        case GB_EQ_binop_code        :  // z = (x == y), is LXNOR for bool

            switch (xcode)
            {
                case GB_BOOL_code    : return ("GrB_LXNOR"    ) ;
                case GB_INT8_code    : return ("GrB_EQ_INT8"  ) ;
                case GB_INT16_code   : return ("GrB_EQ_INT16" ) ;
                case GB_INT32_code   : return ("GrB_EQ_INT32" ) ;
                case GB_INT64_code   : return ("GrB_EQ_INT64" ) ;
                case GB_UINT8_code   : return ("GrB_EQ_UINT8" ) ;
                case GB_UINT16_code  : return ("GrB_EQ_UINT16") ;
                case GB_UINT32_code  : return ("GrB_EQ_UINT32") ;
                case GB_UINT64_code  : return ("GrB_EQ_UINT64") ;
                case GB_FP32_code    : return ("GrB_EQ_FP32"  ) ;
                case GB_FP64_code    : return ("GrB_EQ_FP64"  ) ;
                case GB_FC32_code    : return ("GxB_EQ_FC32"  ) ;
                case GB_FC64_code    : return ("GxB_EQ_FC64"  ) ;
                default :;
            }
            break ;

        case GB_BOR_binop_code       :   // z = (x | y), bitwise or

            switch (xcode)
            {
                case GB_INT8_code    : return ("GrB_BOR_INT8"  ) ;
                case GB_INT16_code   : return ("GrB_BOR_INT16" ) ;
                case GB_INT32_code   : return ("GrB_BOR_INT32" ) ;
                case GB_INT64_code   : return ("GrB_BOR_INT64" ) ;
                case GB_UINT8_code   : return ("GrB_BOR_UINT8" ) ;
                case GB_UINT16_code  : return ("GrB_BOR_UINT16") ;
                case GB_UINT32_code  : return ("GrB_BOR_UINT32") ;
                case GB_UINT64_code  : return ("GrB_BOR_UINT64") ;
                default :;
            }
            break ;

        case GB_BAND_binop_code      :   // z = (x & y), bitwise and

            switch (xcode)
            {
                case GB_INT8_code    : return ("GrB_BAND_INT8"  ) ;
                case GB_INT16_code   : return ("GrB_BAND_INT16" ) ;
                case GB_INT32_code   : return ("GrB_BAND_INT32" ) ;
                case GB_INT64_code   : return ("GrB_BAND_INT64" ) ;
                case GB_UINT8_code   : return ("GrB_BAND_UINT8" ) ;
                case GB_UINT16_code  : return ("GrB_BAND_UINT16") ;
                case GB_UINT32_code  : return ("GrB_BAND_UINT32") ;
                case GB_UINT64_code  : return ("GrB_BAND_UINT64") ;
                default :;
            }
            break ;

        case GB_BXOR_binop_code      :   // z = (x ^ y), bitwise xor

            switch (xcode)
            {
                case GB_INT8_code    : return ("GrB_BXOR_INT8"  ) ;
                case GB_INT16_code   : return ("GrB_BXOR_INT16" ) ;
                case GB_INT32_code   : return ("GrB_BXOR_INT32" ) ;
                case GB_INT64_code   : return ("GrB_BXOR_INT64" ) ;
                case GB_UINT8_code   : return ("GrB_BXOR_UINT8" ) ;
                case GB_UINT16_code  : return ("GrB_BXOR_UINT16") ;
                case GB_UINT32_code  : return ("GrB_BXOR_UINT32") ;
                case GB_UINT64_code  : return ("GrB_BXOR_UINT64") ;
                default :;
            }
            break ;

        case GB_BXNOR_binop_code     :   // z = ~(x ^ y), bitwise xnor

            switch (xcode)
            {
                case GB_INT8_code    : return ("GrB_BXNOR_INT8"  ) ;
                case GB_INT16_code   : return ("GrB_BXNOR_INT16" ) ;
                case GB_INT32_code   : return ("GrB_BXNOR_INT32" ) ;
                case GB_INT64_code   : return ("GrB_BXNOR_INT64" ) ;
                case GB_UINT8_code   : return ("GrB_BXNOR_UINT8" ) ;
                case GB_UINT16_code  : return ("GrB_BXNOR_UINT16") ;
                case GB_UINT32_code  : return ("GrB_BXNOR_UINT32") ;
                case GB_UINT64_code  : return ("GrB_BXNOR_UINT64") ;
                default :;
            }
            break ;

        case GB_BGET_binop_code      :   // z = bitget (x,y)

            switch (xcode)
            {
                case GB_INT8_code    : return ("GxB_BGET_INT8"  ) ;
                case GB_INT16_code   : return ("GxB_BGET_INT16" ) ;
                case GB_INT32_code   : return ("GxB_BGET_INT32" ) ;
                case GB_INT64_code   : return ("GxB_BGET_INT64" ) ;
                case GB_UINT8_code   : return ("GxB_BGET_UINT8" ) ;
                case GB_UINT16_code  : return ("GxB_BGET_UINT16") ;
                case GB_UINT32_code  : return ("GxB_BGET_UINT32") ;
                case GB_UINT64_code  : return ("GxB_BGET_UINT64") ;
                default :;
            }
            break ;

        case GB_BSET_binop_code      :   // z = bitset (x,y)

            switch (xcode)
            {
                case GB_INT8_code    : return ("GxB_BSET_INT8"  ) ;
                case GB_INT16_code   : return ("GxB_BSET_INT16" ) ;
                case GB_INT32_code   : return ("GxB_BSET_INT32" ) ;
                case GB_INT64_code   : return ("GxB_BSET_INT64" ) ;
                case GB_UINT8_code   : return ("GxB_BSET_UINT8" ) ;
                case GB_UINT16_code  : return ("GxB_BSET_UINT16") ;
                case GB_UINT32_code  : return ("GxB_BSET_UINT32") ;
                case GB_UINT64_code  : return ("GxB_BSET_UINT64") ;
                default :;
            }
            break ;

        case GB_BCLR_binop_code      :   // z = bitclr (x,y)

            switch (xcode)
            {
                case GB_INT8_code    : return ("GxB_BCLR_INT8"  ) ;
                case GB_INT16_code   : return ("GxB_BCLR_INT16" ) ;
                case GB_INT32_code   : return ("GxB_BCLR_INT32" ) ;
                case GB_INT64_code   : return ("GxB_BCLR_INT64" ) ;
                case GB_UINT8_code   : return ("GxB_BCLR_UINT8" ) ;
                case GB_UINT16_code  : return ("GxB_BCLR_UINT16") ;
                case GB_UINT32_code  : return ("GxB_BCLR_UINT32") ;
                case GB_UINT64_code  : return ("GxB_BCLR_UINT64") ;
                default :;
            }
            break ;

        case GB_BSHIFT_binop_code    :   // z = bitshift (x,y)

            switch (xcode)
            {
                case GB_INT8_code    : return ("GxB_BSHIFT_INT8"  ) ;
                case GB_INT16_code   : return ("GxB_BSHIFT_INT16" ) ;
                case GB_INT32_code   : return ("GxB_BSHIFT_INT32" ) ;
                case GB_INT64_code   : return ("GxB_BSHIFT_INT64" ) ;
                case GB_UINT8_code   : return ("GxB_BSHIFT_UINT8" ) ;
                case GB_UINT16_code  : return ("GxB_BSHIFT_UINT16") ;
                case GB_UINT32_code  : return ("GxB_BSHIFT_UINT32") ;
                case GB_UINT64_code  : return ("GxB_BSHIFT_UINT64") ;
                default :;
            }
            break ;

        case GB_NE_binop_code        :  // z = (x != y)

            switch (xcode)
            {
                case GB_BOOL_code    : return ("GrB_NE_BOOL"  ) ;
                case GB_INT8_code    : return ("GrB_NE_INT8"  ) ;
                case GB_INT16_code   : return ("GrB_NE_INT16" ) ;
                case GB_INT32_code   : return ("GrB_NE_INT32" ) ;
                case GB_INT64_code   : return ("GrB_NE_INT64" ) ;
                case GB_UINT8_code   : return ("GrB_NE_UINT8" ) ;
                case GB_UINT16_code  : return ("GrB_NE_UINT16") ;
                case GB_UINT32_code  : return ("GrB_NE_UINT32") ;
                case GB_UINT64_code  : return ("GrB_NE_UINT64") ;
                case GB_FP32_code    : return ("GrB_NE_FP32"  ) ;
                case GB_FP64_code    : return ("GrB_NE_FP64"  ) ;
                case GB_FC32_code    : return ("GxB_NE_FC32"  ) ;
                case GB_FC64_code    : return ("GxB_NE_FC64"  ) ;
                default :;
            }
            break ;

        case GB_GT_binop_code        :  // z = (x >  y)

            switch (xcode)
            {
                case GB_BOOL_code    : return ("GrB_GT_BOOL"  ) ;
                case GB_INT8_code    : return ("GrB_GT_INT8"  ) ;
                case GB_INT16_code   : return ("GrB_GT_INT16" ) ;
                case GB_INT32_code   : return ("GrB_GT_INT32" ) ;
                case GB_INT64_code   : return ("GrB_GT_INT64" ) ;
                case GB_UINT8_code   : return ("GrB_GT_UINT8" ) ;
                case GB_UINT16_code  : return ("GrB_GT_UINT16") ;
                case GB_UINT32_code  : return ("GrB_GT_UINT32") ;
                case GB_UINT64_code  : return ("GrB_GT_UINT64") ;
                case GB_FP32_code    : return ("GrB_GT_FP32"  ) ;
                case GB_FP64_code    : return ("GrB_GT_FP64"  ) ;
                default :;
            }
            break ;

        case GB_LT_binop_code        :  // z = (x <  y)

            switch (xcode)
            {
                case GB_BOOL_code    : return ("GrB_LT_BOOL"  ) ;
                case GB_INT8_code    : return ("GrB_LT_INT8"  ) ;
                case GB_INT16_code   : return ("GrB_LT_INT16" ) ;
                case GB_INT32_code   : return ("GrB_LT_INT32" ) ;
                case GB_INT64_code   : return ("GrB_LT_INT64" ) ;
                case GB_UINT8_code   : return ("GrB_LT_UINT8" ) ;
                case GB_UINT16_code  : return ("GrB_LT_UINT16") ;
                case GB_UINT32_code  : return ("GrB_LT_UINT32") ;
                case GB_UINT64_code  : return ("GrB_LT_UINT64") ;
                case GB_FP32_code    : return ("GrB_LT_FP32"  ) ;
                case GB_FP64_code    : return ("GrB_LT_FP64"  ) ;
                default :;
            }
            break ;

        case GB_GE_binop_code        :  // z = (x >= y)

            switch (xcode)
            {
                case GB_BOOL_code    : return ("GrB_GE_BOOL"  ) ;
                case GB_INT8_code    : return ("GrB_GE_INT8"  ) ;
                case GB_INT16_code   : return ("GrB_GE_INT16" ) ;
                case GB_INT32_code   : return ("GrB_GE_INT32" ) ;
                case GB_INT64_code   : return ("GrB_GE_INT64" ) ;
                case GB_UINT8_code   : return ("GrB_GE_UINT8" ) ;
                case GB_UINT16_code  : return ("GrB_GE_UINT16") ;
                case GB_UINT32_code  : return ("GrB_GE_UINT32") ;
                case GB_UINT64_code  : return ("GrB_GE_UINT64") ;
                case GB_FP32_code    : return ("GrB_GE_FP32"  ) ;
                case GB_FP64_code    : return ("GrB_GE_FP64"  ) ;
                default :;
            }
            break ;

        case GB_LE_binop_code        :  // z = (x <= y)

            switch (xcode)
            {
                case GB_BOOL_code    : return ("GrB_LE_BOOL"  ) ;
                case GB_INT8_code    : return ("GrB_LE_INT8"  ) ;
                case GB_INT16_code   : return ("GrB_LE_INT16" ) ;
                case GB_INT32_code   : return ("GrB_LE_INT32" ) ;
                case GB_INT64_code   : return ("GrB_LE_INT64" ) ;
                case GB_UINT8_code   : return ("GrB_LE_UINT8" ) ;
                case GB_UINT16_code  : return ("GrB_LE_UINT16") ;
                case GB_UINT32_code  : return ("GrB_LE_UINT32") ;
                case GB_UINT64_code  : return ("GrB_LE_UINT64") ;
                case GB_FP32_code    : return ("GrB_LE_FP32"  ) ;
                case GB_FP64_code    : return ("GrB_LE_FP64"  ) ;
                default :;
            }
            break ;

        case GB_ATAN2_binop_code     :  // z = atan2 (x,y)

            switch (xcode)
            {
                case GB_FP32_code    : return ("GxB_ATAN2_FP32") ;
                case GB_FP64_code    : return ("GxB_ATAN2_FP64") ;
                default :;
            }
            break ;

        case GB_HYPOT_binop_code     :  // z = hypot (x,y)

            switch (xcode)
            {
                case GB_FP32_code    : return ("GxB_HYPOT_FP32") ;
                case GB_FP64_code    : return ("GxB_HYPOT_FP64") ;
                default :;
            }
            break ;

        case GB_FMOD_binop_code      :  // z = fmod (x,y)

            switch (xcode)
            {
                case GB_FP32_code    : return ("GxB_FMOD_FP32") ;
                case GB_FP64_code    : return ("GxB_FMOD_FP64") ;
                default :;
            }
            break ;

        case GB_REMAINDER_binop_code :  // z = remainder (x,y)

            switch (xcode)
            {
                case GB_FP32_code    : return ("GxB_REMAINDER_FP32") ;
                case GB_FP64_code    : return ("GxB_REMAINDER_FP64") ;
                default :;
            }
            break ;

        case GB_COPYSIGN_binop_code  :  // z = copysign (x,y)

            switch (xcode)
            {
                case GB_FP32_code    : return ("GxB_COPYSIGN_FP32") ;
                case GB_FP64_code    : return ("GxB_COPYSIGN_FP64") ;
                default :;
            }
            break ;

        case GB_LDEXP_binop_code     :  // z = ldexp (x,y)

            switch (xcode)
            {
                case GB_FP32_code    : return ("GxB_LDEXP_FP32") ;
                case GB_FP64_code    : return ("GxB_LDEXP_FP64") ;
                default :;
            }
            break ;

        case GB_CMPLX_binop_code     :  // z = cmplx (x,y)

            switch (xcode)
            {
                case GB_FP32_code    : return ("GxB_CMPLX_FP32") ;
                case GB_FP64_code    : return ("GxB_CMPLX_FP64") ;
                default :;
            }
            break ;

        case GB_FIRSTI_binop_code    :  // z = first_i(A(i,j),y) == i

            switch (zcode)
            {
                case GB_INT32_code   : return ("GxB_FIRSTI_INT32") ;
                case GB_INT64_code   : return ("GxB_FIRSTI_INT64") ;
                default :;
            }
            break ;

        case GB_FIRSTI1_binop_code   :  // z = first_i1(A(i,j),y) == i+1

            switch (zcode)
            {
                case GB_INT32_code   : return ("GxB_FIRSTI1_INT32") ;
                case GB_INT64_code   : return ("GxB_FIRSTI1_INT64") ;
                default :;
            }
            break ;

        case GB_FIRSTJ_binop_code    :  // z = first_j(A(i,j),y) == j

            switch (zcode)
            {
                case GB_INT32_code   : return ("GxB_FIRSTJ_INT32") ;
                case GB_INT64_code   : return ("GxB_FIRSTJ_INT64") ;
                default :;
            }
            break ;

        case GB_FIRSTJ1_binop_code   :  // z = first_j1(A(i,j),y) == j+1

            switch (zcode)
            {
                case GB_INT32_code   : return ("GxB_FIRSTJ1_INT32") ;
                case GB_INT64_code   : return ("GxB_FIRSTJ1_INT64") ;
                default :;
            }
            break ;

        case GB_SECONDI_binop_code   :  // z = second_i(x,B(i,j)) == i

            switch (zcode)
            {
                case GB_INT32_code   : return ("GxB_SECONDI_INT32") ;
                case GB_INT64_code   : return ("GxB_SECONDI_INT64") ;
                default :;
            }
            break ;

        case GB_SECONDI1_binop_code  :  // z = second_i1(x,B(i,j)) == i+1

            switch (zcode)
            {
                case GB_INT32_code   : return ("GxB_SECONDI1_INT32") ;
                case GB_INT64_code   : return ("GxB_SECONDI1_INT64") ;
                default :;
            }
            break ;

        case GB_SECONDJ_binop_code   :  // z = second_j(x,B(i,j)) == j

            switch (zcode)
            {
                case GB_INT32_code   : return ("GxB_SECONDJ_INT32") ;
                case GB_INT64_code   : return ("GxB_SECONDJ_INT64") ;
                default :;
            }
            break ;

        case GB_SECONDJ1_binop_code  :  // z = second_j1(x,B(i,j)) == j+1

            switch (zcode)
            {
                case GB_INT32_code   : return ("GxB_SECONDJ1_INT32") ;
                case GB_INT64_code   : return ("GxB_SECONDJ1_INT64") ;
                default :;
            }
            break ;

        //----------------------------------------------------------------------
        // user-defined operators
        //----------------------------------------------------------------------

        case GB_USER_unop_code :
        case GB_USER_idxunop_code :
        case GB_USER_idxbinop_code :
        case GB_USER_binop_code :   return (op->user_name) ;

        //----------------------------------------------------------------------
        // operator not recognized
        //----------------------------------------------------------------------

        default :;
    }

    return ("") ;
}

