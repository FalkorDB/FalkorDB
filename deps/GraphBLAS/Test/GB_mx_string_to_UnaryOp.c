//------------------------------------------------------------------------------
// GB_mx_string_to_UnaryOp.c: get a GraphBLAS operator from built-in strings
//------------------------------------------------------------------------------

// SuiteSparse:GraphBLAS, Timothy A. Davis, (c) 2017-2023, All Rights Reserved.
// SPDX-License-Identifier: Apache-2.0

//------------------------------------------------------------------------------

#include "GB_mex.h"

// opname_mx: a built-in string defining the operator name

// optype_mx: a built-in string defining the operator type for built-in ops

// default_optype: default type if optype_mx is NULL

bool GB_mx_string_to_UnaryOp            // true if successful, false otherwise
(
    GrB_UnaryOp *op_handle,             // the unary op
    const GrB_Type default_optype,      // default operator type
    const mxArray *opname_mx,           // built-in string with operator name
    const mxArray *optype_mx,           // built-in string with operator type
    const bool user_complex             // true if X is complex
)
{

    (*op_handle) = NULL ;
    GrB_UnaryOp op = NULL ;

    //--------------------------------------------------------------------------
    // get the string
    //--------------------------------------------------------------------------

    #define LEN 256
    char opname [LEN+2] ;
    int len = GB_mx_mxArray_to_string (opname, LEN, opname_mx) ;
    if (len < 0)
    {
        return (false) ;
    }

    // get the optype from the optype_mx string, if present
    GrB_Type optype = GB_mx_string_to_Type (optype_mx, default_optype) ;
    if (optype == NULL)
    {
        mexWarnMsgIdAndTxt ("GB:warn", "unrecognized op type") ;
        return (false) ;
    }

    //--------------------------------------------------------------------------
    // convert the string to a GraphBLAS unary operator, built-in or Complex
    //--------------------------------------------------------------------------

    if (user_complex && optype == Complex)
    {

        //----------------------------------------------------------------------
        // X complex
        //----------------------------------------------------------------------

        if (len == 0)
        {
            op = NULL ;                 // no default Complex operator
        }

        // 7 unary operators z=f(x), both x,z are Complex (6 same as builtin)
        else if (MATCH (opname, "one"     )) { op = Complex_one      ; }
        else if (MATCH (opname, "identity")) { op = Complex_identity ; }
        else if (MATCH (opname, "ainv"    )) { op = Complex_ainv     ; }
        else if (MATCH (opname, "abs"     )) { op = Complex_abs      ; }
        else if (MATCH (opname, "minv"    )) { op = Complex_minv     ; }

        // this is not built-in
        else if (MATCH (opname, "not"     )) { op = Complex_not      ; }

        else if (MATCH (opname, "conj"    )) { op = Complex_conj     ; }
        else if (MATCH (opname, "real"    )) { op = Complex_real     ; }
        else if (MATCH (opname, "imag"    )) { op = Complex_imag     ; }
        else if (MATCH (opname, "carg"    )) { op = Complex_angle    ; }

        else
        {
            mexWarnMsgIdAndTxt ("GB:warn", "Complex op unrecognized") ;
        }

    }
    else
    {

        //----------------------------------------------------------------------
        // X is real (Z might be Complex)
        //----------------------------------------------------------------------

        GB_Opcode opcode ;

             if (MATCH (opname, "one"     )) { opcode = GB_ONE_unop_code ; }
        else if (MATCH (opname, "identity")) { opcode = GB_IDENTITY_unop_code ; }
        else if (MATCH (opname, "ainv"    )) { opcode = GB_AINV_unop_code ; }
        else if (MATCH (opname, "abs"     )) { opcode = GB_ABS_unop_code ; }
        else if (MATCH (opname, "minv"    )) { opcode = GB_MINV_unop_code ; }
        else if (MATCH (opname, "not"     )) { opcode = GB_LNOT_unop_code ; }

        else if (MATCH (opname, "conj"    )) { opcode = GB_CONJ_unop_code ; }
        else if (MATCH (opname, "real"    )) { opcode = GB_CREAL_unop_code ; }
        else if (MATCH (opname, "imag"    )) { opcode = GB_CIMAG_unop_code ; }
        else if (MATCH (opname, "carg"    )) { opcode = GB_CARG_unop_code ; }

        else if (MATCH (opname, "sqrt"    )) { opcode = GB_SQRT_unop_code ; }
        else if (MATCH (opname, "log"     )) { opcode = GB_LOG_unop_code ; }
        else if (MATCH (opname, "exp"     )) { opcode = GB_EXP_unop_code ; }

        else if (MATCH (opname, "sin"     )) { opcode = GB_SIN_unop_code ; }
        else if (MATCH (opname, "cos"     )) { opcode = GB_COS_unop_code ; }
        else if (MATCH (opname, "tan"     )) { opcode = GB_TAN_unop_code ; }

        else if (MATCH (opname, "asin"    )) { opcode = GB_ASIN_unop_code ; }
        else if (MATCH (opname, "acos"    )) { opcode = GB_ACOS_unop_code ; }
        else if (MATCH (opname, "atan"    )) { opcode = GB_ATAN_unop_code ; }

        else if (MATCH (opname, "sinh"    )) { opcode = GB_SINH_unop_code ; }
        else if (MATCH (opname, "cosh"    )) { opcode = GB_COSH_unop_code ; }
        else if (MATCH (opname, "tanh"    )) { opcode = GB_TANH_unop_code ; }

        else if (MATCH (opname, "asinh"   )) { opcode = GB_ASINH_unop_code ; }
        else if (MATCH (opname, "acosh"   )) { opcode = GB_ACOSH_unop_code ; }
        else if (MATCH (opname, "atanh"   )) { opcode = GB_ATANH_unop_code ; }

        else if (MATCH (opname, "signum"  )) { opcode = GB_SIGNUM_unop_code ; }
        else if (MATCH (opname, "ceil"    )) { opcode = GB_CEIL_unop_code ; }
        else if (MATCH (opname, "floor"   )) { opcode = GB_FLOOR_unop_code ; }
        else if (MATCH (opname, "round"   )) { opcode = GB_ROUND_unop_code ; }
        else if (MATCH (opname, "trunc"   )) { opcode = GB_TRUNC_unop_code ; }

        else if (MATCH (opname, "exp2"    )) { opcode = GB_EXP2_unop_code ; }
        else if (MATCH (opname, "expm1"   )) { opcode = GB_EXPM1_unop_code ; }
        else if (MATCH (opname, "log10"   )) { opcode = GB_LOG10_unop_code ; }
        else if (MATCH (opname, "log1p"   )) { opcode = GB_LOG1P_unop_code ; }
        else if (MATCH (opname, "log2"    )) { opcode = GB_LOG2_unop_code ; }

        else if (MATCH (opname, "lgamma"  )) { opcode = GB_LGAMMA_unop_code ; }
        else if (MATCH (opname, "tgamma"  )) { opcode = GB_TGAMMA_unop_code ; }
        else if (MATCH (opname, "erf"     )) { opcode = GB_ERF_unop_code ; }
        else if (MATCH (opname, "erfc"    )) { opcode = GB_ERFC_unop_code ; }
        else if (MATCH (opname, "cbrt"    )) { opcode = GB_CBRT_unop_code ; }
        else if (MATCH (opname, "frexpx"  )) { opcode = GB_FREXPX_unop_code ; }
        else if (MATCH (opname, "frexpe"  )) { opcode = GB_FREXPE_unop_code ; }

        else if (MATCH (opname, "isinf"   )) { opcode = GB_ISINF_unop_code ; }
        else if (MATCH (opname, "isnan"   )) { opcode = GB_ISNAN_unop_code ; }
        else if (MATCH (opname, "isfinite")) { opcode = GB_ISFINITE_unop_code ; }

        else if (MATCH (opname, "bitnot"  )) { opcode = GB_BNOT_unop_code ; }
        else if (MATCH (opname, "bitcmp"  )) { opcode = GB_BNOT_unop_code ; }
        else if (MATCH (opname, "bnot"    )) { opcode = GB_BNOT_unop_code ; }
        else if (MATCH (opname, "bcmp"    )) { opcode = GB_BNOT_unop_code ; }

        else if (MATCH (opname, "positioni" )) { opcode = GB_POSITIONI_unop_code ; }
        else if (MATCH (opname, "i"         )) { opcode = GB_POSITIONI_unop_code ; }
        else if (MATCH (opname, "positioni1")) { opcode = GB_POSITIONI1_unop_code ; }
        else if (MATCH (opname, "i1"        )) { opcode = GB_POSITIONI1_unop_code ; }
        else if (MATCH (opname, "positionj" )) { opcode = GB_POSITIONJ_unop_code ; }
        else if (MATCH (opname, "j"         )) { opcode = GB_POSITIONJ_unop_code ; }
        else if (MATCH (opname, "positionj1")) { opcode = GB_POSITIONJ1_unop_code ; }
        else if (MATCH (opname, "j1"        )) { opcode = GB_POSITIONJ1_unop_code ; }

        else
        {
            mexWarnMsgIdAndTxt ("GB:warn", "unrecognized function name") ;
            return (false) ;
        }

        GB_Type_code xcode = optype->code ;
        bool is64 = (xcode == GB_INT64_code) ;

        if (GB_OPCODE_IS_POSITIONAL (opcode))
        {
            if (! (xcode == GB_INT64_code || xcode == GB_INT32_code))
            {
                mexWarnMsgIdAndTxt ("GB:warn","unknown operator") ;
                return (false) ;
            }
        }

        switch (opcode)
        {

            case GB_ONE_unop_code:

                switch (xcode)
                {
                    case GB_BOOL_code    : op = GxB_ONE_BOOL   ; break ;
                    case GB_INT8_code    : op = GxB_ONE_INT8   ; break ;
                    case GB_INT16_code   : op = GxB_ONE_INT16  ; break ;
                    case GB_INT32_code   : op = GxB_ONE_INT32  ; break ;
                    case GB_INT64_code   : op = GxB_ONE_INT64  ; break ;
                    case GB_UINT8_code   : op = GxB_ONE_UINT8  ; break ;
                    case GB_UINT16_code  : op = GxB_ONE_UINT16 ; break ;
                    case GB_UINT32_code  : op = GxB_ONE_UINT32 ; break ;
                    case GB_UINT64_code  : op = GxB_ONE_UINT64 ; break ;
                    case GB_FP32_code    : op = GxB_ONE_FP32   ; break ;
                    case GB_FP64_code    : op = GxB_ONE_FP64   ; break ;
                    case GB_FC32_code    : op = GxB_ONE_FC32   ; break ;
                    case GB_FC64_code    : op = GxB_ONE_FC64   ; break ;
                    default              : 
                        mexWarnMsgIdAndTxt ("GB:warn","unknown operator") ;
                        return (false) ;
                }
                break ;

            case GB_IDENTITY_unop_code:

                switch (xcode)
                {
                    case GB_BOOL_code    : op = GrB_IDENTITY_BOOL   ; break ;
                    case GB_INT8_code    : op = GrB_IDENTITY_INT8   ; break ;
                    case GB_INT16_code   : op = GrB_IDENTITY_INT16  ; break ;
                    case GB_INT32_code   : op = GrB_IDENTITY_INT32  ; break ;
                    case GB_INT64_code   : op = GrB_IDENTITY_INT64  ; break ;
                    case GB_UINT8_code   : op = GrB_IDENTITY_UINT8  ; break ;
                    case GB_UINT16_code  : op = GrB_IDENTITY_UINT16 ; break ;
                    case GB_UINT32_code  : op = GrB_IDENTITY_UINT32 ; break ;
                    case GB_UINT64_code  : op = GrB_IDENTITY_UINT64 ; break ;
                    case GB_FP32_code    : op = GrB_IDENTITY_FP32   ; break ;
                    case GB_FP64_code    : op = GrB_IDENTITY_FP64   ; break ;
                    case GB_FC32_code    : op = GxB_IDENTITY_FC32   ; break ;
                    case GB_FC64_code    : op = GxB_IDENTITY_FC64   ; break ;
                    default              : 
                        mexWarnMsgIdAndTxt ("GB:warn","unknown operator") ;
                        return (false) ;
                }
                break ;

            case GB_ABS_unop_code:

                switch (xcode)
                {
                    case GB_BOOL_code    : op = GrB_ABS_BOOL   ; break ;
                    case GB_INT8_code    : op = GrB_ABS_INT8   ; break ;
                    case GB_INT16_code   : op = GrB_ABS_INT16  ; break ;
                    case GB_INT32_code   : op = GrB_ABS_INT32  ; break ;
                    case GB_INT64_code   : op = GrB_ABS_INT64  ; break ;
                    case GB_UINT8_code   : op = GrB_ABS_UINT8  ; break ;
                    case GB_UINT16_code  : op = GrB_ABS_UINT16 ; break ;
                    case GB_UINT32_code  : op = GrB_ABS_UINT32 ; break ;
                    case GB_UINT64_code  : op = GrB_ABS_UINT64 ; break ;
                    case GB_FP32_code    : op = GrB_ABS_FP32   ; break ;
                    case GB_FP64_code    : op = GrB_ABS_FP64   ; break ;
                    case GB_FC32_code    : op = GxB_ABS_FC32   ; break ;
                    case GB_FC64_code    : op = GxB_ABS_FC64   ; break ;
                    default              : 
                        mexWarnMsgIdAndTxt ("GB:warn","unknown operator") ;
                        return (false) ;
                }
                break ;

            case GB_AINV_unop_code:

                switch (xcode)
                {
                    case GB_BOOL_code    : op = GrB_AINV_BOOL   ; break ;
                    case GB_INT8_code    : op = GrB_AINV_INT8   ; break ;
                    case GB_INT16_code   : op = GrB_AINV_INT16  ; break ;
                    case GB_INT32_code   : op = GrB_AINV_INT32  ; break ;
                    case GB_INT64_code   : op = GrB_AINV_INT64  ; break ;
                    case GB_UINT8_code   : op = GrB_AINV_UINT8  ; break ;
                    case GB_UINT16_code  : op = GrB_AINV_UINT16 ; break ;
                    case GB_UINT32_code  : op = GrB_AINV_UINT32 ; break ;
                    case GB_UINT64_code  : op = GrB_AINV_UINT64 ; break ;
                    case GB_FP32_code    : op = GrB_AINV_FP32   ; break ;
                    case GB_FP64_code    : op = GrB_AINV_FP64   ; break ;
                    case GB_FC32_code    : op = GxB_AINV_FC32   ; break ;
                    case GB_FC64_code    : op = GxB_AINV_FC64   ; break ;
                    default              : 
                        mexWarnMsgIdAndTxt ("GB:warn","unknown operator") ;
                        return (false) ;
                }
                break ;

            case GB_MINV_unop_code   :

                switch (xcode)
                {
                    case GB_BOOL_code    : op = GrB_MINV_BOOL   ; break ;
                    case GB_INT8_code    : op = GrB_MINV_INT8   ; break ;
                    case GB_INT16_code   : op = GrB_MINV_INT16  ; break ;
                    case GB_INT32_code   : op = GrB_MINV_INT32  ; break ;
                    case GB_INT64_code   : op = GrB_MINV_INT64  ; break ;
                    case GB_UINT8_code   : op = GrB_MINV_UINT8  ; break ;
                    case GB_UINT16_code  : op = GrB_MINV_UINT16 ; break ;
                    case GB_UINT32_code  : op = GrB_MINV_UINT32 ; break ;
                    case GB_UINT64_code  : op = GrB_MINV_UINT64 ; break ;
                    case GB_FP32_code    : op = GrB_MINV_FP32   ; break ;
                    case GB_FP64_code    : op = GrB_MINV_FP64   ; break ;
                    case GB_FC32_code    : op = GxB_MINV_FC32   ; break ;
                    case GB_FC64_code    : op = GxB_MINV_FC64   ; break ;
                    default              : 
                        mexWarnMsgIdAndTxt ("GB:warn","unknown operator") ;
                        return (false) ;
                }
                break ;

            case GB_LNOT_unop_code   :

                switch (xcode)
                {
                    case GB_BOOL_code    : op = GrB_LNOT        ; break ;
                    case GB_INT8_code    : op = GxB_LNOT_INT8   ; break ;
                    case GB_INT16_code   : op = GxB_LNOT_INT16  ; break ;
                    case GB_INT32_code   : op = GxB_LNOT_INT32  ; break ;
                    case GB_INT64_code   : op = GxB_LNOT_INT64  ; break ;
                    case GB_UINT8_code   : op = GxB_LNOT_UINT8  ; break ;
                    case GB_UINT16_code  : op = GxB_LNOT_UINT16 ; break ;
                    case GB_UINT32_code  : op = GxB_LNOT_UINT32 ; break ;
                    case GB_UINT64_code  : op = GxB_LNOT_UINT64 ; break ;
                    case GB_FP32_code    : op = GxB_LNOT_FP32   ; break ;
                    case GB_FP64_code    : op = GxB_LNOT_FP64   ; break ;
                    default              : 
                        mexWarnMsgIdAndTxt ("GB:warn","unknown operator") ;
                        return (false) ;
                }
                break ;

            case GB_BNOT_unop_code   :

                switch (xcode)
                {
                    case GB_INT8_code    : op = GrB_BNOT_INT8   ; break ;
                    case GB_INT16_code   : op = GrB_BNOT_INT16  ; break ;
                    case GB_INT32_code   : op = GrB_BNOT_INT32  ; break ;
                    case GB_INT64_code   : op = GrB_BNOT_INT64  ; break ;
                    case GB_UINT8_code   : op = GrB_BNOT_UINT8  ; break ;
                    case GB_UINT16_code  : op = GrB_BNOT_UINT16 ; break ;
                    case GB_UINT32_code  : op = GrB_BNOT_UINT32 ; break ;
                    case GB_UINT64_code  : op = GrB_BNOT_UINT64 ; break ;
                    default              : 
                        mexWarnMsgIdAndTxt ("GB:warn","unknown operator") ;
                        return (false) ;
                }
                break ;

    //--------------------------------------------------------------------------
    // unary operators for floating-point types (real and complex)
    //--------------------------------------------------------------------------

            case GB_SQRT_unop_code :    // z = sqrt (x)

                switch (xcode)
                {
                    case GB_FP32_code    : op = GxB_SQRT_FP32   ; break ;
                    case GB_FP64_code    : op = GxB_SQRT_FP64   ; break ;
                    case GB_FC32_code    : op = GxB_SQRT_FC32   ; break ;
                    case GB_FC64_code    : op = GxB_SQRT_FC64   ; break ;
                    default              : 
                        mexWarnMsgIdAndTxt ("GB:warn","unknown operator") ;
                        return (false) ;
                }
                break ;

            case GB_LOG_unop_code :     // z = log (x)

                switch (xcode)
                {
                    case GB_FP32_code    : op = GxB_LOG_FP32   ; break ;
                    case GB_FP64_code    : op = GxB_LOG_FP64   ; break ;
                    case GB_FC32_code    : op = GxB_LOG_FC32   ; break ;
                    case GB_FC64_code    : op = GxB_LOG_FC64   ; break ;
                    default              : 
                        mexWarnMsgIdAndTxt ("GB:warn","unknown operator") ;
                        return (false) ;
                }
                break ;

            case GB_EXP_unop_code :     // z = exp (x)

                switch (xcode)
                {
                    case GB_FP32_code    : op = GxB_EXP_FP32   ; break ;
                    case GB_FP64_code    : op = GxB_EXP_FP64   ; break ;
                    case GB_FC32_code    : op = GxB_EXP_FC32   ; break ;
                    case GB_FC64_code    : op = GxB_EXP_FC64   ; break ;
                    default              : 
                        mexWarnMsgIdAndTxt ("GB:warn","unknown operator") ;
                        return (false) ;
                }
                break ;


            case GB_SIN_unop_code :     // z = sin (x)

                switch (xcode)
                {
                    case GB_FP32_code    : op = GxB_SIN_FP32   ; break ;
                    case GB_FP64_code    : op = GxB_SIN_FP64   ; break ;
                    case GB_FC32_code    : op = GxB_SIN_FC32   ; break ;
                    case GB_FC64_code    : op = GxB_SIN_FC64   ; break ;
                    default              : 
                        mexWarnMsgIdAndTxt ("GB:warn","unknown operator") ;
                        return (false) ;
                }
                break ;

            case GB_COS_unop_code :     // z = cos (x)

                switch (xcode)
                {
                    case GB_FP32_code    : op = GxB_COS_FP32   ; break ;
                    case GB_FP64_code    : op = GxB_COS_FP64   ; break ;
                    case GB_FC32_code    : op = GxB_COS_FC32   ; break ;
                    case GB_FC64_code    : op = GxB_COS_FC64   ; break ;
                    default              : 
                        mexWarnMsgIdAndTxt ("GB:warn","unknown operator") ;
                        return (false) ;
                }
                break ;

            case GB_TAN_unop_code :     // z = tan (x)

                switch (xcode)
                {
                    case GB_FP32_code    : op = GxB_TAN_FP32   ; break ;
                    case GB_FP64_code    : op = GxB_TAN_FP64   ; break ;
                    case GB_FC32_code    : op = GxB_TAN_FC32   ; break ;
                    case GB_FC64_code    : op = GxB_TAN_FC64   ; break ;
                    default              : 
                        mexWarnMsgIdAndTxt ("GB:warn","unknown operator") ;
                        return (false) ;
                }
                break ;


            case GB_ASIN_unop_code :    // z = asin (x)

                switch (xcode)
                {
                    case GB_FP32_code    : op = GxB_ASIN_FP32   ; break ;
                    case GB_FP64_code    : op = GxB_ASIN_FP64   ; break ;
                    case GB_FC32_code    : op = GxB_ASIN_FC32   ; break ;
                    case GB_FC64_code    : op = GxB_ASIN_FC64   ; break ;
                    default              : 
                        mexWarnMsgIdAndTxt ("GB:warn","unknown operator") ;
                        return (false) ;
                }
                break ;

            case GB_ACOS_unop_code :    // z = acos (x)

                switch (xcode)
                {
                    case GB_FP32_code    : op = GxB_ACOS_FP32   ; break ;
                    case GB_FP64_code    : op = GxB_ACOS_FP64   ; break ;
                    case GB_FC32_code    : op = GxB_ACOS_FC32   ; break ;
                    case GB_FC64_code    : op = GxB_ACOS_FC64   ; break ;
                    default              : 
                        mexWarnMsgIdAndTxt ("GB:warn","unknown operator") ;
                        return (false) ;
                }
                break ;

            case GB_ATAN_unop_code :    // z = atan (x)

                switch (xcode)
                {
                    case GB_FP32_code    : op = GxB_ATAN_FP32   ; break ;
                    case GB_FP64_code    : op = GxB_ATAN_FP64   ; break ;
                    case GB_FC32_code    : op = GxB_ATAN_FC32   ; break ;
                    case GB_FC64_code    : op = GxB_ATAN_FC64   ; break ;
                    default              : 
                        mexWarnMsgIdAndTxt ("GB:warn","unknown operator") ;
                        return (false) ;
                }
                break ;


            case GB_SINH_unop_code :    // z = sinh (x)

                switch (xcode)
                {
                    case GB_FP32_code    : op = GxB_SINH_FP32   ; break ;
                    case GB_FP64_code    : op = GxB_SINH_FP64   ; break ;
                    case GB_FC32_code    : op = GxB_SINH_FC32   ; break ;
                    case GB_FC64_code    : op = GxB_SINH_FC64   ; break ;
                    default              : 
                        mexWarnMsgIdAndTxt ("GB:warn","unknown operator") ;
                        return (false) ;
                }
                break ;

            case GB_COSH_unop_code :    // z = cosh (x)

                switch (xcode)
                {
                    case GB_FP32_code    : op = GxB_COSH_FP32   ; break ;
                    case GB_FP64_code    : op = GxB_COSH_FP64   ; break ;
                    case GB_FC32_code    : op = GxB_COSH_FC32   ; break ;
                    case GB_FC64_code    : op = GxB_COSH_FC64   ; break ;
                    default              : 
                        mexWarnMsgIdAndTxt ("GB:warn","unknown operator") ;
                        return (false) ;
                }
                break ;

            case GB_TANH_unop_code :    // z = tanh (x)

                switch (xcode)
                {
                    case GB_FP32_code    : op = GxB_TANH_FP32   ; break ;
                    case GB_FP64_code    : op = GxB_TANH_FP64   ; break ;
                    case GB_FC32_code    : op = GxB_TANH_FC32   ; break ;
                    case GB_FC64_code    : op = GxB_TANH_FC64   ; break ;
                    default              : 
                        mexWarnMsgIdAndTxt ("GB:warn","unknown operator") ;
                        return (false) ;
                }
                break ;


            case GB_ASINH_unop_code :   // z = asinh (x)

                switch (xcode)
                {
                    case GB_FP32_code    : op = GxB_ASINH_FP32   ; break ;
                    case GB_FP64_code    : op = GxB_ASINH_FP64   ; break ;
                    case GB_FC32_code    : op = GxB_ASINH_FC32   ; break ;
                    case GB_FC64_code    : op = GxB_ASINH_FC64   ; break ;
                    default              : 
                        mexWarnMsgIdAndTxt ("GB:warn","unknown operator") ;
                        return (false) ;
                }
                break ;

            case GB_ACOSH_unop_code :   // z = acosh (x)

                switch (xcode)
                {
                    case GB_FP32_code    : op = GxB_ACOSH_FP32   ; break ;
                    case GB_FP64_code    : op = GxB_ACOSH_FP64   ; break ;
                    case GB_FC32_code    : op = GxB_ACOSH_FC32   ; break ;
                    case GB_FC64_code    : op = GxB_ACOSH_FC64   ; break ;
                    default              : 
                        mexWarnMsgIdAndTxt ("GB:warn","unknown operator") ;
                        return (false) ;
                }
                break ;

            case GB_ATANH_unop_code :   // z = atanh (x)

                switch (xcode)
                {
                    case GB_FP32_code    : op = GxB_ATANH_FP32   ; break ;
                    case GB_FP64_code    : op = GxB_ATANH_FP64   ; break ;
                    case GB_FC32_code    : op = GxB_ATANH_FC32   ; break ;
                    case GB_FC64_code    : op = GxB_ATANH_FC64   ; break ;
                    default              : 
                        mexWarnMsgIdAndTxt ("GB:warn","unknown operator") ;
                        return (false) ;
                }
                break ;


            case GB_SIGNUM_unop_code :    // z = signum (x)

                switch (xcode)
                {
                    case GB_FP32_code    : op = GxB_SIGNUM_FP32   ; break ;
                    case GB_FP64_code    : op = GxB_SIGNUM_FP64   ; break ;
                    case GB_FC32_code    : op = GxB_SIGNUM_FC32   ; break ;
                    case GB_FC64_code    : op = GxB_SIGNUM_FC64   ; break ;
                    default              : 
                        mexWarnMsgIdAndTxt ("GB:warn","unknown operator") ;
                        return (false) ;
                }
                break ;

            case GB_CEIL_unop_code :    // z = ceil (x)

                switch (xcode)
                {
                    case GB_FP32_code    : op = GxB_CEIL_FP32   ; break ;
                    case GB_FP64_code    : op = GxB_CEIL_FP64   ; break ;
                    case GB_FC32_code    : op = GxB_CEIL_FC32   ; break ;
                    case GB_FC64_code    : op = GxB_CEIL_FC64   ; break ;
                    default              : 
                        mexWarnMsgIdAndTxt ("GB:warn","unknown operator") ;
                        return (false) ;
                }
                break ;

            case GB_FLOOR_unop_code :   // z = floor (x)

                switch (xcode)
                {
                    case GB_FP32_code    : op = GxB_FLOOR_FP32   ; break ;
                    case GB_FP64_code    : op = GxB_FLOOR_FP64   ; break ;
                    case GB_FC32_code    : op = GxB_FLOOR_FC32   ; break ;
                    case GB_FC64_code    : op = GxB_FLOOR_FC64   ; break ;
                    default              : 
                        mexWarnMsgIdAndTxt ("GB:warn","unknown operator") ;
                        return (false) ;
                }
                break ;

            case GB_ROUND_unop_code :   // z = round (x)

                switch (xcode)
                {
                    case GB_FP32_code    : op = GxB_ROUND_FP32   ; break ;
                    case GB_FP64_code    : op = GxB_ROUND_FP64   ; break ;
                    case GB_FC32_code    : op = GxB_ROUND_FC32   ; break ;
                    case GB_FC64_code    : op = GxB_ROUND_FC64   ; break ;
                    default              : 
                        mexWarnMsgIdAndTxt ("GB:warn","unknown operator") ;
                        return (false) ;
                }
                break ;

            case GB_TRUNC_unop_code :   // z = trunc (x)

                switch (xcode)
                {
                    case GB_FP32_code    : op = GxB_TRUNC_FP32   ; break ;
                    case GB_FP64_code    : op = GxB_TRUNC_FP64   ; break ;
                    case GB_FC32_code    : op = GxB_TRUNC_FC32   ; break ;
                    case GB_FC64_code    : op = GxB_TRUNC_FC64   ; break ;
                    default              : 
                        mexWarnMsgIdAndTxt ("GB:warn","unknown operator") ;
                        return (false) ;
                }
                break ;


            case GB_EXP2_unop_code :    // z = exp2 (x)

                switch (xcode)
                {
                    case GB_FP32_code    : op = GxB_EXP2_FP32   ; break ;
                    case GB_FP64_code    : op = GxB_EXP2_FP64   ; break ;
                    case GB_FC32_code    : op = GxB_EXP2_FC32   ; break ;
                    case GB_FC64_code    : op = GxB_EXP2_FC64   ; break ;
                    default              : 
                        mexWarnMsgIdAndTxt ("GB:warn","unknown operator") ;
                        return (false) ;
                }
                break ;

            case GB_EXPM1_unop_code :   // z = expm1 (x)

                switch (xcode)
                {
                    case GB_FP32_code    : op = GxB_EXPM1_FP32   ; break ;
                    case GB_FP64_code    : op = GxB_EXPM1_FP64   ; break ;
                    case GB_FC32_code    : op = GxB_EXPM1_FC32   ; break ;
                    case GB_FC64_code    : op = GxB_EXPM1_FC64   ; break ;
                    default              : 
                        mexWarnMsgIdAndTxt ("GB:warn","unknown operator") ;
                        return (false) ;
                }
                break ;

            case GB_LOG10_unop_code :   // z = log10 (x)

                switch (xcode)
                {
                    case GB_FP32_code    : op = GxB_LOG10_FP32   ; break ;
                    case GB_FP64_code    : op = GxB_LOG10_FP64   ; break ;
                    case GB_FC32_code    : op = GxB_LOG10_FC32   ; break ;
                    case GB_FC64_code    : op = GxB_LOG10_FC64   ; break ;
                    default              : 
                        mexWarnMsgIdAndTxt ("GB:warn","unknown operator") ;
                        return (false) ;
                }
                break ;

            case GB_LOG1P_unop_code :   // z = log1P (x)

                switch (xcode)
                {
                    case GB_FP32_code    : op = GxB_LOG1P_FP32   ; break ;
                    case GB_FP64_code    : op = GxB_LOG1P_FP64   ; break ;
                    case GB_FC32_code    : op = GxB_LOG1P_FC32   ; break ;
                    case GB_FC64_code    : op = GxB_LOG1P_FC64   ; break ;
                    default              : 
                        mexWarnMsgIdAndTxt ("GB:warn","unknown operator") ;
                        return (false) ;
                }
                break ;

            case GB_LOG2_unop_code :    // z = log2 (x)

                switch (xcode)
                {
                    case GB_FP32_code    : op = GxB_LOG2_FP32   ; break ;
                    case GB_FP64_code    : op = GxB_LOG2_FP64   ; break ;
                    case GB_FC32_code    : op = GxB_LOG2_FC32   ; break ;
                    case GB_FC64_code    : op = GxB_LOG2_FC64   ; break ;
                    default              : 
                        mexWarnMsgIdAndTxt ("GB:warn","unknown operator") ;
                        return (false) ;
                }
                break ;

    //--------------------------------------------------------------------------
    // unary operators for real floating-point types
    //--------------------------------------------------------------------------

            case GB_LGAMMA_unop_code :  // z = lgamma (x)

                switch (xcode)
                {
                    case GB_FP32_code    : op = GxB_LGAMMA_FP32   ; break ;
                    case GB_FP64_code    : op = GxB_LGAMMA_FP64   ; break ;
                    default              : 
                        mexWarnMsgIdAndTxt ("GB:warn","unknown operator") ;
                        return (false) ;
                }
                break ;

            case GB_TGAMMA_unop_code :  // z = tgamma (x)

                switch (xcode)
                {
                    case GB_FP32_code    : op = GxB_TGAMMA_FP32   ; break ;
                    case GB_FP64_code    : op = GxB_TGAMMA_FP64   ; break ;
                    default              : 
                        mexWarnMsgIdAndTxt ("GB:warn","unknown operator") ;
                        return (false) ;
                }
                break ;

            case GB_ERF_unop_code :     // z = erf (x)

                switch (xcode)
                {
                    case GB_FP32_code    : op = GxB_ERF_FP32   ; break ;
                    case GB_FP64_code    : op = GxB_ERF_FP64   ; break ;
                    default              : 
                        mexWarnMsgIdAndTxt ("GB:warn","unknown operator") ;
                        return (false) ;
                }
                break ;

            case GB_ERFC_unop_code :    // z = erfc (x)

                switch (xcode)
                {
                    case GB_FP32_code    : op = GxB_ERFC_FP32   ; break ;
                    case GB_FP64_code    : op = GxB_ERFC_FP64   ; break ;
                    default              : 
                        mexWarnMsgIdAndTxt ("GB:warn","unknown operator") ;
                        return (false) ;
                }
                break ;

            case GB_CBRT_unop_code :    // z = cbrt (x)

                switch (xcode)
                {
                    case GB_FP32_code    : op = GxB_CBRT_FP32   ; break ;
                    case GB_FP64_code    : op = GxB_CBRT_FP64   ; break ;
                    default              : 
                        mexWarnMsgIdAndTxt ("GB:warn","unknown operator") ;
                        return (false) ;
                }
                break ;

            case GB_FREXPX_unop_code :  // z = frexpx (x), mantissa from frexp

                switch (xcode)
                {
                    case GB_FP32_code    : op = GxB_FREXPX_FP32   ; break ;
                    case GB_FP64_code    : op = GxB_FREXPX_FP64   ; break ;
                    default              : 
                        mexWarnMsgIdAndTxt ("GB:warn","unknown operator") ;
                        return (false) ;
                }
                break ;

            case GB_FREXPE_unop_code :  // z = frexpe (x), exponent from frexp

                switch (xcode)
                {
                    case GB_FP32_code    : op = GxB_FREXPE_FP32   ; break ;
                    case GB_FP64_code    : op = GxB_FREXPE_FP64   ; break ;
                    default              : 
                        mexWarnMsgIdAndTxt ("GB:warn","unknown operator") ;
                        return (false) ;
                }
                break ;


    //--------------------------------------------------------------------------
    // unary operators for complex types only
    //--------------------------------------------------------------------------

            case GB_CONJ_unop_code :    // z = conj (x)

                switch (xcode)
                {
                    case GB_FC32_code    : op = GxB_CONJ_FC32   ; break ;
                    case GB_FC64_code    : op = GxB_CONJ_FC64   ; break ;
                    default              : 
                        mexWarnMsgIdAndTxt ("GB:warn","unknown operator") ;
                        return (false) ;
                }
                break ;

    //--------------------------------------------------------------------------
    // unary operators where z is real and x is complex
    //--------------------------------------------------------------------------

            case GB_CREAL_unop_code :   // z = creal (x)

                switch (xcode)
                {
                    case GB_FC32_code    : op = GxB_CREAL_FC32   ; break ;
                    case GB_FC64_code    : op = GxB_CREAL_FC64   ; break ;
                    default              : 
                        mexWarnMsgIdAndTxt ("GB:warn","unknown operator") ;
                        return (false) ;
                }
                break ;

            case GB_CIMAG_unop_code :   // z = cimag (x)

                switch (xcode)
                {
                    case GB_FC32_code    : op = GxB_CIMAG_FC32   ; break ;
                    case GB_FC64_code    : op = GxB_CIMAG_FC64   ; break ;
                    default              : 
                        mexWarnMsgIdAndTxt ("GB:warn","unknown operator") ;
                        return (false) ;
                }
                break ;

            case GB_CARG_unop_code :    // z = carg (x)

                switch (xcode)
                {
                    case GB_FC32_code    : op = GxB_CARG_FC32   ; break ;
                    case GB_FC64_code    : op = GxB_CARG_FC64   ; break ;
                    default              : 
                        mexWarnMsgIdAndTxt ("GB:warn","unknown operator") ;
                        return (false) ;
                }
                break ;

    //--------------------------------------------------------------------------
    // unary operators where z is bool and x is any floating-point type
    //--------------------------------------------------------------------------

            case GB_ISINF_unop_code :   // z = isinf (x)

                switch (xcode)
                {
                    case GB_FP32_code    : op = GxB_ISINF_FP32   ; break ;
                    case GB_FP64_code    : op = GxB_ISINF_FP64   ; break ;
                    case GB_FC32_code    : op = GxB_ISINF_FC32   ; break ;
                    case GB_FC64_code    : op = GxB_ISINF_FC64   ; break ;
                    default              : 
                        mexWarnMsgIdAndTxt ("GB:warn","unknown operator") ;
                        return (false) ;
                }
                break ;

            case GB_ISNAN_unop_code :   // z = isnan (x)

                switch (xcode)
                {
                    case GB_FP32_code    : op = GxB_ISNAN_FP32   ; break ;
                    case GB_FP64_code    : op = GxB_ISNAN_FP64   ; break ;
                    case GB_FC32_code    : op = GxB_ISNAN_FC32   ; break ;
                    case GB_FC64_code    : op = GxB_ISNAN_FC64   ; break ;
                    default              : 
                        mexWarnMsgIdAndTxt ("GB:warn","unknown operator") ;
                        return (false) ;
                }
                break ;

            case GB_ISFINITE_unop_code: // z = isfinite (x)

                switch (xcode)
                {
                    case GB_FP32_code    : op = GxB_ISFINITE_FP32   ; break ;
                    case GB_FP64_code    : op = GxB_ISFINITE_FP64   ; break ;
                    case GB_FC32_code    : op = GxB_ISFINITE_FC32   ; break ;
                    case GB_FC64_code    : op = GxB_ISFINITE_FC64   ; break ;
                    default              : 
                        mexWarnMsgIdAndTxt ("GB:warn","unknown operator") ;
                        return (false) ;
                }
                break ;

    //--------------------------------------------------------------------------
    // positional ops
    //--------------------------------------------------------------------------

            case GB_POSITIONI_unop_code  : op = is64 ? GxB_POSITIONI_INT64  : GxB_POSITIONI_INT32  ; break ;
            case GB_POSITIONI1_unop_code : op = is64 ? GxB_POSITIONI1_INT64 : GxB_POSITIONI1_INT32 ; break ;
            case GB_POSITIONJ_unop_code  : op = is64 ? GxB_POSITIONJ_INT64  : GxB_POSITIONJ_INT32  ; break ;
            case GB_POSITIONJ1_unop_code : op = is64 ? GxB_POSITIONJ1_INT64 : GxB_POSITIONJ1_INT32 ; break ;

    //--------------------------------------------------------------------------

            default : 
                mexWarnMsgIdAndTxt ("GB:warn","unknown unary operator") ;
                return (false) ;
        }
    }

    //--------------------------------------------------------------------------
    // return result
    //--------------------------------------------------------------------------

    // return the unary operator to the caller
    ASSERT_UNARYOP_OK (op, "got unary op", GB0) ;
    (*op_handle) = op ;
    return (true) ;
}

