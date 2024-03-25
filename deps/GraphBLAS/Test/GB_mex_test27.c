//------------------------------------------------------------------------------
// GB_mex_test27: test GrB_get and GrB_set (unary ops)
//------------------------------------------------------------------------------

// SuiteSparse:GraphBLAS, Timothy A. Davis, (c) 2017-2023, All Rights Reserved.
// SPDX-License-Identifier: Apache-2.0

//------------------------------------------------------------------------------

#include "GB_mex.h"
#include "GB_mex_errors.h"

#define USAGE "GB_mex_test27"

#define FREE_ALL ;
#define GET_DEEP_COPY ;
#define FREE_DEEP_COPY ;

#define GETOP(op,opname)                                                \
{                                                                       \
    OK (GrB_UnaryOp_get_String (op, name, GrB_NAME)) ;                  \
    CHECK (MATCH (name, opname)) ;                                      \
    OK (GrB_UnaryOp_get_String (op, cname, GxB_JIT_C_NAME)) ;           \
    printf ("%s: %s\n", name, cname) ; \
    OK (GrB_UnaryOp_get_SIZE (op, &size, GrB_NAME)) ;                   \
    CHECK (size == strlen (name) + 1) ;                                 \
    GrB_Info info2, info3 ;                                             \
    info2 = GrB_UnaryOp_get_SIZE (op, &size, GrB_INP0_TYPE_STRING) ;   \
    info3 = GrB_UnaryOp_get_String (op, name, GrB_INP0_TYPE_STRING) ;  \
    CHECK (info2 == info3) ;                                            \
    CHECK (size == strlen (name) + 1) ;                                 \
    if (info2 == GrB_NO_VALUE) { CHECK (size == 1) ; }                  \
    info2 = GrB_UnaryOp_get_SIZE (op, &size, GrB_INP1_TYPE_STRING) ;   \
    info3 = GrB_UnaryOp_get_String (op, name, GrB_INP1_TYPE_STRING) ;  \
    CHECK (info2 == info3) ;                                            \
    CHECK (size == 1) ;                                                 \
    CHECK (info2 == GrB_NO_VALUE) ;                                     \
    info2 = GrB_UnaryOp_get_SIZE (op, &size, GrB_OUTP_TYPE_STRING) ;   \
    info3 = GrB_UnaryOp_get_String (op, name, GrB_OUTP_TYPE_STRING) ;  \
    CHECK (info2 == info3) ;                                            \
    CHECK (size == strlen (name) + 1) ;                                 \
    if (info2 == GrB_NO_VALUE) { CHECK (size == 1) ; }                  \
}

#define GETNAME(op)                                         \
{                                                           \
    GETOP (op, #op) ;                                       \
/*  OK (GxB_UnaryOp_fprint (op, "unop", 3, NULL)) ; */      \
}

#define GETNAM2(op,alias)                                   \
{                                                           \
    GETOP (op,alias) ;                                      \
/*  OK (GxB_UnaryOp_fprint (op, "unop", 3, NULL)) ; */      \
}

void myfunc (float *z, const float *x) ;
void myfunc (float *z, const float *x) { (*z) = -(*x) ; }
#define MYFUNC_DEFN \
"void myfunc (float *z, const float *x) { (*z) = -(*x) ; }"

void mexFunction
(
    int nargout,
    mxArray *pargout [ ],
    int nargin,
    const mxArray *pargin [ ]
)
{

    //--------------------------------------------------------------------------
    // startup GraphBLAS
    //--------------------------------------------------------------------------

    GrB_Info info, expected ;
    bool malloc_debug = GB_mx_get_global (true) ;
    GrB_Scalar s = NULL, s_fp64 = NULL, s_int32 = NULL, s_fp32 = NULL ;
    GrB_UnaryOp unop = NULL ;
    uint8_t stuff [256] ;
    void *nothing = stuff ;
    size_t size ;
    char name [256] ;
    char cname [256] ;
    char defn [2048] ;
    int32_t code, i ;
    float fvalue ;
    double dvalue ;

    OK (GrB_Scalar_new (&s_fp64, GrB_FP64)) ;
    OK (GrB_Scalar_new (&s_fp32, GrB_FP32)) ;
    OK (GrB_Scalar_new (&s_int32, GrB_INT32)) ;

    //--------------------------------------------------------------------------
    // GrB_UnaryOp get name
    //--------------------------------------------------------------------------

    GETNAME (GrB_IDENTITY_BOOL) ;
    GETNAME (GrB_IDENTITY_INT8) ;
    GETNAME (GrB_IDENTITY_INT16) ;
    GETNAME (GrB_IDENTITY_INT32) ;
    GETNAME (GrB_IDENTITY_INT64) ;
    GETNAME (GrB_IDENTITY_UINT8) ;
    GETNAME (GrB_IDENTITY_UINT16) ;
    GETNAME (GrB_IDENTITY_UINT32) ;
    GETNAME (GrB_IDENTITY_UINT64) ;
    GETNAME (GrB_IDENTITY_FP32) ;
    GETNAME (GrB_IDENTITY_FP64) ;
    GETNAME (GxB_IDENTITY_FC32) ;
    GETNAME (GxB_IDENTITY_FC64) ;

    GETNAME (GrB_AINV_BOOL) ;
    GETNAME (GrB_AINV_INT8) ;
    GETNAME (GrB_AINV_INT16) ;
    GETNAME (GrB_AINV_INT32) ;
    GETNAME (GrB_AINV_INT64) ;
    GETNAME (GrB_AINV_UINT8) ;
    GETNAME (GrB_AINV_UINT16) ;
    GETNAME (GrB_AINV_UINT32) ;
    GETNAME (GrB_AINV_UINT64) ;
    GETNAME (GrB_AINV_FP32) ;
    GETNAME (GrB_AINV_FP64) ;
    GETNAME (GxB_AINV_FC32) ;
    GETNAME (GxB_AINV_FC64) ;

    GETNAME (GrB_MINV_BOOL) ;
    GETNAME (GrB_MINV_INT8) ;
    GETNAME (GrB_MINV_INT16) ;
    GETNAME (GrB_MINV_INT32) ;
    GETNAME (GrB_MINV_INT64) ;
    GETNAME (GrB_MINV_UINT8) ;
    GETNAME (GrB_MINV_UINT16) ;
    GETNAME (GrB_MINV_UINT32) ;
    GETNAME (GrB_MINV_UINT64) ;
    GETNAME (GrB_MINV_FP32) ;
    GETNAME (GrB_MINV_FP64) ;
    GETNAME (GxB_MINV_FC32) ;
    GETNAME (GxB_MINV_FC64) ;

    GETNAME (GrB_LNOT) ;
    GETNAM2 (GxB_LNOT_BOOL, "GrB_LNOT") ;
    GETNAME (GxB_LNOT_INT8) ;
    GETNAME (GxB_LNOT_INT16) ;
    GETNAME (GxB_LNOT_INT32) ;
    GETNAME (GxB_LNOT_INT64) ;
    GETNAME (GxB_LNOT_UINT8) ;
    GETNAME (GxB_LNOT_UINT16) ;
    GETNAME (GxB_LNOT_UINT32) ;
    GETNAME (GxB_LNOT_UINT64) ;
    GETNAME (GxB_LNOT_FP32) ;
    GETNAME (GxB_LNOT_FP64) ;

    GETNAME (GxB_ONE_BOOL) ;
    GETNAME (GxB_ONE_INT8) ;
    GETNAME (GxB_ONE_INT16) ;
    GETNAME (GxB_ONE_INT32) ;
    GETNAME (GxB_ONE_INT64) ;
    GETNAME (GxB_ONE_UINT8) ;
    GETNAME (GxB_ONE_UINT16) ;
    GETNAME (GxB_ONE_UINT32) ;
    GETNAME (GxB_ONE_UINT64) ;
    GETNAME (GxB_ONE_FP32) ;
    GETNAME (GxB_ONE_FP64) ;
    GETNAME (GxB_ONE_FC32) ;
    GETNAME (GxB_ONE_FC64) ;

    GETNAME (GrB_ABS_BOOL) ;
    GETNAME (GrB_ABS_INT8) ;
    GETNAME (GrB_ABS_INT16) ;
    GETNAME (GrB_ABS_INT32) ;
    GETNAME (GrB_ABS_INT64) ;
    GETNAME (GrB_ABS_UINT8) ;
    GETNAME (GrB_ABS_UINT16) ;
    GETNAME (GrB_ABS_UINT32) ;
    GETNAME (GrB_ABS_UINT64) ;
    GETNAME (GrB_ABS_FP32) ;
    GETNAME (GrB_ABS_FP64) ;

    GETNAME (GrB_BNOT_INT8) ;
    GETNAME (GrB_BNOT_INT16) ;
    GETNAME (GrB_BNOT_INT32) ;
    GETNAME (GrB_BNOT_INT64) ;
    GETNAME (GrB_BNOT_UINT8) ;
    GETNAME (GrB_BNOT_UINT16) ;
    GETNAME (GrB_BNOT_UINT32) ;
    GETNAME (GrB_BNOT_UINT64) ;

    GETNAM2 (GxB_ABS_BOOL,      "GrB_ABS_BOOL") ;
    GETNAM2 (GxB_ABS_INT8,      "GrB_ABS_INT8") ;
    GETNAM2 (GxB_ABS_INT16,     "GrB_ABS_INT16") ;
    GETNAM2 (GxB_ABS_INT32,     "GrB_ABS_INT32") ;
    GETNAM2 (GxB_ABS_INT64,     "GrB_ABS_INT64") ;
    GETNAM2 (GxB_ABS_UINT8,     "GrB_ABS_UINT8") ;
    GETNAM2 (GxB_ABS_UINT16,    "GrB_ABS_UINT16") ;
    GETNAM2 (GxB_ABS_UINT32,    "GrB_ABS_UINT32") ;
    GETNAM2 (GxB_ABS_UINT64,    "GrB_ABS_UINT64") ;
    GETNAM2 (GxB_ABS_FP32,      "GrB_ABS_FP32") ;
    GETNAM2 (GxB_ABS_FP64,      "GrB_ABS_FP64") ;
    GETNAME (GxB_ABS_FC32) ;
    GETNAME (GxB_ABS_FC64) ;

    GETNAME (GxB_SQRT_FP32) ;
    GETNAME (GxB_SQRT_FP64) ;
    GETNAME (GxB_SQRT_FC32) ;
    GETNAME (GxB_SQRT_FC64) ;

    GETNAME (GxB_LOG_FP32) ;
    GETNAME (GxB_LOG_FP64) ;
    GETNAME (GxB_LOG_FC32) ;
    GETNAME (GxB_LOG_FC64) ;

    GETNAME (GxB_EXP_FP32) ;
    GETNAME (GxB_EXP_FP64) ;
    GETNAME (GxB_EXP_FC32) ;
    GETNAME (GxB_EXP_FC64) ;

    GETNAME (GxB_LOG2_FP32) ;
    GETNAME (GxB_LOG2_FP64) ;
    GETNAME (GxB_LOG2_FC32) ;
    GETNAME (GxB_LOG2_FC64) ;

    GETNAME (GxB_SIN_FP32) ;
    GETNAME (GxB_SIN_FP64) ;
    GETNAME (GxB_SIN_FC32) ;
    GETNAME (GxB_SIN_FC64) ;

    GETNAME (GxB_COS_FP32) ;
    GETNAME (GxB_COS_FP64) ;
    GETNAME (GxB_COS_FC32) ;
    GETNAME (GxB_COS_FC64) ;

    GETNAME (GxB_TAN_FP32) ;
    GETNAME (GxB_TAN_FP64) ;
    GETNAME (GxB_TAN_FC32) ;
    GETNAME (GxB_TAN_FC64) ;

    GETNAME (GxB_ACOS_FP32) ;
    GETNAME (GxB_ACOS_FP64) ;
    GETNAME (GxB_ACOS_FC32) ;
    GETNAME (GxB_ACOS_FC64) ;

    GETNAME (GxB_ASIN_FP32) ;
    GETNAME (GxB_ASIN_FP64) ;
    GETNAME (GxB_ASIN_FC32) ;
    GETNAME (GxB_ASIN_FC64) ;

    GETNAME (GxB_ATAN_FP32) ;
    GETNAME (GxB_ATAN_FP64) ;
    GETNAME (GxB_ATAN_FC32) ;
    GETNAME (GxB_ATAN_FC64) ;

    GETNAME (GxB_SINH_FP32) ;
    GETNAME (GxB_SINH_FP64) ;
    GETNAME (GxB_SINH_FC32) ;
    GETNAME (GxB_SINH_FC64) ;

    GETNAME (GxB_COSH_FP32) ;
    GETNAME (GxB_COSH_FP64) ;
    GETNAME (GxB_COSH_FC32) ;
    GETNAME (GxB_COSH_FC64) ;

    GETNAME (GxB_TANH_FP32) ;
    GETNAME (GxB_TANH_FP64) ;
    GETNAME (GxB_TANH_FC32) ;
    GETNAME (GxB_TANH_FC64) ;

    GETNAME (GxB_ATANH_FP32) ;
    GETNAME (GxB_ATANH_FP64) ;
    GETNAME (GxB_ATANH_FC32) ;
    GETNAME (GxB_ATANH_FC64) ;

    GETNAME (GxB_ASINH_FP32) ;
    GETNAME (GxB_ASINH_FP64) ;
    GETNAME (GxB_ASINH_FC32) ;
    GETNAME (GxB_ASINH_FC64) ;

    GETNAME (GxB_ACOSH_FP32) ;
    GETNAME (GxB_ACOSH_FP64) ;
    GETNAME (GxB_ACOSH_FC32) ;
    GETNAME (GxB_ACOSH_FC64) ;

    GETNAME (GxB_SIGNUM_FP32) ;
    GETNAME (GxB_SIGNUM_FP64) ;
    GETNAME (GxB_SIGNUM_FC32) ;
    GETNAME (GxB_SIGNUM_FC64) ;

    GETNAME (GxB_CEIL_FP32) ;
    GETNAME (GxB_CEIL_FP64) ;
    GETNAME (GxB_CEIL_FC32) ;
    GETNAME (GxB_CEIL_FC64) ;

    GETNAME (GxB_FLOOR_FP32) ;
    GETNAME (GxB_FLOOR_FP64) ;
    GETNAME (GxB_FLOOR_FC32) ;
    GETNAME (GxB_FLOOR_FC64) ;

    GETNAME (GxB_ROUND_FP32) ;
    GETNAME (GxB_ROUND_FP64) ;
    GETNAME (GxB_ROUND_FC32) ;
    GETNAME (GxB_ROUND_FC64) ;

    GETNAME (GxB_TRUNC_FP32) ;
    GETNAME (GxB_TRUNC_FP64) ;
    GETNAME (GxB_TRUNC_FC32) ;
    GETNAME (GxB_TRUNC_FC64) ;

    GETNAME (GxB_EXP2_FP32) ;
    GETNAME (GxB_EXP2_FP64) ;
    GETNAME (GxB_EXP2_FC32) ;
    GETNAME (GxB_EXP2_FC64) ;

    GETNAME (GxB_EXPM1_FP32) ;
    GETNAME (GxB_EXPM1_FP64) ;
    GETNAME (GxB_EXPM1_FC32) ;
    GETNAME (GxB_EXPM1_FC64) ;

    GETNAME (GxB_LOG10_FP32) ;
    GETNAME (GxB_LOG10_FP64) ;
    GETNAME (GxB_LOG10_FC32) ;
    GETNAME (GxB_LOG10_FC64) ;

    GETNAME (GxB_LOG1P_FP32) ;
    GETNAME (GxB_LOG1P_FP64) ;
    GETNAME (GxB_LOG1P_FC32) ;
    GETNAME (GxB_LOG1P_FC64) ;

    GETNAME (GxB_LGAMMA_FP32) ;
    GETNAME (GxB_LGAMMA_FP64) ;

    GETNAME (GxB_TGAMMA_FP32) ;
    GETNAME (GxB_TGAMMA_FP64) ;

    GETNAME (GxB_ERF_FP32) ;
    GETNAME (GxB_ERF_FP64) ;

    GETNAME (GxB_ERFC_FP32) ;
    GETNAME (GxB_ERFC_FP64) ;

    GETNAME (GxB_CBRT_FP32) ;
    GETNAME (GxB_CBRT_FP64) ;

    GETNAME (GxB_FREXPX_FP32) ;
    GETNAME (GxB_FREXPX_FP64) ;

    GETNAME (GxB_FREXPE_FP32) ;
    GETNAME (GxB_FREXPE_FP64) ;

    GETNAME (GxB_CONJ_FC32) ;
    GETNAME (GxB_CONJ_FC64) ;

    GETNAME (GxB_CREAL_FC32) ;
    GETNAME (GxB_CREAL_FC64) ;

    GETNAME (GxB_CIMAG_FC32) ;
    GETNAME (GxB_CIMAG_FC64) ;

    GETNAME (GxB_CARG_FC32) ;
    GETNAME (GxB_CARG_FC64) ;

    GETNAME (GxB_ISINF_FP32) ;
    GETNAME (GxB_ISINF_FP64) ;
    GETNAME (GxB_ISINF_FC32) ;
    GETNAME (GxB_ISINF_FC64) ;

    GETNAME (GxB_ISNAN_FP32) ;
    GETNAME (GxB_ISNAN_FP64) ;
    GETNAME (GxB_ISNAN_FC32) ;
    GETNAME (GxB_ISNAN_FC64) ;

    GETNAME (GxB_ISFINITE_FP32) ;
    GETNAME (GxB_ISFINITE_FP64) ;
    GETNAME (GxB_ISFINITE_FC32) ;
    GETNAME (GxB_ISFINITE_FC64) ;

    GETNAME (GxB_POSITIONI_INT32) ;
    GETNAME (GxB_POSITIONI_INT64) ;
    GETNAME (GxB_POSITIONI1_INT32) ;
    GETNAME (GxB_POSITIONI1_INT64) ;

    GETNAME (GxB_POSITIONJ_INT32) ;
    GETNAME (GxB_POSITIONJ_INT64) ;
    GETNAME (GxB_POSITIONJ1_INT32) ;
    GETNAME (GxB_POSITIONJ1_INT64) ;

    //--------------------------------------------------------------------------
    // other get/set methods for GrB_UnaryOp
    //--------------------------------------------------------------------------

    OK (GrB_UnaryOp_get_INT32_(GrB_ABS_FP32, &code, GrB_INP0_TYPE_CODE)) ;
    CHECK (code == GrB_FP32_CODE) ;

    OK (GrB_UnaryOp_get_String_(GrB_ABS_FP32, name, GrB_INP0_TYPE_STRING)) ;
    CHECK (MATCH (name, "GrB_FP32")) ;

    OK (GrB_UnaryOp_get_INT32_(GrB_ABS_FP64, &code, GrB_OUTP_TYPE_CODE)) ;
    CHECK (code == GrB_FP64_CODE) ;

    OK (GrB_UnaryOp_get_String_(GrB_ABS_FP64, name, GrB_OUTP_TYPE_STRING)) ;
    CHECK (MATCH (name, "GrB_FP64")) ;

    OK (GrB_UnaryOp_get_Scalar_(GrB_ABS_FP32, s_int32, GrB_INP0_TYPE_CODE)) ;
    OK (GrB_Scalar_extractElement_INT32_(&code, s_int32)) ;
    CHECK (code == GrB_FP32_CODE) ;

    OK (GrB_UnaryOp_get_Scalar_(GrB_LNOT, s_int32, GrB_OUTP_TYPE_CODE)) ;
    OK (GrB_Scalar_extractElement_INT32_(&code, s_int32)) ;
    CHECK (code == GrB_BOOL_CODE) ;

    expected = GrB_NO_VALUE ;
    ERR (GrB_UnaryOp_get_INT32_(GrB_BNOT_UINT8, &code, GrB_INP1_TYPE_CODE)) ;
    ERR (GrB_UnaryOp_get_Scalar_(GrB_LNOT, s_int32, GrB_INP1_TYPE_CODE)) ;
    ERR (GrB_UnaryOp_get_String_(GrB_BNOT_UINT8, name, GrB_INP1_TYPE_STRING)) ;
    ERR (GrB_UnaryOp_get_SIZE_(GrB_BNOT_UINT8, &size, GrB_INP1_TYPE_STRING)) ;

    expected = GrB_INVALID_VALUE ;
    ERR (GrB_UnaryOp_get_INT32_(GrB_BNOT_UINT8, &code, GrB_NAME)) ;

    expected = GrB_INVALID_VALUE ;
    ERR (GrB_UnaryOp_get_VOID_(GrB_LNOT, nothing, 0)) ;

    OK (GrB_UnaryOp_new (&unop, myfunc, GrB_FP32, GrB_FP32)) ;
    OK (GrB_UnaryOp_get_SIZE_(unop, &size, GrB_NAME)) ;
    CHECK (size == 1) ;
    OK (GrB_UnaryOp_get_SIZE_(unop, &size, GxB_JIT_C_NAME)) ;
    CHECK (size == 1) ;
    OK (GrB_UnaryOp_get_SIZE_(unop, &size, GxB_JIT_C_DEFINITION)) ;
    CHECK (size == 1) ;

    expected = GrB_INVALID_VALUE ;
    OK (GrB_UnaryOp_set_String_(unop, "myfunc", GxB_JIT_C_NAME)) ;
    OK (GrB_UnaryOp_get_String_(unop, name, GxB_JIT_C_NAME)) ;
    CHECK (MATCH (name, "myfunc")) ;

    CHECK (unop->hash == UINT64_MAX) ;
    METHOD (GrB_UnaryOp_set_String (unop, MYFUNC_DEFN, GxB_JIT_C_DEFINITION)) ;
    OK (GrB_UnaryOp_get_String_(unop, defn, GxB_JIT_C_DEFINITION)) ;
    CHECK (MATCH (defn, MYFUNC_DEFN)) ;
    CHECK (unop->hash != UINT64_MAX) ;
    OK (GxB_print (unop, 3)) ;

    OK (GrB_UnaryOp_set_String_(unop, "user name for myfunc", GrB_NAME)) ;
    OK (GrB_UnaryOp_get_String_(unop, name, GrB_NAME)) ;
    CHECK (MATCH (name, "user name for myfunc")) ;
    expected = GrB_ALREADY_SET ;
    ERR (GrB_UnaryOp_set_String_(unop, "another user name", GrB_NAME)) ;
    printf ("    test GrB_ALREADY_SET: ok\n") ;

    expected = GrB_NO_VALUE ;
    ERR (GrB_UnaryOp_get_INT32_(unop, &code, GrB_INP1_TYPE_CODE)) ;

    expected = GrB_INVALID_VALUE ;
    ERR (GrB_UnaryOp_set_String_(unop, "another_name", 999)) ;
    ERR (GrB_UnaryOp_get_SIZE(unop, &size, 999)) ;

    expected = GrB_ALREADY_SET ;
    ERR (GrB_UnaryOp_set_String_(unop, "another_name", GxB_JIT_C_NAME)) ;
    ERR (GrB_UnaryOp_set_String_(unop, "another_defn", GxB_JIT_C_DEFINITION)) ;
    CHECK (MATCH ("GrB_ALREADY_SET", GB_status_code (GrB_ALREADY_SET))) ;

    expected = GrB_INVALID_VALUE ;
    ERR (GrB_UnaryOp_set_String_(GrB_LNOT, "another_name", GxB_JIT_C_NAME)) ;
    ERR (GrB_UnaryOp_set_Scalar_(unop, s_int32, 0)) ;
    ERR (GrB_UnaryOp_set_INT32_(unop, 0, 0)) ;
    ERR (GrB_UnaryOp_set_VOID_(unop, nothing, 0, 0)) ;

    //--------------------------------------------------------------------------
    // finalize GraphBLAS
    //--------------------------------------------------------------------------

    GrB_free (&s_fp64) ;
    GrB_free (&s_fp32) ;
    GrB_free (&s_int32) ;
    GrB_free (&unop) ;
    GB_mx_put_global (true) ;
    printf ("\nGB_mex_test27:  all tests passed\n\n") ;
}

