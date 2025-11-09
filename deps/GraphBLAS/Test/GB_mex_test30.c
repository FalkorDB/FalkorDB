//------------------------------------------------------------------------------
// GB_mex_test30: test GrB_get and GrB_set (index unary ops)
//------------------------------------------------------------------------------

// SuiteSparse:GraphBLAS, Timothy A. Davis, (c) 2017-2025, All Rights Reserved.
// SPDX-License-Identifier: Apache-2.0

//------------------------------------------------------------------------------

#include "GB_mex.h"
#include "GB_mex_errors.h"

#define FREE_ALL ;
#define GET_DEEP_COPY ;
#define FREE_DEEP_COPY ;

#define GETOP(op,opname)                                                    \
{                                                                           \
    size_t siz1, siz2, siz3 ;                                               \
    OK (GrB_IndexUnaryOp_get_String (op, name, GrB_NAME)) ;                 \
    CHECK (MATCH (name, opname)) ;                                          \
    OK (GrB_IndexUnaryOp_get_String (op, cname, GxB_JIT_C_NAME));           \
    printf ("\n%s: %s\n", name, cname) ;                                    \
    OK (GxB_IndexUnaryOp_fprint (op, "idxunop", 5, stdout)) ;               \
    OK (GrB_IndexUnaryOp_get_SIZE (op, &size, GrB_NAME)) ;                  \
    CHECK (size == strlen (name) + 1) ;                                     \
    GrB_Info info2, info3 ;                                                 \
    info2 = GrB_IndexUnaryOp_get_SIZE (op, &siz1, GrB_INP0_TYPE_STRING) ;   \
    info3 = GrB_IndexUnaryOp_get_String (op, name, GrB_INP0_TYPE_STRING) ;  \
    CHECK (info2 == info3) ;                                                \
    CHECK (siz1 == strlen (name) + 1) ;                                     \
    if (info2 == GrB_NO_VALUE) { CHECK (siz1 == 1) ; }                      \
    info2 = GrB_IndexUnaryOp_get_SIZE (op, &siz2, GrB_INP1_TYPE_STRING) ;   \
    info3 = GrB_IndexUnaryOp_get_String (op, name, GrB_INP1_TYPE_STRING) ;  \
    CHECK (info2 == info3) ;                                                \
    CHECK (siz2 == strlen (name) + 1) ;                                     \
    if (info2 == GrB_NO_VALUE) { CHECK (siz2 == 1) ; }                      \
    info2 = GrB_IndexUnaryOp_get_SIZE (op, &siz3, GrB_OUTP_TYPE_STRING) ;   \
    info3 = GrB_IndexUnaryOp_get_String (op, name, GrB_OUTP_TYPE_STRING) ;  \
    CHECK (info2 == info3) ;                                                \
    CHECK (siz3 == strlen (name) + 1) ;                                     \
    if (info2 == GrB_NO_VALUE) { CHECK (siz3 == 1) ; }                      \
}

#define GETNAME(op)                                         \
{                                                           \
    GETOP (op, #op) ;                                       \
}

void myfunc (bool *z, const float *x, uint64_t i, uint64_t j,
    const float *y) ;
void myfunc (bool *z, const float *x, uint64_t i, uint64_t j,
    const float *y)
{
    (*z) = (*x) > 2 ;
}

#define MYFUNC_DEFN \
"void myfunc (bool *z, const float *x, uint64_t i, uint64_t j,      \n" \
"    const float *y)                                                \n" \
"{                                                                  \n" \
"    (*z) = (*x) > 2 ;                                              \n" \
"}"

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
    GrB_IndexUnaryOp op = NULL ;
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
    // GrB_IndexUnaryOp get name
    //--------------------------------------------------------------------------

    GETNAME (GrB_ROWINDEX_INT32) ;
    GETNAME (GrB_ROWINDEX_INT64) ;
    GETNAME (GrB_COLINDEX_INT32) ;
    GETNAME (GrB_COLINDEX_INT64) ;
    GETNAME (GrB_DIAGINDEX_INT32) ;
    GETNAME (GrB_DIAGINDEX_INT64) ;

    GETNAME (GxB_FLIPDIAGINDEX_INT32) ;
    GETNAME (GxB_FLIPDIAGINDEX_INT64) ;

    GETNAME (GrB_TRIL) ;
    GETNAME (GrB_TRIU) ;
    GETNAME (GrB_DIAG) ;
    GETNAME (GrB_OFFDIAG) ;

    GETNAME (GrB_COLLE) ;
    GETNAME (GrB_COLGT) ;
    GETNAME (GrB_ROWLE) ;
    GETNAME (GrB_ROWGT) ;

    GETNAME (GrB_VALUEEQ_BOOL) ;
    GETNAME (GrB_VALUEEQ_INT8) ;
    GETNAME (GrB_VALUEEQ_INT16) ;
    GETNAME (GrB_VALUEEQ_INT32) ;
    GETNAME (GrB_VALUEEQ_INT64) ;
    GETNAME (GrB_VALUEEQ_UINT8) ;
    GETNAME (GrB_VALUEEQ_UINT16) ;
    GETNAME (GrB_VALUEEQ_UINT32) ;
    GETNAME (GrB_VALUEEQ_UINT64) ;
    GETNAME (GrB_VALUEEQ_FP32) ;
    GETNAME (GrB_VALUEEQ_FP64) ;
    GETNAME (GxB_VALUEEQ_FC32) ;
    GETNAME (GxB_VALUEEQ_FC64) ;

    GETNAME (GrB_VALUENE_BOOL) ;
    GETNAME (GrB_VALUENE_INT8) ;
    GETNAME (GrB_VALUENE_INT16) ;
    GETNAME (GrB_VALUENE_INT32) ;
    GETNAME (GrB_VALUENE_INT64) ;
    GETNAME (GrB_VALUENE_UINT8) ;
    GETNAME (GrB_VALUENE_UINT16) ;
    GETNAME (GrB_VALUENE_UINT32) ;
    GETNAME (GrB_VALUENE_UINT64) ;
    GETNAME (GrB_VALUENE_FP32) ;
    GETNAME (GrB_VALUENE_FP64) ;
    GETNAME (GxB_VALUENE_FC32) ;
    GETNAME (GxB_VALUENE_FC64) ;

    GETNAME (GrB_VALUELT_BOOL) ;
    GETNAME (GrB_VALUELT_INT8) ;
    GETNAME (GrB_VALUELT_INT16) ;
    GETNAME (GrB_VALUELT_INT32) ;
    GETNAME (GrB_VALUELT_INT64) ;
    GETNAME (GrB_VALUELT_UINT8) ;
    GETNAME (GrB_VALUELT_UINT16) ;
    GETNAME (GrB_VALUELT_UINT32) ;
    GETNAME (GrB_VALUELT_UINT64) ;
    GETNAME (GrB_VALUELT_FP32) ;
    GETNAME (GrB_VALUELT_FP64) ;

    GETNAME (GrB_VALUELE_BOOL) ;
    GETNAME (GrB_VALUELE_INT8) ;
    GETNAME (GrB_VALUELE_INT16) ;
    GETNAME (GrB_VALUELE_INT32) ;
    GETNAME (GrB_VALUELE_INT64) ;
    GETNAME (GrB_VALUELE_UINT8) ;
    GETNAME (GrB_VALUELE_UINT16) ;
    GETNAME (GrB_VALUELE_UINT32) ;
    GETNAME (GrB_VALUELE_UINT64) ;
    GETNAME (GrB_VALUELE_FP32) ;
    GETNAME (GrB_VALUELE_FP64) ;

    GETNAME (GrB_VALUEGT_BOOL) ;
    GETNAME (GrB_VALUEGT_INT8) ;
    GETNAME (GrB_VALUEGT_INT16) ;
    GETNAME (GrB_VALUEGT_INT32) ;
    GETNAME (GrB_VALUEGT_INT64) ;
    GETNAME (GrB_VALUEGT_UINT8) ;
    GETNAME (GrB_VALUEGT_UINT16) ;
    GETNAME (GrB_VALUEGT_UINT32) ;
    GETNAME (GrB_VALUEGT_UINT64) ;
    GETNAME (GrB_VALUEGT_FP32) ;
    GETNAME (GrB_VALUEGT_FP64) ;

    GETNAME (GrB_VALUEGE_BOOL) ;
    GETNAME (GrB_VALUEGE_INT8) ;
    GETNAME (GrB_VALUEGE_INT16) ;
    GETNAME (GrB_VALUEGE_INT32) ;
    GETNAME (GrB_VALUEGE_INT64) ;
    GETNAME (GrB_VALUEGE_UINT8) ;
    GETNAME (GrB_VALUEGE_UINT16) ;
    GETNAME (GrB_VALUEGE_UINT32) ;
    GETNAME (GrB_VALUEGE_UINT64) ;
    GETNAME (GrB_VALUEGE_FP32) ;
    GETNAME (GrB_VALUEGE_FP64) ;

    GETNAME (GxB_NONZOMBIE) ;

    //--------------------------------------------------------------------------
    // other get/set methods for GrB_IndexUnaryOp
    //--------------------------------------------------------------------------

    OK (GrB_IndexUnaryOp_get_INT32_(GrB_VALUEGE_FP32, &code, GrB_INP0_TYPE_CODE)) ;
    CHECK (code == GrB_FP32_CODE) ;

    OK (GrB_IndexUnaryOp_get_String_(GrB_VALUEGE_FP32, name, GrB_INP0_TYPE_STRING)) ;
    CHECK (MATCH (name, "GrB_FP32")) ;

    OK (GrB_IndexUnaryOp_get_INT32_(GrB_VALUEGE_FP64, &code, GrB_OUTP_TYPE_CODE)) ;
    CHECK (code == GrB_BOOL_CODE) ;

    OK (GrB_IndexUnaryOp_get_String_(GrB_VALUEGE_FP64, name, GrB_OUTP_TYPE_STRING)) ;
    CHECK (MATCH (name, "GrB_BOOL")) ;

    OK (GrB_IndexUnaryOp_get_Scalar_(GrB_VALUEGE_FP32, s_int32, GrB_INP0_TYPE_CODE)) ;
    OK (GrB_Scalar_extractElement_INT32_(&code, s_int32)) ;
    CHECK (code == GrB_FP32_CODE) ;

    OK (GrB_IndexUnaryOp_get_Scalar_(GrB_VALUEGE_FP32, s_int32, GrB_OUTP_TYPE_CODE)) ;
    OK (GrB_Scalar_extractElement_INT32_(&code, s_int32)) ;
    CHECK (code == GrB_BOOL_CODE) ;

    expected = GrB_NO_VALUE ;
    ERR (GrB_IndexUnaryOp_get_INT32_(GrB_TRIL, &code, GrB_INP0_TYPE_CODE)) ;
    ERR (GrB_IndexUnaryOp_get_Scalar_(GrB_TRIL, s_int32, GrB_INP0_TYPE_CODE)) ;

    expected = GrB_INVALID_VALUE ;
    ERR (GrB_IndexUnaryOp_get_INT32_(GrB_TRIL, &code, GrB_NAME)) ;

    expected = GrB_INVALID_VALUE ;
    ERR (GrB_IndexUnaryOp_get_VOID_(GrB_TRIL, nothing, 0)) ;

    OK (GrB_IndexUnaryOp_new (&op, (GxB_index_unary_function) myfunc,
        GrB_BOOL, GrB_FP32, GrB_FP32)) ;
    OK (GrB_IndexUnaryOp_get_SIZE_(op, &size, GrB_NAME)) ;
    CHECK (size == 1) ;
    OK (GrB_IndexUnaryOp_get_SIZE_(op, &size, GxB_JIT_C_NAME)) ;
    printf ("size %lu\n", size) ;
    CHECK (size == 1) ;
    OK (GrB_IndexUnaryOp_get_SIZE_(op, &size, GxB_JIT_C_DEFINITION)) ;
    CHECK (size == 1) ;

    expected = GrB_INVALID_VALUE ;
    OK (GrB_IndexUnaryOp_set_String_(op, "myfunc", GxB_JIT_C_NAME)) ;
    OK (GrB_IndexUnaryOp_get_String_(op, name, GxB_JIT_C_NAME)) ;
    CHECK (MATCH (name, "myfunc")) ;
    CHECK (op->hash == UINT64_MAX) ;
    METHOD (GrB_IndexUnaryOp_set_String (op, MYFUNC_DEFN, GxB_JIT_C_DEFINITION)) ;
    OK (GrB_IndexUnaryOp_get_String_(op, defn, GxB_JIT_C_DEFINITION)) ;
    CHECK (MATCH (defn, MYFUNC_DEFN)) ;
    CHECK (op->hash != UINT64_MAX) ;
    OK (GxB_print (op, 3)) ;

    OK (GrB_IndexUnaryOp_set_String_(op, "user name for myfunc", GrB_NAME)) ;
    OK (GrB_IndexUnaryOp_get_String_(op, name, GrB_NAME)) ;
    CHECK (MATCH (name, "user name for myfunc")) ;
    expected = GrB_ALREADY_SET ;
    ERR (GrB_IndexUnaryOp_set_String_(op, "another user name", GrB_NAME)) ;
    printf ("    test GrB_ALREADY_SET: ok\n") ;

    expected = GrB_INVALID_VALUE ;
    ERR (GrB_IndexUnaryOp_set_String_(op, "another_name", 999)) ;
    ERR (GrB_IndexUnaryOp_get_SIZE(op, &size, 999)) ;

    expected = GrB_ALREADY_SET ;
    ERR (GrB_IndexUnaryOp_set_String_(op, "another_name", GxB_JIT_C_NAME)) ;
    ERR (GrB_IndexUnaryOp_set_String_(op, "another_defn", GxB_JIT_C_DEFINITION)) ;

    expected = GrB_INVALID_VALUE ;
    ERR (GrB_IndexUnaryOp_set_String_(GrB_LNOT, "newname", GxB_JIT_C_NAME)) ;
    ERR (GrB_IndexUnaryOp_set_Scalar_(op, s_int32, 0)) ;
    ERR (GrB_IndexUnaryOp_set_INT32_(op, 0, 0)) ;
    ERR (GrB_IndexUnaryOp_set_VOID_(op, nothing, 0, 0)) ;

    //--------------------------------------------------------------------------
    // finalize GraphBLAS
    //--------------------------------------------------------------------------

    GrB_free (&s_fp64) ;
    GrB_free (&s_fp32) ;
    GrB_free (&s_int32) ;
    GrB_free (&op) ;
    GB_mx_put_global (true) ;
    printf ("\nGB_mex_test30:  all tests passed\n\n") ;
}

