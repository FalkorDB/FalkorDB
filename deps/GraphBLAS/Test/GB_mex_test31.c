//------------------------------------------------------------------------------
// GB_mex_test31: test GrB_get and GrB_set (monoids)
//------------------------------------------------------------------------------

// SuiteSparse:GraphBLAS, Timothy A. Davis, (c) 2017-2023, All Rights Reserved.
// SPDX-License-Identifier: Apache-2.0

//------------------------------------------------------------------------------

#include "GB_mex.h"
#include "GB_mex_errors.h"

#define FREE_ALL ;
#define GET_DEEP_COPY ;
#define FREE_DEEP_COPY ;

#define GETOP(op,opname)                                                \
{                                                                       \
    size_t siz1, siz2, siz3 ;                                           \
    OK (GrB_Monoid_get_String (op, name, GrB_NAME)) ;                   \
    CHECK (MATCH (name, opname)) ;                                      \
    printf ("\n%s\n", name) ;                                           \
    OK (GxB_Monoid_fprint (op, "monoid", 5, stdout)) ;                  \
    OK (GrB_Monoid_get_SIZE (op, &size, GrB_NAME)) ;                    \
    CHECK (size == strlen (name) + 1) ;                                 \
    GrB_Info info2, info3 ;                                             \
    info2 = GrB_Monoid_get_SIZE (op, &siz1, GrB_INP0_TYPE_STRING) ;     \
    info3 = GrB_Monoid_get_String (op, name, GrB_INP0_TYPE_STRING) ;    \
    CHECK (info2 == info3) ;                                            \
    CHECK (siz1 == strlen (name) + 1) ;                                 \
    CHECK (info2 == GrB_SUCCESS) ;                                      \
    info2 = GrB_Monoid_get_SIZE (op, &siz2, GrB_INP1_TYPE_STRING) ;     \
    info3 = GrB_Monoid_get_String (op, name, GrB_INP1_TYPE_STRING) ;    \
    CHECK (info2 == info3) ;                                            \
    CHECK (siz2 == strlen (name) + 1) ;                                 \
    CHECK (info2 == GrB_SUCCESS) ;                                      \
    info2 = GrB_Monoid_get_SIZE (op, &siz3, GrB_OUTP_TYPE_STRING) ;     \
    info3 = GrB_Monoid_get_String (op, name, GrB_OUTP_TYPE_STRING) ;    \
    CHECK (info2 == info3) ;                                            \
    CHECK (siz3 == strlen (name) + 1) ;                                 \
    CHECK (info2 == GrB_SUCCESS) ;                                      \
}

#define GETNAME(op)                                         \
{                                                           \
    GETOP (op, #op) ;                                       \
}

#define GETNAM2(op,alias)                                   \
{                                                           \
    GETOP (op,alias) ;                                      \
}

void myfunc (float *z, const float *x, const float *y) ;
void myfunc (float *z, const float *x, const float *y) { (*z) = (*x)+(*y) ; }
#define MYFUNC_DEFN \
"void myfunc (float *z, const float *x, const float *y) { (*z) = (*x)+(*y) ; }"

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
    GrB_BinaryOp binop = NULL, op = NULL ;
    GrB_Monoid monoid = NULL ;
    uint8_t stuff [256] ;
    void *nothing = stuff ;
    size_t size ;
    char name [256] ;
    char defn [2048] ;
    int32_t code, i ;
    float fvalue ;
    double dvalue ;
    GrB_Index nvals = 999 ;

    OK (GrB_Scalar_new (&s_fp64, GrB_FP64)) ;
    OK (GrB_Scalar_new (&s_fp32, GrB_FP32)) ;
    OK (GrB_Scalar_new (&s_int32, GrB_INT32)) ;

    //--------------------------------------------------------------------------
    // GrB_Monoid get name
    //--------------------------------------------------------------------------

    GETNAM2 (GxB_MIN_INT8_MONOID,        "GrB_MIN_MONOID_INT8") ;
    GETNAM2 (GxB_MIN_INT16_MONOID,       "GrB_MIN_MONOID_INT16") ;
    GETNAM2 (GxB_MIN_INT32_MONOID,       "GrB_MIN_MONOID_INT32") ;
    GETNAM2 (GxB_MIN_INT64_MONOID,       "GrB_MIN_MONOID_INT64") ;
    GETNAM2 (GxB_MIN_UINT8_MONOID,       "GrB_MIN_MONOID_UINT8") ;
    GETNAM2 (GxB_MIN_UINT16_MONOID,      "GrB_MIN_MONOID_UINT16") ;
    GETNAM2 (GxB_MIN_UINT32_MONOID,      "GrB_MIN_MONOID_UINT32") ;
    GETNAM2 (GxB_MIN_UINT64_MONOID,      "GrB_MIN_MONOID_UINT64") ;
    GETNAM2 (GxB_MIN_FP32_MONOID,        "GrB_MIN_MONOID_FP32") ;
    GETNAM2 (GxB_MIN_FP64_MONOID,        "GrB_MIN_MONOID_FP64") ;

    GETNAME (GrB_MIN_MONOID_INT8) ;
    GETNAME (GrB_MIN_MONOID_INT16) ;
    GETNAME (GrB_MIN_MONOID_INT32) ;
    GETNAME (GrB_MIN_MONOID_INT64) ;
    GETNAME (GrB_MIN_MONOID_UINT8) ;
    GETNAME (GrB_MIN_MONOID_UINT16) ;
    GETNAME (GrB_MIN_MONOID_UINT32) ;
    GETNAME (GrB_MIN_MONOID_UINT64) ;
    GETNAME (GrB_MIN_MONOID_FP32) ;
    GETNAME (GrB_MIN_MONOID_FP64) ;

    GETNAM2 (GxB_MAX_INT8_MONOID,        "GrB_MAX_MONOID_INT8") ;
    GETNAM2 (GxB_MAX_INT16_MONOID,       "GrB_MAX_MONOID_INT16") ;
    GETNAM2 (GxB_MAX_INT32_MONOID,       "GrB_MAX_MONOID_INT32") ;
    GETNAM2 (GxB_MAX_INT64_MONOID,       "GrB_MAX_MONOID_INT64") ;
    GETNAM2 (GxB_MAX_UINT8_MONOID,       "GrB_MAX_MONOID_UINT8") ;
    GETNAM2 (GxB_MAX_UINT16_MONOID,      "GrB_MAX_MONOID_UINT16") ;
    GETNAM2 (GxB_MAX_UINT32_MONOID,      "GrB_MAX_MONOID_UINT32") ;
    GETNAM2 (GxB_MAX_UINT64_MONOID,      "GrB_MAX_MONOID_UINT64") ;
    GETNAM2 (GxB_MAX_FP32_MONOID,        "GrB_MAX_MONOID_FP32") ;
    GETNAM2 (GxB_MAX_FP64_MONOID,        "GrB_MAX_MONOID_FP64") ;

    GETNAME (GrB_MAX_MONOID_INT8) ;
    GETNAME (GrB_MAX_MONOID_INT16) ;
    GETNAME (GrB_MAX_MONOID_INT32) ;
    GETNAME (GrB_MAX_MONOID_INT64) ;
    GETNAME (GrB_MAX_MONOID_UINT8) ;
    GETNAME (GrB_MAX_MONOID_UINT16) ;
    GETNAME (GrB_MAX_MONOID_UINT32) ;
    GETNAME (GrB_MAX_MONOID_UINT64) ;
    GETNAME (GrB_MAX_MONOID_FP32) ;
    GETNAME (GrB_MAX_MONOID_FP64) ;

    GETNAM2 (GxB_PLUS_INT8_MONOID,        "GrB_PLUS_MONOID_INT8") ;
    GETNAM2 (GxB_PLUS_INT16_MONOID,       "GrB_PLUS_MONOID_INT16") ;
    GETNAM2 (GxB_PLUS_INT32_MONOID,       "GrB_PLUS_MONOID_INT32") ;
    GETNAM2 (GxB_PLUS_INT64_MONOID,       "GrB_PLUS_MONOID_INT64") ;
    GETNAM2 (GxB_PLUS_UINT8_MONOID,       "GrB_PLUS_MONOID_UINT8") ;
    GETNAM2 (GxB_PLUS_UINT16_MONOID,      "GrB_PLUS_MONOID_UINT16") ;
    GETNAM2 (GxB_PLUS_UINT32_MONOID,      "GrB_PLUS_MONOID_UINT32") ;
    GETNAM2 (GxB_PLUS_UINT64_MONOID,      "GrB_PLUS_MONOID_UINT64") ;
    GETNAM2 (GxB_PLUS_FP32_MONOID,        "GrB_PLUS_MONOID_FP32") ;
    GETNAM2 (GxB_PLUS_FP64_MONOID,        "GrB_PLUS_MONOID_FP64") ;
    GETNAME (GxB_PLUS_FC32_MONOID) ;
    GETNAME (GxB_PLUS_FC64_MONOID) ;

    GETNAME (GrB_PLUS_MONOID_INT8) ;
    GETNAME (GrB_PLUS_MONOID_INT16) ;
    GETNAME (GrB_PLUS_MONOID_INT32) ;
    GETNAME (GrB_PLUS_MONOID_INT64) ;
    GETNAME (GrB_PLUS_MONOID_UINT8) ;
    GETNAME (GrB_PLUS_MONOID_UINT16) ;
    GETNAME (GrB_PLUS_MONOID_UINT32) ;
    GETNAME (GrB_PLUS_MONOID_UINT64) ;
    GETNAME (GrB_PLUS_MONOID_FP32) ;
    GETNAME (GrB_PLUS_MONOID_FP64) ;

    GETNAM2 (GxB_TIMES_INT8_MONOID,        "GrB_TIMES_MONOID_INT8") ;
    GETNAM2 (GxB_TIMES_INT16_MONOID,       "GrB_TIMES_MONOID_INT16") ;
    GETNAM2 (GxB_TIMES_INT32_MONOID,       "GrB_TIMES_MONOID_INT32") ;
    GETNAM2 (GxB_TIMES_INT64_MONOID,       "GrB_TIMES_MONOID_INT64") ;
    GETNAM2 (GxB_TIMES_UINT8_MONOID,       "GrB_TIMES_MONOID_UINT8") ;
    GETNAM2 (GxB_TIMES_UINT16_MONOID,      "GrB_TIMES_MONOID_UINT16") ;
    GETNAM2 (GxB_TIMES_UINT32_MONOID,      "GrB_TIMES_MONOID_UINT32") ;
    GETNAM2 (GxB_TIMES_UINT64_MONOID,      "GrB_TIMES_MONOID_UINT64") ;
    GETNAM2 (GxB_TIMES_FP32_MONOID,        "GrB_TIMES_MONOID_FP32") ;
    GETNAM2 (GxB_TIMES_FP64_MONOID,        "GrB_TIMES_MONOID_FP64") ;
    GETNAME (GxB_TIMES_FC32_MONOID) ;
    GETNAME (GxB_TIMES_FC64_MONOID) ;

    GETNAME (GrB_TIMES_MONOID_INT8) ;
    GETNAME (GrB_TIMES_MONOID_INT16) ;
    GETNAME (GrB_TIMES_MONOID_INT32) ;
    GETNAME (GrB_TIMES_MONOID_INT64) ;
    GETNAME (GrB_TIMES_MONOID_UINT8) ;
    GETNAME (GrB_TIMES_MONOID_UINT16) ;
    GETNAME (GrB_TIMES_MONOID_UINT32) ;
    GETNAME (GrB_TIMES_MONOID_UINT64) ;
    GETNAME (GrB_TIMES_MONOID_FP32) ;
    GETNAME (GrB_TIMES_MONOID_FP64) ;

    GETNAME (GxB_ANY_BOOL_MONOID) ;
    GETNAME (GxB_ANY_INT8_MONOID) ;
    GETNAME (GxB_ANY_INT16_MONOID) ;
    GETNAME (GxB_ANY_INT32_MONOID) ;
    GETNAME (GxB_ANY_INT64_MONOID) ;
    GETNAME (GxB_ANY_UINT8_MONOID) ;
    GETNAME (GxB_ANY_UINT16_MONOID) ;
    GETNAME (GxB_ANY_UINT32_MONOID) ;
    GETNAME (GxB_ANY_UINT64_MONOID) ;
    GETNAME (GxB_ANY_FP32_MONOID) ;
    GETNAME (GxB_ANY_FP64_MONOID) ;
    GETNAME (GxB_ANY_FC32_MONOID) ;
    GETNAME (GxB_ANY_FC64_MONOID) ;

    GETNAM2 (GxB_LOR_BOOL_MONOID,       "GrB_LOR_MONOID_BOOL") ;
    GETNAM2 (GxB_LAND_BOOL_MONOID,      "GrB_LAND_MONOID_BOOL") ;
    GETNAM2 (GxB_LXOR_BOOL_MONOID,      "GrB_LXOR_MONOID_BOOL") ;
    GETNAM2 (GxB_LXNOR_BOOL_MONOID,     "GrB_LXNOR_MONOID_BOOL") ;
    GETNAM2 (GxB_EQ_BOOL_MONOID,        "GrB_LXNOR_MONOID_BOOL") ;

    GETNAME (GrB_LOR_MONOID_BOOL) ;
    GETNAME (GrB_LAND_MONOID_BOOL) ;
    GETNAME (GrB_LXOR_MONOID_BOOL) ;
    GETNAME (GrB_LXNOR_MONOID_BOOL) ;

    GETNAME (GxB_BOR_UINT8_MONOID) ;
    GETNAME (GxB_BOR_UINT16_MONOID) ;
    GETNAME (GxB_BOR_UINT32_MONOID) ;
    GETNAME (GxB_BOR_UINT64_MONOID) ;

    GETNAME (GxB_BAND_UINT8_MONOID) ;
    GETNAME (GxB_BAND_UINT16_MONOID) ;
    GETNAME (GxB_BAND_UINT32_MONOID) ;
    GETNAME (GxB_BAND_UINT64_MONOID) ;

    GETNAME (GxB_BXOR_UINT8_MONOID) ;
    GETNAME (GxB_BXOR_UINT16_MONOID) ;
    GETNAME (GxB_BXOR_UINT32_MONOID) ;
    GETNAME (GxB_BXOR_UINT64_MONOID) ;

    GETNAME (GxB_BXNOR_UINT8_MONOID) ;
    GETNAME (GxB_BXNOR_UINT16_MONOID) ;
    GETNAME (GxB_BXNOR_UINT32_MONOID) ;
    GETNAME (GxB_BXNOR_UINT64_MONOID) ;

    //--------------------------------------------------------------------------
    // other get/set methods for GrB_Monoid
    //--------------------------------------------------------------------------

    OK (GrB_Monoid_get_INT32_(GrB_MAX_MONOID_FP32, &code, GrB_INP0_TYPE_CODE)) ;
    CHECK (code == GrB_FP32_CODE) ;

    OK (GrB_Monoid_get_String_(GrB_MAX_MONOID_FP32, name,
        GrB_INP0_TYPE_STRING)) ;
    CHECK (MATCH (name, "GrB_FP32")) ;

    OK (GrB_Monoid_get_String_(GrB_MAX_MONOID_INT32, name,
        GrB_INP1_TYPE_STRING)) ;
    CHECK (MATCH (name, "GrB_INT32")) ;

    OK (GrB_Monoid_get_INT32_(GrB_MAX_MONOID_FP64, &code, GrB_OUTP_TYPE_CODE)) ;
    CHECK (code == GrB_FP64_CODE) ;

    OK (GrB_Monoid_get_String_(GrB_MAX_MONOID_FP64, name,
        GrB_OUTP_TYPE_STRING)) ;
    CHECK (MATCH (name, "GrB_FP64")) ;

    OK (GrB_Monoid_get_Scalar_(GrB_MAX_MONOID_FP32, s_int32,
        GrB_INP0_TYPE_CODE)) ;
    OK (GrB_Scalar_extractElement_INT32_(&code, s_int32)) ;
    CHECK (code == GrB_FP32_CODE) ;

    OK (GrB_Monoid_get_Scalar_(GrB_LAND_MONOID_BOOL, s_int32,
        GrB_OUTP_TYPE_CODE)) ;
    OK (GrB_Scalar_extractElement_INT32_(&code, s_int32)) ;
    CHECK (code == GrB_BOOL_CODE) ;

    OK (GrB_Monoid_get_INT32_(GrB_PLUS_MONOID_FP64, &code,
        GrB_INP1_TYPE_CODE)) ;
    CHECK (code == GrB_FP64_CODE) ;

    OK (GrB_Monoid_get_Scalar_(GrB_LAND_MONOID_BOOL, s_int32,
        GrB_INP1_TYPE_CODE)) ;
    OK (GrB_Scalar_extractElement_INT32_(&code, s_int32)) ;
    CHECK (code == GrB_BOOL_CODE) ;

    expected = GrB_INVALID_VALUE ;
    ERR (GrB_Monoid_get_INT32_(GrB_LAND_MONOID_BOOL, &code, GrB_NAME)) ;
    ERR (GrB_Monoid_get_String_(GrB_MAX_MONOID_INT32, name, 999)) ;
    ERR (GrB_Monoid_get_VOID_(GrB_LAND_MONOID_BOOL, nothing, 0)) ;

    OK (GrB_BinaryOp_new (&binop, (GxB_binary_function) myfunc,
        GrB_FP32, GrB_FP32, GrB_FP32)) ;
    OK (GrB_BinaryOp_set_String_(binop, "myfunc", GrB_NAME)) ;
    METHOD (GrB_BinaryOp_set_String (binop, MYFUNC_DEFN, GxB_JIT_C_DEFINITION)) ;

    OK (GrB_Monoid_new_FP32 (&monoid, binop, (float) 0.0)) ;
    OK (GrB_Monoid_get_SIZE_(monoid, &size, GrB_NAME)) ;
    OK (GrB_Monoid_get_String_(monoid, name, GrB_NAME)) ;
    printf ("\nuser monoid: [%s]\n", name) ;
    CHECK (MATCH (name, "")) ;
    CHECK (size == 1) ;
    OK (GxB_print (monoid, 3)) ;

    OK (GrB_Monoid_get_String_(monoid, name, GrB_INP0_TYPE_STRING)) ;
    CHECK (MATCH (name, "GrB_FP32")) ;

    OK (GrB_Monoid_get_SIZE_(monoid, &size, GrB_INP0_TYPE_STRING)) ;
    CHECK (size == strlen ("GrB_FP32") + 1) ;

    expected = GrB_INVALID_VALUE ;
    ERR (GrB_Monoid_get_SIZE_(monoid, &size, GrB_INP0_TYPE_CODE)) ;

    expected = GrB_INVALID_VALUE ;
    ERR (GrB_Monoid_set_Scalar_(monoid, s_int32, 0)) ;
    ERR (GrB_Monoid_set_INT32_(monoid, 0, 0)) ;
    ERR (GrB_Monoid_set_VOID_(monoid, nothing, 0, 0)) ;

    OK (GrB_Monoid_set_String_(monoid, "monoid_stuff", GrB_NAME)) ;
    OK (GrB_Monoid_get_String_(monoid, name, GrB_NAME)) ;
    printf ("\nuser monoid: [%s]\n", name) ;
    CHECK (MATCH (name, "monoid_stuff")) ;
    OK (GrB_Monoid_get_SIZE_(monoid, &size, GrB_NAME)) ;
    CHECK (size == strlen (name) + 1) ;

    expected = GrB_ALREADY_SET ;
    ERR (GrB_Monoid_set_String_(monoid, "another user name", GrB_NAME)) ;
    printf ("    test GrB_ALREADY_SET: ok\n") ;

    printf ("\nterminal monoid:\n") ;
    int32_t id_int32 ;
    OK (GxB_print (GrB_MAX_MONOID_INT32, 3)) ;
    OK (GrB_Monoid_get_Scalar_ (GrB_MAX_MONOID_INT32, s_int32,
        GxB_MONOID_IDENTITY)) ;
    OK (GrB_Scalar_nvals (&nvals, s_int32)) ;
    CHECK (nvals == 1) ;
    OK (GrB_Scalar_extractElement_INT32_(&id_int32, s_int32)) ;
    CHECK (id_int32 == INT32_MIN) ;

    int32_t term_int32 ;
    OK (GrB_Monoid_get_Scalar_ (GrB_MAX_MONOID_INT32, s_int32,
        GxB_MONOID_TERMINAL)) ;
    OK (GrB_Scalar_extractElement_INT32_(&term_int32, s_int32)) ;
    CHECK (term_int32 == INT32_MAX) ;

    printf ("\nmon-terminal monoid:\n") ;
    OK (GxB_print (GrB_PLUS_MONOID_INT32, 3)) ;
    OK (GrB_Monoid_get_Scalar_ (GrB_PLUS_MONOID_INT32, s_int32,
        GxB_MONOID_TERMINAL)) ;
    OK (GrB_Scalar_nvals (&nvals, s_int32)) ;
    CHECK (nvals == 0) ;

    OK (GrB_Monoid_get_Scalar_ (GrB_PLUS_MONOID_INT32, s_int32,
        GxB_MONOID_IDENTITY)) ;
    OK (GrB_Scalar_extractElement_INT32_(&id_int32, s_int32)) ;
    CHECK (id_int32 == 0) ;

    expected = GrB_DOMAIN_MISMATCH ;
    ERR (GrB_Monoid_get_Scalar_ (GrB_PLUS_MONOID_INT32, s_fp64,
        GxB_MONOID_IDENTITY)) ;
    ERR (GrB_Monoid_get_Scalar_ (GrB_PLUS_MONOID_INT32, s_fp64,
        GxB_MONOID_TERMINAL)) ;
    ERR (GrB_Monoid_get_Scalar_ (GrB_MAX_MONOID_INT32, s_fp64,
        GxB_MONOID_IDENTITY)) ;
    ERR (GrB_Monoid_get_Scalar_ (GrB_MAX_MONOID_INT32, s_fp64,
        GxB_MONOID_TERMINAL)) ;

    expected = GrB_INVALID_VALUE ;
    ERR (GrB_Monoid_get_Scalar_ (GrB_MAX_MONOID_INT32, s_fp64,
        GrB_OUTP_FIELD)) ;
    ERR (GrB_Monoid_set_String_(GrB_MAX_MONOID_INT32, "newname", GrB_NAME)) ;

    op = NULL ;
    OK (GrB_Monoid_get_SIZE_ (monoid, &size, GxB_MONOID_OPERATOR)) ;
    CHECK (size == sizeof (GrB_BinaryOp)) ;
    OK (GrB_Monoid_get_VOID (monoid, (void *) (&op), GxB_MONOID_OPERATOR)) ;
    CHECK (op == binop) ;
    OK (GrB_Monoid_get_VOID_ (GrB_PLUS_MONOID_INT32, (void *) &op,
        GxB_MONOID_OPERATOR)) ;
    CHECK (op == GrB_PLUS_INT32) ;

    //--------------------------------------------------------------------------
    // finalize GraphBLAS
    //--------------------------------------------------------------------------

    GrB_free (&s_fp64) ;
    GrB_free (&s_fp32) ;
    GrB_free (&s_int32) ;
    GrB_free (&binop) ;
    GrB_free (&monoid) ;
    GB_mx_put_global (true) ;
    printf ("\nGB_mex_test31:  all tests passed\n\n") ;
}

