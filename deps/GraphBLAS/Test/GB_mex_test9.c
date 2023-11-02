//------------------------------------------------------------------------------
// GB_mex_test9: still more basic tests (not for Windows)
//------------------------------------------------------------------------------

// SuiteSparse:GraphBLAS, Timothy A. Davis, (c) 2017-2023, All Rights Reserved.
// SPDX-License-Identifier: Apache-2.0

//------------------------------------------------------------------------------

// Windows is limited to user-defined types of size 128 or less.

#include "GB_mex.h"
#include "GB_mex_errors.h"
#include "GB_stringify.h"

#define USAGE "GB_mex_test9"
#define FREE_ALL ;
#define GET_DEEP_COPY ;
#define FREE_DEEP_COPY ;

typedef struct
{
    double stuff [32] ;
}
bigtype ;

 void f1 (void *z, const void *x) ;
 void f2 (void *z, const void *x, const void *y) ;
 void i1 (void *z, const void *x, GrB_Index i, GrB_Index j, const void *thunk) ;

#define F1                                                              \
"void f1 (void *z, const void *x) "                                     \
"{ (*((double *)z)) = 2*(*(double *)x) ; } "
 void f1 (void *z, const void *x)
 { (*((double *)z)) = 2*(*(double *)x) ; }

#define F2                                                              \
"void f2 (void *z, const void *x, const void *y) "                      \
"{ (*((double *)z)) = 2*(*(double *)x) + 1 ; }   "
 void f2 (void *z, const void *x, const void *y)
 { (*((double *)z)) = 2*(*(double *)x) + 1 ; }

#define I1                                                          \
"void i1 (void *z, const void *x, GrB_Index i, GrB_Index j, "       \
" const void *thunk) "                                              \
"{ (*((bool *)z)) = (i == j) ; }"
 void i1 (void *z, const void *x, GrB_Index i, GrB_Index j, const void *thunk)
 { (*((bool *)z)) = (i == j) ; }

#define GET_DEEP_COPY ;
#define FREE_DEEP_COPY ;

void mexFunction
(
    int nargout,
    mxArray *pargout [ ],
    int nargin,
    const mxArray *pargin [ ]
)
{

    GrB_Info info ;

    //--------------------------------------------------------------------------
    // startup GraphBLAS
    //--------------------------------------------------------------------------

    bool malloc_debug = GB_mx_get_global (true) ;

    //--------------------------------------------------------------------------
    // user-defined type of 256 bytes
    //--------------------------------------------------------------------------

    GrB_Type BigType ;
    METHOD (GxB_Type_new (&BigType, sizeof (bigtype), "bigtype",
        "typedef struct { double stuff [32] ; } bigtype")) ;
    OK (GxB_Type_fprint (BigType, "(256-byte big type)", GxB_COMPLETE,
        stdout)) ;
    OK (GrB_Type_free (&BigType)) ;

    //--------------------------------------------------------------------------
    // user-defined operators
    //--------------------------------------------------------------------------

    GrB_UnaryOp op1 = NULL ;
    METHOD (GxB_UnaryOp_new (&op1, &f1, GrB_FP64, GrB_FP64, "f1", F1)) ;
    OK (GxB_UnaryOp_fprint (op1, "f1", GxB_COMPLETE, stdout)) ;
    OK (GrB_UnaryOp_free (&op1)) ;

    GrB_BinaryOp op2 = NULL ;
    METHOD (GxB_BinaryOp_new (&op2, &f2, GrB_FP64, GrB_FP64, GrB_FP64,
        "f2", F2)) ;
    OK (GxB_BinaryOp_fprint (op2, "f2", GxB_COMPLETE, stdout)) ;
    OK (GrB_BinaryOp_free (&op2)) ;

    GrB_IndexUnaryOp opi = NULL ;
    METHOD (GxB_IndexUnaryOp_new (&opi, &i1, GrB_FP64, GrB_FP64, GrB_FP64,
        "i1", I1)) ;
    OK (GxB_IndexUnaryOp_fprint (opi, "i1", GxB_COMPLETE, stdout)) ;
    OK (GrB_IndexUnaryOp_free (&opi)) ;

    //--------------------------------------------------------------------------
    // reduce an empty matrix to a scalar using the ANY monoid
    //--------------------------------------------------------------------------

    GrB_Matrix A ;
    OK (GrB_Matrix_new (&A, GrB_FP32, 10, 10)) ;
    OK (GxB_Matrix_fprint (A, "empty matrix", GxB_COMPLETE, stdout)) ;
    float x = 42 ;
    OK (GrB_Matrix_reduce_FP32 (&x, NULL, GxB_ANY_FP32_MONOID, A, NULL)) ;
    printf ("\nreduce empty matrix to non-opaque scalar via ANY: %g\n", x) ;

    GrB_Scalar s ;
    OK (GrB_Scalar_new (&s, GrB_FP32)) ;
    OK (GrB_Scalar_setElement_FP32 (s, 33)) ;
    OK (GxB_Scalar_fprint (s, "scalar == 33", GxB_COMPLETE, stdout)) ;
    OK (GrB_Matrix_reduce_Monoid_Scalar (s, NULL, GxB_ANY_FP32_MONOID, A,
        NULL)) ;
    printf ("\nreduce empty matrix to opaque scalar via ANY:\n") ;
    OK (GxB_Scalar_fprint (s, "empty scalar", GxB_COMPLETE, stdout)) ;

    OK (GrB_Matrix_free (&A)) ;
    OK (GrB_Scalar_free (&s)) ;

    //--------------------------------------------------------------------------
    // enumify, macrofy, and related methods
    //--------------------------------------------------------------------------

    bool has_byte = false ;
    uint8_t byte = 0 ;
    const char *f = GB_macrofy_id (31, 1, &has_byte, &byte) ;
    CHECK (f != NULL && strlen (f) == 0) ;

    // Using GB_boolean_rename results in these cases not being tested.
    int ecode = -1 ;
    GB_enumify_binop (&ecode, GB_MIN_binop_code, GB_BOOL_code, false) ;
    CHECK (ecode == 18) ;
    GB_enumify_binop (&ecode, GB_MAX_binop_code, GB_BOOL_code, false) ;
    CHECK (ecode == 17) ;
    GB_enumify_binop (&ecode, GB_TIMES_binop_code, GB_BOOL_code, false) ;
    CHECK (ecode == 18) ;
    GB_enumify_binop (&ecode, GB_PLUS_binop_code, GB_BOOL_code, false) ;
    CHECK (ecode == 17) ;
    GB_enumify_binop (&ecode, GB_NE_binop_code, GB_BOOL_code, false) ;
    CHECK (ecode == 16) ;
    GB_enumify_binop (&ecode, GB_ISEQ_binop_code, GB_BOOL_code, false) ;
    CHECK (ecode == 15) ;
    GB_enumify_binop (&ecode, GB_ISNE_binop_code, GB_BOOL_code, false) ;
    CHECK (ecode == 16) ;
    GB_enumify_binop (&ecode, GB_DIV_binop_code, GB_BOOL_code, false) ;
    CHECK (ecode == 1) ;
    GB_enumify_binop (&ecode, GB_RDIV_binop_code, GB_BOOL_code, false) ;
    CHECK (ecode == 2) ;
    GB_enumify_binop (&ecode, GB_RMINUS_binop_code, GB_BOOL_code, false) ;
    CHECK (ecode == 16) ;
    GB_enumify_binop (&ecode, GB_POW_binop_code, GB_BOOL_code, false) ;
    CHECK (ecode == 71) ;
    GB_enumify_binop (&ecode, GB_MINUS_binop_code, GB_BOOL_code, false) ;
    CHECK (ecode == 16) ;

    GB_enumify_identity (&ecode, GB_MIN_binop_code, GB_BOOL_code) ;
    CHECK (ecode == 2) ;
    GB_enumify_identity (&ecode, GB_MAX_binop_code, GB_BOOL_code) ;
    CHECK (ecode == 3) ;

    ecode = -1 ;
    GB_enumify_terminal (&ecode, GB_TIMES_binop_code, GB_BOOL_code) ;
    CHECK (ecode == 3) ;
    ecode = -1 ;
    GB_enumify_terminal (&ecode, GB_MIN_binop_code, GB_BOOL_code) ;
    CHECK (ecode == 3) ;
    GB_enumify_terminal (&ecode, GB_MAX_binop_code, GB_BOOL_code) ;
    CHECK (ecode == 2) ;

    FILE *fp = fopen ("/tmp/GB_tcov_gunk.h", "w") ;
    GB_macrofy_binop (fp, "nothing", false, false, false,
        199, false, NULL, NULL, NULL) ;
    fclose (fp) ;

    //--------------------------------------------------------------------------
    // wrapup
    //--------------------------------------------------------------------------

    GB_mx_put_global (true) ;
    printf ("\nGB_mex_test9: all tests passed\n\n") ;
}

