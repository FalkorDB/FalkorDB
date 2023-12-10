//------------------------------------------------------------------------------
// GB_mex_test6: still more basic tests
//------------------------------------------------------------------------------

// SuiteSparse:GraphBLAS, Timothy A. Davis, (c) 2017-2023, All Rights Reserved.
// SPDX-License-Identifier: Apache-2.0

//------------------------------------------------------------------------------

// Test lots of random stuff.  The function otherwise serves no purpose.

#include "GB_mex.h"
#include "GB_mex_errors.h"
#include "GB_serialize.h"

#define USAGE "GB_mex_test6"
#define FREE_ALL ;
#define GET_DEEP_COPY ;
#define FREE_DEEP_COPY ;

typedef int32_t myint ;

void mexFunction
(
    int nargout,
    mxArray *pargout [ ],
    int nargin,
    const mxArray *pargin [ ]
)
{

    GrB_Info info ;
    GrB_Matrix C = NULL, A = NULL, B = NULL, P = NULL ;
    GrB_Scalar alpha = NULL, beta = NULL, S = NULL ;
    GrB_Vector u = NULL ;
    GrB_Type MyType = NULL ;
    const char *err ;

    //--------------------------------------------------------------------------
    // startup GraphBLAS
    //--------------------------------------------------------------------------

    bool malloc_debug = GB_mx_get_global (true) ;
    int expected = GrB_SUCCESS ;

    //--------------------------------------------------------------------------
    // eWiseUnion
    //--------------------------------------------------------------------------

    OK (GrB_Type_new (&MyType, sizeof (myint))) ;

    OK (GrB_Matrix_new (&A, GrB_FP64, 10, 10)) ;
    OK (GrB_Matrix_new (&B, GrB_FP64, 10, 10)) ;
    OK (GrB_Matrix_new (&C, GrB_FP64, 10, 10)) ;

    expected = GrB_NULL_POINTER ;
    ERR (GxB_Matrix_eWiseUnion (C, NULL, NULL, GrB_PLUS_FP64, A, alpha,
        B, beta, NULL)) ;

    OK (GrB_Scalar_new (&alpha, GrB_FP64)) ;
    OK (GrB_Scalar_new (&beta, GrB_FP64)) ;

    expected = GrB_EMPTY_OBJECT ;
    ERR (GxB_Matrix_eWiseUnion (C, NULL, NULL, GrB_PLUS_FP64, A, alpha,
        B, beta, NULL)) ;
    OK (GrB_Matrix_error_(&err, C)) ;
    printf ("expected error:\n%s\n", err) ;

    OK (GrB_Scalar_setElement_FP64_ (alpha, (double) 42)) ;
    ERR (GxB_Matrix_eWiseUnion (C, NULL, NULL, GrB_PLUS_FP64, A, alpha,
        B, beta, NULL)) ;
    OK (GrB_Matrix_error_(&err, C)) ;
    printf ("expected error:\n%s\n", err) ;

    GrB_Scalar_free_(&alpha) ;
    OK (GrB_Scalar_new (&alpha, MyType)) ;
    myint nothing [1] ;
    memset (nothing, 0, sizeof (myint)) ;
    OK (GrB_Scalar_setElement_UDT (alpha, (void *) nothing)) ;
    OK (GxB_Scalar_fprint (alpha, "alpha", 3, NULL)) ;
    OK (GxB_Scalar_fprint (beta, "beta", 3, NULL)) ;
    OK (GrB_Scalar_setElement_FP64 (beta, (double) 99)) ;

    expected = GrB_DOMAIN_MISMATCH ;

    ERR (GxB_Matrix_eWiseUnion (C, NULL, NULL, GrB_PLUS_FP64, A, alpha,
        B, beta, NULL)) ;
    OK (GrB_Matrix_error_(&err, C)) ;
    printf ("expected error:\n%s\n", err) ;

    ERR (GxB_Matrix_eWiseUnion (C, NULL, NULL, GrB_PLUS_FP64, A, beta,
        B, alpha, NULL)) ;
    OK (GrB_Matrix_error_(&err, C)) ;
    printf ("expected error:\n%s\n", err) ;

    GrB_Matrix_free_(&A) ;
    GrB_Matrix_free_(&B) ;
    GrB_Matrix_free_(&C) ;
    GrB_Scalar_free_(&alpha) ;
    GrB_Scalar_free_(&beta) ;
    GrB_Type_free_(&MyType) ;

    //--------------------------------------------------------------------------
    // sort
    //--------------------------------------------------------------------------

    expected = GrB_NULL_POINTER ;
    OK (GrB_Matrix_new (&A, GrB_FP64, 10, 10)) ;
    OK (GrB_Matrix_setElement_FP64 (A, (double) 1.2, 0, 0)) ;
    ERR (GxB_Matrix_sort (NULL, NULL, GrB_LT_FP64, A, NULL)) ;

    OK (GrB_Matrix_new (&C, GrB_FP64, 10, 10)) ;
    OK (GrB_Matrix_new (&P, GrB_INT64, 10, 10)) ;

    expected = GrB_DOMAIN_MISMATCH ;
    ERR (GxB_Matrix_sort (C, P, GrB_PLUS_FP64, A, NULL)) ;

    GrB_Matrix_free_(&C) ;
    OK (GrB_Matrix_new (&C, GrB_FP64, 9, 10)) ;

    expected = GrB_DIMENSION_MISMATCH ;
    ERR (GxB_Matrix_sort (C, P, GrB_LT_FP64, A, NULL)) ;

    GrB_Matrix_free_(&P) ;
    GrB_Matrix_free_(&C) ;

    //--------------------------------------------------------------------------
    // reduce to GrB_Scalar
    //--------------------------------------------------------------------------

    GrB_Scalar_new (&S, GrB_FP64) ;
    OK (GrB_Vector_new (&u, GrB_FP64, 10)) ;

    expected = GrB_DOMAIN_MISMATCH ;

    ERR (GrB_Matrix_reduce_BinaryOp_Scalar_(S, NULL, GrB_LT_FP64, A, NULL)) ;
    OK (GrB_Scalar_error_(&err, S)) ;
    printf ("expected error:\n%s\n", err) ;

    ERR (GrB_Vector_reduce_BinaryOp_Scalar_(S, NULL, GrB_LT_FP64, u, NULL)) ;
    OK (GrB_Scalar_error_(&err, S)) ;
    printf ("expected error:\n%s\n", err) ;

    expected = GrB_NOT_IMPLEMENTED ;

    ERR (GrB_Matrix_reduce_BinaryOp_Scalar_(S, NULL, GrB_DIV_FP64, A, NULL)) ;
    OK (GrB_Scalar_error_(&err, S)) ;
    printf ("expected error:\n%s\n", err) ;

    ERR (GrB_Vector_reduce_BinaryOp_Scalar_(S, NULL, GrB_DIV_FP64, u, NULL)) ;
    OK (GrB_Scalar_error_(&err, S)) ;
    printf ("expected error:\n%s\n", err) ;

    GrB_Scalar_free_(&S) ;
    GrB_Vector_free_(&u) ;
    GrB_Matrix_free_(&A) ;

    //--------------------------------------------------------------------------
    // wrapup
    //--------------------------------------------------------------------------

    GB_mx_put_global (true) ;   
    printf ("\nGB_mex_test6: all tests passed\n\n") ;
}

