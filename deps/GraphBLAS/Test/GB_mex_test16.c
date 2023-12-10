//------------------------------------------------------------------------------
// GB_mex_test16: JIT error handling
//------------------------------------------------------------------------------

// SuiteSparse:GraphBLAS, Timothy A. Davis, (c) 2017-2023, All Rights Reserved.
// SPDX-License-Identifier: Apache-2.0

//------------------------------------------------------------------------------

#include "GB_mex.h"
#include "GB_mex_errors.h"
#include "GB_stringify.h"

#define USAGE "GB_mex_test16"

#define FREE_ALL ;
#define GET_DEEP_COPY ;
#define FREE_DEEP_COPY ;

void myfunc (float *z, const float *x) ;
void myfunc (float *z, const float *x) { (*z) = (*x) + 1 ; }

void mymult (float *z, const float *x, const float *y) ;
void mymult (float *z, const float *x, const float *y) { (*z) = (*x)*(*y) + 1 ;}

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

    GrB_Info info ;
    bool malloc_debug = GB_mx_get_global (true) ;

    //--------------------------------------------------------------------------
    // create some valid matrices
    //--------------------------------------------------------------------------

    GrB_Index n = 4 ;
    GrB_Matrix A = NULL, B = NULL, C = NULL ;
    GrB_Vector v = NULL ;
    OK (GrB_Matrix_new (&A, GrB_FP32, n, n)) ;
    OK (GrB_Matrix_new (&C, GrB_FP32, n, n)) ;
    OK (GrB_Matrix_new (&B, GrB_FP32, n, n)) ;
    OK (GxB_set (A, GxB_SPARSITY_CONTROL, GxB_SPARSE)) ;
    OK (GxB_set (B, GxB_SPARSITY_CONTROL, GxB_BITMAP)) ;
    OK (GrB_assign (A, NULL, NULL, 1, GrB_ALL, n, GrB_ALL, n, NULL)) ;
    OK (GrB_assign (B, NULL, NULL, 1, GrB_ALL, n, GrB_ALL, n, NULL)) ;
    OK (GrB_Matrix_setElement (A, 0, 0, 0)) ;
    OK (GrB_Matrix_setElement (B, 0, 0, 0)) ;
    OK (GrB_Matrix_wait (A, GrB_MATERIALIZE)) ;
    OK (GrB_Matrix_wait (B, GrB_MATERIALIZE)) ;
    OK (GrB_Matrix_wait (C, GrB_MATERIALIZE)) ;

    //--------------------------------------------------------------------------
    // set the JIT to ON, and the Factory Kernels off
    //--------------------------------------------------------------------------

    int save_control ;
    bool save_factory = GB_factory_kernels_enabled ;
    GB_factory_kernels_enabled = false ;
    OK (GxB_Global_Option_get_INT32 (GxB_JIT_C_CONTROL, &save_control)) ;
    OK (GxB_Global_Option_set_INT32 (GxB_JIT_C_CONTROL, GxB_JIT_ON)) ;

    //--------------------------------------------------------------------------
    // try some methods that require the JIT
    //--------------------------------------------------------------------------

    OK (GxB_set (GxB_BURBLE, true)) ;
    GrB_Semiring s ;
    GrB_BinaryOp mult ;
    GrB_Monoid mon ;

    // user type with zero size
    GrB_Type MyType ;
    GrB_Info expected = GrB_INVALID_VALUE ;
    ERR (GxB_Type_new (&MyType, 0, NULL, NULL)) ;
    OK (GxB_Global_Option_set_INT32 (GxB_JIT_C_CONTROL, GxB_JIT_ON)) ;
    OK (GrB_Type_new (&MyType, sizeof (double))) ;
    OK (GxB_print (MyType, 3)) ;
    size_t size ;
    expected = GrB_NO_VALUE ;
    ERR (GB_user_type_jit (&size, MyType)) ;
    OK (GxB_Global_Option_set_INT32 (GxB_JIT_C_CONTROL, GxB_JIT_ON)) ;

    printf ("\nmacrofy type:\n") ;
    GB_macrofy_user_type (NULL, MyType) ;

    // user function with NULL pointer
    GrB_UnaryOp op ;
    expected = GrB_NULL_POINTER ;
    ERR (GxB_UnaryOp_new (&op, NULL, GrB_FP32, GrB_FP32, NULL, NULL)) ;
    OK (GxB_Global_Option_set_INT32 (GxB_JIT_C_CONTROL, GxB_JIT_ON)) ;

    OK (GrB_UnaryOp_new (&op, (GxB_unary_function) myfunc,
        GrB_FP32, GrB_FP32)) ;
    printf ("\nmacrofy op:\n") ;
    GB_macrofy_user_op (NULL, (GB_Operator) op) ;

    OK (GrB_BinaryOp_new (&mult, (GxB_binary_function) mymult,
        GrB_FP32, GrB_FP32, GrB_FP32)) ;
    OK (GxB_print (mult, 3)) ;

    OK (GrB_Monoid_new (&mon, mult, (float) 1)) ;
    OK (GxB_print (mult, 3)) ;

    OK (GrB_Semiring_new (&s, mon, mult)) ;
    OK (GxB_print (s, 3)) ;

    GB_jit_encoding e ;
    char *suffix ;
    uint64_t code = GB_encodify_mxm (&e, &suffix, 0, false, false, GxB_SPARSE,
        GrB_FP32, NULL, false, false, s, false, A, B) ;
    CHECK (code == UINT64_MAX) ;

    code = GB_encodify_reduce (&e, &suffix, mon, A) ;
    CHECK (code == UINT64_MAX) ;

    code = GB_encodify_assign (&e, &suffix, 0, C, false, 0, 0, NULL,
        false, false, mult, A, NULL, 0) ;
    CHECK (code == UINT64_MAX) ;

    code = GB_encodify_build (&e, &suffix, 0, mult, GrB_FP32, GrB_FP32) ;
    CHECK (code == UINT64_MAX) ;

    //--------------------------------------------------------------------------
    // restore the JIT control, Factory Kernels, and renable the JIT
    //--------------------------------------------------------------------------

    OK (GxB_Global_Option_set_INT32 (GxB_JIT_C_CONTROL, save_control)) ;
    GB_factory_kernels_enabled = save_factory ;

    //--------------------------------------------------------------------------
    // GrB_select
    //--------------------------------------------------------------------------

    GrB_free (&C) ;
    OK (GrB_Matrix_new (&C, GrB_FP32, 2*n, 2*n)) ;
    expected = GrB_DIMENSION_MISMATCH ;
    ERR (GrB_select (C, NULL, NULL, GrB_TRIL, A, 0, NULL)) ;
    GrB_free (&C) ;

    //--------------------------------------------------------------------------
    // GrB_assign burble
    //--------------------------------------------------------------------------

    OK (GrB_Vector_new (&v, GrB_FP32, n)) ;
    OK (GrB_Matrix_new (&C, GrB_FP32, 2*n, 2*n)) ;
    OK (GxB_set (A, GxB_SPARSITY_CONTROL, GxB_BITMAP)) ;
    OK (GxB_set (C, GxB_SPARSITY_CONTROL, GxB_BITMAP)) ;
    OK (GrB_Row_assign (A, NULL, NULL, v, 0, GrB_ALL, n, NULL)) ;
    OK (GrB_Col_assign (A, NULL, NULL, v, GrB_ALL, n, 0, NULL)) ;
    GrB_Index I [4] = {0,1,2,3} ;
    OK (GrB_assign (C, NULL, NULL, A, I, 4, I, 4, NULL)) ;
    OK (GxB_subassign (C, NULL, NULL, A, I, 4, I, 4, NULL)) ;

    //--------------------------------------------------------------------------
    // finalize GraphBLAS
    //--------------------------------------------------------------------------

    GrB_free (&A) ;
    GrB_free (&B) ;
    GrB_free (&C) ;
    GrB_free (&v) ;
    GrB_free (&MyType) ;
    GrB_free (&op) ;
    GrB_free (&mult) ;
    GrB_free (&s) ;
    GrB_free (&mon) ;

    OK (GxB_set (GxB_BURBLE, false)) ;
    GB_mx_put_global (true) ;
    printf ("\nGB_mex_test16:  all tests passed\n\n") ;
}

