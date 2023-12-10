//------------------------------------------------------------------------------
// GB_mex_test15: JIT error handling
//------------------------------------------------------------------------------

// SuiteSparse:GraphBLAS, Timothy A. Davis, (c) 2017-2023, All Rights Reserved.
// SPDX-License-Identifier: Apache-2.0

//------------------------------------------------------------------------------

#include "GB_mex.h"
#include "GB_mex_errors.h"

#define USAGE "GB_mex_test15"

#define FREE_ALL ;
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

    //--------------------------------------------------------------------------
    // startup GraphBLAS
    //--------------------------------------------------------------------------

    GrB_Info info ;
    bool malloc_debug = GB_mx_get_global (true) ;

    //--------------------------------------------------------------------------
    // create some valid matrices
    //--------------------------------------------------------------------------

    GrB_Index n = 4 ;
    GrB_Matrix A = NULL, B = NULL, F = NULL, C = NULL, D = NULL, G = NULL,
        S = NULL, H = NULL, F2 = NULL, F3 = NULL ;
    OK (GrB_Matrix_new (&A, GrB_FP32, n, n)) ;
    OK (GrB_Matrix_new (&C, GrB_FP32, n, n)) ;
    OK (GrB_Matrix_new (&B, GrB_FP32, n, n)) ;
    OK (GrB_Matrix_new (&D, GrB_FP32, n, n)) ;
    OK (GrB_Matrix_new (&F, GrB_FP32, n, n)) ;
    OK (GrB_Matrix_new (&F2, GrB_FP32, n, n)) ;
    OK (GrB_Matrix_new (&G, GrB_FP32, 2*n, 2*n)) ;
    OK (GrB_Matrix_new (&S, GrB_FP32, 200, 200)) ;
    OK (GrB_Matrix_new (&H, GrB_FP32, 400, 400)) ;
    OK (GxB_set (A, GxB_SPARSITY_CONTROL, GxB_SPARSE)) ;
    OK (GxB_set (B, GxB_SPARSITY_CONTROL, GxB_BITMAP)) ;
    OK (GxB_set (D, GxB_SPARSITY_CONTROL, GxB_SPARSE)) ;
    OK (GxB_set (H, GxB_SPARSITY_CONTROL, GxB_SPARSE)) ;
    OK (GxB_set (F, GxB_SPARSITY_CONTROL, GxB_FULL)) ;
    OK (GxB_set (F2, GxB_SPARSITY_CONTROL, GxB_FULL)) ;
    OK (GrB_assign (A, NULL, NULL, 1, GrB_ALL, n, GrB_ALL, n, NULL)) ;
    OK (GrB_assign (B, NULL, NULL, 1, GrB_ALL, n, GrB_ALL, n, NULL)) ;
    OK (GrB_assign (F, NULL, NULL, 1, GrB_ALL, n, GrB_ALL, n, NULL)) ;
    OK (GrB_assign (F2, NULL, NULL, 1, GrB_ALL, n, GrB_ALL, n, NULL)) ;
    OK (GrB_Matrix_setElement (A, 0, 0, 0)) ;
    OK (GrB_Matrix_setElement (B, 0, 0, 0)) ;
    OK (GrB_Matrix_setElement (F, 0, 0, 0)) ;
    OK (GrB_Matrix_setElement (F2, 0, 0, 0)) ;
    OK (GrB_Matrix_setElement (S, 2, 10, 4)) ;
    for (int i = 0 ; i < 200 ; i++)
    {
        OK (GrB_Matrix_setElement (S, 3, i, i)) ;
    }
    OK (GrB_select (D, NULL, NULL, GrB_DIAG, A, 0, NULL)) ;
    OK (GrB_Matrix_dup (&F3, F2)) ;
    OK (GrB_Matrix_wait (A, GrB_MATERIALIZE)) ;
    OK (GrB_Matrix_wait (B, GrB_MATERIALIZE)) ;
    OK (GrB_Matrix_wait (C, GrB_MATERIALIZE)) ;
    OK (GrB_Matrix_wait (F, GrB_MATERIALIZE)) ;
    OK (GrB_Matrix_wait (F2, GrB_MATERIALIZE)) ;
    OK (GrB_Matrix_wait (F3, GrB_MATERIALIZE)) ;
    OK (GrB_Matrix_wait (D, GrB_MATERIALIZE)) ;
    OK (GrB_Matrix_wait (G, GrB_MATERIALIZE)) ;
    OK (GrB_Matrix_wait (H, GrB_MATERIALIZE)) ;
    OK (GrB_Matrix_wait (S, GrB_MATERIALIZE)) ;

    //--------------------------------------------------------------------------
    // set the JIT to ON, and the Factory Kernels off, then disable the JIT
    //--------------------------------------------------------------------------

    int save_control ;
    bool save_factory = GB_factory_kernels_enabled ;
    GB_factory_kernels_enabled = false ;
    OK (GxB_Global_Option_get_INT32 (GxB_JIT_C_CONTROL, &save_control)) ;
    OK (GxB_Global_Option_set_INT32 (GxB_JIT_C_CONTROL, GxB_JIT_ON)) ;
    GB_Global_hack_set (3, 1) ;     // JIT will return GrB_NOT_IMPLEMENTED

    //--------------------------------------------------------------------------
    // try some methods that require the JIT
    //--------------------------------------------------------------------------

    OK (GxB_set (GxB_BURBLE, true)) ;
    GrB_Semiring s = GrB_PLUS_TIMES_SEMIRING_FP32 ;
    GrB_BinaryOp add = GrB_PLUS_FP32 ;
    GrB_Monoid mon = GrB_PLUS_MONOID_FP32 ;

    GrB_Info expected = GrB_NOT_IMPLEMENTED ;

    // dot2
    ERR (GrB_mxm (C, NULL, NULL, s, A, A, GrB_DESC_T0)) ;

    // dot3
    ERR (GrB_mxm (C, A, NULL, s, A, A, GrB_DESC_T0)) ;

    // saxpy3
    ERR (GrB_mxm (C, NULL, NULL, s, A, A, NULL)) ;

    // saxpy4
    ERR (GrB_mxm (F2, NULL, add, s, A, F, NULL)) ;

    // saxpy5
    ERR (GrB_mxm (F3, NULL, add, s, F, A, NULL)) ;

    // row scale
    ERR (GrB_mxm (C, NULL, NULL, s, D, A, NULL)) ;

    // col scale
    ERR (GrB_mxm (C, NULL, NULL, s, A, D, NULL)) ;

    // saxbit
    ERR (GrB_mxm (C, NULL, NULL, s, B, B, NULL)) ;

    // add
    ERR (GrB_eWiseAdd (C, NULL, NULL, add, A, A, NULL)) ;

    // emult 08
    ERR (GrB_eWiseMult (C, NULL, NULL, add, A, A, NULL)) ;

    // emult 02
    ERR (GrB_eWiseMult (C, NULL, NULL, add, A, F, NULL)) ;

    // emult 03
    ERR (GrB_eWiseMult (C, NULL, NULL, GrB_DIV_FP32, F, A, NULL)) ;

    // emult 04
    ERR (GrB_eWiseMult (C, D, NULL, add, F, B, NULL)) ;

    // emult bitmap
    ERR (GrB_eWiseMult (C, NULL, NULL, add, B, B, NULL)) ;

    // reduce
    float x = 0 ;
    ERR (GrB_reduce (&x, NULL, mon, A, NULL)) ;

    // select (convert sparse to bitmap)
    ERR (GrB_select (C, NULL, NULL, GrB_DIAG, A, 0, NULL)) ;

    // select sparse
    ERR (GrB_select (C, NULL, NULL, GrB_VALUENE_FP32, D, 0, NULL)) ;

    // select bitmap
    ERR (GrB_select (C, NULL, NULL, GrB_VALUENE_FP32, B, 0, NULL)) ;

    // concat full
    GrB_Matrix Tiles [4] ;
    Tiles [0] = F ;
    Tiles [1] = F ;
    Tiles [2] = F ;
    Tiles [3] = F ;
    ERR (GxB_Matrix_concat (G, Tiles, 2, 2, NULL)) ;

    // concat sparse
    Tiles [0] = S ;
    Tiles [1] = S ;
    Tiles [2] = S ;
    Tiles [3] = S ;
    ERR (GxB_Matrix_concat (H, Tiles, 2, 2, NULL)) ;

    // concat bitmap
    Tiles [0] = B ;
    Tiles [1] = B ;
    Tiles [2] = B ;
    Tiles [3] = B ;
    GrB_free (&G) ;
    OK (GrB_Matrix_new (&G, GrB_FP32, 2*n, 2*n)) ;
    ERR (GxB_Matrix_concat (G, Tiles, 2, 2, NULL)) ;

    // split full
    GrB_Index Tile_rows [2] = {2, 2} ;
    GrB_Index Tile_cols [2] = {2, 2} ;
    ERR (GxB_Matrix_split (Tiles, 2, 2, Tile_rows, Tile_cols, F, NULL)) ;

    // split sparse
    ERR (GxB_Matrix_split (Tiles, 2, 2, Tile_rows, Tile_cols, A, NULL)) ;

    // split bitmap
    ERR (GxB_Matrix_split (Tiles, 2, 2, Tile_rows, Tile_cols, B, NULL)) ;

    //--------------------------------------------------------------------------
    // restore the JIT control, Factory Kernels, and renable the JIT
    //--------------------------------------------------------------------------

    OK (GxB_set (GxB_BURBLE, false)) ;
    OK (GxB_Global_Option_set_INT32 (GxB_JIT_C_CONTROL, save_control)) ;
    GB_factory_kernels_enabled = save_factory ;
    GB_Global_hack_set (3, 0) ;

    //--------------------------------------------------------------------------
    // finalize GraphBLAS
    //--------------------------------------------------------------------------

    GrB_free (&A) ;
    GrB_free (&B) ;
    GrB_free (&C) ;
    GrB_free (&D) ;
    GrB_free (&F) ;
    GrB_free (&G) ;
    GrB_free (&H) ;
    GrB_free (&S) ;
    GrB_free (&F2) ;
    GrB_free (&F3) ;

    GB_mx_put_global (true) ;
    printf ("\nGB_mex_test15:  all tests passed\n\n") ;
}

