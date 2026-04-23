//------------------------------------------------------------------------------
// GB_mex_test45: test extractElement when input is jumbled
//------------------------------------------------------------------------------

// SuiteSparse:GraphBLAS, Timothy A. Davis, (c) 2017-2026, All Rights Reserved.
// SPDX-License-Identifier: Apache-2.0

//------------------------------------------------------------------------------

#include "GB_mex.h"
#include "GB_mex_errors.h"

#undef  FREE_ALL
#define FREE_ALL                    \
{                                   \
    GrB_free (&A) ;                 \
    GrB_free (&C) ;                 \
    GrB_free (&X) ;                 \
    GrB_free (&Y) ;                 \
    GB_mx_put_global (true) ;       \
}

//------------------------------------------------------------------------------
// GB_mex_test45 mexFunction
//------------------------------------------------------------------------------

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
    GrB_Matrix A = NULL, C = NULL ;
    GrB_Vector X = NULL, Y = NULL ;

    //--------------------------------------------------------------------------
    // create a random matrix A
    //--------------------------------------------------------------------------

    #define N 10000
    #define ANZ 100000
    #define XNZ 5
    OK (GrB_Matrix_new (&A, GrB_FP64, N, N)) ;
    OK (GrB_Matrix_new (&C, GrB_FP64, N, N)) ;
    OK (GrB_Vector_new (&X, GrB_FP64, N)) ;
    OK (GrB_Vector_new (&Y, GrB_FP64, N)) ;
    OK (GrB_Matrix_set_INT32 (A, GxB_HYPERSPARSE, GxB_SPARSITY_CONTROL)) ;
    OK (GrB_Matrix_set_INT32 (C, GxB_HYPERSPARSE, GxB_SPARSITY_CONTROL)) ;
    OK (GrB_Vector_set_INT32 (X, GxB_SPARSE, GxB_SPARSITY_CONTROL)) ;
    OK (GrB_Vector_set_INT32 (Y, GxB_SPARSE, GxB_SPARSITY_CONTROL)) ;

    srand (42) ;
    for (int k = 0 ; k < ANZ ; k++)
    {
        int64_t i = rand ( ) % N ;
        int64_t j = rand ( ) % N ;
        OK (GrB_Matrix_setElement_FP64 (A, 1, i, j)) ;
    }

    for (int k = 0 ; k < XNZ ; k++)
    {
        int64_t i = rand ( ) % N ;
        OK (GrB_Vector_setElement_FP64 (X, 1, i)) ;
    }

    GxB_print (A, 2) ;
    GxB_print (X, 2) ;
    OK (GrB_set (GrB_GLOBAL, true, GxB_BURBLE)) ;

    //--------------------------------------------------------------------------
    // C = A*A and Y = C*X
    //--------------------------------------------------------------------------

    OK (GrB_mxm (C, NULL, NULL, GrB_PLUS_TIMES_SEMIRING_FP64, A, A, NULL)) ;
    OK (GrB_mxv (Y, NULL, NULL, GrB_PLUS_TIMES_SEMIRING_FP64, C, X, NULL)) ;
    Y->jumbled = true ; // hack Y to force it to be jumbled

    printf ("\n======================= C:\n") ;
    GxB_print (C, 2) ;
    int quick = 10 ;
    for (int i = 0 ; i < N ; i++)
    {
        // a = C (i,0)
        double a ;
        info = GrB_Matrix_extractElement_FP64 (&a, C, i, 0) ;
        if (info == GrB_SUCCESS && quick > 0)
        {
            quick-- ;
            printf ("C(%d,0) = %g\n", i, a) ;
        }
    }
    printf ("...\n") ;

    printf ("\n======================= Y:\n") ;
    quick = 10 ;
    GxB_print (Y, 2) ;
    for (int i = 0 ; i < N ; i++)
    {
        double a ;
        info = GrB_Vector_extractElement_FP64 (&a, Y, i) ;
        if (info == GrB_SUCCESS && quick > 0)
        {
            quick-- ;
            printf ("Y(%d) = %g\n", i, a) ;
        }
    }
    printf ("...\n") ;

    //--------------------------------------------------------------------------
    // finalize GraphBLAS
    //--------------------------------------------------------------------------

    FREE_ALL ;
    printf ("\nGB_mex_test45:  all tests passed\n\n") ;
}

