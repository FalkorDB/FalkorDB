//------------------------------------------------------------------------------
// GraphBLAS/Demo/Program/grow_demo.c: grow a matrix row-by-row
//------------------------------------------------------------------------------

// SuiteSparse:GraphBLAS, Timothy A. Davis, (c) 2017-2024, All Rights Reserved.
// SPDX-License-Identifier: Apache-2.0

//------------------------------------------------------------------------------

// Reads in a matrix A, then does C = A one row at a time.

#include "graphblas_demos.h"
#include "simple_rand.c"
#include "usercomplex.h"
#include "usercomplex.c"
#include "wathen.c"
#include "get_matrix.c"
#include "random_matrix.c"
#include "import_test.c"
#include "read_matrix.c"

#include "omp.h"
#if defined ( _OPENMP )
#define WALLCLOCK omp_get_wtime ( )
#else
#define WALLCLOCK 0
#endif

// macro used by OK(...) to free workspace if an error occurs
#undef  FREE_ALL
#define FREE_ALL                            \
    GrB_Matrix_free (&A) ;                  \
    GrB_Matrix_free (&C) ;                  \
    GrB_Matrix_free (&T) ;                  \
    GrB_Matrix_free (&W) ;                  \
    GrB_Vector_free (&w) ;                  \

GrB_Info check_result (GrB_Matrix A1, GrB_Matrix C1, GrB_BinaryOp eq) ;
GrB_Info check_result (GrB_Matrix A1, GrB_Matrix C1, GrB_BinaryOp eq)
{
    double t = WALLCLOCK ;
    GrB_Info info ;
    GrB_Matrix A = NULL, C = NULL, T = NULL, W = NULL ;
    GrB_Vector w = NULL ;
    GrB_Index anvals, cnvals, tnvals, anrows, ancols ;
    OK (GrB_Matrix_nrows (&anrows, A1)) ;
    OK (GrB_Matrix_ncols (&ancols, A1)) ;
    OK (GrB_Matrix_nvals (&anvals, A1)) ;
    OK (GrB_Matrix_nvals (&cnvals, C1)) ;
    CHECK (anvals == cnvals, GrB_PANIC) ;
    OK (GrB_Matrix_new (&T, GrB_BOOL, anrows, ancols)) ;
    OK (GrB_Matrix_eWiseMult_BinaryOp (T, NULL, NULL, eq, A1, C1, NULL)) ;
    // OK (GxB_print (T, 2)) ;
    OK (GrB_Matrix_nvals (&tnvals, T)) ;
    CHECK (anvals == tnvals, GrB_PANIC) ;
    bool ok = true ;
    OK (GrB_Matrix_reduce_BOOL (&ok, NULL, GrB_LAND_MONOID_BOOL, T, NULL)) ;
    CHECK (ok, GrB_PANIC) ;
    t = (WALLCLOCK - t) ;
    FREE_ALL ;
    printf ("A and C match, time %g\n", t) ;
    return (GrB_SUCCESS) ;
}

int main (int argc, char **argv)
{
    //--------------------------------------------------------------------------
    // check inputs
    //--------------------------------------------------------------------------

    GrB_Matrix A = NULL, C = NULL, T = NULL, W = NULL ;
    GrB_Vector w = NULL ;
    GrB_Info info ;

    OK (GrB_init (GrB_NONBLOCKING)) ;
    int32_t nthreads ;
    OK (GxB_Global_Option_get (GxB_GLOBAL_NTHREADS, &nthreads)) ;
    fprintf (stderr, "grow demo: nthreads %d\n", nthreads) ;

    //--------------------------------------------------------------------------
    // get A matrix
    //--------------------------------------------------------------------------

    OK (get_matrix (&A, argc, argv, false, false, false)) ;
    GrB_Index anrows, ancols ;
    OK (GrB_Matrix_nrows (&anrows, A)) ;
    OK (GrB_Matrix_ncols (&ancols, A)) ;

    int32_t type_code ;
    OK (GrB_Matrix_get_INT32 (A, &type_code, GrB_EL_TYPE_CODE)) ;
    GrB_Type atype = NULL ;
    // OK (GxB_print (A, 5)) ;
    // printf ("type_code: %d\n", type_code) ;
    GrB_BinaryOp eq = NULL ;

    switch (type_code)
    {
        case GrB_BOOL_CODE   : atype = GrB_BOOL   ; eq = GrB_EQ_BOOL   ; break ;
        case GrB_INT8_CODE   : atype = GrB_INT8   ; eq = GrB_EQ_INT8   ; break ;
        case GrB_UINT8_CODE  : atype = GrB_UINT8  ; eq = GrB_EQ_UINT8  ; break ;
        case GrB_INT16_CODE  : atype = GrB_INT16  ; eq = GrB_EQ_INT16  ; break ;
        case GrB_UINT16_CODE : atype = GrB_UINT16 ; eq = GrB_EQ_UINT16 ; break ;
        case GrB_INT32_CODE  : atype = GrB_INT32  ; eq = GrB_EQ_INT32  ; break ;
        case GrB_UINT32_CODE : atype = GrB_UINT32 ; eq = GrB_EQ_UINT32 ; break ;
        case GrB_INT64_CODE  : atype = GrB_INT64  ; eq = GrB_EQ_INT64  ; break ;
        case GrB_UINT64_CODE : atype = GrB_UINT64 ; eq = GrB_EQ_UINT64 ; break ;
        case GrB_FP32_CODE   : atype = GrB_FP32   ; eq = GrB_EQ_FP32   ; break ;
        case GrB_FP64_CODE   : atype = GrB_FP64   ; eq = GrB_EQ_FP64   ; break ;
        case GxB_FC32_CODE   : atype = GxB_FC32   ; eq = GxB_EQ_FC32   ; break ;
        case GxB_FC64_CODE   : atype = GxB_FC64   ; eq = GxB_EQ_FC64   ; break ;
        default              : ;
    }

    // OK (GxB_print (atype, 5)) ;
    CHECK (atype != NULL, GrB_INVALID_VALUE) ;
    OK (GxB_Matrix_fprint (A, "A", 1, stdout)) ;

    //--------------------------------------------------------------------------
    // C = A, one row at a time
    //--------------------------------------------------------------------------

    OK (GrB_Matrix_new (&C, atype, anrows, ancols)) ;
    OK (GrB_Vector_new (&w, atype, ancols)) ;
    // OK (GrB_Global_set_INT32 (GrB_GLOBAL, true, GxB_BURBLE)) ;
    OK (GrB_Matrix_set_INT32 (C, (int32_t) false, (GrB_Field) GxB_HYPER_HASH)) ;
    OK (GrB_Matrix_set_INT32 (C, GxB_HYPERSPARSE, (GrB_Field) GxB_SPARSITY_CONTROL)) ;
    OK (GrB_Vector_set_INT32 (w, GxB_SPARSE, (GrB_Field) GxB_SPARSITY_CONTROL)) ;
    // printf ("\n\nC empty:\n") ;
    // OK (GxB_print (C, 1)) ;

    double t, tt [4] = {0, 0, 0, 0}, t2 [4] = {0, 0, 0, 0} ;
    tt [0] = WALLCLOCK ;

    for (int64_t i = 0 ; i < anrows ; i++)
    {
        // printf ("\n\ni = %ld\n", i) ;

        // w = A (i,:), using A' via the descriptor
        t = WALLCLOCK ;
        OK (GrB_Col_extract (w, NULL, NULL, A, GrB_ALL, ancols, i,
            GrB_DESC_T0)) ;
        tt [1] += (WALLCLOCK - t) ;
        // OK (GxB_print (w, 3)) ;

        // C (i,:) = w
        t = WALLCLOCK ;
        OK (GrB_Row_assign (C, NULL, NULL, w, i, GrB_ALL, ancols, NULL)) ;
        tt [2] += (WALLCLOCK - t) ;

        // ensure C is finished
        t = WALLCLOCK ;
        OK (GrB_Matrix_wait (C, GrB_MATERIALIZE)) ;
        tt [3] += (WALLCLOCK - t) ;
        // OK (GxB_print (C, 1)) ;
    }

    OK (GrB_Global_set_INT32 (GrB_GLOBAL, false, (GrB_Field) GxB_BURBLE)) ;
    // OK (GxB_print (C, 1)) ;

    tt [0] = WALLCLOCK - tt [0] ;
    printf ("one row at a time:\n") ;
    printf ("total time: %g\n", tt [0]) ;
    printf ("extract:    %g\n", tt [1]) ;
    printf ("assign:     %g\n", tt [2]) ;
    printf ("wait:       %g\n", tt [3]) ;

    // check to see if A and C are equal
    OK (check_result (A, C, eq)) ;

    //--------------------------------------------------------------------------
    // C = A, multiple rows at a time
    //--------------------------------------------------------------------------

    // OK (GrB_Global_set_INT32 (GrB_GLOBAL, true, GxB_BURBLE)) ;
    GrB_Matrix_free (&C) ;
    OK (GrB_Matrix_new (&C, atype, anrows, ancols)) ;
    OK (GrB_Matrix_set_INT32 (C, GxB_HYPERSPARSE, (GrB_Field) GxB_SPARSITY_CONTROL)) ;

    t2 [0] = WALLCLOCK ;
    int64_t i1 = 0 ;
    int64_t ilast = anrows - 1 ;
    while (i1 <= ilast)
    {
        // determine the rows i1:i2 of A to append
        int64_t irows = (ilast - i1) / 2 ;
        if (irows == 0) irows = 1 ;
        int64_t i2 = i1 + irows - 1 ;
        if (i2 > ilast) i2 = ilast ;
        irows = i2 - i1 + 1 ;
        // printf ("i1: %ld i2: %ld irows %ld ilast: %ld\n",
        //     i1, i2, irows, ilast) ;

        // W = A (i1:i2,:)
        t = WALLCLOCK ;
        OK (GrB_Matrix_new (&W, atype, irows, ancols)) ;
        OK (GrB_Matrix_set_INT32 (W, GxB_SPARSE, (GrB_Field) GxB_SPARSITY_CONTROL)) ;
        GrB_Index Icolon [3] ;
        Icolon [GxB_BEGIN] = i1 ;
        Icolon [GxB_INC  ] = 1 ;
        Icolon [GxB_END  ] = i2 ;
        // OK (GxB_print (W, 3)) ;
        info = GrB_Matrix_extract (W, NULL, NULL, A, Icolon, GxB_RANGE,
            GrB_ALL, ancols, NULL) ;
        // printf ("got here: %d\n", info) ;
        OK (info) ;
        t2 [1] += (WALLCLOCK - t) ;
        // OK (GxB_print (W, 3)) ;

        // C (i1:i2,:) = W
        t = WALLCLOCK ;
        OK (GrB_Matrix_assign (C, NULL, NULL, W,
            Icolon, GxB_RANGE, GrB_ALL, ancols, NULL)) ;
        t2 [2] += (WALLCLOCK - t) ;

        // ensure C is finished
        t = WALLCLOCK ;
        OK (GrB_Matrix_wait (C, GrB_MATERIALIZE)) ;
        t2 [3] += (WALLCLOCK - t) ;
        // OK (GxB_print (C, 1)) ;

        GrB_Matrix_free (&W) ;

        i1 += irows ;
    }
    OK (GrB_Global_set_INT32 (GrB_GLOBAL, false, (GrB_Field) GxB_BURBLE)) ;

    t2 [0] = WALLCLOCK - t2 [0] ;
    printf ("\nmany rows at a time:\n") ;
    printf ("total time: %g\n", t2 [0]) ;
    printf ("extract:    %g\n", t2 [1]) ;
    printf ("assign:     %g\n", t2 [2]) ;
    printf ("wait:       %g\n", t2 [3]) ;

    // check to see if A and C are equal
    OK (check_result (A, C, eq)) ;

    //--------------------------------------------------------------------------
    // C = A, using dup (1 threads and all threads)
    //--------------------------------------------------------------------------

    printf ("\nsingle call to dup:\n") ;
    for (int trial = 1 ; trial <= 2 ; trial++)
    {
        int32_t threads ;
        GrB_Global_get_INT32 (GrB_GLOBAL, &threads, (GrB_Field) GxB_GLOBAL_NTHREADS) ;
        GrB_Matrix_free (&T) ;
        t = WALLCLOCK ;
        OK (GrB_Matrix_dup (&T, A)) ;
        t = (WALLCLOCK - t) ;
        printf ("dup:        %g (%d threads)\n", t, threads) ;
        GrB_Global_set_INT32 (GrB_GLOBAL, (int) 1, (GrB_Field) GxB_GLOBAL_NTHREADS) ;
    }

    printf ("grow_demo: all tests passed\n") ;
    FREE_ALL ;
    GrB_finalize ( ) ;
    return (0) ;
}

