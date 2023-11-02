//------------------------------------------------------------------------------
// GraphBLAS/Demo/Program/reduce_demo: reduce a matrix to a scalar
//------------------------------------------------------------------------------

// SuiteSparse:GraphBLAS, Timothy A. Davis, (c) 2017-2023, All Rights Reserved.
// SPDX-License-Identifier: Apache-2.0

//------------------------------------------------------------------------------

// TODO for GPU: add this to CMakelists.txt, or merge with reduce_demo.c

#include "GraphBLAS.h"

// #define N 65536
   #define N 16384

int main (void)
{

    #if defined ( _OPENMP )
    double t0 = omp_get_wtime ( ) ;
    #endif

    // start GraphBLAS
    GrB_init (GrB_NONBLOCKING) ;
    printf ("demo: reduce a matrix to a scalar\n") ;

    int nthreads_max ;
    GxB_get (GxB_NTHREADS, &nthreads_max) ;
    printf ("# of threads: %d\n", nthreads_max) ;

    #if defined ( _OPENMP )
    t0 = omp_get_wtime ( ) - t0 ;
    printf ("GPU warmup time: %g\n", t0) ;
    t0 = omp_get_wtime ( ) ;
    #endif

    GrB_Index nrows = N ;
    GrB_Index ncols = N ;
    GrB_Matrix A ;
    GrB_Matrix_new (&A, GrB_INT64, nrows, ncols) ;

    GrB_Index *I = malloc (nrows * ncols * sizeof (GrB_Index)) ;
    GrB_Index *J = malloc (nrows * ncols * sizeof (GrB_Index)) ;
    int64_t   *X = malloc (nrows * ncols * sizeof (int64_t)) ;

    #pragma omp parallel for num_threads(nthreads_max) collapse(2) \
        schedule(static)
    for (int64_t i = 0 ; i < nrows ; i++)
    {
        for (int64_t j = 0 ; j < ncols ; j++)
        {
            int64_t k = i * N + j ;
            // int x = (int) (rand ( ) & 0xFF) ;
            int x = (int) (k & 0xFF) ;
            I [k] = i ;
            J [k] = j ;
            X [k] = x ;
        }
    }

    GrB_Index nvals = ((size_t) N) * ((size_t) N) ;
    GrB_Matrix_build (A, I, J, X, nvals, GrB_PLUS_INT64) ;

    GxB_print (A, 2) ;

    free (I) ;
    free (J) ;
    free (X) ;

    #if defined ( _OPENMP )
    t0 = omp_get_wtime ( ) - t0 ;
    printf ("time to create matrix: %g\n", t0) ;
    #endif

    GrB_Index result ;

    GrB_Matrix B ;
    GrB_Matrix_new (&B, GrB_INT64, 2000, 2000) ;
    for (int64_t i = 0 ; i < 2000 ; i++)
    {
        for (int64_t j = 0 ; j < 2000 ; j++)
        {
            GrB_Matrix_setElement (B, 1, i, j) ;
        }
    }
    GrB_Index ignore ;
    GrB_Matrix_nvals (&ignore, B) ;

    #if defined ( _OPENMP )
    double tfirst = omp_get_wtime ( ) ;
    #endif
    GrB_reduce (&result, NULL, GxB_PLUS_INT64_MONOID, B, NULL) ;
    #if defined ( _OPENMP )
    tfirst = omp_get_wtime ( ) - tfirst ;
    printf ("warmup %g sec (on all threads/gpu, small matrix)\n", tfirst) ;
    printf ("result: %"PRIu64"\n", result) ;
    #endif

    double t1 ;

    printf ("\nreduce to a scalar:\n") ;

    for (int nthreads = 1 ; nthreads <= nthreads_max ; nthreads++)
    {
        GxB_set (GxB_NTHREADS, nthreads) ;
        #if defined ( _OPENMP )
        double t = omp_get_wtime ( ) ;
        #endif
        GrB_reduce (&result, NULL, GxB_PLUS_INT64_MONOID, A, NULL) ;
        #if defined ( _OPENMP )
        t = omp_get_wtime ( ) - t ;
        if (nthreads == 1) t1 = t ;
        printf ("nthreads %3d time: %12.6f speedup %8.2f\n", 
            nthreads, t, t1/t) ;
        #endif
    }

    // printf ("result %d\n", result) ;
    printf ("result %"PRId64"\n", (int64_t) result) ;

    // free everyting
    GrB_free (&A) ;
    GrB_finalize ( ) ;
}

