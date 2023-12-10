//------------------------------------------------------------------------------
// GraphBLAS/Demo/Program/reduce_demo: reduce a matrix to a scalar
//------------------------------------------------------------------------------

// SuiteSparse:GraphBLAS, Timothy A. Davis, (c) 2017-2023, All Rights Reserved.
// SPDX-License-Identifier: Apache-2.0

//------------------------------------------------------------------------------

#include "GraphBLAS.h"
#if defined ( _OPENMP )
#include <omp.h>
#endif

// #define N 65536
   #define N 16382

int main (void)
{

    #if defined ( _OPENMP )
    double t0 = omp_get_wtime ( ) ;
    #endif

    // start GraphBLAS
    GrB_init (GrB_NONBLOCKING) ;
    int nthreads ;
    GxB_Global_Option_get (GxB_GLOBAL_NTHREADS, &nthreads) ;
    printf ("demo: reduce a matrix to a scalar, nthreads: %d\n", nthreads) ;

    int nthreads_max ;
    GxB_Global_Option_get (GxB_GLOBAL_NTHREADS, &nthreads_max) ;
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

    GrB_Index *I = (GrB_Index *) malloc (nrows * ncols * sizeof (GrB_Index)) ;
    GrB_Index *J = (GrB_Index *) malloc (nrows * ncols * sizeof (GrB_Index)) ;
    int64_t   *X = (int64_t   *) malloc (nrows * ncols * sizeof (int64_t)) ;

    int64_t k ;
    #pragma omp parallel for num_threads(nthreads_max) schedule(static)
    for (k = 0 ; k < N*N ; k++)
    {
        // k = i * N + j ;
        int64_t i = k / N ;
        int64_t j = k % N ;
        // int x = (int) (rand ( ) & 0xFF) ;
        int x = (int) (k & 0xFF) ;
        I [k] = i ;
        J [k] = j ;
        X [k] = x ;
    }

    GrB_Index nvals = N*N ;
    GrB_Matrix_build_INT64 (A, I, J, X, nvals, GrB_PLUS_INT64) ;

    free (I) ;
    free (J) ;
    free (X) ;

    #if defined ( _OPENMP )
    t0 = omp_get_wtime ( ) - t0 ;
    printf ("time to create matrix: %g\n", t0) ;
    #endif

    GrB_Index result ;

    double t1 ;

    printf ("\nreduce to a scalar:\n") ;

    for (int nthreads = 1 ; nthreads <= nthreads_max ; nthreads++)
    {
        GxB_Global_Option_set (GxB_GLOBAL_NTHREADS, nthreads) ;
        #if defined ( _OPENMP )
        double t = omp_get_wtime ( ) ;
        #endif
        GrB_Matrix_reduce_UINT64 (&result, NULL, GrB_PLUS_MONOID_INT64,
            A, NULL) ;
        #if defined ( _OPENMP )
        t = omp_get_wtime ( ) - t ;
        if (nthreads == 1) t1 = t ;
        printf ("nthreads %3d time: %12.6f speedup %8.2f\n", 
            nthreads, t, t1/t) ;
        #endif
    }

    printf ("result %" PRId64 "\n", result) ;

    // free everyting
    GrB_Matrix_free (&A) ;
    GrB_finalize ( ) ;
}

