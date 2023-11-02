//------------------------------------------------------------------------------
// GraphBLAS/Demo/Program/openmp2_demo: example of user multithreading
//------------------------------------------------------------------------------

// SuiteSparse:GraphBLAS, Timothy A. Davis, (c) 2017-2023, All Rights Reserved.
// SPDX-License-Identifier: Apache-2.0

//------------------------------------------------------------------------------

// This demo uses OpenMP, and illustrates how GraphBLAS can be called from
// a multi-threaded user program.

#include "GraphBLAS.h"

#ifdef _OPENMP
#include <omp.h>
#endif

#if defined __INTEL_COMPILER
#pragma warning (disable: 58 167 144 177 181 186 188 589 593 869 981 1418 1419 1572 1599 2259 2282 2557 2547 3280 )
#elif defined __GNUC__
#pragma GCC diagnostic ignored "-Wunknown-pragmas"
#pragma GCC diagnostic ignored "-Wunused-parameter"
#if !defined ( __cplusplus )
#pragma GCC diagnostic ignored "-Wincompatible-pointer-types"
#endif
#endif

#define NTRIALS 1000
#define N 6

#undef  OK
#define OK(method)                                                  \
{                                                                   \
    GrB_Info info = method ;                                        \
    if (! (info == GrB_SUCCESS || info == GrB_NO_VALUE))            \
    {                                                               \
        fprintf (stderr, "Failure (id: %d, info: %d):\n", id, info) ;   \
        /* return to caller (do not use inside critical section) */ \
        return (0) ;                                                \
    }                                                               \
}

//------------------------------------------------------------------------------
// worker
//------------------------------------------------------------------------------

int worker (int id)
{

    GrB_Matrix A = NULL ;
//  printf ("\nworker %d\n", id) ;

    for (int hammer_hard = 0 ; hammer_hard < NTRIALS ; hammer_hard++)
    {
        OK (GrB_Matrix_free (&A)) ;
        OK (GrB_Matrix_new (&A, GrB_FP64, N, N)) ;

        for (int i = 0 ; i < N ; i++)
        {
            for (int j = 0 ; j < N ; j++)
            {
                double x = (i+1)*100000 + (j+1)*1000 + id ;
                OK (GrB_Matrix_setElement_FP64 (A, x, i, j)) ;
            } 
        }

        // force completion
        OK (GrB_Matrix_wait (A, GrB_MATERIALIZE)) ;
    }
    OK (GrB_Matrix_free (&A)) ;

    return (0) ;
}

//------------------------------------------------------------------------------
// openmp_demo main program
//------------------------------------------------------------------------------

int main (int argc, char **argv)
{
    printf ("Demo: %s:\n", argv [0]) ;
    int id = 0 ;

    // start GraphBLAS
    OK (GrB_init (GrB_NONBLOCKING)) ;
    int nthreads ;
    OK (GxB_Global_Option_get (GxB_GLOBAL_NTHREADS, &nthreads)) ;
    int ver [3] ;
    char *date ;
    OK (GxB_Global_Option_get (GxB_LIBRARY_VERSION , ver)) ;
    OK (GxB_Global_Option_get (GxB_LIBRARY_DATE , &date)) ;
    printf ("openmp demo, nthreads %d (v%d.%d.%d, %s)\n",
        nthreads, ver [0], ver [1], ver [2], date) ;

    OK (GxB_Global_Option_set (GxB_NTHREADS, 1)) ;
    double t1 = 0 ;

    for (int nth = 1 ; nth <= nthreads ; nth++)
    {
        printf ("nthreads %2d: ", nth) ;

        for (int trial = 0 ; trial <= 0 ; trial++)
        {
            #ifdef _OPENMP
            double t = omp_get_wtime ( ) ;
            #endif

            // create the threads
            #pragma omp parallel for num_threads(nth) 
            for (id = 0 ; id < 128 ; id++)
            {
                worker (id) ;
            }

            #ifdef _OPENMP
            t = omp_get_wtime ( ) - t ;
            if (nth == 1) t1 = t ;
            printf (" time: %10.4f sec speedup: %10.4f", t, t1/t) ;
            #endif
        }
        printf ("\n") ;
    }

    // finish GraphBLAS
    GrB_finalize ( ) ;

    // finish OpenMP
    exit (0) ;
}

