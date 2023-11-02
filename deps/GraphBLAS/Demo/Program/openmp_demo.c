//------------------------------------------------------------------------------
// GraphBLAS/Demo/Program/openmp_demo: example of user multithreading
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

#define NTHREADS 8
#define NTRIALS 10
#define N 6

#undef  OK
#define OK(method)                                                  \
{                                                                   \
    GrB_Info info = method ;                                        \
    if (! (info == GrB_SUCCESS || info == GrB_NO_VALUE))            \
    {                                                               \
        printf ("Failure (id: %d, info: %d):\n", id, info) ;        \
        /* return to caller (do not use inside critical section) */ \
        return (0) ;                                                \
    }                                                               \
}

//------------------------------------------------------------------------------
// worker
//------------------------------------------------------------------------------

int worker (GrB_Matrix *Ahandle, int id)
{

    printf ("\n================= worker %d starts:\n", id) ;
    fprintf (stderr, "worker %d\n", id) ;

    OK (GrB_Matrix_new (Ahandle, GrB_FP64, N, N)) ;
    GrB_Matrix A = *Ahandle ;

    // worker generates an intentional error message
    GrB_Matrix_setElement_INT32 (A, 42, 1000+id, 1000+id) ;

    // print the intentional error generated when the worker started
    #pragma omp critical
    {
        // critical section
        printf ("\n----------------- worker %d intentional error:\n", id) ;
        const char *s ;
        GrB_Matrix_error (&s, A) ;
        printf ("%s\n", s) ;
    }

    for (int hammer_hard = 0 ; hammer_hard < NTRIALS ; hammer_hard++)
    {
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

    // Printing is done in a critical section, just so it is not overly
    // jumbled.  Each matrix and error will print in a single body of text,
    // but the order of the matrices and errors printed will be out of order
    // because the critical section does not enforce the order that the
    // threads enter.

    GrB_Info info2 ;
    #pragma omp critical
    {
        // critical section
        printf ("\n----------------- worker %d is done:\n", id) ;
        info2 = GxB_Matrix_fprint (A, "A", GxB_SHORT, stdout) ;
    }
    OK (info2) ;

    // worker generates an intentional error message
    GrB_Matrix_setElement_INT32 (A, 42, 1000+id, 1000+id) ;

    // print the intentional error generated when the worker started
    // It should be unchanged.
    #pragma omp critical
    {
        // critical section
        printf ("\n----------------- worker %d error should be same:\n", id) ;
        const char *s ;
        GrB_Matrix_error (&s, A) ;
        printf ("%s\n", s) ;
    }
    return (0) ;
}

//------------------------------------------------------------------------------
// openmp_demo main program
//------------------------------------------------------------------------------

int main (int argc, char **argv)
{
    fprintf (stderr, "Demo: %s:\n", argv [0]) ;
    printf ("Demo: %s:\n", argv [0]) ;

    // initialize the mutex
    int id = -1 ;

    // start GraphBLAS
    OK (GrB_init (GrB_NONBLOCKING)) ;
    int nthreads ;
    OK (GxB_Global_Option_get (GxB_GLOBAL_NTHREADS, &nthreads)) ;
    fprintf (stderr, "openmp demo, nthreads %d\n", nthreads) ;

    // Determine which user-threading model is being used.
    #ifdef _OPENMP
    printf ("User threads in this program are OpenMP threads.\n") ;
    #else
    printf ("This user program is single threaded.\n") ;
    #endif

    GrB_Matrix Aarray [NTHREADS] ;

    // create the threads
    #pragma omp parallel for num_threads(NTHREADS) 
    for (id = 0 ; id < NTHREADS ; id++)
    {
        worker (&Aarray [id], id) ;
    }

    // the leader thread prints them again, and frees them
    for (int id = 0 ; id < NTHREADS ; id++)
    {
        GrB_Matrix A = Aarray [id] ;
        printf ("\n---- Leader prints matrix %d\n", id) ;
        OK (GxB_Matrix_fprint (A, "A", GxB_SHORT, stdout)) ;
        GrB_Matrix_free (&A) ;
    }

    // finish GraphBLAS
    GrB_finalize ( ) ;

    // finish OpenMP
    exit (0) ;
}

