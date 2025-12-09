//------------------------------------------------------------------------------
// GraphBLAS/Demo/Program/context_demo: example for the GxB_Context
//------------------------------------------------------------------------------

// SuiteSparse:GraphBLAS, Timothy A. Davis, (c) 2017-2025, All Rights Reserved.
// SPDX-License-Identifier: Apache-2.0

//------------------------------------------------------------------------------

#include "graphblas_demos.h"
#define FREE_ALL ;

#define TRY(method)                                                     \
{                                                                       \
    GrB_Info info = method ;                                            \
    if (info != GrB_SUCCESS) printf ("Failure: %d, %s, %d\n", info,     \
        __FILE__, __LINE__) ;                                           \
}

int main (void)
{
#if defined ( _MSC_VER )
    printf ("context_demo: requires OpenMP 4.0\n") ;
#else

    // start GraphBLAS
    GrB_Info info ;
    OK (GrB_init (GrB_NONBLOCKING)) ;
    int nthreads_max = 0 ;
    OK (GrB_Global_get_INT32 (GrB_GLOBAL, &nthreads_max, GxB_NTHREADS)) ;
    nthreads_max = MIN (nthreads_max, 256) ;
    printf ("context demo: nthreads_max %d\n", nthreads_max) ;
    OK (GxB_Context_fprint (GxB_CONTEXT_WORLD, "World", GxB_COMPLETE, stdout)) ;

    // use only a power of 2 number of threads
    int nthreads = 1 ;
    while (1)
    {
        if (2*nthreads > nthreads_max) break ;
        nthreads = 2 * nthreads ;
    }

    printf ("\nnthreads to use: %d\n", nthreads) ;
    OK (GrB_Global_set_INT32 (GrB_GLOBAL, nthreads, GxB_NTHREADS)) ;

    #ifdef _OPENMP
    omp_set_max_active_levels (2) ;
    #endif

    //--------------------------------------------------------------------------
    // construct tuples for a decent-sized random matrix
    //--------------------------------------------------------------------------

    GrB_Index n = 100000 ;
    GrB_Index nvals = 20000000 ;
    uint64_t state = 1 ;
    GrB_Index *I = malloc (nvals * sizeof (GrB_Index)) ;
    GrB_Index *J = malloc (nvals * sizeof (GrB_Index)) ;
    double    *X = malloc (nvals * sizeof (double)) ;
    for (int k = 0 ; k < nvals ; k++)
    {
        I [k] = simple_rand (&state) % n ;
        J [k] = simple_rand (&state) % n ;
        X [k] = simple_rand_x (&state) ;
    }

    //--------------------------------------------------------------------------
    // create random matrices parallel
    //--------------------------------------------------------------------------

    int nmats = MIN (2*nthreads, 256) ;

    for (int nmat = 1 ; nmat <= nmats ; nmat = 2*nmat)
    {
        printf ("\nnmat %d\n", nmat) ;
        double t1 = 0 ;

        // create nmat matrices, each in parallel with varying # of threads
        for (int nthreads2 = 1 ; nthreads2 <= nthreads ; nthreads2 *= 2)
        {
            int nouter = 1 ;            // # of user threads in outer loop
            int ninner = nthreads2 ;    // # of threads each user thread can use

            printf ("\n") ;

            while (ninner >= 1)
            {
                if (nouter <= nmat)
                {

                    double t = WALLCLOCK ;

                    #pragma omp parallel for num_threads (nouter) \
                        schedule (dynamic, 1)
                    for (int k = 0 ; k < nmat ; k++)
                    {
                        // each user thread constructs its own context
                        GxB_Context Context = NULL ;
                        TRY (GxB_Context_new (&Context)) ;
                        TRY (GxB_Context_set_INT (Context, ninner,
                            GxB_NTHREADS)) ;
//                      TRY (GxB_Context_set (Context, GxB_NTHREADS, ninner)) ;
                        TRY (GxB_Context_engage (Context)) ;

                        // kth user thread builds kth matrix with ninner threads
                        GrB_Matrix A = NULL ;
                        TRY (GrB_Matrix_new (&A, GrB_FP64, n, n)) ;
                        TRY (GrB_Matrix_build_FP64 (A, I, J, X, nvals,
                            GrB_PLUS_FP64)) ;

                        // free the matrix just built
                        TRY (GrB_Matrix_free (&A)) ;

                        // each user thread frees its own context
                        TRY (GxB_Context_disengage (Context)) ;
                        TRY (GxB_Context_free (&Context)) ;
                    }

                    t = WALLCLOCK - t ;
                    if (nouter == 1 && ninner == 1) t1 = t ;

                    printf ("   threads (%4d,%4d): %4d "
                        "time: %8.4f sec speedup: %8.3f\n",
                        nouter, ninner, nouter * ninner, t, t1/t) ;
                }
                nouter = nouter * 2 ;
                ninner = ninner / 2 ;
            }
        }
    }

    free (I) ;
    free (J) ;
    free (X) ;
    OK (GrB_finalize ( )) ;
#endif
}

