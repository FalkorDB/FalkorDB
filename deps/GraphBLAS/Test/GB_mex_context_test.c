//------------------------------------------------------------------------------
// GB_mex_context_text: based on Demo/Programcontext_demo
//------------------------------------------------------------------------------

// SuiteSparse:GraphBLAS, Timothy A. Davis, (c) 2017-2023, All Rights Reserved.
// SPDX-License-Identifier: Apache-2.0

//------------------------------------------------------------------------------

#include "GB_mex.h"
#include "GB_mex_errors.h"
#define MIN(x,y) ((x) < (y)) ? (x) : (y)
#define MAX(x,y) ((x) > (y)) ? (x) : (y)

#define FREE_ALL { GB_mx_put_global (true) ; }
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
    GrB_Info info ;
    bool malloc_debug = GB_mx_get_global (true) ;

    //==========================================================================
    // variant or context_demo
    //==========================================================================

    int nthreads_max = 0 ;
    OK (GxB_Global_Option_get (GxB_GLOBAL_NTHREADS, &nthreads_max)) ;
    nthreads_max = MIN (nthreads_max, 256) ;
    printf ("context demo: nthreads_max %d\n", nthreads_max) ;
    OK (GxB_print (GxB_CONTEXT_WORLD, 3)) ;

    // use only a power of 2 number of threads
    int nthreads = 1 ;
    while (1)
    {
        if (2*nthreads > nthreads_max) break ;
        nthreads = 2 * nthreads ;
    }

    nthreads = MIN (nthreads, 8) ;

    printf ("\nnthreads to use: %d\n", nthreads) ;
    OK (GxB_Global_Option_set (GxB_GLOBAL_NTHREADS, nthreads)) ;

    #ifdef _OPENMP
    omp_set_max_active_levels (2) ;
    #endif

    //==========================================================================
    // more tests
    //==========================================================================

    GxB_Context Context = NULL ;
    METHOD (GxB_Context_new (&Context)) ;
    OK (GxB_Context_set (Context, GxB_NTHREADS, 5)) ;
    OK (GxB_Context_fprint (Context, "context", 3, stdout)) ;
    int nth = 0 ;
    OK (GxB_Context_get (GxB_CONTEXT_WORLD, GxB_NTHREADS, &nth)) ;
    printf ("nth %d nthreads %d\n", nth, nthreads) ;
    CHECK (nth == nthreads) ;
    OK (GxB_Context_engage (Context)) ;
    OK (GxB_Context_get (Context, GxB_NTHREADS, &nth)) ;
    CHECK (nth == 5) ;
    double chunk = 4096 ;
    OK (GxB_Context_set (Context, GxB_CHUNK, chunk)) ;
    chunk = 0 ;
    OK (GxB_Context_get (Context, GxB_CHUNK, &chunk)) ;
    CHECK (chunk == 4096) ;
    chunk = 0 ;
    OK (GxB_Context_get_FP64 (Context, GxB_CHUNK, &chunk)) ;
    CHECK (chunk == 4096) ;
    OK (GxB_Context_set (Context, GxB_CHUNK, -1)) ;
    OK (GxB_Context_get (Context, GxB_CHUNK, &chunk)) ;
    OK (chunk == GB_CHUNK_DEFAULT) ;
    int gpu = 9 ;
    OK (GxB_Context_get (Context, GxB_CONTEXT_GPU_ID, &gpu)) ;
    CHECK (gpu == -1) ;
    OK (GxB_Context_set (Context, GxB_CONTEXT_GPU_ID, 3)) ;
    OK (GxB_Context_get (Context, GxB_CONTEXT_GPU_ID, &gpu)) ;
    printf ("gpu now %d\n", gpu) ;
    CHECK (gpu == -1) ;
    OK (GxB_Context_engage (GxB_CONTEXT_WORLD)) ;

    OK (GxB_Context_fprint (Context, "context", 3, stdout)) ;

    chunk = -1 ;
    GrB_Info expected = GrB_INVALID_VALUE ;
    ERR (GxB_Context_get_FP64 (Context, GxB_NTHREADS, &chunk)) ;
    ERR (GxB_Context_set_FP64 (Context, GxB_NTHREADS, chunk)) ;

    nth = 99 ;
    ERR (GxB_Context_get_INT32 (Context, GxB_CHUNK, &nth)) ;
    ERR (GxB_Context_set_INT32 (Context, GxB_CHUNK, nth)) ;

    ERR (GxB_Context_get (Context, 999, &nth)) ;
    ERR (GxB_Context_set (Context, 999, nth)) ;

    GrB_free (&Context) ;

    expected = GrB_NULL_POINTER ;
    ERR (GxB_Context_fprint (Context, "context", 3, stdout)) ;

    //==========================================================================
    // context_demo continued
    //==========================================================================

    //--------------------------------------------------------------------------
    // construct tuples for a decent-sized random matrix
    //--------------------------------------------------------------------------

    GrB_Index n = 1000 ; // 10000 ;
    GrB_Index nvals = 20000 ; // 2000000 ;
    simple_rand_seed (1) ;
    GrB_Index *I = mxMalloc (nvals * sizeof (GrB_Index)) ;
    GrB_Index *J = mxMalloc (nvals * sizeof (GrB_Index)) ;
    double    *X = mxMalloc (nvals * sizeof (double)) ;
    for (int k = 0 ; k < nvals ; k++)
    {
        I [k] = simple_rand_i ( ) % n ;
        J [k] = simple_rand_i ( ) % n ;
        X [k] = simple_rand_x ( ) ;
    }

    GrB_Matrix C = NULL ;
    OK (GrB_Matrix_new (&C, GrB_FP64, n, n)) ;
    OK (GrB_Matrix_build (C, I, J, X, nvals, GrB_PLUS_FP64)) ;

    //--------------------------------------------------------------------------
    // create random matrices parallel
    //--------------------------------------------------------------------------

    int nmats = MIN (2*nthreads, 256) ;
    GrB_Matrix G [256] ;

    for (int nmat = 1 ; nmat <= nmats ; nmat = 2*nmat)
    {
        double t1 = 0 ;

        // create nmat matrices, each in parallel with varying # of threads
        for (int nthreads2 = 1 ; nthreads2 <= nthreads ; nthreads2 *= 2)
        {
            int nouter = 1 ;            // # of user threads in outer loop
            int ninner = nthreads2 ;    // # of threads each user thread can use

            printf ("\nnmat: %4d nthreads2: %4d\n", nmat, nthreads2) ;

            while (ninner >= 1)
            {
                if (nouter <= nmat)
                {
                    double t = GB_OPENMP_GET_WTIME ;
                    #pragma omp parallel for num_threads (nouter) \
                        schedule (dynamic, 1)
                    for (int k = 0 ; k < nmat ; k++)
                    {
                        // each user thread constructs its own context
                        GxB_Context Context = NULL ;
                        OK (GxB_Context_new (&Context)) ;
                        OK (GxB_Context_set (Context, GxB_NTHREADS, ninner)) ;
                        OK (GxB_Context_engage (Context)) ;

                        // kth user thread builds kth matrix with ninner threads
                        GrB_Matrix A = NULL ;
                        OK (GrB_Matrix_new (&A, GrB_FP64, n, n)) ;
                        OK (GrB_Matrix_build (A, I, J, X, nvals,
                            GrB_PLUS_FP64)) ;

                        // save the matrix just built
                        G [k] = A ;
                        A = NULL ;

                        // each user thread frees its own context
                        OK (GxB_Context_disengage (Context)) ;
                        OK (GxB_Context_free (&Context)) ;
                    }

                    t = GB_OPENMP_GET_WTIME - t ;
                    if (nouter == 1 && ninner == 1) t1 = t ;

                    printf ("   threads (%4d,%4d): %4d "
                        "time: %8.4f sec speedup: %8.3f\n",
                        nouter, ninner, nouter * ninner, t, t1/t) ;

                    // check results
                    for (int k = 0 ; k < nmat ; k++)
                    {
                        bool ok = GB_mx_isequal (C, G [k], 1e-14) ;
                        GrB_free (&(G [k])) ;
                        CHECK (ok) ;
                    }

                }
                nouter = nouter * 2 ;
                ninner = ninner / 2 ;
            }
        }
    }

    mxFree (I) ;
    mxFree (J) ;
    mxFree (X) ;
    GrB_free (&C) ;

    GB_mx_put_global (true) ;
}

