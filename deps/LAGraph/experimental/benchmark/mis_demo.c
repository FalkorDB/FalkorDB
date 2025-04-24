//------------------------------------------------------------------------------
// LAGraph/experimental/benchmark/mis_demo.c: benchmark for triangle centrality
//------------------------------------------------------------------------------

// LAGraph, (c) 2019-2022 by The LAGraph Contributors, All Rights Reserved.
// SPDX-License-Identifier: BSD-2-Clause
//
// For additional details (including references to third party source code and
// other files) see the LICENSE file or contact permission@sei.cmu.edu. See
// Contributors.txt for a full list of contributors. Created, in part, with
// funding and support from the U.S. Government (see Acknowledgments.txt file).
// DM22-0790

// Contributed by Tim Davis, Texas A&M

//------------------------------------------------------------------------------

// Usage:  mis_demo < matrixmarketfile.mtx
//         mis_demo matrixmarketfile.mtx
//         mis_demo matrixmarketfile.grb

#include "../../src/benchmark/LAGraph_demo.h"
#include "LAGraphX.h"
#include "LG_Xtest.h"

// #define NTHREAD_LIST 2

#define NTHREAD_LIST 1
#define THREAD_LIST 0

// #define NTHREAD_LIST 6
// #define THREAD_LIST 64, 32, 24, 12, 8, 4

// #define NTHREAD_LIST 7
// #define THREAD_LIST 40, 20, 16, 8, 4, 2, 1

#define LG_FREE_ALL                 \
{                                   \
    LAGraph_Delete (&G, NULL) ;     \
    GrB_free (&A) ;                 \
    GrB_free (&mis) ;               \
}

int main (int argc, char **argv)
{

    //--------------------------------------------------------------------------
    // initialize LAGraph and GraphBLAS
    //--------------------------------------------------------------------------

    char msg [LAGRAPH_MSG_LEN] ;

    GrB_Vector mis = NULL ;
    GrB_Matrix A = NULL ;
    LAGraph_Graph G = NULL ;

    // start GraphBLAS and LAGraph
    bool burble = false ;
    demo_init (burble) ;
    LAGRAPH_TRY (LAGraph_Random_Init (msg)) ;

    int ntrials = 3 ;
    ntrials = 3 ;
    printf ("# of trials: %d\n", ntrials) ;

    int nt = NTHREAD_LIST ;
    int Nthreads [20] = { 0, THREAD_LIST } ;

    int nthreads_max, nthreads_outer, nthreads_inner ;
    LAGRAPH_TRY (LAGraph_GetNumThreads (&nthreads_outer, &nthreads_inner, msg)) ;
    nthreads_max = nthreads_outer * nthreads_inner ;

    if (Nthreads [1] == 0)
    {
        // create thread list automatically
        Nthreads [1] = nthreads_max ;
        for (int t = 2 ; t <= nt ; t++)
        {
            Nthreads [t] = Nthreads [t-1] / 2 ;
            if (Nthreads [t] == 0) nt = t-1 ;
        }
    }
    printf ("threads to test: ") ;
    for (int t = 1 ; t <= nt ; t++)
    {
        int nthreads = Nthreads [t] ;
        if (nthreads > nthreads_max) continue ;
        printf (" %d", nthreads) ;
    }
    printf ("\n") ;

    //--------------------------------------------------------------------------
    // read in the graph
    //--------------------------------------------------------------------------

    char *matrix_name = (argc > 1) ? argv [1] : "stdin" ;
    LAGRAPH_TRY (readproblem (&G, NULL,
        true, true, true, NULL, false, argc, argv)) ;

    GrB_Index n, nvals ;
    GRB_TRY (GrB_Matrix_nrows (&n, G->A)) ;
    GRB_TRY (GrB_Matrix_nvals (&nvals, G->A)) ;
    // LAGRAPH_TRY (LAGraph_Graph_Print (G, LAGraph_SHORT, stdout, msg)) ;
    LAGRAPH_TRY (LAGraph_Cached_OutDegree (G, msg)) ;

    //--------------------------------------------------------------------------
    // maximal independent set
    //--------------------------------------------------------------------------

    // warmup for more accurate timing
    double tt = LAGraph_WallClockTime ( ) ;
    LAGRAPH_TRY (LAGraph_MaximalIndependentSet (&mis, G, 1, NULL, msg)) ;
    tt = LAGraph_WallClockTime ( ) - tt ;
    LAGRAPH_TRY (LG_check_mis (G->A, mis, NULL, msg)) ;
    GRB_TRY (GrB_free (&mis)) ;
    printf ("warmup time %g sec\n", tt) ;

    for (int t = 1 ; t <= nt ; t++)
    {
        int nthreads = Nthreads [t] ;
        if (nthreads > nthreads_max) continue ;
        LAGRAPH_TRY (LAGraph_SetNumThreads (1, nthreads, msg)) ;
        double ttot = 0, ttrial [100] ;
        for (int trial = 0 ; trial < ntrials ; trial++)
        {
            int64_t seed = trial * n + 1 ;
            double tt = LAGraph_WallClockTime ( ) ;
            LAGRAPH_TRY (LAGraph_MaximalIndependentSet (&mis, G, seed, NULL,
                msg)) ;
            LAGRAPH_TRY (LG_check_mis (G->A, mis, NULL, msg)) ;
            GRB_TRY (GrB_free (&mis)) ;
            ttrial [trial] = LAGraph_WallClockTime ( ) - tt ;
            ttot += ttrial [trial] ;
            printf ("seed %g threads %2d trial %2d: %12.6f sec\n",
                (double) seed, nthreads, trial, ttrial [trial]) ;
            fprintf (stderr,
                "seed %g threads %2d trial %2d: %12.6f sec\n",
                (double) seed, nthreads, trial, ttrial [trial]) ;
        }
        ttot = ttot / ntrials ;

        printf ("Avg: MIS nthreads: %3d time: %12.6f matrix: %s\n",
            nthreads, ttot, matrix_name) ;

        fprintf (stderr, "Avg: MIS nthreads: %3d time: %12.6f matrix: %s\n",
            nthreads, ttot, matrix_name) ;
    }

    fflush (stdout) ;
    LG_FREE_ALL ;
    LAGRAPH_TRY (LAGraph_Random_Finalize (msg)) ;
    LAGRAPH_TRY (LAGraph_Finalize (msg)) ;
    return (GrB_SUCCESS) ;
}
