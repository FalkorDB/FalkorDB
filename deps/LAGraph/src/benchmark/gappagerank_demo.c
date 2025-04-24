//------------------------------------------------------------------------------
// LAGraph/src/benchmark/gappagerank_demo.c: benchmark GAP PageRank
//------------------------------------------------------------------------------

// LAGraph, (c) 2019-2022 by The LAGraph Contributors, All Rights Reserved.
// SPDX-License-Identifier: BSD-2-Clause
//
// For additional details (including references to third party source code and
// other files) see the LICENSE file or contact permission@sei.cmu.edu. See
// Contributors.txt for a full list of contributors. Created, in part, with
// funding and support from the U.S. Government (see Acknowledgments.txt file).
// DM22-0790

// Contributed by Timothy A. Davis, Texas A&M University, and Gabor Szarnyas,
// BME

//------------------------------------------------------------------------------

#include "LAGraph_demo.h"

#define NTHREAD_LIST 1
// #define NTHREAD_LIST 2
#define THREAD_LIST 0

// #define NTHREAD_LIST 6
// #define THREAD_LIST 64, 32, 24, 12, 8, 4

#define LG_FREE_ALL                             \
{                                               \
    GrB_free (&A) ;                             \
    GrB_free (&Abool) ;                         \
    GrB_free (&PR) ;                            \
    LAGraph_Delete (&G, msg) ;                  \
    if (f != NULL) fclose (f) ;                 \
}

int main (int argc, char **argv)
{

    char msg [LAGRAPH_MSG_LEN] ;

    LAGraph_Graph G = NULL ;

    GrB_Matrix A = NULL ;
    GrB_Matrix Abool = NULL ;
    GrB_Vector PR = NULL ;
    FILE *f = NULL ;

    // start GraphBLAS and LAGraph
    bool burble = false ;
    demo_init (burble) ;

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
        false, false, true, NULL, false, argc, argv)) ;
    GrB_Index n, nvals ;
    GRB_TRY (GrB_Matrix_nrows (&n, G->A)) ;
    GRB_TRY (GrB_Matrix_nvals (&nvals, G->A)) ;

    // determine the cached out degree property
    LAGRAPH_TRY (LAGraph_Cached_OutDegree (G, msg)) ;

    // check # of sinks:
    GrB_Index nsinks ;
    GRB_TRY (GrB_Vector_nvals (&nvals, G->out_degree)) ;
    nsinks = n - nvals ;
    printf ("nsinks: %" PRIu64 "\n", nsinks) ;

    //--------------------------------------------------------------------------
    // compute the GAP pagerank
    //--------------------------------------------------------------------------

    // the GAP benchmark requires 16 trials
    int ntrials = 16 ;
    // ntrials = 1 ;    // HACK to run just one trial
    printf ("# of trials: %d\n", ntrials) ;

    float damping = 0.85 ;
    float tol = 1e-4 ;
    int iters = 0, itermax = 100 ;

    for (int kk = 1 ; kk <= nt ; kk++)
    {
        int nthreads = Nthreads [kk] ;
        if (nthreads > nthreads_max) continue ;
        LAGRAPH_TRY (LAGraph_SetNumThreads (1, nthreads, msg)) ;
        printf ("\n--------------------------- nthreads: %2d\n", nthreads) ;

        double total_time = 0 ;

        for (int trial = 0 ; trial < ntrials ; trial++)
        {
            GrB_free (&PR) ;
            double t1 = LAGraph_WallClockTime ( ) ;
            LAGRAPH_TRY (LAGr_PageRankGAP (&PR, &iters, G,
                damping, tol, itermax, msg)) ;
            t1 = LAGraph_WallClockTime ( ) - t1 ;
            printf ("trial: %2d time: %10.4f sec\n", trial, t1) ;
            total_time += t1 ;
        }

        float rsum ;
        GRB_TRY (GrB_reduce (&rsum, NULL, GrB_PLUS_MONOID_FP32, PR, NULL)) ;

        double t = total_time / ntrials ;
        printf ("GAP: %3d: avg time: %10.3f (sec), "
                "rate: %10.3f iters: %d rsum: %e\n", nthreads,
                t, 1e-6*((double) nvals) * iters / t, iters, rsum) ;
        fprintf (stderr, "GAP: Avg: PR %3d: %10.3f sec: %s rsum: %e\n",
             nthreads, t, matrix_name, rsum) ;

    }

    //--------------------------------------------------------------------------
    // compute the standard pagerank
    //--------------------------------------------------------------------------

    // the STD pagerank may be slower than the GAP-style pagerank, because it
    // must do extra work to handle sinks.  sum(PR) will always equal 1.

    for (int kk = 1 ; kk <= nt ; kk++)
    {
        int nthreads = Nthreads [kk] ;
        if (nthreads > nthreads_max) continue ;
        LAGRAPH_TRY (LAGraph_SetNumThreads (1, nthreads, msg)) ;
        printf ("\n--------------------------- nthreads: %2d\n", nthreads) ;

        double total_time = 0 ;

        for (int trial = 0 ; trial < ntrials ; trial++)
        {
            GrB_free (&PR) ;
            double t1 = LAGraph_WallClockTime ( ) ;
            LAGRAPH_TRY (LAGr_PageRank (&PR, &iters, G,
                damping, tol, itermax, msg)) ;
            t1 = LAGraph_WallClockTime ( ) - t1 ;
            printf ("trial: %2d time: %10.4f sec\n", trial, t1) ;
            total_time += t1 ;
        }

        float rsum ;
        GRB_TRY (GrB_reduce (&rsum, NULL, GrB_PLUS_MONOID_FP32, PR, NULL)) ;

        double t = total_time / ntrials ;
        printf ("STD: %3d: avg time: %10.3f (sec), "
                "rate: %10.3f iters: %d rsum: %e\n", nthreads,
                t, 1e-6*((double) nvals) * iters / t, iters, rsum) ;
        fprintf (stderr, "STD: Avg: PR %3d: %10.3f sec: %s rsum: %e\n",
             nthreads, t, matrix_name, rsum) ;

    }

    //--------------------------------------------------------------------------
    // free all workspace and finish
    //--------------------------------------------------------------------------

    LG_FREE_ALL ;
    LAGRAPH_TRY (LAGraph_Finalize (msg)) ;
    return (GrB_SUCCESS) ;
}
