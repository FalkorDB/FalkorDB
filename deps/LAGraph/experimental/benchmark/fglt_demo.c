//------------------------------------------------------------------------------
// LAGraph/experimental/benchmark/fglt_demo.c: benchmark Fast Graphlet Transform
//------------------------------------------------------------------------------

// LAGraph, (c) 2019-2022 by The LAGraph Contributors, All Rights Reserved.
// SPDX-License-Identifier: BSD-2-Clause
//
// For additional details (including references to third party source code and
// other files) see the LICENSE file or contact permission@sei.cmu.edu. See
// Contributors.txt for a full list of contributors. Created, in part, with
// funding and support from the U.S. Government (see Acknowledgments.txt file).
// DM22-0790

// Contributed by Tanner Hoke

//------------------------------------------------------------------------------

#include "../../src/benchmark/LAGraph_demo.h"
#include "LAGraphX.h"
#include "LG_Xtest.h"

#define NTHREAD_LIST 1
// #define NTHREAD_LIST 2
#define THREAD_LIST 0

// #define NTHREAD_LIST 6
// #define THREAD_LIST 64, 32, 24, 12, 8, 4

#define LG_FREE_ALL                             \
{                                               \
    GrB_free (&A) ;                             \
    GrB_free (&Fnet) ;                          \
    LAGraph_Delete (&G, msg) ;                  \
    if (f != NULL) fclose (f) ;                 \
}

int main (int argc, char **argv)
{

    char msg [LAGRAPH_MSG_LEN] ;

    LAGraph_Graph G = NULL ;

    GrB_Matrix A = NULL ;
    GrB_Matrix Fnet = NULL ;
    GrB_Matrix Abool = NULL ;
    FILE *f = NULL ;

    // start GraphBLAS and LAGraph
    bool burble = false ;
    demo_init (burble) ;

    int nt = NTHREAD_LIST ;
    int Nthreads [20] = { 0, THREAD_LIST } ;
    int nthreads_max, nthreads_outer, nthreads_inner ;
    LAGRAPH_TRY (LAGraph_GetNumThreads (&nthreads_outer, &nthreads_inner, NULL)) ;
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
    LAGRAPH_TRY (LAGraph_Cached_OutDegree (G, msg)) ;

    //--------------------------------------------------------------------------
    // compute the GAP pagerank
    //--------------------------------------------------------------------------

    // warmup for more accurate timing
    double tt = LAGraph_WallClockTime ( ) ;
    LAGRAPH_TRY (LAGraph_FastGraphletTransform (&Fnet, G, true, msg)) ;
    tt = LAGraph_WallClockTime ( ) - tt ;
    GRB_TRY (GrB_free (&Fnet)) ;
    printf ("warmup time %g sec\n", tt) ;


    // the GAP benchmark requires 16 trials
    int ntrials = 16 ;
    ntrials = 1 ;    // HACK to run just one trial
    printf ("# of trials: %d\n", ntrials) ;

    for (int kk = 1 ; kk <= nt ; kk++)
    {
        int nthreads = Nthreads [kk] ;
        if (nthreads > nthreads_max) continue ;
        LAGRAPH_TRY (LAGraph_SetNumThreads (1, nthreads, msg)) ;
        printf ("\n--------------------------- nthreads: %2d\n", nthreads) ;

        double total_time = 0 ;

        for (int trial = 0 ; trial < ntrials ; trial++)
        {
            int64_t seed = trial * n + 1 ;
            double tt = LAGraph_WallClockTime ( ) ;
            LAGRAPH_TRY (LAGraph_FastGraphletTransform (&Fnet, G, true, msg)) ;
            tt = LAGraph_WallClockTime ( ) - tt ;
            GRB_TRY (GrB_free (&Fnet)) ;
            printf ("trial: %2d time: %10.4f sec\n", trial, tt) ;
            total_time += tt ;
        }

        double t = total_time / ntrials ;
        printf ("FGLT: %3d: avg time: %10.3f (sec) matrix: %s\n",
                nthreads, t, matrix_name) ;
        fprintf (stderr, "FGLT: %3d: avg time: %10.3f (sec) matrix: %s\n",
                nthreads, t, matrix_name) ;

    }

    //--------------------------------------------------------------------------
    // free all workspace and finish
    //--------------------------------------------------------------------------

    LG_FREE_ALL ;
    LAGRAPH_TRY (LAGraph_Finalize (msg)) ;
    return (GrB_SUCCESS) ;
}
