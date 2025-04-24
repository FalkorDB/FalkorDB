//------------------------------------------------------------------------------
// LAGraph/src/benchmark/sssp_demo: test for LAGraph SSSP
//------------------------------------------------------------------------------

// LAGraph, (c) 2019-2022 by The LAGraph Contributors, All Rights Reserved.
// SPDX-License-Identifier: BSD-2-Clause
//
// For additional details (including references to third party source code and
// other files) see the LICENSE file or contact permission@sei.cmu.edu. See
// Contributors.txt for a full list of contributors. Created, in part, with
// funding and support from the U.S. Government (see Acknowledgments.txt file).
// DM22-0790

// Contributed by Jinhao Chen, Scott Kolodziej and Timothy A. Davis, Texas A&M
// University

//------------------------------------------------------------------------------

// Usage:
// test_sssp matrix.mtx sourcenodes.mtx delta
// test_sssp matrix.grb sourcenodes.mtx delta

#include "LAGraph_demo.h"

// #define NTHREAD_LIST 1
// #define NTHREAD_LIST 2
// #define THREAD_LIST 0

#define NTHREAD_LIST 1
#define THREAD_LIST 0

#define LG_FREE_ALL                 \
{                                   \
    LAGraph_Delete (&G, NULL) ;     \
    GrB_free (&SourceNodes) ;       \
    GrB_free (&pathlen) ;           \
}

int main (int argc, char **argv)
{
    char msg [LAGRAPH_MSG_LEN] ;

    LAGraph_Graph G = NULL ;
    GrB_Matrix SourceNodes = NULL ;
    GrB_Vector pathlen = NULL ;

    // start GraphBLAS and LAGraph
    bool burble = false ;
    demo_init (burble) ;

    //--------------------------------------------------------------------------
    // determine # of threads to use
    //--------------------------------------------------------------------------

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
    LAGRAPH_TRY (readproblem (&G, &SourceNodes,
        false, false, false, GrB_INT32, false, argc, argv)) ;
    GrB_Index n, nvals ;
    GRB_TRY (GrB_Matrix_nrows (&n, G->A)) ;
    GRB_TRY (GrB_Matrix_nvals (&nvals, G->A)) ;
    LAGRAPH_TRY (LAGraph_Cached_EMin (G, msg)) ;

    //--------------------------------------------------------------------------
    // get delta
    //--------------------------------------------------------------------------

    int32_t delta ;
    if (argc > 3)
    {
        // usage:  ./test_sssp matrix sourcenodes delta
        delta = atoi (argv [3]) ;
    }
    else
    {
        // usage:  ./test_sssp matrix sourcenodes
        // or:     ./test_sssp < matrix
        delta = 2 ;
    }
    printf ("delta: %d\n", delta) ;
    GrB_Scalar Delta = NULL ;
    GRB_TRY (GrB_Scalar_new (&Delta, GrB_INT32)) ;
    GRB_TRY (GrB_Scalar_setElement (Delta, delta)) ;

    //--------------------------------------------------------------------------
    // begin tests
    //--------------------------------------------------------------------------

    // get the number of source nodes
    GrB_Index nsource ;
    GRB_TRY (GrB_Matrix_nrows (&nsource, SourceNodes)) ;

    int ntrials = (int) nsource ;

    for (int tt = 1 ; tt <= nt ; tt++)
    {
        int nthreads = Nthreads [tt] ;
        if (nthreads > nthreads_max) continue ;
        LAGRAPH_TRY (LAGraph_SetNumThreads (1, nthreads, msg)) ;
        double total_time = 0 ;

        for (int trial = 0 ; trial < ntrials ; trial++)
        {

            //------------------------------------------------------------------
            // get the source node for this trial
            //------------------------------------------------------------------

            // src = SourceNodes [trial]
            GrB_Index src = -1 ;
            GRB_TRY (GrB_Matrix_extractElement (&src, SourceNodes, trial, 0)) ;
            src-- ;     // convert from 1-based to 0-based

            //------------------------------------------------------------------
            // sssp
            //------------------------------------------------------------------

            GrB_free (&pathlen) ;
            double ttrial = LAGraph_WallClockTime ( ) ;
            LAGRAPH_TRY (LAGr_SingleSourceShortestPath (&pathlen, G, src,
                Delta, msg)) ;
            ttrial = LAGraph_WallClockTime ( ) - ttrial ;

            printf ("sssp15:  threads: %2d trial: %2d source %12" PRId64
                " time: %10.4f sec\n", nthreads, trial, src, ttrial) ;
            total_time += ttrial ;

#if LG_CHECK_RESULT
            // check result
            if (trial == 0)
            {
                // all trials can be checked, but this is slow so do just
                // for the first trial
                double tcheck = LAGraph_WallClockTime ( ) ;
                LAGRAPH_TRY (LG_check_sssp (pathlen, G, src, msg)) ;
                tcheck = LAGraph_WallClockTime ( ) - tcheck ;
                printf ("total check time: %g sec\n", tcheck) ;
            }
#endif
        }

        //----------------------------------------------------------------------
        // report results
        //----------------------------------------------------------------------

        printf ("\n") ;
        double e = (double) nvals ;
        total_time = total_time / ntrials ;
        printf ("%2d: SSSP    time: %14.6f sec  rate: %8.2f (delta %d)\n",
            nthreads, total_time, 1e-6 * e / total_time, delta);
        fprintf (stderr, "Avg: SSSP         %3d: %10.3f sec: %s\n",
             nthreads, total_time, matrix_name) ;
    }

    //--------------------------------------------------------------------------
    // free all workspace and finish
    //--------------------------------------------------------------------------

    LG_FREE_ALL ;
    LAGRAPH_TRY (LAGraph_Finalize (msg)) ;
    return (GrB_SUCCESS) ;
}
