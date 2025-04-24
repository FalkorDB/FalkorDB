//------------------------------------------------------------------------------
// LAGraph/src/benchmark/cc_demo.c: benchmark LAGr_ConnectedComponents
//------------------------------------------------------------------------------

// LAGraph, (c) 2019-2022 by The LAGraph Contributors, All Rights Reserved.
// SPDX-License-Identifier: BSD-2-Clause
//
// For additional details (including references to third party source code and
// other files) see the LICENSE file or contact permission@sei.cmu.edu. See
// Contributors.txt for a full list of contributors. Created, in part, with
// funding and support from the U.S. Government (see Acknowledgments.txt file).
// DM22-0790

// Contributed by Timothy A. Davis, Texas A&M University

//------------------------------------------------------------------------------

// Usage: test_cc can be used with both stdin or a file as its input,
// in either grb or mtx format.

//------------------------------------------------------------------------------

#include "LAGraph_demo.h"
#include "LAGraphX.h"
#include "LG_alg_internal.h"

#undef  LG_FREE_ALL
#define LG_FREE_ALL                 \
{                                   \
    LAGraph_Delete (&G, NULL) ;     \
    GrB_free (&components) ;        \
    GrB_free (&components2) ;       \
}

#define NTHREAD_LIST 1
// #define NTHREAD_LIST 2
#define THREAD_LIST 0

// #define NTHREAD_LIST 6
// #define THREAD_LIST 64, 32, 24, 12, 8, 4

GrB_Index countCC (GrB_Vector f, GrB_Index n)
{
    GrB_Index nCC = 0;
    GrB_Index *w_val = NULL ;
    LAGraph_Malloc ((void **) &w_val, n, sizeof (GrB_Index), NULL) ;
    if (w_val == NULL) { printf ("out of memory\n") ; abort ( ) ; }
    GrB_Index *i_val = NULL ;
    #if LAGRAPH_SUITESPARSE
    // SuiteSparse:GraphBLAS allows NULL inputs to GrB_Vector_extractTuples
    #else
    LAGraph_Malloc ((void **) &i_val, n, sizeof (GrB_Index), NULL) ;
    if (i_val == NULL) { printf ("out of memory\n") ; abort ( ) ; }
    #endif
    GrB_Vector_extractTuples (i_val, w_val, &n, f) ;
    for (GrB_Index i = 0; i < n; i++)
    {
        if (w_val[i] == i)
        {
            nCC++ ;
        }
    }
    LAGraph_Free ((void **) &i_val, NULL) ;
    LAGraph_Free ((void **) &w_val, NULL) ;
    return nCC;
}

int main (int argc, char **argv)
{

    char msg [LAGRAPH_MSG_LEN] ;

    LAGraph_Graph G = NULL ;
    GrB_Vector components = NULL, components2 = NULL ;

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
    fprintf (stderr, "\n%s:\n", matrix_name) ;
    LAGRAPH_TRY (readproblem (&G,
        NULL,   // no source nodes
        true,   // make the graph undirected, and symmetrize the matrix
        false,  // do not remove self-edges
        true,   // structural only, no values needed
        NULL,   // no type preference
        false,  // do not ensure all entries positive
        argc, argv)) ;
    GrB_Index n, nvals ;
    GRB_TRY (GrB_Matrix_nrows (&n, G->A)) ;
    GRB_TRY (GrB_Matrix_nvals (&nvals, G->A)) ;

    //--------------------------------------------------------------------------
    // begin tests
    //--------------------------------------------------------------------------

    // warmup
    LAGRAPH_TRY (LAGr_ConnectedComponents (&components, G, msg)) ;
    GrB_Index nCC = countCC (components, n) ;
    printf ("nCC: %20.0g\n", (double) nCC) ;

#if 0 & LG_CHECK_RESULT
    double tcheck = LAGraph_WallClockTime ( ) ;
    int result = LG_check_cc (components, G, msg) ;
    if (result != 0)
    {
        printf ("test failure: (%d) %s\n", result, msg) ;
    }
    tcheck = LAGraph_WallClockTime ( ) - tcheck ;
    LAGRAPH_TRY (result) ;
    printf ("LG_check_cc passed, time: %g\n", tcheck) ;
#endif

    #define NTRIALS 16
    // #define NTRIALS 1
    printf ("# of trials: %d\n\n", NTRIALS) ;

    //--------------------------------------------------------------------------
    // LAGr_ConnectedComponents
    //--------------------------------------------------------------------------

    for (int trial = 1 ; trial <= nt ; trial++)
    {
        int nthreads = Nthreads [trial] ;
        if (nthreads > nthreads_max) continue ;
        LAGRAPH_TRY (LAGraph_SetNumThreads (1, nthreads, NULL)) ;
        double ttt = 0 ;
        int ntrials = NTRIALS ;
        for (int k = 0 ; k < ntrials ; k++)
        {
            GrB_free (&components2) ;
            double ttrial = LAGraph_WallClockTime ( ) ;
            LAGRAPH_TRY (LAGr_ConnectedComponents (&components2, G, msg)) ;
            ttrial = LAGraph_WallClockTime ( ) - ttrial ;
            ttt += ttrial ;
            printf ("SV6:      nthreads: %2d trial: %2d time: %10.4f sec\n",
                nthreads, k, ttrial) ;
            GrB_Index nCC2 = countCC (components2, n) ;
            if (nCC != nCC2) printf ("failure! %g %g diff %g\n",
                (double) nCC, (double) nCC2, (double) (nCC-nCC2)) ;
        }
        ttt = ttt / ntrials ;
        printf ("SV6:      nthreads: %2d Avg: time: %10.4f sec ntrials %d\n\n",
                nthreads, ttt, ntrials) ;
        fprintf (stderr,
                "SV6:      nthreads: %2d Avg: time: %10.4f sec ntrials %d\n",
                nthreads, ttt, ntrials) ;
    }

    //--------------------------------------------------------------------------
    // 7: draft version
    //--------------------------------------------------------------------------

#if 0
    for (int trial = 1 ; trial <= nt ; trial++)
    {
        int nthreads = Nthreads [trial] ;
        if (nthreads > nthreads_max) continue ;
        LAGRAPH_TRY (LAGraph_SetNumThreads (1, nthreads, NULL)) ;
        double ttt = 0 ;
        int ntrials = NTRIALS ;
        for (int k = 0 ; k < ntrials ; k++)
        {
            GrB_free (&components2) ;
            double ttrial = LAGraph_WallClockTime ( ) ;
            LAGRAPH_TRY (LG_CC_7 (&components2, G, msg)) ;
            ttrial = LAGraph_WallClockTime ( ) - ttrial ;
            ttt += ttrial ;
            printf ("SV7:      nthreads: %2d trial: %2d time: %10.4f sec\n",
                nthreads, k, ttrial) ;
            GrB_Index nCC2 = countCC (components2, n) ;
            if (nCC != nCC2) printf ("failure! %g %g diff %g\n",
                (double) nCC, (double) nCC2, (double) (nCC-nCC2)) ;
        }
        ttt = ttt / ntrials ;
        printf ("SV7:      nthreads: %2d Avg: time: %10.4f sec ntrials %d\n\n",
                nthreads, ttt, ntrials) ;
        fprintf (stderr,
                "SV7:      nthreads: %2d Avg: time: %10.4f sec ntrials %d\n",
                nthreads, ttt, ntrials) ;
    }
#endif

    //--------------------------------------------------------------------------
    // LG_CC_FastSV5: using 32-bit integers
    //--------------------------------------------------------------------------

#if 0
    for (int trial = 1 ; trial <= nt ; trial++)
    {
        int nthreads = Nthreads [trial] ;
        if (nthreads > nthreads_max) continue ;
        LAGRAPH_TRY (LAGraph_SetNumThreads (1, nthreads, NULL)) ;
        double ttt = 0 ;
        int ntrials = NTRIALS ;
        for (int k = 0 ; k < ntrials ; k++)
        {
            GrB_free (&components2) ;
            double ttrial = LAGraph_WallClockTime ( ) ;
            LAGRAPH_TRY (LG_CC_FastSV5 (&components2, G, msg)) ;
            ttrial = LAGraph_WallClockTime ( ) - ttrial ;
            ttt += ttrial ;
            printf ("SV5b:     nthreads: %2d trial: %2d time: %10.4f sec\n",
                nthreads, k, ttrial) ;
            GrB_Index nCC2 = countCC (components2, n) ;
            if (nCC != nCC2) printf ("failure! %g %g diff %g\n",
                (double) nCC, (double) nCC2, (double) (nCC-nCC2)) ;
        }
        ttt = ttt / ntrials ;
        printf ("SV5b:     nthreads: %2d Avg: time: %10.4f sec ntrials %d\n\n",
                nthreads, ttt, ntrials) ;
        fprintf (stderr,
                "SV5b:     nthreads: %2d Avg: time: %10.4f sec ntrials %d\n",
                nthreads, ttt, ntrials) ;
    }
#endif

    //--------------------------------------------------------------------------
    // LG_CC_Boruvka
    //--------------------------------------------------------------------------

#if 0
    for (int trial = 1 ; trial <= nt ; trial++)
    {
        int nthreads = Nthreads [trial] ;
        if (nthreads > nthreads_max) continue ;
        LAGRAPH_TRY (LAGraph_SetNumThreads (1, nthreads, NULL)) ;
        double ttt = 0 ;
        int ntrials = 1 /* NTRIALS */ ;
        for (int k = 0 ; k < ntrials ; k++)
        {
            GrB_free (&components2) ;
            double ttrial = LAGraph_WallClockTime ( ) ;
            LAGRAPH_TRY (LG_CC_Boruvka (&components2, G, msg)) ;
            ttrial = LAGraph_WallClockTime ( ) - ttrial ;
            ttt += ttrial ;
            printf ("Boruvka:  nthreads: %2d trial: %2d time: %10.4f sec\n",
                nthreads, k, ttrial) ;
            GrB_Index nCC2 = countCC (components2, n) ;
            if (nCC != nCC2) printf ("failure! %g %g diff %g\n",
                (double) nCC, (double) nCC2, (double) (nCC-nCC2)) ;
        }
        ttt = ttt / ntrials ;
        printf ("Boruvka:  nthreads: %2d Avg: time: %10.4f sec ntrials %d\n\n",
                nthreads, ttt, ntrials) ;
        fprintf (stderr,
                "Boruvka:  nthreads: %2d Avg: time: %10.4f sec ntrials %d\n",
                nthreads, ttt, ntrials) ;
    }
#endif

    //--------------------------------------------------------------------------
    // LAGraph_cc_lacc
    //--------------------------------------------------------------------------

#if 0
    for (int trial = 1 ; trial <= nt ; trial++)
    {
        int nthreads = Nthreads [trial] ;
        if (nthreads > nthreads_max) continue ;
        LAGRAPH_TRY (LAGraph_SetNumThreads (1, nthreads, NULL)) ;
        double ttt = 0 ;
        int ntrials = 1 /* NTRIALS */ ;
        for (int k = 0 ; k < ntrials ; k++)
        {
            GrB_free (&components2) ;
            double ttrial = LAGraph_WallClockTime ( ) ;
            LAGRAPH_TRY (LAGraph_cc_lacc (&components2, G->A, false, msg)) ;
            ttrial = LAGraph_WallClockTime ( ) - ttrial ;
            ttt += ttrial ;
            printf ("LACC:     nthreads: %2d trial: %2d time: %10.4f sec\n",
                nthreads, k, ttrial) ;
            GrB_Index nCC2 = countCC (components2, n) ;
            if (nCC != nCC2) printf ("failure! %g %g diff %g\n",
                (double) nCC, (double) nCC2, (double) (nCC-nCC2)) ;
        }
        ttt = ttt / ntrials ;
        printf ("LACC:     nthreads: %2d Avg: time: %10.4f sec ntrials %d\n\n",
                nthreads, ttt, ntrials) ;
        fprintf (stderr,
                "LACC:     nthreads: %2d Avg: time: %10.4f sec ntrials %d\n",
                nthreads, ttt, ntrials) ;
    }
#endif

    //--------------------------------------------------------------------------
    // free all workspace and finish
    //--------------------------------------------------------------------------

    LG_FREE_ALL ;
    LAGRAPH_TRY (LAGraph_Finalize (msg)) ;
    return (GrB_SUCCESS) ;
}
