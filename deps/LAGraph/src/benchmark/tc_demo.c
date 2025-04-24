//------------------------------------------------------------------------------
// LAGraph/src/benchmark/tc_demo.c: benchmark for LAGr_TriangleCount
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

// Usage:  test_tc < matrixmarketfile.mtx
//         test_tc matrixmarketfile.mtx
//         test_tc matrixmarketfile.grb

//  Known triangle counts:
//      kron:       106873365648
//      urand:      5378
//      twitter:    34824916864
//      web:        84907041475
//      road:       438804

#include "LAGraph_demo.h"

#define NTHREAD_LIST 1
// #define NTHREAD_LIST 2
#define THREAD_LIST 0

// #define NTHREAD_LIST 6
// #define THREAD_LIST 64, 32, 24, 12, 8, 4

#define LG_FREE_ALL                 \
{                                   \
    LAGraph_Delete (&G, NULL) ;     \
    GrB_free (&A) ;                 \
}

char t [256] ;

char *method_name (int method, int sorting)
{
    char *s ;
    switch (method)
    {
        case LAGr_TriangleCount_AutoMethod: s = "default (Sandia_LUT)           " ; break ;
        case LAGr_TriangleCount_Burkhardt:  s = "Burkhardt: sum ((A^2) .* A) / 6" ; break ;
        case LAGr_TriangleCount_Cohen:      s = "Cohen:     sum ((L*U) .* A) / 2" ; break ;
        case LAGr_TriangleCount_Sandia_LL:  s = "Sandia_LL: sum ((L*L) .* L)    " ; break ;
        case LAGr_TriangleCount_Sandia_UU:  s = "Sandia_UU: sum ((U*U) .* U)    " ; break ;
        case LAGr_TriangleCount_Sandia_LUT: s = "Sandia_LUT: sum ((L*U') .* L)  " ; break ;
        case LAGr_TriangleCount_Sandia_ULT: s = "Sandia_ULT: sum ((U*L') .* U)  " ; break ;
        default: abort ( ) ;
    }

    if (sorting == LAGr_TriangleCount_Descending) sprintf (t, "%s sort: descending degree", s) ;
    else if (sorting == LAGr_TriangleCount_Ascending) sprintf (t, "%s ascending degree", s) ;
    else if (sorting == LAGr_TriangleCount_AutoSort) sprintf (t, "%s auto-sort", s) ;
    else sprintf (t, "%s sort: none", s) ;
    return (t) ;
}


void print_method (FILE *f, int method, int sorting)
{
    fprintf (f, "%s\n", method_name (method, sorting)) ;
}

int main (int argc, char **argv)
{

    //--------------------------------------------------------------------------
    // initialize LAGraph and GraphBLAS
    //--------------------------------------------------------------------------

    char msg [LAGRAPH_MSG_LEN] ;

    GrB_Matrix A = NULL ;
    LAGraph_Graph G = NULL ;

    // start GraphBLAS and LAGraph
    bool burble = false ;
    demo_init (burble) ;

    int ntrials = 3 ;
    // ntrials = 1 ;        // HACK
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
    LAGRAPH_TRY (LAGraph_Graph_Print (G, LAGraph_SHORT, stdout, msg)) ;

    // determine the cached out degree property
    LAGRAPH_TRY (LAGraph_Cached_OutDegree (G, msg)) ;

    GrB_Index n, nvals ;
    GRB_TRY (GrB_Matrix_nrows (&n, G->A)) ;
    GRB_TRY (GrB_Matrix_nvals (&nvals, G->A)) ;

    //--------------------------------------------------------------------------
    // triangle counting
    //--------------------------------------------------------------------------

    GrB_Index ntriangles, ntsimple = 0 ;

#if 0
    // check # of triangles
    double tsimple = LAGraph_WallClockTime ( ) ;
    LAGRAPH_TRY (LG_check_tri (&ntsimple, G, NULL)) ;
    tsimple = LAGraph_WallClockTime ( ) - tsimple ;
    printf ("# of triangles: %" PRId64 " slow time: %g sec\n",
        ntsimple, tsimple) ;
#endif

    // warmup for more accurate timing, and also print # of triangles
    double ttot = LAGraph_WallClockTime ( ) ;
    printf ("\nwarmup method: ") ;
    LAGr_TriangleCount_Presort presort = LAGr_TriangleCount_AutoSort ;
    print_method (stdout, 6, presort) ;

    // warmup method:
    // LAGr_TriangleCount_Sandia_ULT: sum (sum ((U * L') .* U))
    LAGr_TriangleCount_Method method = LAGr_TriangleCount_Sandia_ULT ;
    LAGRAPH_TRY (LAGr_TriangleCount (&ntriangles, G, &method, &presort, msg)) ;
    printf ("# of triangles: %" PRIu64 "\n", ntriangles) ;
    print_method (stdout, 6, presort) ;
    ttot = LAGraph_WallClockTime ( ) - ttot ;
    printf ("nthreads: %3d time: %12.6f rate: %6.2f (Sandia_ULT, one trial)\n",
            nthreads_max, ttot, 1e-6 * nvals / ttot) ;

#if 0
    if (ntriangles != ntsimple)
    {
        printf ("wrong # triangles: %g %g\n", (double) ntriangles,
            (double) ntsimple) ;
        abort ( ) ;
    }
#endif

    double t_best = INFINITY ;
    int method_best = -1 ;
    int nthreads_best = -1 ;
    int sorting_best = 0 ;

    // kron: input graph: nodes: 134217726 edges: 4223264644
    // fails on methods 3 and 4.

    // just try methods 5 and 6
    // for (int method = 5 ; method <= 6 ; method++)

    // try all methods 3 to 5
    for (int method = 3 ; method <= 5 ; method++)
    {
        // for (int sorting = -1 ; sorting <= 2 ; sorting++)

        int sorting = LAGr_TriangleCount_AutoSort ; // just use auto-sort
        {
            printf ("\nMethod: ") ; print_method (stdout, method, sorting) ;
            if (n == 134217726 && method < 5)
            {
                printf ("kron fails on method %d; skipped\n", method) ;
                continue ;
            }
            if (n != 134217728 && method < 5)
            {
                printf ("all but urand is slow with method %d: skipped\n",
                        method) ;
                continue ;
            }

            for (int t = 1 ; t <= nt ; t++)
            {
                int nthreads = Nthreads [t] ;
                if (nthreads > nthreads_max) continue ;
                LAGRAPH_TRY (LAGraph_SetNumThreads (1, nthreads, msg)) ;
                GrB_Index nt2 ;
                double ttot = 0, ttrial [100] ;
                LAGr_TriangleCount_Presort p ;
                LAGr_TriangleCount_Method m ;
                for (int trial = 0 ; trial < ntrials ; trial++)
                {
                    double tt = LAGraph_WallClockTime ( ) ;
                    m = method ;
                    p = sorting ;
                    LAGRAPH_TRY(LAGr_TriangleCount (&nt2, G, &m, &p, msg));
                    ttrial [trial] = LAGraph_WallClockTime ( ) - tt ;
                    ttot += ttrial [trial] ;
                    printf ("trial %2d: %12.6f sec rate %6.2f  # triangles: "
                        "%g\n", trial, ttrial [trial],
                        1e-6 * nvals / ttrial [trial], (double) nt2) ;
                }
                ttot = ttot / ntrials ;
                printf ("nthreads: %3d time: %12.6f rate: %6.2f", nthreads,
                        ttot, 1e-6 * nvals / ttot) ;
                printf ("   # of triangles: %" PRId64 " presort: %d\n",
                        ntriangles, (int) p) ;
                if (nt2 != ntriangles)
                {
                    printf ("Test failure!\n") ;
                    abort ( ) ;
                }
                fprintf (stderr, "\nMethod used: ") ;
                print_method (stderr, m, p) ;
                fprintf (stderr, "Avg: TC method%d.%d %3d: %10.3f sec: %s\n",
                         method, sorting, nthreads, ttot, matrix_name) ;

                if (ttot < t_best)
                {
                    t_best = ttot ;
                    method_best = method ;
                    nthreads_best = nthreads ;
                    sorting_best = sorting ;
                }
            }
        }
    }

    printf ("\nBest method: ") ;
    print_method (stdout, method_best, sorting_best) ;
    printf ("nthreads: %3d time: %12.6f rate: %6.2f\n",
        nthreads_best, t_best, 1e-6 * nvals / t_best) ;
    LG_FREE_ALL ;
    LAGRAPH_TRY (LAGraph_Finalize (msg)) ;
    return (GrB_SUCCESS) ;
}
