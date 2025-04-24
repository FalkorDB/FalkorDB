//------------------------------------------------------------------------------
// LAGraph/src/benchmark/bc_demo.c:  Benchmark for LAGr_Betweenness
//------------------------------------------------------------------------------

// LAGraph, (c) 2019-2022 by The LAGraph Contributors, All Rights Reserved.
// SPDX-License-Identifier: BSD-2-Clause
//
// For additional details (including references to third party source code and
// other files) see the LICENSE file or contact permission@sei.cmu.edu. See
// Contributors.txt for a full list of contributors. Created, in part, with
// funding and support from the U.S. Government (see Acknowledgments.txt file).
// DM22-0790

// Contributed by Scott Kolodziej and Timothy A. Davis, Texas A&M University

//------------------------------------------------------------------------------

// usage:
// bc_demo < matrixfile.mtx
// bc_demo matrixfile.mtx sourcenodes.mtx

#include "LAGraph_demo.h"

// NTHREAD_LIST and THREAD_LIST are used together to select the # of OpenMP
// threads to use in this demo.  If THREAD_LIST is zero, then the # of threads
// is chosen automatically.  Let p = omp_get_max_threads ( ).  Then if
// NTHREAD_LIST is 4, the experiment will be run 4 times, with p, p/2, p/4, and
// p/8 threads.  If you instead wish to exactly specify the number of threads
// for each run, the define THREAD_LIST as a list of integers.  Each of the
// *_demo.c main programs has its own NTHREAD_LIST and THREAD_LIST definitions.

// to run just once, with p = omp_get_max_threads() threads
#define NTHREAD_LIST 1
#define THREAD_LIST 0

// to run with p and p/2 threads, if p = omp_get_max_threads()
// #define NTHREAD_LIST 2
// #define THREAD_LIST 0

// to run with 40, 20, 10, and 1 thread:
// #define NTHREAD_LIST 4
// #define THREAD_LIST 40, 20, 10, 1

// to run with 64, 32, 24, 12, 8, and 4 threads:
// #define NTHREAD_LIST 6
// #define THREAD_LIST 64, 32, 24, 12, 8, 4

#define LG_FREE_ALL                 \
{                                   \
    LAGraph_Delete (&G, NULL) ;     \
    GrB_free (&c2) ;                \
    GrB_free (&centrality) ;        \
    GrB_free (&SourceNodes) ;       \
}

#define BATCH_SIZE 4

int main (int argc, char **argv)
{

    char msg [LAGRAPH_MSG_LEN] ;

    LAGraph_Graph G = NULL ;
    GrB_Vector centrality = NULL, c2 = NULL ;
    GrB_Matrix SourceNodes = NULL ;

    // start GraphBLAS and LAGraph
    bool burble = false ;
    demo_init (burble) ;

    int batch_size = BATCH_SIZE ;

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

    double *tt = malloc ((nthreads_max+1) *sizeof (double));

    //--------------------------------------------------------------------------
    // read in the graph
    //--------------------------------------------------------------------------

    char *matrix_name = (argc > 1) ? argv [1] : "stdin" ;
    LAGRAPH_TRY (readproblem (&G, &SourceNodes,
        false, false, true, NULL, false, argc, argv)) ;
    GrB_Index n, nvals ;
    GRB_TRY (GrB_Matrix_nrows (&n, G->A)) ;
    GRB_TRY (GrB_Matrix_nvals (&nvals, G->A)) ;

    //--------------------------------------------------------------------------
    // get the source nodes
    //--------------------------------------------------------------------------

    GrB_Index nsource ;
    GRB_TRY (GrB_Matrix_nrows (&nsource, SourceNodes)) ;
    if (nsource % batch_size != 0)
    {
        printf ("SourceNode size must be multiple of batch_size (%d)\n",
            batch_size) ;
        exit (1) ;
    }

    //--------------------------------------------------------------------------
    // Begin tests
    //--------------------------------------------------------------------------

    int ntrials = 0 ;

    for (int64_t kstart = 0 ; kstart < nsource ; kstart += batch_size)
    {

        //----------------------------------------------------------------------
        // Create batch of vertices to use in traversal
        //----------------------------------------------------------------------

        ntrials++ ;
        printf ("\nTrial %d : sources: [", ntrials) ;
        GrB_Index vertex_list [BATCH_SIZE] ;
        for (int64_t k = 0 ; k < batch_size ; k++)
        {
            // get the kth source node
            int64_t source = -1 ;
            GRB_TRY (GrB_Matrix_extractElement (&source, SourceNodes,
                k + kstart, 0)) ;
            // subtract one to convert from 1-based to 0-based
            source-- ;
            vertex_list [k] = source ;
            printf (" %"PRId64, source) ;
        }
        printf (" ]\n") ;

        //----------------------------------------------------------------------
        // Compute betweenness centrality using batch algorithm
        //----------------------------------------------------------------------

        // back to default
        LAGRAPH_TRY (LAGraph_SetNumThreads (nthreads_outer, nthreads_inner, msg)) ;

        for (int t = 1 ; t <= nt ; t++)
        {
            if (Nthreads [t] > nthreads_max) continue ;
            LAGRAPH_TRY (LAGraph_SetNumThreads (1, Nthreads [t], msg)) ;
            GrB_free (&centrality) ;
            double t2 = LAGraph_WallClockTime ( ) ;
            LAGRAPH_TRY (LAGr_Betweenness (&centrality, G, vertex_list,
                batch_size, msg)) ;
            t2 = LAGraph_WallClockTime ( ) - t2 ;
            printf ("BC time %2d: %12.4f (sec)\n", Nthreads [t], t2) ;
            fflush (stdout) ;
            tt [t] += t2 ;

        }

        GrB_free (&centrality) ;

        // if burble is on, just do the first batch
        if (burble) break ;
    }

    //--------------------------------------------------------------------------
    // free all workspace and finish
    //--------------------------------------------------------------------------

    printf ("\nntrials: %d\n", ntrials) ;

    printf ("\n") ;
    for (int t = 1 ; t <= nt ; t++)
    {
        if (Nthreads [t] > nthreads_max) continue ;
        double t2 = tt [t] / ntrials ;
        printf ("Ave BC %2d: %10.3f sec, rate %10.3f\n",
            Nthreads [t], t2, 1e-6*((double) nvals) / t2) ;
        fprintf (stderr, "Avg: BC %3d: %10.3f sec: %s\n",
            Nthreads [t], t2, matrix_name) ;
    }

    free ((void *) tt);
    LG_FREE_ALL;
    LAGRAPH_TRY (LAGraph_Finalize (msg)) ;
    return (GrB_SUCCESS) ;
}
