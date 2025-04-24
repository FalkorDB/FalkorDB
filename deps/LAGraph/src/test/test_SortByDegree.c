//------------------------------------------------------------------------------
// LAGraph/src/test/test_SortByDegree  test LAGr_SortByDegree
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

#include "LAGraph_test.h"

//------------------------------------------------------------------------------
// global variables
//------------------------------------------------------------------------------

LAGraph_Graph G = NULL, H = NULL ;
char msg [LAGRAPH_MSG_LEN] ;
GrB_Matrix A = NULL, B = NULL ;
GrB_Vector d = NULL ;
#define LEN 512
char filename [LEN+1] ;
int64_t *P = NULL ;
bool *W = NULL ;
GrB_Index n, nrows, ncols ;
bool is_symmetric ;
int kind ;

//------------------------------------------------------------------------------
// setup: start a test
//------------------------------------------------------------------------------

void setup (void)
{
    OK (LAGraph_Init (msg)) ;
}

//------------------------------------------------------------------------------
// teardown: finalize a test
//------------------------------------------------------------------------------

void teardown (void)
{
    OK (LAGraph_Finalize (msg)) ;
}

//------------------------------------------------------------------------------
// test_SortByDegree  test LAGr_SortByDegree
//------------------------------------------------------------------------------

const char *files [ ] =
{
    "A.mtx",
    "LFAT5.mtx",
    "cover.mtx",
    "full.mtx",
    "full_symmetric.mtx",
    "karate.mtx",
    "ldbc-cdlp-directed-example.mtx",
    "ldbc-cdlp-undirected-example.mtx",
    "ldbc-directed-example-bool.mtx",
    "ldbc-directed-example-unweighted.mtx",
    "ldbc-directed-example.mtx",
    "ldbc-undirected-example-bool.mtx",
    "ldbc-undirected-example-unweighted.mtx",
    "ldbc-undirected-example.mtx",
    "ldbc-wcc-example.mtx",
    "matrix_int16.mtx",
    "msf1.mtx",
    "msf2.mtx",
    "msf3.mtx",
    "structure.mtx",
    "sample.mtx",
    "sample2.mtx",
    "skew_fp32.mtx",
    "tree-example.mtx",
    "west0067.mtx",
    "",
} ;

void test_SortByDegree (void)
{
    setup ( ) ;

    for (int kk = 0 ; ; kk++)
    {

        // get the name of the test matrix
        const char *aname = files [kk] ;
        if (strlen (aname) == 0) break;
        TEST_CASE (aname) ;
        printf ("\n############################################# %s\n", aname) ;
        snprintf (filename, LEN, LG_DATA_DIR "%s", aname) ;

        for (int outer = 0 ; outer <= 1 ; outer++)
        {

            // load the matrix as A
            FILE *f = fopen (filename, "r") ;
            TEST_CHECK (f != NULL) ;
            OK (LAGraph_MMRead (&A, f, msg)) ;
            OK (fclose (f)) ;
            TEST_MSG ("Loading of adjacency matrix failed") ;

            // ensure the matrix is square
            OK (GrB_Matrix_nrows (&nrows, A)) ;
            OK (GrB_Matrix_ncols (&ncols, A)) ;
            TEST_CHECK (nrows == ncols) ;
            n = nrows ;

            // decide if the graph G is directed or undirected
            if (outer == 0)
            {
                kind = LAGraph_ADJACENCY_DIRECTED ;
                printf ("\n#### case: directed graph\n\n") ;
            }
            else
            {
                kind = LAGraph_ADJACENCY_UNDIRECTED ;
                printf ("\n#### case: undirected graph\n\n") ;
            }

            // construct a graph G with adjacency matrix A
            TEST_CHECK (A != NULL) ;
            OK (LAGraph_New (&G, &A, kind, msg)) ;
            TEST_CHECK (A == NULL) ;
            TEST_CHECK (G != NULL) ;

            // create the cached properties
            int ok_result = (kind == LAGraph_ADJACENCY_UNDIRECTED) ?
                LAGRAPH_CACHE_NOT_NEEDED : GrB_SUCCESS ;
            int result = LAGraph_Cached_AT (G, msg) ;
            TEST_CHECK (result == ok_result) ;
            OK (LAGraph_Cached_OutDegree (G, msg)) ;
            result = LAGraph_Cached_InDegree (G, msg) ;
            TEST_CHECK (result == ok_result) ;
            OK (LAGraph_Cached_IsSymmetricStructure (G, msg)) ;
            OK (LAGraph_Graph_Print (G, LAGraph_SHORT, stdout, msg)) ;

            // sort 4 different ways
            for (int trial = 0 ; trial <= 3 ; trial++)
            {
                bool byout = (trial == 0 || trial == 1) ;
                bool ascending = (trial == 0 || trial == 2) ;

                // sort the graph by degree
                TEST_CHECK (P == NULL) ;
                OK (LAGr_SortByDegree (&P, G, byout, ascending, msg)) ;
                TEST_CHECK (P != NULL) ;

                // ensure P is a permutation of 0..n-1
                OK (LAGraph_Calloc ((void **) &W, n, sizeof (bool), msg)) ;
                for (int k = 0 ; k < n ; k++)
                {
                    int64_t j = P [k] ;
                    TEST_CHECK (j >= 0 && j < n) ;
                    TEST_CHECK (W [j] == false) ;
                    W [j] = true ;
                }

                // check the result by constructing a new graph with adjacency
                // matrix B = A (P,P)
                OK (GrB_Matrix_new (&B, GrB_BOOL, n, n)) ;
                OK (GrB_extract (B, NULL, NULL, G->A,
                    (GrB_Index *) P, n, (GrB_Index *) P, n, NULL)) ;
                OK (LAGraph_New (&H, &B, kind, msg)) ;
                TEST_CHECK (B == NULL) ;
                TEST_CHECK (H != NULL) ;

                // get the cached properties of H
                OK (LAGraph_Cached_OutDegree (H, msg)) ;
                result = LAGraph_Cached_InDegree (H, msg) ;
                TEST_CHECK (result == ok_result) ;
                OK (LAGraph_Cached_IsSymmetricStructure (H, msg)) ;
                TEST_CHECK (G->is_symmetric_structure ==
                            H->is_symmetric_structure) ;
                printf ("\nTrial %d, graph H, sorted (%s) by (%s) degrees:\n",
                    trial, ascending ? "ascending" : "descending",
                    byout ? "row" : "column") ;
                OK (LAGraph_Graph_Print (H, LAGraph_SHORT, stdout, msg)) ;

                d = (byout || G->is_symmetric_structure == LAGraph_TRUE) ?
                    H->out_degree : H->in_degree ;

                // ensure d is sorted in ascending or descending order
                int64_t last_deg = (ascending) ? (-1) : (n+1) ;
                for (int k = 0 ; k < n ; k++)
                {
                    int64_t deg = 0 ;
                    GrB_Info info = GrB_Vector_extractElement (&deg, d, k) ;
                    TEST_CHECK (info == GrB_NO_VALUE || info == GrB_SUCCESS) ;
                    if (info == GrB_NO_VALUE) deg = 0 ;
                    if (ascending)
                    {
                        TEST_CHECK (last_deg <= deg) ;
                    }
                    else
                    {
                        TEST_CHECK (last_deg >= deg) ;
                    }
                    last_deg = deg ;
                }

                // free workspace and the graph H
                OK (LAGraph_Free ((void **) &W, NULL)) ;
                OK (LAGraph_Free ((void **) &P, NULL)) ;
                OK (LAGraph_Delete (&H, msg)) ;
            }

            // check if the adjacency matrix is symmetric
            if (outer == 0)
            {
                // if G->A is symmetric, then continue the outer iteration to
                // create an undirected graph.  Otherwise just do the directed
                // graph
                OK (LAGraph_Matrix_IsEqual (&is_symmetric, G->A, G->AT, msg)) ;
                if (!is_symmetric)
                {
                    printf ("matrix is unsymmetric; skip undirected case\n") ;
                    OK (LAGraph_Delete (&G, msg)) ;
                    break ;
                }
            }
            OK (LAGraph_Delete (&G, msg)) ;
        }
    }

    teardown ( ) ;
}


//------------------------------------------------------------------------------
// test_SortByDegree_brutal
//------------------------------------------------------------------------------

#if LAGRAPH_SUITESPARSE
void test_SortByDegree_brutal (void)
{
    OK (LG_brutal_setup (msg)) ;
    printf ("\n") ;

    for (int kk = 0 ; ; kk++)
    {

        // get the name of the test matrix
        const char *aname = files [kk] ;
        if (strlen (aname) == 0) break;
        TEST_CASE (aname) ;
        printf ("%s\n", aname) ;
        snprintf (filename, LEN, LG_DATA_DIR "%s", aname) ;

        for (int outer = 0 ; outer <= 1 ; outer++)
        {

            // load the matrix as A
            FILE *f = fopen (filename, "r") ;
            TEST_CHECK (f != NULL) ;
            OK (LAGraph_MMRead (&A, f, msg)) ;
            OK (fclose (f)) ;
            TEST_MSG ("Loading of adjacency matrix failed") ;

            // ensure the matrix is square
            OK (GrB_Matrix_nrows (&nrows, A)) ;
            OK (GrB_Matrix_ncols (&ncols, A)) ;
            TEST_CHECK (nrows == ncols) ;
            n = nrows ;

            // decide if the graph G is directed or undirected
            if (outer == 0)
            {
                kind = LAGraph_ADJACENCY_DIRECTED ;
            }
            else
            {
                kind = LAGraph_ADJACENCY_UNDIRECTED ;
            }

            // construct a graph G with adjacency matrix A
            TEST_CHECK (A != NULL) ;
            OK (LAGraph_New (&G, &A, kind, msg)) ;
            TEST_CHECK (A == NULL) ;
            TEST_CHECK (G != NULL) ;

            // create the cached properties
            int ok_result = (kind == LAGraph_ADJACENCY_UNDIRECTED) ?
                LAGRAPH_CACHE_NOT_NEEDED : GrB_SUCCESS ;
            int result = LAGraph_Cached_AT (G, msg) ;
            TEST_CHECK (result == ok_result) ;
            OK (LAGraph_Cached_OutDegree (G, msg)) ;
            result = LAGraph_Cached_InDegree (G, msg) ;
            TEST_CHECK (result == ok_result) ;
            OK (LAGraph_Cached_IsSymmetricStructure (G, msg)) ;
            // OK (LAGraph_Graph_Print (G, LAGraph_SHORT, stdout, msg)) ;

            // sort 4 different ways
            for (int trial = 0 ; trial <= 3 ; trial++)
            {
                bool byout = (trial == 0 || trial == 1) ;
                bool ascending = (trial == 0 || trial == 2) ;

                // sort the graph by degree
                TEST_CHECK (P == NULL) ;
                OK (LAGr_SortByDegree (&P, G, byout, ascending, msg)) ;
                TEST_CHECK (P != NULL) ;

                // ensure P is a permutation of 0..n-1
                OK (LAGraph_Calloc ((void **) &W, n, sizeof (bool), msg)) ;
                for (int k = 0 ; k < n ; k++)
                {
                    int64_t j = P [k] ;
                    TEST_CHECK (j >= 0 && j < n) ;
                    TEST_CHECK (W [j] == false) ;
                    W [j] = true ;
                }

                // check the result by constructing a new graph with adjacency
                // matrix B = A (P,P)
                OK (GrB_Matrix_new (&B, GrB_BOOL, n, n)) ;
                OK (GrB_extract (B, NULL, NULL, G->A,
                    (GrB_Index *) P, n, (GrB_Index *) P, n, NULL)) ;
                OK (LAGraph_New (&H, &B, kind, msg)) ;
                TEST_CHECK (B == NULL) ;
                TEST_CHECK (H != NULL) ;

                // get the cached properties of H
                OK (LAGraph_Cached_OutDegree (H, msg)) ;
                result = LAGraph_Cached_InDegree (H, msg) ;
                TEST_CHECK (result == ok_result) ;
                OK (LAGraph_Cached_IsSymmetricStructure (H, msg)) ;
                TEST_CHECK (G->is_symmetric_structure ==
                            H->is_symmetric_structure) ;

                d = (byout || G->is_symmetric_structure == LAGraph_TRUE) ?
                    H->out_degree : H->in_degree ;

                // ensure d is sorted in ascending or descending order
                int64_t last_deg = (ascending) ? (-1) : (n+1) ;
                for (int k = 0 ; k < n ; k++)
                {
                    int64_t deg = 0 ;
                    GrB_Info info = GrB_Vector_extractElement (&deg, d, k) ;
                    TEST_CHECK (info == GrB_NO_VALUE || info == GrB_SUCCESS) ;
                    if (info == GrB_NO_VALUE) deg = 0 ;
                    if (ascending)
                    {
                        TEST_CHECK (last_deg <= deg) ;
                    }
                    else
                    {
                        TEST_CHECK (last_deg >= deg) ;
                    }
                    last_deg = deg ;
                }

                // free workspace and the graph H
                OK (LAGraph_Free ((void **) &W, NULL)) ;
                OK (LAGraph_Free ((void **) &P, NULL)) ;
                OK (LAGraph_Delete (&H, msg)) ;
            }

            // check if the adjacency matrix is symmetric
            if (outer == 0)
            {
                // if G->A is symmetric, then continue the outer iteration to
                // create an undirected graph.  Otherwise just do the directed
                // graph
                OK (LAGraph_Matrix_IsEqual (&is_symmetric, G->A, G->AT, msg)) ;
                if (!is_symmetric)
                {
                    OK (LAGraph_Delete (&G, msg)) ;
                    break ;
                }
            }

            OK (LAGraph_Delete (&G, msg)) ;
        }
    }

    OK (LG_brutal_teardown (msg)) ;
}
#endif

//-----------------------------------------------------------------------------
// test_SortByDegree_failures:  test error handling of LAGr_SortByDegree
//-----------------------------------------------------------------------------

void test_SortByDegree_failures (void)
{
    setup ( ) ;

    int result = LAGr_SortByDegree (NULL, NULL, true, true, msg) ;
    printf ("\nresult %d: msg: %s\n", result, msg) ;
    TEST_CHECK (result == GrB_NULL_POINTER) ;

    result = LAGr_SortByDegree (&P, NULL, true, true, msg) ;
    TEST_CHECK (result == GrB_NULL_POINTER) ;
    printf ("msg: %s\n", msg) ;

    // create the karate graph
    FILE *f = fopen (LG_DATA_DIR "karate.mtx", "r") ;
    TEST_CHECK (f != NULL) ;
    OK (LAGraph_MMRead (&A, f, msg)) ;
    OK (fclose (f)) ;
    TEST_MSG ("Loading of adjacency matrix failed") ;
    OK (LAGraph_New (&G, &A, LAGraph_ADJACENCY_UNDIRECTED, msg)) ;

    // cached degree property must first be computed
    result = LAGr_SortByDegree (&P, G, true, true, msg) ;
    printf ("\nresult %d: msg: %s\n", result, msg) ;
    TEST_CHECK (result == LAGRAPH_NOT_CACHED) ;

    teardown ( ) ;
}

//-----------------------------------------------------------------------------
// TEST_LIST: the list of tasks for this entire test
//-----------------------------------------------------------------------------

TEST_LIST =
{
    { "SortByDegree", test_SortByDegree },
    { "SortByDegree_failures", test_SortByDegree_failures },
    #if LAGRAPH_SUITESPARSE
    { "SortByDegree_brutal", test_SortByDegree_brutal },
    #endif
    { NULL, NULL }
} ;
