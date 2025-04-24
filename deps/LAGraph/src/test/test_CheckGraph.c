//------------------------------------------------------------------------------
// LAGraph/src/test/test_CheckGraph.c:  test LAGraph_CheckGraph
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

LAGraph_Graph G = NULL ;
char msg [LAGRAPH_MSG_LEN] ;
GrB_Matrix A = NULL, B_bool = NULL, B_int32 = NULL ;
GrB_Vector d_int64 = NULL, d_bool = NULL ;
#define LEN 512
char filename [LEN+1] ;

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
// test_CheckGraph:  test LAGraph_CheckGraph
//------------------------------------------------------------------------------

typedef struct
{
    LAGraph_Kind kind ;
    const char *name ;
}
matrix_info ;

const matrix_info files [ ] =
{
    LAGraph_ADJACENCY_DIRECTED,   "cover.mtx",
    LAGraph_ADJACENCY_DIRECTED,   "ldbc-directed-example.mtx",
    LAGraph_ADJACENCY_UNDIRECTED, "ldbc-undirected-example.mtx",
    LAGRAPH_UNKNOWN,              ""
} ;

void test_CheckGraph (void)
{
    setup ( ) ;

    for (int k = 0 ; ; k++)
    {

        // load the adjacency matrix as A
        const char *aname = files [k].name ;
        LAGraph_Kind kind = files [k].kind ;
        if (strlen (aname) == 0) break;
        TEST_CASE (aname) ;
        snprintf (filename, LEN, LG_DATA_DIR "%s", aname) ;
        FILE *f = fopen (filename, "r") ;
        TEST_CHECK (f != NULL) ;
        OK (LAGraph_MMRead (&A, f, msg)) ;
        OK (fclose (f)) ;
        TEST_MSG ("Loading of adjacency matrix failed") ;

        // create the graph
        OK (LAGraph_New (&G, &A, kind, msg)) ;
        TEST_CHECK (A == NULL) ;    // A has been moved into G->A

        // check the graph
        OK (LAGraph_CheckGraph (G, msg)) ;
        TEST_CHECK (G->kind == kind) ;
        if (kind == LAGraph_ADJACENCY_DIRECTED)
        {
            TEST_CHECK (G->is_symmetric_structure == LAGRAPH_UNKNOWN) ;
        }
        else
        {
            TEST_CHECK (G->is_symmetric_structure == LAGraph_TRUE) ;
        }

        // create its cached properties
        int ok_result = (kind == LAGraph_ADJACENCY_UNDIRECTED) ?
            LAGRAPH_CACHE_NOT_NEEDED : GrB_SUCCESS ;
        int result = LAGraph_Cached_AT (G, msg) ;
        OK (LAGraph_CheckGraph (G, msg)) ;
        TEST_CHECK (result == ok_result) ;

        OK (LAGraph_Cached_OutDegree (G, msg)) ;
        OK (LAGraph_CheckGraph (G, msg)) ;

        result = LAGraph_Cached_InDegree (G, msg) ;
        TEST_CHECK (result == ok_result) ;
        OK (LAGraph_CheckGraph (G, msg)) ;

        // free the graph
        OK (LAGraph_Delete (&G, msg)) ;
        TEST_CHECK (G == NULL) ;

    }

    teardown ( ) ;
}

//------------------------------------------------------------------------------
// test_CheckGraph_failures:  test error handling of LAGraph_CheckGraph
//------------------------------------------------------------------------------

void test_CheckGraph_failures (void)
{
    setup ( ) ;

    printf ("\nTesting LAGraph_CheckGraph error handling:\n") ;

    // construct an invalid graph with a rectangular adjacency matrix
    TEST_CASE ("lp_afiro") ;
    FILE *f = fopen (LG_DATA_DIR "lp_afiro.mtx", "r") ;
    TEST_CHECK (f != NULL) ;
    OK (LAGraph_MMRead (&A, f, msg)) ;
    OK (fclose (f)) ;
    TEST_MSG ("Loading of lp_afiro.mtx failed") ;

    // create an invalid graph
    OK (LAGraph_New (&G, &A, LAGraph_ADJACENCY_DIRECTED, msg)) ;
    TEST_CHECK (A == NULL) ;    // A has been moved into G->A

    // adjacency matrix invalid
    TEST_CHECK (LAGraph_CheckGraph (G, msg) == LAGRAPH_INVALID_GRAPH) ;
    printf ("msg: %s\n", msg) ;

    // free the graph
    OK (LAGraph_Delete (&G, msg)) ;
    TEST_CHECK (G == NULL) ;

    // load a valid adjacency matrix
    TEST_CASE ("cover") ;
    f = fopen (LG_DATA_DIR "cover.mtx", "r") ;
    TEST_CHECK (f != NULL) ;
    OK (LAGraph_MMRead (&A, f, msg)) ;
    OK (fclose (f)) ;
    TEST_MSG ("Loading of cover.mtx failed") ;

    // create an valid graph
    OK (LAGraph_New (&G, &A, LAGraph_ADJACENCY_DIRECTED, msg)) ;
    TEST_CHECK (A == NULL) ;    // A has been moved into G->A
    OK (LAGraph_CheckGraph (G, msg)) ;

    OK (GrB_Vector_new (&d_bool,  GrB_BOOL, 7)) ;
    OK (GrB_Vector_new (&d_int64, GrB_INT64, 1000)) ;
    OK (GrB_Matrix_new (&B_bool,  GrB_INT64, 7, 7)) ;
    OK (GrB_Matrix_new (&B_int32, GrB_INT32, 3, 4)) ;

    // G->AT has the right type, but wrong size
    G->AT = B_int32 ;
    TEST_CHECK (LAGraph_CheckGraph (G, msg) == LAGRAPH_INVALID_GRAPH) ;
    printf ("msg: %s\n", msg) ;

    // G->AT has the right size, but wrong type
    G->AT = B_bool ;
    TEST_CHECK (LAGraph_CheckGraph (G, msg) == LAGRAPH_INVALID_GRAPH) ;
    printf ("msg: %s\n", msg) ;

    #if LAGRAPH_SUITESPARSE
    // G->AT must be by-row
    OK (GxB_set (G->AT, GxB_FORMAT, GxB_BY_COL)) ;
    TEST_CHECK (LAGraph_CheckGraph (G, msg) == LAGRAPH_INVALID_GRAPH) ;
    printf ("msg: %s\n", msg) ;
    #endif

    G->AT = NULL ;

    // G->out_degree has the right type, but wrong size
    G->out_degree = d_int64 ;
    TEST_CHECK (LAGraph_CheckGraph (G, msg) == LAGRAPH_INVALID_GRAPH) ;
    printf ("msg: %s\n", msg) ;

    // G->out_degree has the right size, but wrong type
    G->out_degree = d_bool ;
    TEST_CHECK (LAGraph_CheckGraph (G, msg) == LAGRAPH_INVALID_GRAPH) ;
    printf ("msg: %s\n", msg) ;

    G->out_degree = NULL ;

    // G->in_degree has the right type, but wrong size
    G->in_degree = d_int64 ;
    TEST_CHECK (LAGraph_CheckGraph (G, msg) == LAGRAPH_INVALID_GRAPH) ;
    printf ("msg: %s\n", msg) ;

    // G->in_degree has the right size, but wrong type
    G->in_degree = d_bool ;
    TEST_CHECK (LAGraph_CheckGraph (G, msg) == LAGRAPH_INVALID_GRAPH) ;
    printf ("msg: %s\n", msg) ;

    G->in_degree = NULL ;

    #if LAGRAPH_SUITESPARSE
    // G->A must be by-row
    OK (GxB_set (G->A, GxB_FORMAT, GxB_BY_COL)) ;
    TEST_CHECK (LAGraph_CheckGraph (G, msg) == LAGRAPH_INVALID_GRAPH) ;
    printf ("msg: %s\n", msg) ;
    #endif

    GrB_free (&d_bool) ;
    GrB_free (&d_int64) ;
    GrB_free (&B_bool) ;
    GrB_free (&B_int32) ;

    // mangle G->kind
    G->kind = LAGRAPH_UNKNOWN ;
    TEST_CHECK (LAGraph_CheckGraph (G, msg) == LAGRAPH_INVALID_GRAPH) ;
    printf ("msg: %s\n", msg) ;
    G->kind = LAGraph_ADJACENCY_DIRECTED ;

    // free the adjacency matrix
    GrB_free (&(G->A)) ;
    TEST_CHECK (G->A == NULL) ;

    int result = LAGraph_CheckGraph (G, msg) ;
    printf ("result : %d msg: %s\n", result, msg) ;
    TEST_CHECK (result == LAGRAPH_INVALID_GRAPH) ;

    // free the graph
    OK (LAGraph_Delete (&G, msg)) ;
    TEST_CHECK (G == NULL) ;

    TEST_CHECK (LAGraph_CheckGraph (NULL, msg) == GrB_NULL_POINTER) ;
    printf ("msg: %s\n", msg) ;

    teardown ( ) ;
}

//------------------------------------------------------------------------------
// test_CheckGraph_brutal:
//------------------------------------------------------------------------------

#if LAGRAPH_SUITESPARSE
void test_CheckGraph_brutal (void)
{
    OK (LG_brutal_setup (msg)) ;

    // load a valid adjacency matrix
    TEST_CASE ("karate") ;
    FILE *f = fopen (LG_DATA_DIR "karate.mtx", "r") ;
    TEST_CHECK (f != NULL) ;
    OK (LAGraph_MMRead (&A, f, msg)) ;
    OK (fclose (f)) ;
    TEST_MSG ("Loading of karate.mtx failed") ;
    printf ("\n") ;

    // create an valid graph
    OK (LAGraph_New (&G, &A, LAGraph_ADJACENCY_UNDIRECTED, msg)) ;
    TEST_CHECK (A == NULL) ;    // A has been moved into G->A
    LG_BRUTAL_BURBLE (LAGraph_CheckGraph (G, msg)) ;

    // create its cached properties
    LG_BRUTAL_BURBLE (LAGraph_Cached_AT (G, msg)) ;
    LG_BRUTAL_BURBLE (LAGraph_CheckGraph (G, msg)) ;
    LG_BRUTAL_BURBLE (LAGraph_Cached_OutDegree (G, msg)) ;
    LG_BRUTAL_BURBLE (LAGraph_CheckGraph (G, msg)) ;
    LG_BRUTAL_BURBLE (LAGraph_Cached_InDegree (G, msg)) ;
    LG_BRUTAL_BURBLE (LAGraph_CheckGraph (G, msg)) ;
    LG_BRUTAL_BURBLE (LAGraph_Delete (&G, msg)) ;

    OK (LG_brutal_teardown (msg)) ;
}
#endif

//-----------------------------------------------------------------------------
// TEST_LIST: the list of tasks for this entire test
//-----------------------------------------------------------------------------

TEST_LIST =
{
    { "CheckGraph", test_CheckGraph },
    { "CheckGraph_failures", test_CheckGraph_failures },
    #if LAGRAPH_SUITESPARSE
    { "CheckGraph_brutal", test_CheckGraph_brutal },
    #endif
    { NULL, NULL }
} ;
