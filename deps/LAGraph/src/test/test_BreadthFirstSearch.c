//----------------------------------------------------------------------------
// LAGraph/src/test/test_BreadthFirstSearch.c: test cases for triangle
// counting algorithms
// ----------------------------------------------------------------------------

// LAGraph, (c) 2019-2022 by The LAGraph Contributors, All Rights Reserved.
// SPDX-License-Identifier: BSD-2-Clause
//
// For additional details (including references to third party source code and
// other files) see the LICENSE file or contact permission@sei.cmu.edu. See
// Contributors.txt for a full list of contributors. Created, in part, with
// funding and support from the U.S. Government (see Acknowledgments.txt file).
// DM22-0790

// Contributed by Scott McMillan, SEI/CMU, and Timothy A. Davis, Texas A&M
// University

//-----------------------------------------------------------------------------

#include <stdio.h>
#include <acutest.h>

#include <LAGraph_test.h>
#include <graph_zachary_karate.h>
#include "LG_alg_internal.h"

char msg[LAGRAPH_MSG_LEN];
LAGraph_Graph G = NULL;

//-----------------------------------------------------------------------------
// Valid results for Karate graph:
//-----------------------------------------------------------------------------

GrB_Index const SRC = 30;
// the levels of the tree for the Karate graph, assuming source node 30:
GrB_Index const LEVELS30[] = {2, 1, 2, 2, 3, 3, 3, 2, 1, 2, 3, 3,
                              3, 2, 2, 2, 4, 2, 2, 2, 2, 2, 2, 2,
                              3, 3, 2, 2, 2, 2, 0, 2, 1, 1};
// Karate BFS parents, with source node 30.  This assumes the parent is the min
// of the valid set of parents:
// GrB_Index const PARENT30[] = { 1, 30,  1,  1,  0,  0,  0,  1, 30, 33,  0,  0,
//                                0,  1, 32, 32,  5,  1, 32,  1, 32,  1, 32, 32,
//                               27, 23, 33, 33, 33, 32, 30, 32, 30, 30};
#define xx (-1)
// The following are valid parents for each node, with source node of 30:
GrB_Index const PARENT30 [34][3] = {
    {  1,  8, xx },     // node 0 can have parents 1 or 8
    { 30, xx, xx },     // node 1, parent 30
    {  1,  8, 32 },     // node 2, parents 1, 8, or 32, etc
    {  1, xx, xx },     // node 3
    {  0, xx, xx },     // node 4
    {  0, xx, xx },     // node 5
    {  0, xx, xx },     // node 6
    {  1, xx, xx },     // node 7
    { 30, xx, xx },     // node 8
    { 33, xx, xx },     // node 9
    {  0, xx, xx },     // node 10
    {  0, xx, xx },     // node 11
    {  0,  3, xx },     // node 12
    {  1, 33, xx },     // node 13
    { 32, 33, xx },     // node 14
    { 32, 33, xx },     // node 15
    {  5,  6, xx },     // node 16
    {  1, xx, xx },     // node 17
    { 32, 33, xx },     // node 18
    {  1, 33, xx },     // node 19
    { 32, 33, xx },     // node 20
    {  1, xx, xx },     // node 21
    { 32, 33, xx },     // node 22
    { 32, 33, xx },     // node 23
    { 27, 31, xx },     // node 24
    { 23, 31, xx },     // node 25
    { 33, xx, xx },     // node 26
    { 33, xx, xx },     // node 27
    { 33, xx, xx },     // node 28
    { 32, 33, xx },     // node 29
    { 30, xx, xx },     // node 30, source node
    { 32, 33, xx },     // node 31
    { 30, xx, xx },     // node 32
    { 30, xx, xx }} ;   // node 33
#undef xx

//-----------------------------------------------------------------------------

#define LEN 512
char filename [LEN+1] ;

typedef struct
{
    LAGraph_Kind kind ;
    const char *name ;
}
matrix_info ;

const matrix_info files [ ] =
{
    { LAGraph_ADJACENCY_UNDIRECTED, "A.mtx" },
    { LAGraph_ADJACENCY_DIRECTED,   "cover.mtx" },
    { LAGraph_ADJACENCY_UNDIRECTED, "jagmesh7.mtx" },
    { LAGraph_ADJACENCY_DIRECTED,   "ldbc-cdlp-directed-example.mtx" },
    { LAGraph_ADJACENCY_UNDIRECTED, "ldbc-cdlp-undirected-example.mtx" },
    { LAGraph_ADJACENCY_DIRECTED,   "ldbc-directed-example.mtx" },
    { LAGraph_ADJACENCY_UNDIRECTED, "ldbc-undirected-example.mtx" },
    { LAGraph_ADJACENCY_UNDIRECTED, "ldbc-wcc-example.mtx" },
    { LAGraph_ADJACENCY_UNDIRECTED, "LFAT5.mtx" },
    { LAGraph_ADJACENCY_DIRECTED,   "msf1.mtx" },
    { LAGraph_ADJACENCY_DIRECTED,   "msf2.mtx" },
    { LAGraph_ADJACENCY_DIRECTED,   "msf3.mtx" },
    { LAGraph_ADJACENCY_DIRECTED,   "sample2.mtx" },
    { LAGraph_ADJACENCY_DIRECTED,   "sample.mtx" },
    { LAGraph_ADJACENCY_DIRECTED,   "olm1000.mtx" },
    { LAGraph_ADJACENCY_UNDIRECTED, "bcsstk13.mtx" },
    { LAGraph_ADJACENCY_DIRECTED,   "cryg2500.mtx" },
    { LAGraph_ADJACENCY_UNDIRECTED, "tree-example.mtx" },
    { LAGraph_ADJACENCY_DIRECTED,   "west0067.mtx" },
    { LAGraph_ADJACENCY_UNDIRECTED, "karate.mtx" },
    { LAGraph_ADJACENCY_DIRECTED,   "matrix_bool.mtx" },
    { LAGraph_ADJACENCY_DIRECTED,   "skew_fp32.mtx" },
    { LAGraph_ADJACENCY_UNDIRECTED, "pushpull.mtx" },
    { LAGRAPH_UNKNOWN, "" },
} ;

//****************************************************************************
bool check_karate_parents30(GrB_Vector parents)
{
    // An update to SS:GrB can result in different, yet valid, parent vectors
    // (even single-threaded).  The LG_check_bfs works fine and those tests
    // pass.  This parent test looks for any valid parent vector.

    GrB_Index n = 0;
    TEST_CHECK(0 == GrB_Vector_size(&n, parents));
    TEST_CHECK(ZACHARY_NUM_NODES == n);
    TEST_CHECK(0 == GrB_Vector_nvals(&n, parents));
    TEST_CHECK(ZACHARY_NUM_NODES == n);

    bool ok = false ;
    int64_t parent_id;
    for (GrB_Index ix = 0; ix < ZACHARY_NUM_NODES; ++ix)
    {
        TEST_CHECK(0 == GrB_Vector_extractElement(&parent_id, parents, ix));
        // prior test:
//      TEST_CHECK(parent_id == PARENT30[ix][0]);
//      TEST_MSG("Parent check failed for node %ld: ans,comp = %ld,%ld\n",
//          ix, PARENT30[ix][0], parent_id);
        // more general test:
        ok = false ;
        for (int k = 0 ; k <= 2 ; k++)
        {
            int valid_parent_id = PARENT30 [ix][k] ;
            if (valid_parent_id < 0)
            {
                // end of the list of valid parent ids
                ok = false ;
                break ;
            }
            if (parent_id == valid_parent_id)
            {
                // a match is found
                ok = true ;
                break ;
            }
        }
        if (!ok) break ;
    }

    return ok;
}

//****************************************************************************
bool check_karate_levels30(GrB_Vector levels)
{
    GrB_Index n = 0;
    TEST_CHECK(0 == GrB_Vector_size(&n, levels) );
    TEST_CHECK(ZACHARY_NUM_NODES == n);
    TEST_CHECK(0 == GrB_Vector_nvals(&n, levels) );
    TEST_CHECK(ZACHARY_NUM_NODES == n);

    int64_t lvl;
    for (GrB_Index ix = 0; ix < ZACHARY_NUM_NODES; ++ix)
    {
        TEST_CHECK(0 == GrB_Vector_extractElement(&lvl, levels, ix) );
        TEST_CHECK(lvl == LEVELS30[ix] );
        TEST_MSG("Level check failed for node %g: ans,comp = %g,%g\n",
                 (double) ix, (double) LEVELS30[ix], (double) lvl);
    }

    return true;
}

//****************************************************************************
void setup(void)
{
    LAGraph_Init(msg);
    int retval;
    GrB_Matrix A = NULL;

    TEST_CHECK(0 == GrB_Matrix_new(&A, GrB_UINT32,
                                   ZACHARY_NUM_NODES, ZACHARY_NUM_NODES) );
    TEST_CHECK(0 == GrB_Matrix_build(A, ZACHARY_I, ZACHARY_J, ZACHARY_V,
                                     ZACHARY_NUM_EDGES, GrB_LOR) );

    retval = LAGraph_New(&G, &A, LAGraph_ADJACENCY_UNDIRECTED, msg);
    TEST_CHECK(retval == 0);
    TEST_MSG("retval = %d (%s)", retval, msg);
}

//****************************************************************************
void teardown(void)
{
    int retval = LAGraph_Delete(&G, msg);
    TEST_CHECK(retval == 0);
    TEST_MSG("retval = %d (%s)", retval, msg);

    G = NULL;
    LAGraph_Finalize(msg);
}

//****************************************************************************
void test_BreadthFirstSearch_invalid_graph(void)
{
    setup();
    int retval;
    LAGraph_Graph graph = NULL;

    retval = LAGr_BreadthFirstSearch(NULL, NULL, graph, 0, msg);
    TEST_CHECK(retval == GrB_NULL_POINTER);
    TEST_MSG("retval = %d (%s)", retval, msg);

    retval = LG_BreadthFirstSearch_vanilla(NULL, NULL, graph, 0, msg);
    TEST_CHECK(retval == GrB_NULL_POINTER);
    TEST_MSG("retval = %d (%s)", retval, msg);

    teardown();
}

//****************************************************************************
void test_BreadthFirstSearch_invalid_src(void)
{
    setup();
    int retval;
    GrB_Index n;
    TEST_CHECK(0 == GrB_Matrix_nrows(&n, (G->A)));

    GrB_Vector parent = NULL ;
    GrB_Vector level  = NULL ;

    retval = LAGr_BreadthFirstSearch(&level, NULL, G, n, msg);
    TEST_CHECK(retval == GrB_INVALID_INDEX);
    TEST_MSG("retval = %d (%s)", retval, msg);

    retval = LG_BreadthFirstSearch_vanilla(&level, NULL, G, n, msg);
    TEST_CHECK(retval == GrB_INVALID_INDEX);
    TEST_MSG("retval = %d (%s)", retval, msg);

    retval = LAGr_BreadthFirstSearch(NULL, &parent, G, n, msg);
    TEST_CHECK(retval == GrB_INVALID_INDEX);
    TEST_MSG("retval = %d (%s)", retval, msg);

    retval = LG_BreadthFirstSearch_vanilla(NULL, &parent, G, n, msg);
    TEST_CHECK(retval == GrB_INVALID_INDEX);
    TEST_MSG("retval = %d (%s)", retval, msg);

    teardown();
}

//****************************************************************************
void test_BreadthFirstSearch_neither(void)
{
    setup();
    int retval;

    printf ("\nTest level and parent both NULL:\n") ;

    LAGraph_PrintLevel pr = LAGraph_COMPLETE_VERBOSE ;
    retval = LAGraph_Graph_Print (G, pr, stdout, msg) ;
    TEST_CHECK(retval == GrB_SUCCESS);

    retval = LAGr_BreadthFirstSearch(NULL, NULL, G, 0, msg);
    TEST_CHECK(retval == GrB_NULL_POINTER);
    TEST_MSG("retval = %d (%s)", retval, msg);

    retval = LG_BreadthFirstSearch_vanilla(NULL, NULL, G, 0, msg);
    TEST_CHECK(retval == GrB_NULL_POINTER);
    TEST_MSG("retval = %d (%s)", retval, msg);

    retval = LAGr_BreadthFirstSearch(NULL, NULL, G, 0, msg);
    TEST_CHECK(retval == GrB_NULL_POINTER);
    TEST_MSG("retval = %d (%s)", retval, msg);

    retval = LG_BreadthFirstSearch_vanilla(NULL, NULL, G, 0, msg);
    TEST_CHECK(retval == GrB_NULL_POINTER);
    TEST_MSG("retval = %d (%s)", retval, msg);

    teardown();
}

//****************************************************************************
void test_BreadthFirstSearch_parent(void)
{
    setup();
    int retval;

    GrB_Vector parent    = NULL;
    GrB_Vector parent_do = NULL;

    OK (LAGraph_Cached_OutDegree (G, msg)) ;

    retval = LAGr_BreadthFirstSearch(NULL, &parent, G, 30, msg);
    TEST_CHECK(retval == 0);
    TEST_MSG("retval = %d (%s)", retval, msg);
    TEST_CHECK(check_karate_parents30(parent));
    retval = LG_check_bfs (NULL, parent, G, 30, msg) ;
    TEST_CHECK (retval == 0) ;

    // mangle the parent vector, just to check check_karate_parents30
    OK (GrB_Vector_setElement (parent, 0, 0)) ;
    TEST_CHECK(!check_karate_parents30(parent));
    TEST_CHECK(0 == GrB_free(&parent));

    retval = LG_BreadthFirstSearch_vanilla(NULL, &parent, G, 30, msg);
    TEST_CHECK(retval == 0);
    TEST_MSG("retval = %d (%s)", retval, msg);
    TEST_CHECK(check_karate_parents30(parent));
    retval = LG_check_bfs (NULL, parent, G, 30, msg) ;
    TEST_CHECK (retval == 0) ;
    TEST_CHECK(0 == GrB_free(&parent));

    retval = LAGr_BreadthFirstSearch(NULL, &parent_do, G, 30, msg);
    TEST_CHECK(retval == 0);
    TEST_MSG("retval = %d (%s)", retval, msg);
    TEST_CHECK(check_karate_parents30(parent_do));
    retval = LG_check_bfs (NULL, parent_do, G, 30, msg) ;
    TEST_CHECK (retval == 0) ;
    TEST_CHECK(0 == GrB_free(&parent_do));

    retval = LG_BreadthFirstSearch_vanilla(NULL, &parent_do, G, 30, msg);
    TEST_CHECK(retval == 0);
    TEST_MSG("retval = %d (%s)", retval, msg);
    TEST_CHECK(check_karate_parents30(parent_do));
    retval = LG_check_bfs (NULL, parent_do, G, 30, msg) ;
    TEST_CHECK (retval == 0) ;
    TEST_CHECK(0 == GrB_free(&parent_do));

    GrB_Index n = 0 ;
    TEST_CHECK (0 == GrB_Matrix_nrows (&n, G->A)) ;
    for (GrB_Index src = 0 ; src < n ; src++)
    {
        retval = LAGr_BreadthFirstSearch(NULL, &parent, G, src, msg);
        TEST_CHECK(retval == 0);
        retval = LG_check_bfs (NULL, parent, G, src, msg) ;
        TEST_CHECK (retval == 0) ;
        TEST_CHECK(0 == GrB_free(&parent));

        retval = LG_BreadthFirstSearch_vanilla(NULL, &parent, G, src, msg);
        TEST_CHECK(retval == 0);
        retval = LG_check_bfs (NULL, parent, G, src, msg) ;
        TEST_CHECK (retval == 0) ;
        TEST_CHECK(0 == GrB_free(&parent));
    }

    teardown();
}

//****************************************************************************
void test_BreadthFirstSearch_level(void)
{
    setup();
    int retval;

    GrB_Vector level    = NULL;
    GrB_Vector level_do = NULL;
    OK (LAGraph_Cached_OutDegree (G, msg)) ;

    retval = LAGr_BreadthFirstSearch(&level, NULL, G, 30, msg);
    TEST_CHECK(retval == 0);
    TEST_MSG("retval = %d (%s)", retval, msg);
    TEST_CHECK(check_karate_levels30(level));
    retval = LG_check_bfs (level, NULL, G, 30, msg) ;
    TEST_CHECK (retval == 0) ;
    TEST_CHECK(0 == GrB_free(&level));

    retval = LG_BreadthFirstSearch_vanilla(&level, NULL, G, 30, msg);
    TEST_CHECK(retval == 0);
    TEST_MSG("retval = %d (%s)", retval, msg);
    TEST_CHECK(check_karate_levels30(level));
    retval = LG_check_bfs (level, NULL, G, 30, msg) ;
    TEST_CHECK (retval == 0) ;
    TEST_CHECK(0 == GrB_free(&level));

    retval = LAGr_BreadthFirstSearch(&level_do, NULL, G, 30, msg);
    TEST_CHECK(retval == 0);
    TEST_MSG("retval = %d (%s)", retval, msg);
    TEST_CHECK(check_karate_levels30(level_do));
    retval = LG_check_bfs (level_do, NULL, G, 30, msg) ;
    TEST_CHECK (retval == 0) ;
    TEST_CHECK(0 == GrB_free(&level_do));

    GrB_Index n = 0 ;
    TEST_CHECK (0 == GrB_Matrix_nrows (&n, G->A)) ;
    for (GrB_Index src = 0 ; src < n ; src++)
    {

        retval = LAGr_BreadthFirstSearch(&level, NULL, G, src, msg);
        TEST_CHECK(retval == 0);
        retval = LG_check_bfs (level, NULL, G, src, msg) ;
        TEST_CHECK (retval == 0) ;
        TEST_CHECK(0 == GrB_free(&level));

        retval = LG_BreadthFirstSearch_vanilla(&level, NULL, G, src, msg);
        TEST_CHECK(retval == 0);
        retval = LG_check_bfs (level, NULL, G, src, msg) ;
        TEST_CHECK (retval == 0) ;
        TEST_CHECK(0 == GrB_free(&level));

    }

    teardown();
}

//****************************************************************************
void test_BreadthFirstSearch_both(void)
{
    setup();
    int retval;

    OK (LAGraph_Cached_OutDegree (G, msg)) ;
    GrB_Vector parent    = NULL;
    GrB_Vector level    = NULL;
    retval = LAGr_BreadthFirstSearch(&level, &parent, G, 30, msg);
    TEST_CHECK(retval == 0);
    TEST_MSG("retval = %d (%s)", retval, msg);
    TEST_CHECK(check_karate_levels30(level));
    TEST_CHECK(check_karate_parents30(parent));

    retval = LG_check_bfs (level, parent, G, 30, msg) ;
    TEST_CHECK (retval == 0) ;

    TEST_CHECK(0 == GrB_free(&parent));
    TEST_CHECK(0 == GrB_free(&level));

    GrB_Vector parent_do = NULL;
    GrB_Vector level_do = NULL;
    retval = LAGr_BreadthFirstSearch(&level_do, &parent_do, G, 30, msg);
    TEST_CHECK(retval == 0);
    TEST_MSG("retval = %d (%s)", retval, msg);
    TEST_CHECK(check_karate_levels30(level_do));
    TEST_CHECK(check_karate_parents30(parent_do));
    retval = LG_check_bfs (level_do, parent_do, G, 30, msg) ;
    TEST_CHECK (retval == 0) ;
    TEST_CHECK(0 == GrB_free(&parent_do));
    TEST_CHECK(0 == GrB_free(&level_do));

    GrB_Index n = 0 ;
    TEST_CHECK (0 == GrB_Matrix_nrows (&n, G->A)) ;

    for (GrB_Index src = 0 ; src < n ; src++)
    {
        retval = LAGr_BreadthFirstSearch(&level, &parent, G, src, msg);
        TEST_CHECK(retval == 0);
        retval = LG_check_bfs (level, parent, G, src, msg) ;
        TEST_CHECK (retval == 0) ;
        TEST_CHECK(0 == GrB_free(&parent));
        TEST_CHECK(0 == GrB_free(&level));
    }

    teardown();
}

//****************************************************************************
void test_BreadthFirstSearch_many(void)
{
    LAGraph_Init(msg);
    GrB_Matrix A = NULL ;

    for (int k = 0 ; ; k++)
    {

        // load the adjacency matrix as A
        const char *aname = files [k].name ;
        LAGraph_Kind kind = files [k].kind ;
        if (strlen (aname) == 0) break;
        TEST_CASE (aname) ;
        printf ("\nMatrix: %s\n", aname) ;
        snprintf (filename, LEN, LG_DATA_DIR "%s", aname) ;
        FILE *f = fopen (filename, "r") ;
        TEST_CHECK (f != NULL) ;
        OK (LAGraph_MMRead (&A, f, msg)) ;
        OK (fclose (f)) ;
        TEST_MSG ("Loading of adjacency matrix failed") ;

        // create the graph
        OK (LAGraph_New (&G, &A, kind, msg)) ;
        TEST_CHECK (A == NULL) ;    // A has been moved into G->A

        GrB_Index n = 0 ;
        OK (GrB_Matrix_nrows (&n, G->A)) ;

        for (int caching = 0 ; caching <= 1 ; caching++)
        {
            // run the BFS
            int64_t step = (n > 100) ? (3*n/4) : ((n/4) + 1) ;
            for (int64_t src = 0 ; src < n ; src += step)
            {
                GrB_Vector parent = NULL ;
                GrB_Vector level = NULL ;

                int64_t maxlevel ;
                GrB_Index nvisited ;

                OK (LAGr_BreadthFirstSearch (&level, &parent, G, src, msg)) ;
                OK (LG_check_bfs (level, parent, G, src, msg)) ;
                OK (GrB_reduce (&maxlevel, NULL, GrB_MAX_MONOID_INT64,
                    level, NULL)) ;
                OK (GrB_Vector_nvals (&nvisited, level)) ;
                {
                    printf ("src %g n: %g max level: %g nvisited: %g\n",
                        (double) src, (double) n, (double) maxlevel,
                        (double) nvisited) ;
                }
                OK (GrB_free(&parent));
                OK (GrB_free(&level));

                OK (LG_BreadthFirstSearch_vanilla (&level, &parent,
                    G, src, msg)) ;
                OK (LG_check_bfs (level, parent, G, src, msg)) ;
                OK (GrB_reduce (&maxlevel, NULL, GrB_MAX_MONOID_INT64,
                    level, NULL)) ;
                OK (GrB_Vector_nvals (&nvisited, level)) ;
                {
                    printf ("src %g n: %g max level: %g nvisited: %g\n",
                        (double) src, (double) n, (double) maxlevel,
                        (double) nvisited) ;
                }
                OK (GrB_free(&parent));
                OK (GrB_free(&level));

                OK (LAGr_BreadthFirstSearch (NULL, &parent, G, src, msg)) ;
                OK (LG_check_bfs (NULL, parent, G, src, msg)) ;
                OK (GrB_free(&parent));

                OK (LG_BreadthFirstSearch_vanilla (NULL, &parent,
                    G, src, msg)) ;
                OK (LG_check_bfs (NULL, parent, G, src, msg)) ;
                OK (GrB_free(&parent));

                OK (LAGr_BreadthFirstSearch (&level, NULL, G, src, msg)) ;
                OK (LG_check_bfs (level, NULL, G, src, msg)) ;
                OK (GrB_free(&level));

                OK (LG_BreadthFirstSearch_vanilla (&level, NULL, G, src, msg)) ;
                OK (LG_check_bfs (level, NULL, G, src, msg)) ;
                OK (GrB_free(&level));

            }

            // create its cached properties
            int ok_result = (kind == LAGraph_ADJACENCY_UNDIRECTED) ?
                LAGRAPH_CACHE_NOT_NEEDED : GrB_SUCCESS ;
            int result = LAGraph_Cached_AT (G, msg) ;
            TEST_CHECK (result == ok_result) ;
            OK (LAGraph_CheckGraph (G, msg)) ;
            OK (LAGraph_Cached_OutDegree (G, msg)) ;
            OK (LAGraph_CheckGraph (G, msg)) ;
            result = LAGraph_Cached_InDegree (G, msg) ;
            TEST_CHECK (result == ok_result) ;
            OK (LAGraph_CheckGraph (G, msg)) ;
        }

        OK (LAGraph_Delete (&G, msg)) ;
    }

    LAGraph_Finalize(msg);
}

//------------------------------------------------------------------------------
// test_bfs_brutal
//------------------------------------------------------------------------------

#if LAGRAPH_SUITESPARSE
void test_bfs_brutal (void)
{
    OK (LG_brutal_setup (msg)) ;
    GrB_Matrix A = NULL ;

    for (int k = 0 ; ; k++)
    {
        // load the adjacency matrix as A
        const char *aname = files [k].name ;
        LAGraph_Kind kind = files [k].kind ;
        if (strlen (aname) == 0) break;
        TEST_CASE (aname) ;
        printf ("\nMatrix: %s\n", aname) ;
        snprintf (filename, LEN, LG_DATA_DIR "%s", aname) ;
        FILE *f = fopen (filename, "r") ;
        TEST_CHECK (f != NULL) ;
        OK (LAGraph_MMRead (&A, f, msg)) ;
        OK (fclose (f)) ;
        TEST_MSG ("Loading of adjacency matrix failed") ;
        // create the graph
        OK (LAGraph_New (&G, &A, kind, msg)) ;
        TEST_CHECK (A == NULL) ;    // A has been moved into G->A
        GrB_Index n = 0 ;
        OK (GrB_Matrix_nrows (&n, G->A)) ;
        if (n >= 1000)
        {
            // only do the small graphs
            printf ("skipped\n") ;
            OK (LAGraph_Delete (&G, msg)) ;
            continue ;
        }

        for (int caching = 0 ; caching <= 1 ; caching++)
        {
            // run the BFS
            int64_t step = (n > 100) ? (3*n/4) : ((n/4) + 1) ;
            for (int64_t src = 0 ; src < n ; src += step)
            {
                GrB_Vector parent = NULL ;
                GrB_Vector level = NULL ;

                // parent and level with SS:GrB
                LG_BRUTAL_BURBLE (LAGr_BreadthFirstSearch (&level, &parent, G, src, msg)) ;
                OK (LG_check_bfs (level, parent, G, src, msg)) ;
                OK (GrB_free (&parent)) ;
                OK (GrB_free (&level)) ;

                // level only with SS:GrB
                LG_BRUTAL (LAGr_BreadthFirstSearch (&level, NULL, G, src, msg)) ;
                OK (LG_check_bfs (level, NULL, G, src, msg)) ;
                OK (GrB_free (&level)) ;

                // parent and level with vanilla
                LG_BRUTAL (LG_BreadthFirstSearch_vanilla (&level,
                    &parent, G, src, msg)) ;
                OK (LG_check_bfs (level, parent, G, src, msg)) ;
                OK (GrB_free (&parent)) ;
                OK (GrB_free (&level)) ;

                // level-only with vanilla
                LG_BRUTAL (LG_BreadthFirstSearch_vanilla (&level, NULL,
                        G, src, msg)) ;
                OK (LG_check_bfs (level, NULL, G, src, msg)) ;
                OK (GrB_free (&level)) ;
            }

            // create its cached properties
            int ok_result = (kind == LAGraph_ADJACENCY_UNDIRECTED) ?
                LAGRAPH_CACHE_NOT_NEEDED : GrB_SUCCESS ;
            int result = LAGraph_Cached_AT (G, msg) ;
            TEST_CHECK (result == ok_result) ;
            OK (LAGraph_Cached_OutDegree (G, msg)) ;
            result = LAGraph_Cached_InDegree (G, msg) ;
            TEST_CHECK (result == ok_result) ;
        }

        OK (LAGraph_Delete (&G, msg)) ;
    }

    OK (LG_brutal_teardown (msg)) ;
}
#endif

//****************************************************************************
//****************************************************************************
TEST_LIST = {
    {"BreadthFirstSearch_invalid_graph", test_BreadthFirstSearch_invalid_graph},
    {"BreadthFirstSearch_invalid_src", test_BreadthFirstSearch_invalid_src},
    {"BreadthFirstSearch_neither", test_BreadthFirstSearch_neither},
    {"BreadthFirstSearch_parent", test_BreadthFirstSearch_parent},
    {"BreadthFirstSearch_level", test_BreadthFirstSearch_level},
    {"BreadthFirstSearch_both", test_BreadthFirstSearch_both},
    {"BreadthFirstSearch_many", test_BreadthFirstSearch_many},
    #if LAGRAPH_SUITESPARSE
    {"BreadthFirstSearch_brutal", test_bfs_brutal },
    #endif
    {NULL, NULL}
} ;
