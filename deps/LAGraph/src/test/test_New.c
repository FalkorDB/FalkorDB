//------------------------------------------------------------------------------
// LAGraph/src/test/test_New.c:  test LAGraph_New and LAGraph_Delete
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
GrB_Matrix A = NULL ;
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
// test_New:  test LAGraph_New
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

void test_New (void)
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

        // free the graph
        OK (LAGraph_Delete (&G, msg)) ;
        TEST_CHECK (G == NULL) ;
    }
    teardown ( ) ;
}

//------------------------------------------------------------------------------
// test_New_brutal
//------------------------------------------------------------------------------

#if LAGRAPH_SUITESPARSE
void test_New_brutal (void)
{
    OK (LG_brutal_setup (msg)) ;
    printf ("\n") ;

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
        LG_BRUTAL_BURBLE (LAGraph_New (&G, &A, kind, msg)) ;
        TEST_CHECK (A == NULL) ;    // A has been moved into G->A

        // check the graph
        LG_BRUTAL_BURBLE (LAGraph_CheckGraph (G, msg)) ;

        // free the graph
        LG_BRUTAL_BURBLE (LAGraph_Delete (&G, msg)) ;
        TEST_CHECK (G == NULL) ;
    }

    OK (LG_brutal_teardown (msg)) ;
}
#endif

//------------------------------------------------------------------------------
// test_New_failures:  test error handling of LAGraph_New
//------------------------------------------------------------------------------

void test_New_failures (void)
{
    setup ( ) ;

    // G cannot be NULL
    TEST_CHECK (LAGraph_New (NULL, NULL, 0, msg) == GrB_NULL_POINTER) ;
    printf ("\nmsg: %s\n", msg) ;

    // create a graph with no adjacency matrix; this is OK, since the intent is
    // to create a graph for which the adjacency matrix can be defined later,
    // via assigning it to G->A.  However, the graph will be declared invalid
    // by LAGraph_CheckGraph since G->A is NULL.
    OK (LAGraph_New (&G, NULL, 0, msg)) ;
    TEST_CHECK (LAGraph_CheckGraph (G, msg) == LAGRAPH_INVALID_GRAPH) ;
    printf ("msg: %s\n", msg) ;
    OK (LAGraph_Delete (&G, msg)) ;
    TEST_CHECK (G == NULL) ;
    OK (LAGraph_Delete (&G, msg)) ;
    TEST_CHECK (G == NULL) ;
    OK (LAGraph_Delete (NULL, msg)) ;
    teardown ( ) ;
}

//-----------------------------------------------------------------------------
// TEST_LIST: the list of tasks for this entire test
//-----------------------------------------------------------------------------

TEST_LIST =
{
    { "New", test_New },
    { "New_failures", test_New_failures },
    #if LAGRAPH_SUITESPARSE
    { "New_brutal", test_New_brutal },
    #endif
    { NULL, NULL }
} ;
