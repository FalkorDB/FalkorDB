//------------------------------------------------------------------------------
// LAGraph/src/test/test_Graph_Print.c:  test LAGraph_Graph_Print
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
#include "LG_internal.h"

//------------------------------------------------------------------------------
// global variables
//------------------------------------------------------------------------------

LAGraph_Graph G = NULL ;
char msg [LAGRAPH_MSG_LEN] ;
GrB_Matrix A = NULL, AT = NULL ;
#define LEN 512
char filename [LEN+1] ;
char atype_name [LAGRAPH_MAX_NAME_LEN] ;

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
// prwhat: print what should be printed
//------------------------------------------------------------------------------

const char *prwhat (int pr)
{
    switch (pr)
    {
        case  0: return ("nothing") ;
        case  1: return ("terse") ;
        case  2: return ("summary") ;
        case  3: return ("all") ;
        case  4: return ("summary (doubles in full precision)") ;
        case  5: return ("all (doubles in full precision)") ;
        default: ;
    }
    return (NULL) ;
}

//------------------------------------------------------------------------------
// test_Graph_Print:  test LAGraph_Graph_Print
//------------------------------------------------------------------------------

typedef struct
{
    LAGraph_Kind kind ;
    int nself_edges ;
    const char *name ;
}
matrix_info ;

const matrix_info files [ ] =
{
    LAGraph_ADJACENCY_DIRECTED,   0, "cover.mtx",
    LAGraph_ADJACENCY_DIRECTED,   0, "ldbc-directed-example.mtx",
    LAGraph_ADJACENCY_UNDIRECTED, 0, "ldbc-undirected-example.mtx",
    LAGraph_ADJACENCY_DIRECTED,   2, "west0067.mtx",
    LAGRAPH_UNKNOWN,              0, ""
} ;

void test_Graph_Print (void)
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

        OK (LAGraph_Matrix_TypeName (atype_name, A, msg)) ;
        if (MATCHNAME (atype_name, "double"))
        {
            OK (GrB_Matrix_setElement (A, 3.14159265358979323, 0, 1)) ;
        }

        // create the graph
        OK (LAGraph_New (&G, &A, kind, msg)) ;
        TEST_CHECK (A == NULL) ;    // A has been moved into G->A

        // display the graph
        for (int trial = 0 ; trial <= 1 ; trial++)
        {
            printf ("\n############################# TRIAL: %d\n", trial) ;
            for (int pr = 0 ; pr <= 5 ; pr++)
            {
                printf ("\n########### %s: pr: %d (%s)\n",
                    aname, pr, prwhat (pr)) ;
                LAGraph_PrintLevel prl = pr ;
                OK (LAGraph_Graph_Print (G, prl, stdout, msg)) ;
            }
            int ok_result = (kind == LAGraph_ADJACENCY_UNDIRECTED) ?
                LAGRAPH_CACHE_NOT_NEEDED : GrB_SUCCESS ;
            int result = LAGraph_Cached_AT (G, msg) ;
            TEST_CHECK (result == ok_result) ;
            OK (LAGraph_Cached_IsSymmetricStructure (G, msg)) ;
            OK (LAGraph_Cached_NSelfEdges (G, msg)) ;
            TEST_CHECK (G->nself_edges == files [k].nself_edges) ;
        }

        // free the graph
        OK (LAGraph_Delete (&G, msg)) ;
        TEST_CHECK (G == NULL) ;
    }

    TEST_CHECK (prwhat (999) == NULL) ;
    teardown ( ) ;
}

//------------------------------------------------------------------------------
// test_Graph_Print_failures:  test error handling of LAGraph_Graph_Print
//------------------------------------------------------------------------------

void test_Graph_Print_failures (void)
{
    setup ( ) ;

    // G cannot be NULL
    int result = LAGraph_New (NULL, NULL, 0, msg) ;
    printf ("\nresult: %d, msg: %s\n", result, msg) ;
    TEST_CHECK (result == GrB_NULL_POINTER) ;

    // create a graph with no adjacency matrix; this is OK, since the intent is
    // to create a graph for which the adjacency matrix can be defined later,
    // via assigning it to G->A.  However, the graph will be declared invalid
    // by LAGraph_CheckGraph since G->A is NULL.
    OK (LAGraph_New (&G, NULL, LAGraph_ADJACENCY_UNDIRECTED, msg)) ;

    // G->A is NULL
    LAGraph_PrintLevel pr = LAGraph_COMPLETE_VERBOSE ;
    result = LAGraph_Graph_Print (G, pr, stdout, msg) ;
    printf ("result: %d, msg: %s\n", result, msg) ;
    TEST_CHECK (result == LAGRAPH_INVALID_GRAPH) ;

    OK (LAGraph_Delete (&G, msg)) ;
    TEST_CHECK (G == NULL) ;

    // valid graph
    OK (GrB_Matrix_new (&A, GrB_FP32, 5, 5)) ;
    OK (LAGraph_New (&G, &A, LAGraph_ADJACENCY_UNDIRECTED, msg)) ;
    result = LAGraph_Graph_Print (G, pr, stdout, msg) ;
    printf ("result: %d, msg: %s\n", result, msg) ;
    TEST_CHECK (result == GrB_SUCCESS) ;

    // mangled G->kind
    G->kind = -1 ;
    result = LAGraph_Graph_Print (G, pr, stdout, msg) ;
    printf ("result: %d, msg: %s\n", result, msg) ;
    TEST_CHECK (result == LAGRAPH_INVALID_GRAPH) ;
    G->kind = LAGraph_ADJACENCY_UNDIRECTED ;

    // G->AT has the wrong size
    OK (GrB_Matrix_new (&(G->AT), GrB_FP32, 6, 5)) ;
    result = LAGraph_Graph_Print (G, pr, stdout, msg) ;
    printf ("result: %d, msg: %s\n", result, msg) ;
    TEST_CHECK (result == LAGRAPH_INVALID_GRAPH) ;

    OK (GrB_free (&G->AT)) ;
    OK (GrB_Matrix_new (&(G->AT), GrB_FP32, 5, 5)) ;

    #if LAGRAPH_SUITESPARSE
    // G->AT must be held by row, not by column
    OK (GxB_set (G->AT, GxB_FORMAT, GxB_BY_COL)) ;
    result = LAGraph_Graph_Print (G, pr, stdout, msg) ;
    printf ("result: %d, msg: %s\n", result, msg) ;
    TEST_CHECK (result == LAGRAPH_INVALID_GRAPH) ;
    #endif

    // G->A and G->AT must have the same types
    OK (GrB_free (&G->AT)) ;
    OK (GrB_Matrix_new (&(G->AT), GrB_FP64, 5, 5)) ;
    result = LAGraph_Graph_Print (G, pr, stdout, msg) ;
    printf ("result: %d, msg: %s\n", result, msg) ;
    TEST_CHECK (result == LAGRAPH_INVALID_GRAPH) ;

    OK (LAGraph_Delete (&G, msg)) ;
    TEST_CHECK (G == NULL) ;

    OK (LAGraph_Delete (NULL, msg)) ;
    teardown ( ) ;
}

//-----------------------------------------------------------------------------
// test_Graph_Print_brutal
//-----------------------------------------------------------------------------

#if LAGRAPH_SUITESPARSE
void test_Graph_Print_brutal (void)
{
    OK (LG_brutal_setup (msg)) ;

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

        OK (LAGraph_Matrix_TypeName (atype_name, A, msg)) ;
        if (MATCHNAME (atype_name, "double"))
        {
            OK (GrB_Matrix_setElement (A, 3.14159265358979323, 0, 1)) ;
        }
        OK (GrB_wait (A, GrB_MATERIALIZE)) ;

        // create the graph
        OK (LAGraph_New (&G, &A, kind, msg)) ;
        OK (LAGraph_CheckGraph (G, msg)) ;

        // display the graph
        for (int trial = 0 ; trial <= 1 ; trial++)
        {
            printf ("\n############################# TRIAL: %d\n", trial) ;
            for (int pr = 0 ; pr <= 5 ; pr++)
            {
                printf ("\n########### %s: pr: %d (%s)\n",
                    aname, pr, prwhat (pr)) ;
                LAGraph_PrintLevel prl = pr ;
                if (pr == 3 || pr == 5)
                {
                    printf ("skipped for brutal tests\n") ;
                }
                else
                {
                    LG_BRUTAL (LAGraph_Graph_Print (G, prl, stdout, msg)) ;
                }
            }
            int ok_result = (kind == LAGraph_ADJACENCY_UNDIRECTED) ?
                LAGRAPH_CACHE_NOT_NEEDED : GrB_SUCCESS ;
            int result = LAGraph_Cached_AT (G, msg) ;
            TEST_CHECK (result == ok_result) ;
            OK (LAGraph_Cached_IsSymmetricStructure (G, msg)) ;
            OK (LAGraph_Cached_NSelfEdges (G, msg)) ;
        }

        // free the graph
        OK (LAGraph_Delete (&G, msg)) ;
    }

    OK (LG_brutal_teardown (msg)) ;
}
#endif

//-----------------------------------------------------------------------------
// TEST_LIST: the list of tasks for this entire test
//-----------------------------------------------------------------------------

TEST_LIST =
{
    { "Graph_Print", test_Graph_Print },
    #if LAGRAPH_SUITESPARSE
    { "Graph_Print_brutal", test_Graph_Print_brutal },
    #endif
    { "Graph_Print_failures", test_Graph_Print_failures },
    { NULL, NULL }
} ;
