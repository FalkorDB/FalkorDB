//------------------------------------------------------------------------------
// LAGraph/src/test/test_DeleteCached.c:  test LAGraph_DeleteCached
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
// test_DeleteCached:  test LAGraph_DeleteCached
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
    LAGraph_ADJACENCY_UNDIRECTED, "A.mtx",
    LAGraph_ADJACENCY_UNDIRECTED, "bcsstk13.mtx",
    LAGRAPH_UNKNOWN,              ""
} ;

void test_DeleteCached (void)
{
    setup ( ) ;

    for (int k = 0 ; ; k++)
    {

        // load the matrix as A
        const char *aname = files [k].name ;
        if (strlen (aname) == 0) break;
        LAGraph_Kind kind = files [k].kind ;
        TEST_CASE (aname) ;
        snprintf (filename, LEN, LG_DATA_DIR "%s", aname) ;
        FILE *f = fopen (filename, "r") ;
        TEST_CHECK (f != NULL) ;
        OK (LAGraph_MMRead (&A, f, msg)) ;
        OK (fclose (f)) ;
        TEST_MSG ("Loading of adjacency matrix failed") ;

        // construct the graph G with adjacency matrix A
        OK (LAGraph_New (&G, &A, kind, msg)) ;
        TEST_CHECK (A == NULL) ;

        // create all cached properties (see test_Cached_* for tests of content)
        int ok_result = (kind == LAGraph_ADJACENCY_UNDIRECTED) ?
            LAGRAPH_CACHE_NOT_NEEDED : GrB_SUCCESS ;
        OK (LAGraph_Cached_OutDegree (G, msg)) ;
        int result = LAGraph_Cached_InDegree (G, msg) ;
        TEST_CHECK (result == ok_result) ;
        result = LAGraph_Cached_AT (G, msg) ;
        TEST_CHECK (result == ok_result) ;
        OK (LAGraph_Cached_IsSymmetricStructure (G, msg)) ;

        // print them
        printf ("\nGraph: nself_edges %g, symmetric structure: %d\n",
            (double) G->nself_edges, G->is_symmetric_structure) ;
        printf ("  adj matrix: ") ;
        int rr = (LAGraph_Matrix_Print (G->A, LAGraph_SHORT, stdout, msg)) ;
        printf ("result: %d msg: %s\n", rr, msg) ;
        printf ("  out degree: ") ;
        OK (LAGraph_Vector_Print (G->out_degree, LAGraph_SHORT, stdout, msg)) ;
        if (kind == LAGraph_ADJACENCY_DIRECTED)
        {
            printf ("  adj transposed: ") ;
            OK (LAGraph_Matrix_Print (G->AT, LAGraph_SHORT, stdout, msg)) ;
            printf ("  in degree: ") ;
            OK (LAGraph_Vector_Print (G->in_degree, LAGraph_SHORT, stdout,
                msg)) ;
        }
        else
        {
            TEST_CHECK (G->AT == NULL) ;
            TEST_CHECK (G->in_degree == NULL) ;
        }

        for (int trial = 0 ; trial <= 1 ; trial++)
        {
            // delete all the cached properties
            OK (LAGraph_DeleteCached (G, msg)) ;
            TEST_CHECK (G->AT == NULL) ;
            TEST_CHECK (G->out_degree == NULL) ;
            TEST_CHECK (G->in_degree == NULL) ;
        }

        OK (LAGraph_Delete (&G, msg)) ;
    }

    OK (LAGraph_DeleteCached (NULL, msg)) ;

    teardown ( ) ;
}

//-----------------------------------------------------------------------------

#if LAGRAPH_SUITESPARSE
void test_del_brutal (void)
{
    OK (LG_brutal_setup (msg)) ;

    for (int k = 0 ; ; k++)
    {

        // load the matrix as A
        const char *aname = files [k].name ;
        if (strlen (aname) == 0) break;
        LAGraph_Kind kind = files [k].kind ;
        TEST_CASE (aname) ;
        snprintf (filename, LEN, LG_DATA_DIR "%s", aname) ;
        FILE *f = fopen (filename, "r") ;
        TEST_CHECK (f != NULL) ;
        OK (LAGraph_MMRead (&A, f, msg)) ;
        OK (fclose (f)) ;
        TEST_MSG ("Loading of adjacency matrix failed") ;

        // construct the graph G with adjacency matrix A
        LG_BRUTAL (LAGraph_New (&G, &A, kind, msg)) ;
        TEST_CHECK (A == NULL) ;

        // create all cached properties (see test_Cached_* for tests of content)
        LG_BRUTAL (LAGraph_Cached_OutDegree (G, msg)) ;
        LG_BRUTAL (LAGraph_Cached_InDegree (G, msg)) ;
        LG_BRUTAL (LAGraph_Cached_AT (G, msg)) ;
        LG_BRUTAL (LAGraph_Cached_IsSymmetricStructure (G, msg)) ;

        for (int trial = 0 ; trial <= 1 ; trial++)
        {
            // delete all the cached properties
            LG_BRUTAL (LAGraph_DeleteCached (G, msg)) ;
            TEST_CHECK (G->AT == NULL) ;
            TEST_CHECK (G->out_degree == NULL) ;
            TEST_CHECK (G->in_degree == NULL) ;
        }

        LG_BRUTAL (LAGraph_Delete (&G, msg)) ;
        LG_BRUTAL (LAGraph_DeleteCached (NULL, msg)) ;
    }

    OK (LG_brutal_teardown (msg)) ;
}
#endif

//-----------------------------------------------------------------------------
// TEST_LIST: the list of tasks for this entire test
//-----------------------------------------------------------------------------

TEST_LIST =
{
    { "test_DeleteCached", test_DeleteCached },
    #if LAGRAPH_SUITESPARSE
    { "test_DeleteCached_brutal", test_del_brutal },
    #endif
    { NULL, NULL }
} ;
