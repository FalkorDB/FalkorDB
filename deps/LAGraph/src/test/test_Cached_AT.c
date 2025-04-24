//------------------------------------------------------------------------------
// LAGraph/src/test/test_Cached_AT.c:  test LAGraph_Cached_AT
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
GrB_Matrix A = NULL, B = NULL ;
GrB_Type atype = NULL ;
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
// test_Cached_AT:  test LAGraph_Cached_AT
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

//-----------------------------------------------------------------------------
// test_Cached_AT
//-----------------------------------------------------------------------------

void test_Cached_AT (void)
{
    setup ( ) ;

    for (int k = 0 ; ; k++)
    {

        // load the matrix as A
        const char *aname = files [k].name ;
        int kind = files [k].kind ;
        if (strlen (aname) == 0) break;
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

        // create the G->AT cached property
        int ok_result = (kind == LAGraph_ADJACENCY_UNDIRECTED) ?
            LAGRAPH_CACHE_NOT_NEEDED : GrB_SUCCESS ;
        int result = LAGraph_Cached_AT (G, msg) ;
        TEST_CHECK (result == ok_result) ;

        // try to create it again; this should safely do nothing
        result = LAGraph_Cached_AT (G, msg) ;
        TEST_CHECK (result == ok_result) ;

        // check the result
        if (kind == LAGraph_ADJACENCY_UNDIRECTED)
        {
            TEST_CHECK (G->AT == NULL) ;
        }
        else
        {
            // ensure G->A and G->AT are transposed of each other;
            // B = (G->AT)'
            GrB_Index nrows, ncols ;
            OK (GrB_Matrix_nrows (&nrows, G->A)) ;
            OK (GrB_Matrix_nrows (&ncols, G->A)) ;
            OK (LAGraph_Matrix_TypeName (atype_name, G->A, msg)) ;
            OK (LAGraph_TypeFromName (&atype, atype_name, msg)) ;
            OK (GrB_Matrix_new (&B, atype, nrows, ncols)) ;
            OK (GrB_transpose (B, NULL, NULL, G->AT, NULL)) ;

            // ensure B and G->A are the same
            bool ok ;
            OK (LAGraph_Matrix_IsEqual (&ok, G->A, B, msg)) ;
            TEST_CHECK (ok) ;
            TEST_MSG ("Test for G->A and B equal failed") ;
            OK (GrB_free (&B)) ;
        }

        OK (LAGraph_Delete (&G, msg)) ;
    }

    teardown ( ) ;
}

//-----------------------------------------------------------------------------
// test_Cached_AT_brutal
//-----------------------------------------------------------------------------

#if LAGRAPH_SUITESPARSE
void test_Cached_AT_brutal (void)
{
    OK (LG_brutal_setup (msg)) ;

    for (int k = 0 ; ; k++)
    {

        // load the matrix as A
        const char *aname = files [k].name ;
        int kind = files [k].kind ;
        if (strlen (aname) == 0) break;
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

        // create the G->AT cached property
        LG_BRUTAL (LAGraph_Cached_AT (G, msg)) ;

        // try to create it again; this should safely do nothing
        LG_BRUTAL (LAGraph_Cached_AT (G, msg)) ;

        // check the result
        if (kind == LAGraph_ADJACENCY_UNDIRECTED)
        {
            TEST_CHECK (G->AT == NULL) ;
        }
        else
        {
            // ensure G->A and G->AT are transposed of each other;
            // B = (G->AT)'
            GrB_Index nrows, ncols ;
            OK (GrB_Matrix_nrows (&nrows, G->A)) ;
            OK (GrB_Matrix_nrows (&ncols, G->A)) ;
            OK (LAGraph_Matrix_TypeName (atype_name, G->A, msg)) ;
            OK (LAGraph_TypeFromName (&atype, atype_name, msg)) ;
            OK (GrB_Matrix_new (&B, atype, nrows, ncols)) ;
            OK (GrB_transpose (B, NULL, NULL, G->AT, NULL)) ;

            // ensure B and G->A are the same
            bool ok ;
            LG_BRUTAL (LAGraph_Matrix_IsEqual (&ok, G->A, B, msg)) ;
            TEST_CHECK (ok) ;
            TEST_MSG ("Test for G->A and B equal failed") ;
            OK (GrB_free (&B)) ;
        }

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
    { "test_AT", test_Cached_AT },
    #if LAGRAPH_SUITESPARSE
    { "test_AT_brutal", test_Cached_AT_brutal },
    #endif
    { NULL, NULL }
} ;
