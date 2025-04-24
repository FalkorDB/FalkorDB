//----------------------------------------------------------------------------
// LAGraph/expirimental/test/test_KCore.c: test cases for single k-core
// ----------------------------------------------------------------------------

// LAGraph, (c) 2019-2022 by The LAGraph Contributors, All Rights Reserved.
// SPDX-License-Identifier: BSD-2-Clause
//
// For additional details (including references to third party source code and
// other files) see the LICENSE file or contact permission@sei.cmu.edu. See
// Contributors.txt for a full list of contributors. Created, in part, with
// funding and support from the U.S. Government (see Acknowledgments.txt file).
// DM22-0790

// Contributed by Pranav Konduri, Texas A&M University

//-----------------------------------------------------------------------------

#include <stdio.h>
#include <acutest.h>

#include <LAGraphX.h>
#include <LAGraph_test.h>
#include "LG_Xtest.h"

char msg [LAGRAPH_MSG_LEN] ;
LAGraph_Graph G = NULL ;
GrB_Matrix A = NULL ;
GrB_Vector c1 = NULL, c2 = NULL;
#define LEN 512
char filename [LEN+1] ;

typedef struct
{
    const char *name ;
}
matrix_info ;

const matrix_info files [ ] =
{
    {"karate.mtx" },
    {"west0067.mtx" },
    // {"amazon0601.mtx" },
    // {"cit-Patents.mtx"},
    // {"hollywood-2009.mtx"},
    // {"as-Skitter.mtx"},
    {""},
} ;


void test_KCore (void)
{
    LAGraph_Init (msg) ;

    for (int k = 0 ; ; k++)
    {
        // load the matrix as A
        const char *aname = files [k].name ;
        if (strlen (aname) == 0) break;
        printf ("\n================================== %s: ==================================\n", aname) ;
        TEST_CASE (aname) ;
        snprintf (filename, LEN, LG_DATA_DIR "%s", aname) ;
        FILE *f = fopen (filename, "r") ;
        TEST_CHECK (f != NULL) ;
        OK (LAGraph_MMRead (&A, f, msg)) ;
        TEST_MSG ("Loading of adjacency matrix failed") ;

        // construct an undirected graph G with adjacency matrix A
        OK (LAGraph_New (&G, &A, LAGraph_ADJACENCY_DIRECTED, msg)) ;
        TEST_CHECK (A == NULL) ;

        // check if the pattern is symmetric - if it isn't make it.
        OK (LAGraph_Cached_IsSymmetricStructure (G, msg)) ;

        if (G->is_symmetric_structure == LAGraph_FALSE)
        {
            printf("This matrix is not symmetric. \n");
            // make the adjacency matrix symmetric
            OK (LAGraph_Cached_AT (G, msg)) ;
            OK (GrB_eWiseAdd (G->A, NULL, NULL, GrB_LOR, G->A, G->AT, NULL)) ;
            G->is_symmetric_structure = true ;
            // consider the graph as directed
            G->kind = LAGraph_ADJACENCY_DIRECTED ;
        }
        else
        {
            G->kind = LAGraph_ADJACENCY_UNDIRECTED ;
        }

        // check for self-edges, and remove them.
        OK (LAGraph_Cached_NSelfEdges (G, msg)) ;
        if (G->nself_edges != 0)
        {
            // remove self-edges
            printf ("graph has %g self edges\n", (double) G->nself_edges) ;
            OK (LAGraph_DeleteSelfEdges (G, msg)) ;
            printf ("now has %g self edges\n", (double) G->nself_edges) ;
            TEST_CHECK (G->nself_edges == 0) ;
        }

        uint64_t kmax;
        bool ok;
        //test the k-core
        OK(LAGraph_KCore(&c1, G, 2, msg)) ;
        OK(LG_check_kcore(&c2, &kmax, G, 2, msg)) ;

        OK (LAGraph_Vector_IsEqual (&ok, c1, c2, msg)) ;
        TEST_CHECK (ok) ;

        OK (LAGraph_Delete (&G, msg)) ;
    }

    LAGraph_Finalize (msg) ;
}

//------------------------------------------------------------------------------
// test_errors
//------------------------------------------------------------------------------

void test_errors (void)
{
    LAGraph_Init (msg) ;

    snprintf (filename, LEN, LG_DATA_DIR "%s", "karate.mtx") ;
    FILE *f = fopen (filename, "r") ;
    TEST_CHECK (f != NULL) ;
    OK (LAGraph_MMRead (&A, f, msg)) ;
    TEST_MSG ("Loading of adjacency matrix failed") ;

    // construct an undirected graph G with adjacency matrix A
    OK (LAGraph_New (&G, &A, LAGraph_ADJACENCY_UNDIRECTED, msg)) ;
    TEST_CHECK (A == NULL) ;

    OK (LAGraph_Cached_NSelfEdges (G, msg)) ;

    uint64_t k = 1; //some test k
    GrB_Vector c = NULL ;

    // c is NULL
    int result = LAGraph_KCore (NULL, G, k, msg) ;
    printf ("\nresult: %d %s\n", result, msg) ;
    TEST_CHECK (result == GrB_NULL_POINTER) ;

    // G is invalid
    result = LAGraph_KCore (&c, NULL, k, msg) ;
    printf ("\nresult: %d %s\n", result, msg) ;
    TEST_CHECK (result == GrB_NULL_POINTER) ;
    TEST_CHECK (c == NULL) ;

    // G may have self edges
    G->nself_edges = LAGRAPH_UNKNOWN ;
    result = LAGraph_KCore (&c, G, k, msg) ;
    printf ("\nresult: %d %s\n", result, msg) ;
    TEST_CHECK (result == -1004) ;
    TEST_CHECK (c == NULL) ;

    // G is undirected
    G->nself_edges = 0 ;
    G->kind = LAGraph_ADJACENCY_DIRECTED ;
    G->is_symmetric_structure = LAGraph_FALSE ;
    result = LAGraph_KCore (&c, G, k, msg) ;
    printf ("\nresult: %d %s\n", result, msg) ;
    TEST_CHECK (result == -1005) ;
    TEST_CHECK (c == NULL) ;

    OK (LAGraph_Delete (&G, msg)) ;
    LAGraph_Finalize (msg) ;
}

TEST_LIST = {
    {"KCore", test_KCore},
    {"KCore_errors", test_errors},
    {NULL, NULL}
};
