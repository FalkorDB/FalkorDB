//------------------------------------------------------------------------------
// LAGraph/src/test/test_TriangleCentrality.c: test cases for triangle
// centrality
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

#include <stdio.h>
#include <acutest.h>

#include <LAGraphX.h>
#include <LAGraph_test.h>

char msg [LAGRAPH_MSG_LEN] ;
LAGraph_Graph G = NULL ;
GrB_Matrix A = NULL ;
GrB_Matrix C = NULL ;
#define LEN 512
char filename [LEN+1] ;

typedef struct
{
    uint64_t ntriangles ;
    const char *name ;
}
matrix_info ;

const matrix_info files [ ] =
{
    {     11, "A.mtx" },
    {   2016, "jagmesh7.mtx" },
    { 342300, "bcsstk13.mtx" },
    {     45, "karate.mtx" },
    {      6, "ldbc-cdlp-undirected-example.mtx" },
    {      4, "ldbc-undirected-example-bool.mtx" },
    {      4, "ldbc-undirected-example-unweighted.mtx" },
    {      4, "ldbc-undirected-example.mtx" },
    {      5, "ldbc-wcc-example.mtx" },
    { 0, "" },
} ;

//****************************************************************************
void test_TriangleCentrality (void)
{
    LAGraph_Init (msg) ;

    for (int k = 0 ; ; k++)
    {

        // load the matrix as A
        const char *aname = files [k].name ;
        uint64_t ntriangles = files [k].ntriangles ;
        if (strlen (aname) == 0) break;
        printf ("\n================================== %s:\n", aname) ;
        TEST_CASE (aname) ;
        snprintf (filename, LEN, LG_DATA_DIR "%s", aname) ;
        FILE *f = fopen (filename, "r") ;
        TEST_CHECK (f != NULL) ;
        OK (LAGraph_MMRead (&A, f, msg)) ;

        // C = spones (A), in FP64, required for methods 1 and 1.5
        GrB_Index n ;
        OK (GrB_Matrix_nrows (&n, A)) ;
        OK (GrB_Matrix_new (&C, GrB_FP64, n, n)) ;
        OK (GrB_assign (C, A, NULL, (double) 1, GrB_ALL, n, GrB_ALL, n,
            GrB_DESC_S)) ;
        OK (GrB_free (&A)) ;
        TEST_CHECK (A == NULL) ;
        OK (fclose (f)) ;
        TEST_MSG ("Loading of adjacency matrix failed") ;

        // construct an undirected graph G with adjacency matrix C
        OK (LAGraph_New (&G, &C, LAGraph_ADJACENCY_UNDIRECTED, msg)) ;
        TEST_CHECK (C == NULL) ;

        // check for self-edges
        OK (LAGraph_Cached_NSelfEdges (G, msg)) ;
        if (G->nself_edges != 0)
        {
            // remove self-edges
            printf ("graph has %g self edges\n", (double) G->nself_edges) ;
            OK (LAGraph_DeleteSelfEdges (G, msg)) ;
            printf ("now has %g self edges\n", (double) G->nself_edges) ;
            TEST_CHECK (G->nself_edges == 0) ;
        }

        uint64_t ntri ;
        GrB_Vector c = NULL ;
        for (int method = 0 ; method <= 3 ; method++)
        {
            printf ("\nMethod: %d\n", method) ;

            // compute the triangle centrality
            OK (LAGraph_VertexCentrality_Triangle (&c, &ntri, method, G, msg)) ;
            printf ("# of triangles: %g\n", (double) ntri) ;
            TEST_CHECK (ntri == ntriangles) ;

            LAGraph_PrintLevel
                pr = (n <= 100) ? LAGraph_COMPLETE : LAGraph_SHORT ;
            printf ("\ncentrality:\n") ;
            OK (LAGraph_Vector_Print (c, pr, stdout, msg)) ;
            OK (GrB_free (&c)) ;
        }

        // convert to directed with symmetric structure and recompute
        G->kind = LAGraph_ADJACENCY_DIRECTED ;
        OK (LAGraph_VertexCentrality_Triangle (&c, &ntri, 0, G, msg)) ;
        TEST_CHECK (ntri == ntriangles) ;
        GrB_free (&c) ;

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

    uint64_t ntri ;
    GrB_Vector c = NULL ;

    // c is NULL
    int result = LAGraph_VertexCentrality_Triangle (NULL, &ntri, 3, G, msg) ;
    printf ("\nresult: %d %s\n", result, msg) ;
    TEST_CHECK (result == GrB_NULL_POINTER) ;

    // G is invalid
    result = LAGraph_VertexCentrality_Triangle (&c, &ntri, 3, NULL, msg) ;
    printf ("\nresult: %d %s\n", result, msg) ;
    TEST_CHECK (result == GrB_NULL_POINTER) ;
    TEST_CHECK (c == NULL) ;

    // G may have self edges
    G->nself_edges = LAGRAPH_UNKNOWN ;
    result = LAGraph_VertexCentrality_Triangle (&c, &ntri, 3, G, msg) ;
    printf ("\nresult: %d %s\n", result, msg) ;
    TEST_CHECK (result == -1004) ;
    TEST_CHECK (c == NULL) ;

    // G is undirected
    G->nself_edges = 0 ;
    G->kind = LAGraph_ADJACENCY_DIRECTED ;
    G->is_symmetric_structure = LAGraph_FALSE ;
    result = LAGraph_VertexCentrality_Triangle (&c, &ntri, 3, G, msg) ;
    printf ("\nresult: %d %s\n", result, msg) ;
    TEST_CHECK (result == -1005) ;
    TEST_CHECK (c == NULL) ;

    OK (LAGraph_Delete (&G, msg)) ;
    LAGraph_Finalize (msg) ;
}


//****************************************************************************

TEST_LIST = {
    {"TriangleCentrality", test_TriangleCentrality},
    {"TriangleCentrality_errors", test_errors},
    {NULL, NULL}
};
