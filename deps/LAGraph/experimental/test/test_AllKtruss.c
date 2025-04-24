//----------------------------------------------------------------------------
// LAGraph/experimental/test/test_AllKtest.c: test cases for all-k-truss
// ----------------------------------------------------------------------------

// LAGraph, (c) 2019-2022 by The LAGraph Contributors, All Rights Reserved.
// SPDX-License-Identifier: BSD-2-Clause
//
// For additional details (including references to third party source code and
// other files) see the LICENSE file or contact permission@sei.cmu.edu. See
// Contributors.txt for a full list of contributors. Created, in part, with
// funding and support from the U.S. Government (see Acknowledgments.txt file).
// DM22-0790

// Contributed by Timothy A. Davis, Texas A&M University

//-----------------------------------------------------------------------------

#include <stdio.h>
#include <acutest.h>

#include "LAGraphX.h"
#include "LAGraph_test.h"
#include "LG_Xtest.h"

char msg [LAGRAPH_MSG_LEN] ;
LAGraph_Graph G = NULL ;
GrB_Matrix A = NULL ;
GrB_Matrix C1 = NULL ;
#define LEN 512
char filename [LEN+1] ;

typedef struct
{
    uint32_t ntriangles ;
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
void test_AllKTruss (void)
{
    LAGraph_Init (msg) ;

    for (int id = 0 ; ; id++)
    {

        // load the matrix as A
        const char *aname = files [id].name ;
        uint32_t ntriangles = files [id].ntriangles ;
        if (strlen (aname) == 0) break;
        printf ("\n================================== %s:\n", aname) ;
        TEST_CASE (aname) ;
        snprintf (filename, LEN, LG_DATA_DIR "%s", aname) ;
        FILE *f = fopen (filename, "r") ;
        TEST_CHECK (f != NULL) ;
        OK (LAGraph_MMRead (&A, f, msg)) ;
        TEST_MSG ("Loading of adjacency matrix failed") ;
        fclose (f) ;

        // construct an undirected graph G with adjacency matrix A
        OK (LAGraph_New (&G, &A, LAGraph_ADJACENCY_UNDIRECTED, msg)) ;
        TEST_CHECK (A == NULL) ;

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

        // compute each k-truss
        bool ok = false ;
        GrB_Index n ;
        int64_t kmax ;
        OK (GrB_Matrix_nrows (&n, G->A)) ;
        GrB_Matrix *Cset ;
        int64_t *ntris, *nedges, *nsteps ;
        OK (LAGraph_Calloc ((void **) &Cset  , n, sizeof (GrB_Matrix), msg)) ;
        OK (LAGraph_Malloc ((void **) &ntris , n, sizeof (int64_t), msg)) ;
        OK (LAGraph_Malloc ((void **) &nedges, n, sizeof (int64_t), msg)) ;
        OK (LAGraph_Malloc ((void **) &nsteps, n, sizeof (int64_t), msg)) ;

        OK (LAGraph_AllKTruss (Cset, &kmax, ntris, nedges, nsteps, G, msg)) ;
        printf ("all k-truss: kmax %g\n", (double) kmax) ;

        // compute each k-truss using LAGraph_KTruss, and compare
        for (int k = 3 ; k < n ; k++)
        {
            // printf ("\n%d-truss:\n", k) ;
            TEST_CHECK (k <= kmax) ;
            // compute the k-truss
            OK (LAGraph_KTruss (&C1, G, k, msg)) ;

            // check the result
            GrB_Index nvals ;
            OK (GrB_Matrix_nvals (&nvals, C1)) ;
            OK (LAGraph_Matrix_IsEqual (&ok, C1, Cset [k], msg)) ;
            TEST_CHECK (ok) ;

            // count the triangles in the 3-truss
            uint32_t nt = 0 ;
            OK (GrB_reduce (&nt, NULL, GrB_PLUS_MONOID_UINT32, C1, NULL)) ;
            nt = nt / 6 ;
            if (k == 3)
            {
                TEST_CHECK (nt == ntriangles) ;
            }
            TEST_CHECK (nt == ntris [k]) ;
            TEST_CHECK (nvals == 2 * nedges [k]) ;
            TEST_CHECK (nsteps [k] >= 0) ;

            // free C1, and break if C1 is empty
            OK (GrB_free (&C1)) ;
            if (nvals == 0)
            {
                TEST_CHECK (k == kmax) ;
                break ;
            }
        }

        // convert to directed with symmetric structure and recompute
        G->kind = LAGraph_ADJACENCY_DIRECTED ;
        G->is_symmetric_structure = LAGraph_TRUE ;
        int64_t k2 ;
        GrB_Matrix *Cset2 ;
        int64_t *ntris2, *nedges2, *nsteps2 ;
        OK (LAGraph_Calloc ((void **) &Cset2  , n, sizeof (GrB_Matrix), msg)) ;
        OK (LAGraph_Malloc ((void **) &ntris2 , n, sizeof (int64_t), msg)) ;
        OK (LAGraph_Malloc ((void **) &nedges2, n, sizeof (int64_t), msg)) ;
        OK (LAGraph_Malloc ((void **) &nsteps2, n, sizeof (int64_t), msg)) ;

        OK (LAGraph_AllKTruss (Cset2, &k2, ntris2, nedges2, nsteps2, G, msg)) ;
        TEST_CHECK (k2 == kmax) ;
        for (int k = 0 ; k <= kmax ; k++)
        {
            TEST_CHECK (ntris2  [k] == ntris  [k]) ;
            TEST_CHECK (nedges2 [k] == nedges [k]) ;
            TEST_CHECK (nsteps2 [k] == nsteps [k]) ;
            if (k < 3)
            {
                TEST_CHECK (Cset [k] == NULL) ;
                TEST_CHECK (Cset2 [k] == NULL) ;
            }
            else
            {
                OK (LAGraph_Matrix_IsEqual (&ok, Cset [k], Cset2 [k], msg)) ;
            }
//          if (!ok)
//          {
//              GxB_print (Cset [k], 2) ;
//              GxB_print (Cset2 [k], 2) ;
//          }
            TEST_CHECK (ok) ;
            OK (GrB_free (&(Cset [k]))) ;
            OK (GrB_free (&(Cset2 [k]))) ;
        }

        LAGraph_Free ((void **) &Cset, NULL) ;
        LAGraph_Free ((void **) &ntris, NULL) ;
        LAGraph_Free ((void **) &nedges, NULL) ;
        LAGraph_Free ((void **) &nsteps, NULL) ;

        LAGraph_Free ((void **) &Cset2, NULL) ;
        LAGraph_Free ((void **) &ntris2, NULL) ;
        LAGraph_Free ((void **) &nedges2, NULL) ;
        LAGraph_Free ((void **) &nsteps2, NULL) ;

        OK (LAGraph_Delete (&G, msg)) ;
    }

    LAGraph_Finalize (msg) ;
}

//------------------------------------------------------------------------------
// test_AllKTruss_errors
//------------------------------------------------------------------------------

void test_allktruss_errors (void)
{
    LAGraph_Init (msg) ;

    snprintf (filename, LEN, LG_DATA_DIR "%s", "karate.mtx") ;
    FILE *f = fopen (filename, "r") ;
    TEST_CHECK (f != NULL) ;
    OK (LAGraph_MMRead (&A, f, msg)) ;
    TEST_MSG ("Loading of adjacency matrix failed") ;
    fclose (f) ;

    // construct an undirected graph G with adjacency matrix A
    OK (LAGraph_New (&G, &A, LAGraph_ADJACENCY_UNDIRECTED, msg)) ;
    TEST_CHECK (A == NULL) ;

    OK (LAGraph_Cached_NSelfEdges (G, msg)) ;

    GrB_Index n ;
    int64_t kmax ;
    OK (GrB_Matrix_nrows (&n, G->A)) ;
    int64_t *ntris, *nedges, *nsteps ;
    GrB_Matrix *Cset ;
    OK (LAGraph_Calloc ((void **) &Cset  , n, sizeof (GrB_Matrix), msg)) ;
    OK (LAGraph_Malloc ((void **) &ntris , n, sizeof (int64_t), msg)) ;
    OK (LAGraph_Malloc ((void **) &nedges, n, sizeof (int64_t), msg)) ;
    OK (LAGraph_Malloc ((void **) &nsteps, n, sizeof (int64_t), msg)) ;

    // kmax is NULL
    int result = LAGraph_AllKTruss (Cset, NULL, ntris, nedges, nsteps, G, msg) ;
    printf ("\nresult: %d %s\n", result, msg) ;
    TEST_CHECK (result == GrB_NULL_POINTER) ;

    // G is invalid
    result = LAGraph_AllKTruss (Cset, &kmax, ntris, nedges, nsteps, NULL, msg) ;
    printf ("\nresult: %d %s\n", result, msg) ;
    TEST_CHECK (result == GrB_NULL_POINTER) ;

    // G may have self edges
    G->nself_edges = LAGRAPH_UNKNOWN ;
    result = LAGraph_AllKTruss (Cset, &kmax, ntris, nedges, nsteps, G, msg) ;
    printf ("\nresult: %d %s\n", result, msg) ;
    TEST_CHECK (result == -1004) ;

    // G is undirected
    G->nself_edges = 0 ;
    G->kind = LAGraph_ADJACENCY_DIRECTED ;
    G->is_symmetric_structure = LAGraph_FALSE ;
    result = LAGraph_AllKTruss (Cset, &kmax, ntris, nedges, nsteps, G, msg) ;
    printf ("\nresult: %d %s\n", result, msg) ;
    TEST_CHECK (result == -1005) ;

    LAGraph_Free ((void **) &Cset, NULL) ;
    LAGraph_Free ((void **) &ntris, NULL) ;
    LAGraph_Free ((void **) &nedges, NULL) ;
    LAGraph_Free ((void **) &nsteps, NULL) ;

    OK (LAGraph_Delete (&G, msg)) ;
    LAGraph_Finalize (msg) ;
}

//****************************************************************************

TEST_LIST = {
    {"allktruss", test_AllKTruss},
    {"allktruss_errors", test_allktruss_errors},
    {NULL, NULL}
};
