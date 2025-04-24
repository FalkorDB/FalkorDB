//----------------------------------------------------------------------------
// LAGraph/experimental/test/test_lcc.c: tests for Local Clustering Coefficient
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

// todo: write a simple lcc method, as LG_check_lcc, and compare its results
// with LAGraph_lcc

#include <stdio.h>
#include <acutest.h>

#include <LAGraphX.h>
#include <LAGraph_test.h>

char msg [LAGRAPH_MSG_LEN] ;
LAGraph_Graph G = NULL ;
GrB_Matrix A = NULL ;
#define LEN 512
char filename [LEN+1] ;

typedef struct
{
    bool symmetric ;
    const char *name ;
}
matrix_info ;

// this known solution is listed here in lower precision than double allows
double lcc_west0067 [67] = {
    0.0909091,
    0.0238095,
    0.0714286,
    0.0238095,
    0.125,
    0.119048,
    0.178571,
    0.133333,
    0.0952381,
    0.0833333,
    0.0666667,
    0.1,
    0.0892857,
    0.0714286,
    0.138889,
    0.0555556,
    0.0694444,
    0.0555556,
    0.0714286,
    0.0769231,
    0.0333333,
    0.0666667,
    0.0666667,
    0.0333333,
    0.0694444,
    0.0444444,
    0.0555556,
    0.0777778,
    0.0333333,
            0,
    0.0904762,
    0.0535714,
    0.0714286,
    0.107143,
    0.0892857,
            0,
    0.0681818,
    0.0357143,
    0.0178571,
    0.0666667,
    0.0555556,
    0.0694444,
    0.0666667,
    0.111111,
    0.0444444,
    0.142857,
    0.166667,
    0.2,
    0.0448718,
    0.166667,
    0.166667,
    0.119048,
    0.166667,
    0.190476,
    0.104167,
    0.0333333,
    0.0444444,
    0.0444444,
    0.0777778,
    0.0416667,
    0.0222222,
    0.0972222,
    0.142857,
    0.0416667,
    0.0555556,
    0.196429,
    0.155556
} ;

const matrix_info files [ ] =
{
    { 1, "A.mtx" },
    { 1, "jagmesh7.mtx" },
    { 0, "west0067.mtx" }, // unsymmetric
    { 1, "bcsstk13.mtx" },
    { 1, "karate.mtx" },
    { 1, "ldbc-cdlp-undirected-example.mtx" },
    { 1, "ldbc-undirected-example-bool.mtx" },
    { 1, "ldbc-undirected-example-unweighted.mtx" },
    { 1, "ldbc-undirected-example.mtx" },
    { 1, "ldbc-wcc-example.mtx" },
    { 0, "" },
} ;

//****************************************************************************
void test_lcc (void)
{
    LAGraph_Init (msg) ;
    #if LAGRAPH_SUITESPARSE

    for (int k = 0 ; ; k++)
    {

        // load the matrix as A
        const char *aname = files [k].name ;
        bool symmetric = files [k].symmetric ;
        if (strlen (aname) == 0) break;
        printf ("\n================================== %s:\n", aname) ;
        TEST_CASE (aname) ;
        snprintf (filename, LEN, LG_DATA_DIR "%s", aname) ;
        FILE *f = fopen (filename, "r") ;
        TEST_CHECK (f != NULL) ;
        OK (LAGraph_MMRead (&A, f, msg)) ;

        // construct a directed graph G with adjacency matrix A
        OK (LAGraph_New (&G, &A, LAGraph_ADJACENCY_DIRECTED, msg)) ;
        TEST_CHECK (A == NULL) ;

        // check for self-edges
        OK (LAGraph_Cached_NSelfEdges (G, msg)) ;
        bool sanitize = (G->nself_edges != 0) ;

        GrB_Vector c = NULL ;
        double t [2] ;

        // compute the local clustering coefficient
        OK (LAGraph_lcc (&c, G->A, symmetric, sanitize, t, msg)) ;

        GrB_Index n ;
        OK (GrB_Vector_size (&n, c)) ;
        LAGraph_PrintLevel pr = (n <= 100) ? LAGraph_COMPLETE : LAGraph_SHORT ;

        // check result c for west0067
        if (strcmp (aname, "west0067.mtx") == 0)
        {
            GrB_Vector cgood = NULL ;
            OK (GrB_Vector_new (&cgood, GrB_FP64, n)) ;
            for (int k = 0 ; k < n ; k++)
            {
                OK (GrB_Vector_setElement (cgood, lcc_west0067 [k], k)) ;
            }
            OK (GrB_wait (cgood, GrB_MATERIALIZE)) ;
            printf ("\nlcc (known result, but with float precision):\n") ;
            OK (LAGraph_Vector_Print (cgood, pr, stdout, msg)) ;
            // cgood = abs (cgood - c)
            OK (GrB_eWiseAdd (cgood, NULL, NULL, GrB_MINUS_FP64, cgood, c,
                NULL)) ;
            OK (GrB_apply (cgood, NULL, NULL, GrB_ABS_FP64, cgood, NULL)) ;
            double err = 0 ;
            // err = max (cgood)
            OK (GrB_reduce (&err, NULL, GrB_MAX_MONOID_FP64, cgood, NULL)) ;
            printf ("err: %g\n", err) ;
            TEST_CHECK (err < 1e-6) ;
            OK (GrB_free (&cgood)) ;
        }

        printf ("\nlcc:\n") ;
        OK (LAGraph_Vector_Print (c, pr, stdout, msg)) ;
        OK (GrB_free (&c)) ;

        OK (LAGraph_Delete (&G, msg)) ;
    }

    #else
    printf ("test skipped\n") ;
    #endif
    LAGraph_Finalize (msg) ;
}

//------------------------------------------------------------------------------
// test_errors
//------------------------------------------------------------------------------

void test_errors (void)
{
    LAGraph_Init (msg) ;
    #if LAGRAPH_SUITESPARSE

    snprintf (filename, LEN, LG_DATA_DIR "%s", "karate.mtx") ;
    FILE *f = fopen (filename, "r") ;
    TEST_CHECK (f != NULL) ;
    OK (LAGraph_MMRead (&A, f, msg)) ;
    TEST_MSG ("Loading of adjacency matrix failed") ;

    // construct an undirected graph G with adjacency matrix A
    OK (LAGraph_New (&G, &A, LAGraph_ADJACENCY_UNDIRECTED, msg)) ;
    TEST_CHECK (A == NULL) ;

    OK (LAGraph_Cached_NSelfEdges (G, msg)) ;

    GrB_Vector c = NULL ;
    double t [2] ;

    // c is NULL
    int result = LAGraph_lcc (NULL, G->A, true, false, t, msg) ;
    printf ("\nresult: %d\n", result) ;
    TEST_CHECK (result == GrB_NULL_POINTER) ;

    // G->A is held by column
    OK (GxB_set (G->A, GxB_FORMAT, GxB_BY_COL)) ;
    result = LAGraph_lcc (&c, G->A, true, true, t, msg) ;
    printf ("\nresult: %d\n", result) ;
    TEST_CHECK (result == GrB_INVALID_VALUE) ;
    TEST_CHECK (c == NULL) ;

    OK (LAGraph_Delete (&G, msg)) ;
    #else
    printf ("test skipped\n") ;
    #endif
    LAGraph_Finalize (msg) ;
}

//****************************************************************************

TEST_LIST = {
    {"lcc", test_lcc},
    {"lcc_errors", test_errors},
    {NULL, NULL}
};
