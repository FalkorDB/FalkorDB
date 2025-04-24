//------------------------------------------------------------------------------
// LAGraph/experimental/test/test_msf.c: test cases for Min Spanning Forest
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

// todo: write a simple msf method, as LG_check_msf, and compare its results
// with LAGraph_msf

#include <stdio.h>
#include <acutest.h>

#include <LAGraphX.h>
#include <LAGraph_test.h>

char msg [LAGRAPH_MSG_LEN] ;
LAGraph_Graph G = NULL ;
GrB_Matrix A = NULL ;
GrB_Matrix S = NULL ;
GrB_Matrix C = NULL ;
#define LEN 512
char filename [LEN+1] ;

typedef struct
{
    bool symmetric ;
    const char *name ;
}
matrix_info ;

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
void test_msf (void)
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

        // ensure A is uint64
        GrB_Index nrows, ncols ;
        OK (GrB_Matrix_nrows (&nrows, A)) ;
        OK (GrB_Matrix_ncols (&ncols, A)) ;
        OK (GrB_Matrix_new (&S, GrB_UINT64, nrows, ncols)) ;
        OK (GrB_assign (S, NULL, NULL, A, GrB_ALL, nrows, GrB_ALL, ncols,
            NULL)) ;
        GrB_Index n = nrows ;
        OK (GrB_free (&A)) ;

        // construct a directed graph G with adjacency matrix S
        OK (LAGraph_New (&G, &S, LAGraph_ADJACENCY_DIRECTED, msg)) ;
        TEST_CHECK (S == NULL) ;

        bool sanitize = (!symmetric) ;

        // compute the min spanning forest
        C = NULL ;
        int result = LAGraph_msf (&C, G->A, sanitize, msg) ;
        printf ("result: %d\n", result) ;
        LAGraph_PrintLevel pr = (n <= 100) ? LAGraph_COMPLETE : LAGraph_SHORT ;

        // check result C for A.mtx
        if (strcmp (aname, "A.mtx") == 0)
        {
            GrB_Matrix Cgood = NULL ;
            OK (GrB_Matrix_new (&Cgood, GrB_UINT64, n, n)) ;
            OK (GrB_Matrix_setElement (Cgood, 1, 1, 0)) ;
            OK (GrB_Matrix_setElement (Cgood, 1, 2, 0)) ;
            OK (GrB_Matrix_setElement (Cgood, 1, 3, 1)) ;
            OK (GrB_Matrix_setElement (Cgood, 1, 4, 1)) ;
            OK (GrB_Matrix_setElement (Cgood, 1, 5, 1)) ;
            OK (GrB_Matrix_setElement (Cgood, 1, 6, 0)) ;
            OK (GrB_wait (Cgood, GrB_MATERIALIZE)) ;
            printf ("\nmsf (known result):\n") ;
            OK (LAGraph_Matrix_Print (Cgood, pr, stdout, msg)) ;
            bool ok = false ;
            OK (LAGraph_Matrix_IsEqual (&ok, C, Cgood, msg)) ;
            TEST_CHECK (ok) ;
            OK (GrB_free (&Cgood)) ;
        }

        printf ("\nmsf:\n") ;
        OK (LAGraph_Matrix_Print (C, pr, stdout, msg)) ;
        OK (GrB_free (&C)) ;
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

    // C and A are NULL
    int result = LAGraph_msf (NULL, NULL, true, msg) ;
    TEST_CHECK (result == GrB_NULL_POINTER) ;

    // A must be square
    OK (GrB_Matrix_new (&A, GrB_UINT64, 3, 4)) ;
    result = LAGraph_msf (&C, A, true, msg) ;
    TEST_CHECK (result == GrB_DIMENSION_MISMATCH) ;

    OK (GrB_free (&A)) ;
    #else
    printf ("test skipped\n") ;
    #endif
    LAGraph_Finalize (msg) ;
}

//****************************************************************************

TEST_LIST = {
    {"msf", test_msf},
    {"msf_errors", test_errors},
    {NULL, NULL}
};
