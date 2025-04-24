//------------------------------------------------------------------------------
// LAGraph/experimental/test/test_scc.c: tests for Strongly Connected Components
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

// todo: write a simple scc method, as LG_check_scc, and compare its results
// with LAGraph_scc

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
    const char *name ;
}
matrix_info ;

int scc_cover [7] = { 0, 0, 2, 0, 4, 2, 0 } ;

const matrix_info files [ ] =
{
    { "A2.mtx" },
    { "A.mtx" },
    { "bcsstk13.mtx" },
    { "cover.mtx" },
    { "cover_structure.mtx" },
    { "cryg2500.mtx" },
    { "full.mtx" },
    { "full_noheader.mtx" },
    { "full_symmetric.mtx" },
    { "jagmesh7.mtx" },
    { "karate.mtx" },
    { "ldbc-cdlp-directed-example.mtx" },
    { "ldbc-cdlp-undirected-example.mtx" },
    { "ldbc-directed-example-bool.mtx" },
    { "ldbc-directed-example.mtx" },
    { "ldbc-directed-example-unweighted.mtx" },
    { "ldbc-undirected-example-bool.mtx" },
    { "ldbc-undirected-example.mtx" },
    { "ldbc-undirected-example-unweighted.mtx" },
    { "ldbc-wcc-example.mtx" },
    { "LFAT5.mtx" },
    { "LFAT5_two.mtx" },
    { "matrix_bool.mtx" },
    { "matrix_fp32.mtx" },
    { "matrix_fp32_structure.mtx" },
    { "matrix_fp64.mtx" },
    { "matrix_int16.mtx" },
    { "matrix_int32.mtx" },
    { "matrix_int64.mtx" },
    { "matrix_int8.mtx" },
    { "matrix_uint16.mtx" },
    { "matrix_uint32.mtx" },
    { "matrix_uint64.mtx" },
    { "matrix_uint8.mtx" },
    { "msf1.mtx" },
    { "msf2.mtx" },
    { "msf3.mtx" },
    { "olm1000.mtx" },
    { "pushpull.mtx" },
    { "sample2.mtx" },
    { "sample.mtx" },
    { "structure.mtx" },
    { "test_BF.mtx" },
    { "test_FW_1000.mtx" },
    { "test_FW_2003.mtx" },
    { "test_FW_2500.mtx" },
    { "tree-example.mtx" },
    { "west0067_jumbled.mtx" },
    { "west0067.mtx" },
    { "west0067_noheader.mtx" },
    { "zenios.mtx" },
    { "" },
} ;

//****************************************************************************
void test_scc (void)
{
    LAGraph_Init (msg) ;
    #if LAGRAPH_SUITESPARSE

    for (int k = 0 ; ; k++)
    {

        // load the matrix as A
        const char *aname = files [k].name ;
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

        GrB_Vector c = NULL ;

        // find the strongly connected components with LAGraph_scc
        OK (LAGraph_scc (&c, G->A, msg)) ;

        GrB_Index n ;
        OK (GrB_Vector_size (&n, c)) ;
        LAGraph_PrintLevel pr = (n <= 100) ? LAGraph_COMPLETE : LAGraph_SHORT ;

        // check result c for cover
        if (strcmp (aname, "cover.mtx") == 0)
        {
            GrB_Vector cgood = NULL ;
            OK (GrB_Vector_new (&cgood, GrB_UINT64, n)) ;
            for (int k = 0 ; k < n ; k++)
            {
                OK (GrB_Vector_setElement (cgood, scc_cover [k], k)) ;
            }
            OK (GrB_wait (cgood, GrB_MATERIALIZE)) ;
            printf ("\nscc (known result):\n") ;
            OK (LAGraph_Vector_Print (cgood, pr, stdout, msg)) ;
            bool ok = false ;
            OK (LAGraph_Vector_IsEqual (&ok, c, cgood, msg)) ;
            TEST_CHECK (ok) ;
            OK (GrB_free (&cgood)) ;
        }

        printf ("\nscc:\n") ;
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

    GrB_Vector c = NULL ;
    GrB_Matrix A = NULL ;

    // c and A are NULL
    int result = LAGraph_scc (NULL, A, msg) ;
    printf ("\nresult: %d\n", result) ;
    TEST_CHECK (result == GrB_NULL_POINTER) ;

    // A is rectangular
    OK (GrB_Matrix_new (&A, GrB_BOOL, 3, 4)) ;
    result = LAGraph_scc (&c, A, msg) ;
    TEST_CHECK (result == GrB_DIMENSION_MISMATCH) ;

    OK (GrB_free (&c)) ;
    OK (GrB_free (&A)) ;
    #else
    printf ("test skipped\n") ;
    #endif
    LAGraph_Finalize (msg) ;
}

//****************************************************************************

TEST_LIST = {
    {"scc", test_scc},
    {"scc_errors", test_errors},
    {NULL, NULL}
};
