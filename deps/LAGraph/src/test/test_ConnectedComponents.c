//------------------------------------------------------------------------------
// LAGraph/src/test/test_ConnectedComponents.c: test cases for CC
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

#include "LAGraph_test.h"
// also test LG_CC_FastSV5 and LAGraph_cc_lacc
#include "LAGraphX.h"
#include "LG_alg_internal.h"

#undef NDEBUG
#include <assert.h>

char msg [LAGRAPH_MSG_LEN] ;
LAGraph_Graph G = NULL ;
#define LEN 512
char filename [LEN+1] ;
GrB_Vector C = NULL, C2 = NULL ;
GrB_Matrix A = NULL ;

typedef struct
{
    uint32_t ncomponents ;
    const char *name ;
}
matrix_info ;

const matrix_info files [ ] =
{
    {      1, "karate.mtx" },
    {      1, "A.mtx" },
    {      1, "jagmesh7.mtx" },
    {      1, "ldbc-cdlp-undirected-example.mtx" },
    {      1, "ldbc-undirected-example.mtx" },
    {      1, "ldbc-wcc-example.mtx" },
    {      3, "LFAT5.mtx" },
    {   1989, "LFAT5_hypersparse.mtx" },
    {      6, "LFAT5_two.mtx" },
    {      1, "bcsstk13.mtx" },
    {      1, "tree-example.mtx" },
    {   1391, "zenios.mtx" },
    {      0, "" },
} ;

//------------------------------------------------------------------------------
// count_connected_components: count the # of components in a component vector
//------------------------------------------------------------------------------

int count_connected_components (GrB_Vector C) ;

int count_connected_components (GrB_Vector C)
{
    GrB_Index n = 0 ;
    OK (GrB_Vector_size (&n, C)) ;
    int ncomponents = 0 ;
    for (int i = 0 ; i < n ; i++)
    {
        int64_t comp = -1 ;
        int result = GrB_Vector_extractElement (&comp, C, i) ;
        if (result == GrB_SUCCESS && comp == i) ncomponents++ ;
    }
    return (ncomponents) ;
}

//----------------------------------------------------------------------------
// test_cc_matrices: test with several matrices
//----------------------------------------------------------------------------

void test_cc_matrices (void)
{

    OK (LAGraph_Init (msg)) ;
    // OK (GxB_set (GxB_BURBLE, true)) ;

    for (int k = 0 ; ; k++)
    {

        // load the adjacency matrix as A
        const char *aname = files [k].name ;
        uint32_t ncomp = files [k].ncomponents ;
        if (strlen (aname) == 0) break;
        printf ("\nMatrix: %s\n", aname) ;
        TEST_CASE (aname) ;
        snprintf (filename, LEN, LG_DATA_DIR "%s", aname) ;
        FILE *f = fopen (filename, "r") ;
        TEST_CHECK (f != NULL) ;
        OK (LAGraph_MMRead (&A, f, msg)) ;
        OK (fclose (f)) ;
        TEST_MSG ("Loading of adjacency matrix failed") ;
        GrB_Index n ;
        OK (GrB_Matrix_nrows (&n, A)) ;

        // create the graph
        OK (LAGraph_New (&G, &A, LAGraph_ADJACENCY_UNDIRECTED, msg)) ;
        TEST_CHECK (A == NULL) ;    // A has been moved into G->A

        for (int trial = 0 ; trial <= 1 ; trial++)
        {
            // find the connected components
            printf ("\n--- CC: FastSV6 if SuiteSparse, Boruvka if vanilla:\n") ;
            OK (LAGr_ConnectedComponents (&C, G, msg)) ;
            OK (LAGraph_Vector_Print (C, 2, stdout, msg)) ;

            // count the # of connected components
            int ncomponents = count_connected_components (C) ;
            printf ("# components: %6u Matrix: %s\n", ncomponents, aname) ;
            TEST_CHECK (ncomponents == ncomp) ;
            GrB_Index cnvals ;
            OK (GrB_Vector_nvals (&cnvals, C)) ;
            TEST_CHECK (cnvals == n) ;

            // check the result
            OK (LG_check_cc (C, G, msg)) ;
            OK (GrB_free (&C)) ;

            // find the connected components with LG_CC_FastSV5
            #if LAGRAPH_SUITESPARSE
            printf ("\n------ CC_FastSV5:\n") ;
            OK (LG_CC_FastSV5 (&C2, G, msg)) ;
            ncomponents = count_connected_components (C2) ;
            TEST_CHECK (ncomponents == ncomp) ;
            OK (LG_check_cc (C2, G, msg)) ;
            OK (GrB_free (&C2)) ;
            #endif

            // find the connected components with LG_CC_Boruvka
            int result = GrB_SUCCESS ;
            printf ("\n------ CC_BORUVKA:\n") ;
            result = LG_CC_Boruvka (&C2, G, msg) ;
            OK (result) ;
            ncomponents = count_connected_components (C2) ;
            TEST_CHECK (ncomponents == ncomp) ;
            OK (LG_check_cc (C2, G, msg)) ;
            OK (GrB_free (&C2)) ;

            result = LG_CC_Boruvka (NULL, G, msg) ;
            TEST_CHECK (result == GrB_NULL_POINTER) ;

            if (trial == 0)
            {
                for (int sanitize = 0 ; sanitize <= 1 ; sanitize++)
                {
                    // find the connected components with cc_lacc
                    printf ("\n------ CC_LACC:\n") ;
                    OK (LAGraph_cc_lacc (&C2, G->A, sanitize, msg)) ;
                    ncomponents = count_connected_components (C2) ;
                    TEST_CHECK (ncomponents == ncomp) ;
                    OK (LG_check_cc (C2, G, msg)) ;
                    OK (GrB_free (&C2)) ;
                }

                result = LAGraph_cc_lacc (NULL, G->A, false, msg) ;
                TEST_CHECK (result == GrB_NULL_POINTER) ;
            }

            // convert to directed with symmetric pattern for next trial
            G->kind = LAGraph_ADJACENCY_DIRECTED ;
            G->is_symmetric_structure = LAGraph_TRUE ;
        }

        OK (LAGraph_Delete (&G, msg)) ;
    }

    OK (LAGraph_Finalize (msg)) ;
}

//------------------------------------------------------------------------------
// test_CC_errors:
//------------------------------------------------------------------------------

void test_cc_errors (void)
{
    OK (LAGraph_Init (msg)) ;
    printf ("\n") ;

    // check for null pointers
    int result = GrB_SUCCESS ;

    result = LG_CC_Boruvka (NULL, NULL, msg) ;
    TEST_CHECK (result == GrB_NULL_POINTER) ;

    #if LAGRAPH_SUITESPARSE
    result = LG_CC_FastSV6 (NULL, NULL, msg) ;
    TEST_CHECK (result == GrB_NULL_POINTER) ;
    #endif

    // load a valid matrix
    FILE *f = fopen (LG_DATA_DIR "LFAT5_two.mtx", "r") ;
    TEST_CHECK (f != NULL) ;
    OK (LAGraph_MMRead (&A, f, msg)) ;
    OK (fclose (f)) ;

    // create an valid directed graph (not known to be symmetric)
    OK (LAGraph_New (&G, &A, LAGraph_ADJACENCY_DIRECTED, msg)) ;
    TEST_CHECK (A == NULL) ;    // A has been moved into G->A

    result = LG_CC_Boruvka (&C, G, msg) ;
    TEST_CHECK (result == -1001) ;
    printf ("result expected: %d msg:\n%s\n", result, msg) ;

    #if LAGRAPH_SUITESPARSE
    result = LG_CC_FastSV6 (&C, G, msg) ;
    TEST_CHECK (result == -1001) ;
    printf ("result expected: %d msg:\n%s\n", result, msg) ;
    #endif

    OK (LAGraph_Finalize (msg)) ;
}

//------------------------------------------------------------------------------
// test_CC_brutal:
//------------------------------------------------------------------------------

#if LAGRAPH_SUITESPARSE
void test_cc_brutal (void)
{
    OK (LG_brutal_setup (msg)) ;

    // load a valid adjacency matrix
    TEST_CASE ("LFAT5_two") ;
    uint32_t ncomp = 6 ;
    FILE *f = fopen (LG_DATA_DIR "LFAT5_two.mtx", "r") ;
    TEST_CHECK (f != NULL) ;
    OK (LAGraph_MMRead (&A, f, msg)) ;
    OK (fclose (f)) ;
    TEST_MSG ("Loading of LFAT5_two.mtx failed") ;
    printf ("\n") ;

    // create an valid graph
    OK (LAGraph_New (&G, &A, LAGraph_ADJACENCY_UNDIRECTED, msg)) ;
    TEST_CHECK (A == NULL) ;    // A has been moved into G->A
    LG_BRUTAL_BURBLE (LAGraph_CheckGraph (G, msg)) ;

    // find the connected components
    printf ("\n--- CC: FastSV6 if SuiteSparse, Boruvka if vanilla:\n") ;
    LG_BRUTAL_BURBLE (LAGr_ConnectedComponents (&C, G, msg)) ;
    LG_BRUTAL_BURBLE (LAGraph_Vector_Print (C, LAGraph_SHORT, stdout, msg)) ;

    // count the # of connected components
    int ncomponents = count_connected_components (C) ;
    printf ("# components: %6u Matrix: %s\n", ncomponents, "LFAT_two") ;
    TEST_CHECK (ncomponents == ncomp) ;
    OK (LG_check_cc (C, G, msg)) ;

    OK (LAGraph_Delete (&G, msg)) ;
    OK (GrB_free (&C)) ;
    OK (LG_brutal_teardown (msg)) ;
}
#endif

//****************************************************************************
//****************************************************************************
TEST_LIST = {
    {"cc", test_cc_matrices},
    #if LAGRAPH_SUITESPARSE
    {"cc_brutal", test_cc_brutal},
    #endif
    {"cc_errors", test_cc_errors},
    {NULL, NULL}
};
