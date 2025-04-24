//----------------------------------------------------------------------------
// LAGraph/src/test/test_TriangleCount.c: test cases for triangle
// counting algorithms
// ----------------------------------------------------------------------------

// LAGraph, (c) 2019-2022 by The LAGraph Contributors, All Rights Reserved.
// SPDX-License-Identifier: BSD-2-Clause
//
// For additional details (including references to third party source code and
// other files) see the LICENSE file or contact permission@sei.cmu.edu. See
// Contributors.txt for a full list of contributors. Created, in part, with
// funding and support from the U.S. Government (see Acknowledgments.txt file).
// DM22-0790

// Contributed by Scott McMillan, SEI, and Timothy A. Davis, Texas A&M
// University

//-----------------------------------------------------------------------------

#include <stdio.h>
#include <acutest.h>

#include <LAGraph_test.h>
#include <graph_zachary_karate.h>

char msg[LAGRAPH_MSG_LEN];
LAGraph_Graph G = NULL;

#define LEN 512
char filename [LEN+1] ;

typedef struct
{
    uint64_t ntriangles ;           // # triangles in original matrix
    const char *name ;              // matrix filename
}
matrix_info ;

const matrix_info files [ ] =
{
    {     45, "karate.mtx" },
    {     11, "A.mtx" },
    {   2016, "jagmesh7.mtx" },
    {      6, "ldbc-cdlp-undirected-example.mtx" },
    {      4, "ldbc-undirected-example.mtx" },
    {      5, "ldbc-wcc-example.mtx" },
    {      0, "LFAT5.mtx" },
    { 342300, "bcsstk13.mtx" },
    {      0, "tree-example.mtx" },
    {      0, "" },
} ;

//****************************************************************************
void setup(void)
{
    LAGraph_Init(msg);
    int retval;

    GrB_Matrix A = NULL;

    GrB_Matrix_new(&A, GrB_UINT32, ZACHARY_NUM_NODES, ZACHARY_NUM_NODES);
    GrB_Matrix_build(A, ZACHARY_I, ZACHARY_J, ZACHARY_V, ZACHARY_NUM_EDGES,
                     GrB_LOR);

    retval = LAGraph_New(&G, &A, LAGraph_ADJACENCY_UNDIRECTED, msg);
    TEST_CHECK(retval == 0);
    TEST_MSG("retval = %d (%s)", retval, msg);

    retval = LAGraph_Cached_NSelfEdges(G, msg);
    TEST_CHECK(retval == 0);
    TEST_MSG("retval = %d (%s)", retval, msg);

    TEST_CHECK(G->nself_edges == 0);
}

//****************************************************************************
void teardown(void)
{
    int retval = LAGraph_Delete(&G, msg);
    TEST_CHECK(retval == 0);
    TEST_MSG("retval = %d (%s)", retval, msg);

    G = NULL;
    LAGraph_Finalize(msg);
}

//****************************************************************************
//****************************************************************************
void test_TriangleCount_Methods1(void)
{
    setup();
    int retval;
    uint64_t ntriangles = 0UL;

    // with presort
    LAGr_TriangleCount_Presort presort = LAGr_TriangleCount_AutoSort ;
    LAGr_TriangleCount_Method method = LAGr_TriangleCount_Burkhardt ;
    ntriangles = 0UL;
    // LAGr_TriangleCount_Burkhardt: sum (sum ((A^2) .* A)) / 6
    retval = LAGr_TriangleCount(&ntriangles, G, &method, &presort, msg);

    TEST_CHECK(retval == 0);
    TEST_MSG("retval = %d (%s)", retval, msg);

    TEST_CHECK( ntriangles == 45 );
    TEST_MSG("numtri = %g", (double) ntriangles);

    teardown();
}

//****************************************************************************
void test_TriangleCount_Methods2(void)
{
    setup();
    int retval;
    uint64_t ntriangles = 0UL;

    LAGr_TriangleCount_Presort presort = LAGr_TriangleCount_AutoSort ;
    LAGr_TriangleCount_Method method = LAGr_TriangleCount_Cohen ;
    ntriangles = 0UL;
    // LAGr_TriangleCount_Cohen: sum (sum ((L * U) .* A)) / 2
    retval = LAGr_TriangleCount(&ntriangles, G, &method, &presort, msg);
    TEST_CHECK(retval == 0);
    TEST_MSG("retval = %d (%s)", retval, msg);

    TEST_CHECK( ntriangles == 45 );
    TEST_MSG("numtri = %g", (double) ntriangles);

    teardown();
}

//****************************************************************************
void test_TriangleCount_Methods3(void)
{
    setup();
    int retval;
    uint64_t ntriangles = 0UL;

    LAGr_TriangleCount_Presort presort = LAGr_TriangleCount_AutoSort ;
    LAGr_TriangleCount_Method method = LAGr_TriangleCount_Sandia_LL ;
    ntriangles = 0UL;
    // LAGr_TriangleCount_Sandia_LL: sum (sum ((L * L) .* L))
    retval = LAGr_TriangleCount (&ntriangles, G, &method, &presort, msg) ;
    // should fail (out_degrees needs to be defined)
    TEST_CHECK(retval == LAGRAPH_NOT_CACHED);
    TEST_MSG("retval = %d (%s)", retval, msg);

    retval = LAGraph_Cached_OutDegree(G, msg);
    TEST_CHECK(retval == 0);
    TEST_MSG("retval = %d (%s)", retval, msg);

    presort = LAGr_TriangleCount_AutoSort ;
    method = LAGr_TriangleCount_Sandia_LL ;
    retval = LAGr_TriangleCount(&ntriangles, G, &method, &presort, msg);
    TEST_CHECK(retval == 0);
    TEST_MSG("retval = %d (%s)", retval, msg);

    TEST_CHECK( ntriangles == 45 );
    TEST_MSG("numtri = %g", (double) ntriangles);

    teardown();
}

//****************************************************************************
void test_TriangleCount_Methods4(void)
{
    setup();
    int retval;
    uint64_t ntriangles = 0UL;

    LAGr_TriangleCount_Presort presort = LAGr_TriangleCount_AutoSort ;
    LAGr_TriangleCount_Method method = LAGr_TriangleCount_Sandia_UU ;
    ntriangles = 0UL;
    // LAGr_TriangleCount_Sandia_UU: sum (sum ((U * U) .* U))
    retval = LAGr_TriangleCount(&ntriangles, G, &method, &presort, msg);
    // should fail (out_degrees needs to be defined)
    TEST_CHECK(retval == LAGRAPH_NOT_CACHED);
    TEST_MSG("retval = %d (%s)", retval, msg);

    retval = LAGraph_Cached_OutDegree(G, msg);
    TEST_CHECK(retval == 0);
    TEST_MSG("retval = %d (%s)", retval, msg);

    presort = LAGr_TriangleCount_AutoSort ;
    method = LAGr_TriangleCount_Sandia_UU ;
    retval = LAGr_TriangleCount(&ntriangles, G, &method, &presort, msg);
    TEST_CHECK(retval == 0);
    TEST_MSG("retval = %d (%s)", retval, msg);

    TEST_CHECK( ntriangles == 45 );
    TEST_MSG("numtri = %g", (double) ntriangles) ;

    teardown();
}

//****************************************************************************
void test_TriangleCount_Methods5(void)
{
    setup();
    int retval;
    uint64_t ntriangles = 0UL;

    LAGr_TriangleCount_Presort presort = LAGr_TriangleCount_AutoSort ;
    LAGr_TriangleCount_Method method = LAGr_TriangleCount_Sandia_LUT ;
    ntriangles = 0UL;
    // LAGr_TriangleCount_Sandia_LUT: sum (sum ((L * U') .* L))
    retval = LAGr_TriangleCount(&ntriangles, G, &method, &presort, msg);
    // should fail (out_degrees needs to be defined)
    TEST_CHECK(retval == LAGRAPH_NOT_CACHED);
    TEST_MSG("retval = %d (%s)", retval, msg);

    retval = LAGraph_Cached_OutDegree(G, msg);
    TEST_CHECK(retval == 0);
    TEST_MSG("retval = %d (%s)", retval, msg);

    presort = LAGr_TriangleCount_AutoSort ;
    method = LAGr_TriangleCount_Sandia_LUT ;
    retval = LAGr_TriangleCount(&ntriangles, G, &method, &presort, msg);
    TEST_CHECK(retval == 0);
    TEST_MSG("retval = %d (%s)", retval, msg);

    TEST_CHECK( ntriangles == 45 );
    TEST_MSG("numtri = %g", (double) ntriangles) ;

    teardown();
}

//****************************************************************************
void test_TriangleCount_Methods6(void)
{
    setup();
    int retval;
    uint64_t ntriangles = 0UL;

    LAGr_TriangleCount_Presort presort = LAGr_TriangleCount_AutoSort ;
    LAGr_TriangleCount_Method method = LAGr_TriangleCount_Sandia_ULT ;
    ntriangles = 0UL;
    // LAGr_TriangleCount_Sandia_ULT: sum (sum ((U * L') .* U))
    retval = LAGr_TriangleCount(&ntriangles, G, &method, &presort, msg);
    // should fail (out_degrees needs to be defined)
    TEST_CHECK(retval == LAGRAPH_NOT_CACHED) ;
    TEST_MSG("retval = %d (%s)", retval, msg);

    retval = LAGraph_Cached_OutDegree(G, msg);
    TEST_CHECK(retval == 0);
    TEST_MSG("retval = %d (%s)", retval, msg);

    presort = LAGr_TriangleCount_AutoSort ;
    method = LAGr_TriangleCount_Sandia_ULT ;
    retval = LAGr_TriangleCount(&ntriangles, G, &method, &presort, msg);
    TEST_CHECK(retval == 0);
    TEST_MSG("retval = %d (%s)", retval, msg);

    TEST_CHECK( ntriangles == 45 );
    TEST_MSG("numtri = %g", (double) ntriangles) ;

    teardown();
}

//****************************************************************************
void test_TriangleCount(void)
{
    setup();

    uint64_t ntriangles = 0UL;
    int retval = LAGraph_TriangleCount(&ntriangles, G, msg);
    // should not fail (out_degrees will be calculated)
    TEST_CHECK(retval == 0);
    TEST_MSG("retval = %d (%s)", retval, msg);

    TEST_CHECK( ntriangles == 45 );
    TEST_MSG("numtri = %g", (double) ntriangles) ;

    OK (LG_check_tri (&ntriangles, G, msg)) ;
    TEST_CHECK( ntriangles == 45 );

    teardown();
}

//****************************************************************************
void test_TriangleCount_many (void)
{
    LAGraph_Init(msg);
    GrB_Matrix A = NULL ;
    printf ("\n") ;

    for (int k = 0 ; ; k++)
    {

        // load the adjacency matrix as A
        const char *aname = files [k].name ;
        uint64_t ntriangles = files [k].ntriangles ;
        if (strlen (aname) == 0) break;
        TEST_CASE (aname) ;
        snprintf (filename, LEN, LG_DATA_DIR "%s", aname) ;
        FILE *f = fopen (filename, "r") ;
        TEST_CHECK (f != NULL) ;
        OK (LAGraph_MMRead (&A, f, msg)) ;
        OK (fclose (f)) ;
        TEST_MSG ("Loading of adjacency matrix failed") ;

        // create the graph
        OK (LAGraph_New (&G, &A, LAGraph_ADJACENCY_UNDIRECTED, msg)) ;
        TEST_CHECK (A == NULL) ;    // A has been moved into G->A

        // delete any diagonal entries
        OK (LAGraph_DeleteSelfEdges (G, msg)) ;
        TEST_CHECK (G->nself_edges == 0) ;
        OK (LAGraph_DeleteSelfEdges (G, msg)) ;
        TEST_CHECK (G->nself_edges == 0) ;

        // get the # of triangles
        uint64_t nt0, nt1 ;
        OK (LAGraph_TriangleCount (&nt1, G, msg)) ;
        printf ("# triangles: %g Matrix: %s\n", (double) nt1, aname) ;
        TEST_CHECK (nt1 == ntriangles) ;
        OK (LG_check_tri (&nt0, G, msg)) ;
        TEST_CHECK (nt0 == nt1) ;

        // convert to directed but with symmetric pattern
        G->kind = LAGraph_ADJACENCY_DIRECTED ;
        G->is_symmetric_structure = LAGraph_TRUE ;
        OK (LAGraph_TriangleCount (&nt1, G, msg)) ;
        TEST_CHECK (nt1 == ntriangles) ;

        OK (LG_check_tri (&nt0, G, msg)) ;
        TEST_CHECK (nt0 == nt1) ;

        // try each method
        for (int method = 0 ; method <= 6 ; method++)
        {
            for (int presort = 0 ; presort <= 2 ; presort++)
            {
                LAGr_TriangleCount_Presort s = presort ;
                LAGr_TriangleCount_Method m = method ;
                OK (LAGr_TriangleCount (&nt1, G, &m, &s, msg)) ;
                TEST_CHECK (nt1 == ntriangles) ;
            }
        }

        // invalid method
        LAGr_TriangleCount_Method method = 99 ;
        int result = LAGr_TriangleCount (&nt1, G, &method, NULL, msg) ;
        TEST_CHECK (result == GrB_INVALID_VALUE) ;
        LAGr_TriangleCount_Presort presort = 99 ;
        result = LAGr_TriangleCount (&nt1, G, NULL, &presort, msg) ;
        TEST_CHECK (result == GrB_INVALID_VALUE) ;

        OK (LAGraph_Delete (&G, msg)) ;
    }

    LAGraph_Finalize(msg);
}

//****************************************************************************

void test_TriangleCount_autosort (void)
{
    OK (LAGraph_Init(msg)) ;

    // create a banded matrix with a some dense rows/columns
    GrB_Index n = 50000 ;
    GrB_Matrix A = NULL ;
    OK (GrB_Matrix_new (&A, GrB_BOOL, n, n)) ;

    for (int k = 0 ; k <= 10 ; k++)
    {
        for (int i = 0 ; i < n ; i++)
        {
            OK (GrB_Matrix_setElement_BOOL (A, true, i, k)) ;
            OK (GrB_Matrix_setElement_BOOL (A, true, k, i)) ;
        }
    }

    // create the graph
    OK (LAGraph_New (&G, &A, LAGraph_ADJACENCY_UNDIRECTED, msg)) ;
    TEST_CHECK (A == NULL) ;    // A has been moved into G->A

    OK (LAGraph_DeleteSelfEdges (G, msg)) ;
    TEST_CHECK (G->nself_edges == 0) ;

    OK (LAGraph_Cached_OutDegree (G, msg)) ;

    // try each method; with autosort
    GrB_Index nt1 = 0 ;
    for (int method = 0 ; method <= 6 ; method++)
    {
        LAGr_TriangleCount_Presort presort = LAGr_TriangleCount_AutoSort ;
        LAGr_TriangleCount_Method m = method ;
        nt1 = 0 ;
        OK (LAGr_TriangleCount (&nt1, G, &m, &presort, msg)) ;
        TEST_CHECK (nt1 == 2749560) ;
    }

    nt1 = 0 ;
    OK (LAGraph_TriangleCount (&nt1, G, msg)) ;
    TEST_CHECK (nt1 == 2749560) ;

    OK (LAGraph_Finalize(msg)) ;
}

//------------------------------------------------------------------------------
// test_TriangleCount_brutal
//------------------------------------------------------------------------------

#if LAGRAPH_SUITESPARSE
void test_TriangleCount_brutal (void)
{
    OK (LG_brutal_setup (msg)) ;

    GrB_Matrix A = NULL ;
    printf ("\n") ;

    for (int k = 0 ; ; k++)
    {

        // load the adjacency matrix as A
        const char *aname = files [k].name ;
        uint64_t ntriangles = files [k].ntriangles ;
        if (strlen (aname) == 0) break;
        printf ("\n================== Matrix: %s\n", aname) ;
        TEST_CASE (aname) ;
        snprintf (filename, LEN, LG_DATA_DIR "%s", aname) ;
        FILE *f = fopen (filename, "r") ;
        TEST_CHECK (f != NULL) ;
        OK (LAGraph_MMRead (&A, f, msg)) ;
        OK (fclose (f)) ;
        TEST_MSG ("Loading of adjacency matrix failed") ;

        // create the graph
        OK (LAGraph_New (&G, &A, LAGraph_ADJACENCY_UNDIRECTED, msg)) ;

        // delete any diagonal entries
        OK (LAGraph_DeleteSelfEdges (G, msg)) ;

        // get the # of triangles
        uint64_t nt0, nt1 ;
        LG_BRUTAL_BURBLE (LAGraph_TriangleCount (&nt1, G, msg)) ;
        printf ("# triangles: %g Matrix: %s\n", (double) nt1, aname) ;
        TEST_CHECK (nt1 == ntriangles) ;

        LG_BRUTAL_BURBLE (LG_check_tri (&nt0, G, msg)) ;
        TEST_CHECK (nt0 == nt1) ;

        // convert to directed but with symmetric pattern
        G->kind = LAGraph_ADJACENCY_DIRECTED ;
        G->is_symmetric_structure = LAGraph_TRUE ;
        LG_BRUTAL (LAGraph_TriangleCount (&nt1, G, msg)) ;
        TEST_CHECK (nt1 == ntriangles) ;

        LG_BRUTAL_BURBLE (LG_check_tri (&nt0, G, msg)) ;
        TEST_CHECK (nt0 == nt1) ;

        // try each method
        for (int method = 0 ; method <= 6 ; method++)
        {
            for (int presort = 0 ; presort <= 2 ; presort++)
            {
                LAGr_TriangleCount_Presort s = presort ;
                LAGr_TriangleCount_Method m = method ;
                LG_BRUTAL_BURBLE (LAGr_TriangleCount (&nt1, G, &m, &s, msg)) ;
                TEST_CHECK (nt1 == ntriangles) ;
            }
        }

        OK (LAGraph_Delete (&G, msg)) ;
    }

    OK (LG_brutal_teardown (msg)) ;
}
#endif


//****************************************************************************
//****************************************************************************
TEST_LIST = {
    {"TriangleCount_Methods1", test_TriangleCount_Methods1},
    {"TriangleCount_Methods2", test_TriangleCount_Methods2},
    {"TriangleCount_Methods3", test_TriangleCount_Methods3},
    {"TriangleCount_Methods4", test_TriangleCount_Methods4},
    {"TriangleCount_Methods5", test_TriangleCount_Methods5},
    {"TriangleCount_Methods6", test_TriangleCount_Methods6},
    {"TriangleCount"         , test_TriangleCount},
    {"TriangleCount_many"    , test_TriangleCount_many},
    {"TriangleCount_autosort", test_TriangleCount_autosort},
    #if LAGRAPH_SUITESPARSE
    {"TriangleCount_brutal"  , test_TriangleCount_brutal},
    #endif
    {NULL, NULL}
};
