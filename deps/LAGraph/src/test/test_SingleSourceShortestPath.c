//----------------------------------------------------------------------------
// LAGraph/src/test/test_SingleSourceShortestPath.c: test cases for SSSP
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
#include <LAGraph_test.h>
#include "LG_internal.h"

char msg [LAGRAPH_MSG_LEN] ;
LAGraph_Graph G = NULL ;

#define LEN 512
char filename [LEN+1] ;
char atype_name [LAGRAPH_MAX_NAME_LEN] ;

typedef struct
{
    const char *name ;
}
matrix_info ;

const matrix_info files [ ] =
{
    { "A.mtx" },
    { "cover.mtx" },
    { "jagmesh7.mtx" },
    { "ldbc-cdlp-directed-example.mtx" },
    { "ldbc-cdlp-undirected-example.mtx" },
    { "ldbc-directed-example.mtx" },
    { "ldbc-undirected-example.mtx" },
    { "ldbc-wcc-example.mtx" },
    { "LFAT5.mtx" },
    { "LFAT5_hypersparse.mtx" },
    { "msf1.mtx" },
    { "msf2.mtx" },
    { "msf3.mtx" },
    { "sample2.mtx" },
    { "sample.mtx" },
    { "olm1000.mtx" },
    { "bcsstk13.mtx" },
    { "cryg2500.mtx" },
    { "tree-example.mtx" },
    { "west0067.mtx" },
    { "karate.mtx" },
    { "matrix_bool.mtx" },
    { "test_BF.mtx" },
    { "test_FW_1000.mtx" },
    { "test_FW_2003.mtx" },
    { "test_FW_2500.mtx" },
    { "skew_fp32.mtx" },
    { "matrix_uint32.mtx" },
    { "matrix_uint64.mtx" },
    { "" },
} ;

//****************************************************************************
void test_SingleSourceShortestPath(void)
{
    LAGraph_Init(msg);
    GrB_Matrix A = NULL, T = NULL ;
    GrB_Scalar Delta = NULL ;
    OK (GrB_Scalar_new (&Delta, GrB_INT32)) ;

    for (int k = 0 ; ; k++)
    {

        // load the adjacency matrix as A
        const char *aname = files [k].name ;
        LAGraph_Kind kind = LAGraph_ADJACENCY_DIRECTED ;
        // LAGraph_Kind kind = files [k].kind ;
        if (strlen (aname) == 0) break;
        TEST_CASE (aname) ;
        printf ("\nMatrix: %s\n", aname) ;
        snprintf (filename, LEN, LG_DATA_DIR "%s", aname) ;
        FILE *f = fopen (filename, "r") ;
        TEST_CHECK (f != NULL) ;
        OK (LAGraph_MMRead (&A, f, msg)) ;
        OK (fclose (f)) ;
        TEST_MSG ("Loading of adjacency matrix failed") ;

        GrB_Index n = 0, ncols = 1 ;
        OK (GrB_Matrix_nrows (&n, A)) ;
        OK (GrB_Matrix_ncols (&ncols, A)) ;
        TEST_CHECK (n == ncols) ;

        // convert A to int32
        OK (LAGraph_Matrix_TypeName (atype_name, A, msg)) ;
        if (!MATCHNAME (atype_name, "int32_t"))
        {
            OK (GrB_Matrix_new (&T, GrB_INT32, n, n)) ;
            OK (GrB_assign (T, NULL, NULL, A, GrB_ALL, n, GrB_ALL, n, NULL)) ;
            OK (GrB_free (&A)) ;
            A = T ;
        }

        // ensure all entries are positive, and in the range 1 to 255
        OK (GrB_Matrix_apply_BinaryOp2nd_INT32 (A, NULL, NULL,
            GrB_BAND_INT32, A, 255, NULL)) ;
        int32_t x ;
        OK (GrB_reduce (&x, NULL, GrB_MIN_MONOID_INT32, A, NULL)) ;
        if (x < 1)
        {
            OK (GrB_Matrix_apply_BinaryOp2nd_INT32 (A, NULL, NULL,
                GrB_MAX_INT32, A, 1, NULL)) ;
        }

        // create the graph
        OK (LAGraph_New (&G, &A, kind, msg)) ;
        OK (LAGraph_CheckGraph (G, msg)) ;
        TEST_CHECK (A == NULL) ;    // A has been moved into G->A

        // no need to compute emin; just use a bound
        OK (GrB_Scalar_new (&(G->emin), GrB_INT32)) ;
        OK (GrB_Scalar_setElement (G->emin, 1)) ;

        // delta values to try
        int32_t Deltas [ ] = { 30, 100, 50000 } ;

        // run the SSSP
        GrB_Vector path_length = NULL ;
        int64_t step = (n > 100) ? (3*n/4) : ((n/4) + 1) ;
        // int64_t src = 0 ;
        for (int64_t src = 0 ; src < n ; src += step)
        {
            // int32_t kk = 0 ;
            for (int32_t kk = 0 ; kk < ((n > 100) ? 1 : 3) ; kk++)
            {
                int32_t delta = Deltas [kk] ;
                printf ("src %d delta %d n %d\n", (int) src, delta, (int) n) ;
                OK (GrB_Scalar_setElement (Delta, delta)) ;
                OK (LAGr_SingleSourceShortestPath (&path_length, G,
                    src, Delta, msg)) ;
                int res = LG_check_sssp (path_length, G, src, msg) ;
                if (res != GrB_SUCCESS) printf ("res: %d msg: %s\n", res, msg) ;
                OK (res) ;
                OK (GrB_free(&path_length)) ;
            }
        }

        // add a single negative edge and try again
        OK (GrB_free (&(G->emin))) ;
        G->emin_state = LAGRAPH_UNKNOWN ;
        OK (GrB_Matrix_setElement_INT32 (G->A, -1, 0, 1)) ;
        OK (GrB_Scalar_setElement (Delta, 30)) ;
        OK (LAGr_SingleSourceShortestPath (&path_length, G, 0, Delta, msg)) ;
        OK (LAGraph_Vector_Print (path_length, LAGraph_SHORT, stdout, msg)) ;
        int32_t len = 0 ;
        OK (GrB_Vector_extractElement (&len, path_length, 1)) ;
        TEST_CHECK (len == -1) ;
        OK (GrB_free(&path_length)) ;

        OK (LAGraph_Delete (&G, msg)) ;
    }

    GrB_free (&Delta) ;
    LAGraph_Finalize(msg);
}

//------------------------------------------------------------------------------
// test_SingleSourceShortestPath_types
//------------------------------------------------------------------------------

void test_SingleSourceShortestPath_types (void)
{
    LAGraph_Init (msg) ;
    GrB_Matrix A = NULL, T = NULL ;
    GrB_Scalar Delta = NULL ;
    OK (GrB_Scalar_new (&Delta, GrB_INT32)) ;

    for (int k = 0 ; ; k++)
    {

        // load the adjacency matrix as A
        const char *aname = files [k].name ;
        LAGraph_Kind kind = LAGraph_ADJACENCY_DIRECTED ;
        // LAGraph_Kind kind = files [k].kind ;
        if (strlen (aname) == 0) break;
        TEST_CASE (aname) ;
        printf ("\nMatrix: %s\n", aname) ;
        snprintf (filename, LEN, LG_DATA_DIR "%s", aname) ;
        FILE *f = fopen (filename, "r") ;
        TEST_CHECK (f != NULL) ;
        OK (LAGraph_MMRead (&A, f, msg)) ;
        OK (fclose (f)) ;
        TEST_MSG ("Loading of adjacency matrix failed") ;

        GrB_Index n = 0, ncols = 1 ;
        OK (GrB_Matrix_nrows (&n, A)) ;
        OK (GrB_Matrix_ncols (&ncols, A)) ;
        TEST_CHECK (n == ncols) ;

        // ensure A has the right type
        OK (LAGraph_Matrix_TypeName (atype_name, A, msg)) ;
//      fprintf (stderr, "matrix %s type: %s\n", aname, atype_name) ;
        if (MATCHNAME (atype_name, "int32_t"))
        {
            // use A as-is, but ensure it's positive and nonzero,
            // and in range 1 to 255
            OK (GrB_apply (A, NULL, NULL, GrB_ABS_INT32, A, NULL)) ;
            OK (GrB_apply (A, NULL, NULL, GrB_MAX_INT32, A, 1, NULL)) ;
            OK (GrB_apply (A, NULL, NULL, GrB_MIN_INT32, A, 255, NULL)) ;
        }
        else if (MATCHNAME (atype_name, "int64_t"))
        {
            // use A as-is, but ensure it's positive and nonzero,
            // and in range 1 to 255
            OK (GrB_apply (A, NULL, NULL, GrB_ABS_INT64, A, NULL)) ;
            OK (GrB_apply (A, NULL, NULL, GrB_MAX_INT64, A, 1, NULL)) ;
            OK (GrB_apply (A, NULL, NULL, GrB_MIN_INT64, A, 255, NULL)) ;
        }
        else if (MATCHNAME (atype_name, "uint32_t"))
        {
            // use A as-is, but ensure it's nonzero
            // and in range 1 to 255
            OK (GrB_apply (A, NULL, NULL, GrB_MAX_UINT32, A, 1, NULL)) ;
            OK (GrB_apply (A, NULL, NULL, GrB_MIN_UINT32, A, 255, NULL)) ;
        }
        else if (MATCHNAME (atype_name, "uint64_t"))
        {
            // use A as-is, but ensure it's nonzero
            // and in range 1 to 255
            OK (GrB_apply (A, NULL, NULL, GrB_MAX_UINT64, A, 1, NULL)) ;
            OK (GrB_apply (A, NULL, NULL, GrB_MIN_UINT64, A, 255, NULL)) ;
        }
        else if (MATCHNAME (atype_name, "float"))
        {
            // use A as-is, but ensure it's positive, with values in range
            // 1 to 255
            OK (GrB_apply (A, NULL, NULL, GrB_ABS_FP32, A, NULL)) ;
            float emax = 0 ;
            OK (GrB_reduce (&emax, NULL, GrB_MAX_MONOID_FP32, A, NULL)) ;
            emax = 255. / emax ;
            OK (GrB_apply (A, NULL, NULL, GrB_TIMES_FP32, A, emax, NULL)) ;
            OK (GrB_apply (A, NULL, NULL, GrB_MAX_FP32, A, 1, NULL)) ;
        }
        else if (MATCHNAME (atype_name, "double"))
        {
            // use A as-is, but ensure it's positive, with values in range
            // 1 to 255
            OK (GrB_apply (A, NULL, NULL, GrB_ABS_FP64, A, NULL)) ;
            double emax = 0 ;
            OK (GrB_reduce (&emax, NULL, GrB_MAX_MONOID_FP64, A, NULL)) ;
            emax = 255. / emax ;
            OK (GrB_apply (A, NULL, NULL, GrB_TIMES_FP64, A, emax, NULL)) ;
            OK (GrB_apply (A, NULL, NULL, GrB_MAX_FP64, A, 1, NULL)) ;
        }
        else
        {
            // T = max (abs (double (A)), 0.1)
            OK (GrB_Matrix_new (&T, GrB_FP64, n, n)) ;
            OK (GrB_apply (T, NULL, NULL, GrB_ABS_FP64, A, NULL)) ;
            OK (GrB_apply (T, NULL, NULL, GrB_MAX_FP64, A, 0.1, NULL)) ;
            OK (GrB_free (&A)) ;
            A = T ;
        }

        // create the graph
        OK (LAGraph_New (&G, &A, kind, msg)) ;
        OK (LAGraph_CheckGraph (G, msg)) ;
        TEST_CHECK (A == NULL) ;    // A has been moved into G->A

        // find the smallest entry
        OK (LAGraph_Cached_EMin (G, msg)) ;

        // delta values to try
        int32_t Deltas [ ] = { 30, 100, 50000 } ;

        // run the SSSP
        GrB_Vector path_length = NULL ;
        int64_t step = (n > 100) ? (3*n/4) : ((n/4) + 1) ;
        for (int64_t src = 0 ; src < n ; src += step)
        {
            for (int32_t kk = 0 ; kk < ((n > 100) ? 1 : 3) ; kk++)
            {
                int32_t delta = Deltas [kk] ;
                printf ("src %d delta %d n %d\n", (int) src, delta, (int) n) ;
                OK (GrB_Scalar_setElement (Delta, delta)) ;
                OK (LAGr_SingleSourceShortestPath (&path_length, G, src,
                    Delta, msg)) ;
                int res = LG_check_sssp (path_length, G, src, msg) ;
                if (res != GrB_SUCCESS) printf ("res: %d msg: %s\n", res, msg) ;
                OK (res) ;
                OK (GrB_free(&path_length)) ;
            }
        }

        OK (LAGraph_Delete (&G, msg)) ;
    }

    GrB_free (&Delta) ;
    LAGraph_Finalize (msg) ;
}

//------------------------------------------------------------------------------
// test_SingleSourceShortestPath_failure
//------------------------------------------------------------------------------

void test_SingleSourceShortestPath_failure (void)
{
    LAGraph_Init (msg) ;
    GrB_Scalar Delta = NULL ;
    OK (GrB_Scalar_new (&Delta, GrB_INT32)) ;
    OK (GrB_Scalar_setElement (Delta, 1)) ;

    // load the karate adjacency matrix as A
    GrB_Matrix A = NULL ;
    FILE *f = fopen (LG_DATA_DIR "karate.mtx", "r") ;
    TEST_CHECK (f != NULL) ;
    OK (LAGraph_MMRead (&A, f, msg)) ;
    OK (fclose (f)) ;
    TEST_MSG ("Loading of adjacency matrix failed") ;

    // create the graph
    OK (LAGraph_New (&G, &A, LAGraph_ADJACENCY_DIRECTED, msg)) ;
    OK (LAGraph_CheckGraph (G, msg)) ;
    TEST_CHECK (A == NULL) ;    // A has been moved into G->A

    GrB_Vector path_length = NULL ;
    int result = LAGr_SingleSourceShortestPath (&path_length, G, 0, Delta, msg) ;
    printf ("\nres: %d msg: %s\n", result, msg) ;
    TEST_CHECK (path_length == NULL) ;
    TEST_CHECK (result == GrB_NOT_IMPLEMENTED) ;

    OK (GrB_Scalar_clear (Delta)) ;
    result = LAGr_SingleSourceShortestPath (&path_length, G, 0, Delta, msg) ;
    printf ("\nres: %d msg: %s\n", result, msg) ;
    TEST_CHECK (path_length == NULL) ;
    TEST_CHECK (result == GrB_EMPTY_OBJECT) ;

    OK (GrB_free (&Delta)) ;
    OK (LAGraph_Delete (&G, msg)) ;
    LAGraph_Finalize (msg) ;
}

//------------------------------------------------------------------------------
// test_SingleSourceShortestPath_brutal
//------------------------------------------------------------------------------

#if LAGRAPH_SUITESPARSE
void test_SingleSourceShortestPath_brutal (void)
{
    OK (LG_brutal_setup (msg)) ;
    GrB_Scalar Delta = NULL ;
    OK (GrB_Scalar_new (&Delta, GrB_INT32)) ;

    GrB_Matrix A = NULL, T = NULL ;

    // just test with the first 8 matrices
    for (int k = 0 ; k < 8 ; k++)
    {

        // load the adjacency matrix as A
        const char *aname = files [k].name ;
        LAGraph_Kind kind = LAGraph_ADJACENCY_DIRECTED ;
        // LAGraph_Kind kind = files [k].kind ;
        if (strlen (aname) == 0) break;
        TEST_CASE (aname) ;
        printf ("\nMatrix: %s\n", aname) ;
        snprintf (filename, LEN, LG_DATA_DIR "%s", aname) ;
        FILE *f = fopen (filename, "r") ;
        TEST_CHECK (f != NULL) ;
        OK (LAGraph_MMRead (&A, f, msg)) ;
        OK (fclose (f)) ;
        TEST_MSG ("Loading of adjacency matrix failed") ;

        GrB_Index n = 0 ;
        OK (GrB_Matrix_nrows (&n, A)) ;
        if (n > 30)
        {
            printf ("skipped -- only using small matrices for brutal test\n") ;
            OK (GrB_free (&A)) ;
            continue ;
        }

        // convert A to int32
        OK (LAGraph_Matrix_TypeName (atype_name, A, msg)) ;
        if (!MATCHNAME (atype_name, "int32_t"))
        {
            OK (GrB_Matrix_new (&T, GrB_INT32, n, n)) ;
            OK (GrB_assign (T, NULL, NULL, A, GrB_ALL, n, GrB_ALL, n, NULL)) ;
            OK (GrB_free (&A)) ;
            A = T ;
        }

        // ensure all entries are positive, and in the range 1 to 255
        OK (GrB_Matrix_apply_BinaryOp2nd_INT32 (A, NULL, NULL,
            GrB_BAND_INT32, A, 255, NULL)) ;
        int32_t x ;
        OK (GrB_reduce (&x, NULL, GrB_MIN_MONOID_INT32, A, NULL)) ;
        if (x < 1)
        {
            OK (GrB_Matrix_apply_BinaryOp2nd_INT32 (A, NULL, NULL,
                GrB_MAX_INT32, A, 1, NULL)) ;
        }

        // create the graph
        OK (LAGraph_New (&G, &A, kind, msg)) ;
        OK (LAGraph_CheckGraph (G, msg)) ;

        // run the SSSP on a single source node with one delta
        GrB_Vector path_length = NULL ;
        int64_t src = 0 ;
        int32_t delta = 30 ;
        printf ("src %d delta %d n %d\n", (int) src, delta, (int) n) ;
        OK (GrB_Scalar_setElement (Delta, delta)) ;
        LG_BRUTAL (LAGr_SingleSourceShortestPath (&path_length, G, src,
            Delta, msg)) ;
        int rr = (LG_check_sssp (path_length, G, src, msg)) ;
        printf ("rr %d msg %s\n", rr, msg) ;
        OK (rr) ;
        OK (GrB_free(&path_length)) ;

        // add a single negative edge and try again
        OK (GrB_Matrix_setElement_INT32 (G->A, -1, 0, 1)) ;
        OK (GrB_wait (G->A, GrB_MATERIALIZE)) ;
        LG_BRUTAL (LAGr_SingleSourceShortestPath (&path_length, G, 0,
            Delta, msg)) ;
        int32_t len = 0 ;
        OK (GrB_Vector_extractElement (&len, path_length, 1)) ;
        TEST_CHECK (len == -1) ;
        OK (GrB_free(&path_length)) ;

        OK (LAGraph_Delete (&G, msg)) ;
    }

    GrB_free (&Delta) ;
    OK (LG_brutal_teardown (msg)) ;
}
#endif

//****************************************************************************
//****************************************************************************

TEST_LIST = {
    {"SSSP", test_SingleSourceShortestPath},
    {"SSSP_types", test_SingleSourceShortestPath_types},
    {"SSSP_failure", test_SingleSourceShortestPath_failure},
    #if LAGRAPH_SUITESPARSE
    {"SSSP_brutal", test_SingleSourceShortestPath_brutal },
    #endif
    {NULL, NULL}
};
