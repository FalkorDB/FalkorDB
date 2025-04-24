//------------------------------------------------------------------------------
// LAGraph/src/test/test_minmax.c:  test LAGraph_Cached_EMin/EMax
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
#include "LG_internal.h"

//------------------------------------------------------------------------------
// global variables
//------------------------------------------------------------------------------

int status ;
GrB_Info info ;
char msg [LAGRAPH_MSG_LEN] ;
GrB_Matrix A = NULL ;
LAGraph_Graph G = NULL ;
const char *name ;
#define LEN 512
char filename [LEN+1] ;
char atype_name [LAGRAPH_MAX_NAME_LEN] ;

//------------------------------------------------------------------------------
// test matrices
//------------------------------------------------------------------------------

typedef struct
{
    double emin ;
    double emax ;
    const char *name ;
}
matrix_info ;

const matrix_info files [ ] =
{
    //              amin                amax name
    {                  1,                  1, "A2.mtx" } ,
    {                  1,                  1, "A.mtx" } ,
    {      -583929119292,      1191785641270, "bcsstk13.mtx" } ,
    {                  1,                  9, "comments_cover.mtx" } ,
    {              0.118,              0.754, "comments_full.mtx" } ,
    {          -1.863354,           1.863354, "comments_west0067.mtx" } ,
    {                  1,                  9, "cover.mtx" } ,
    {                  1,                  1, "cover_structure.mtx" } ,
    { -5679.837539484813,  4615.532487504805, "cryg2500.mtx" } ,
    {                  0,                  0, "empty.mtx" } ,
    {              0.118,              0.754, "full.mtx" } ,
    {              0.118,              0.754, "full_noheader.mtx" } ,
    { 3.9635860919952393, 28.239410400390625, "full_symmetric.mtx" } ,
    {                  1,                  1, "jagmesh7.mtx" } ,
    {                  1,                  1, "karate.mtx" } ,
    {                  1,                  1, "ldbc-cdlp-directed-example.mtx" } ,
    {                  1,                  1, "ldbc-cdlp-undirected-example.mtx" } ,
    {                  1,                  1, "ldbc-directed-example-bool.mtx" } ,
    {                0.1,               0.83, "ldbc-directed-example.mtx" } ,
    {                  1,                  1, "ldbc-directed-example-unweighted.mtx" } ,
    {                  1,                  1, "ldbc-undirected-example-bool.mtx" } ,
    {               0.12,                0.9, "ldbc-undirected-example.mtx" } ,
    {                  1,                  1, "ldbc-undirected-example-unweighted.mtx" } ,
    {                  1,                  1, "ldbc-wcc-example.mtx" } ,
    {           -6283200,           12566400, "LFAT5.mtx" } ,
    {           -6283200,           12566400, "LFAT5_two.mtx" } ,
    {              -1.06,              2.429, "lp_afiro.mtx" } ,
    {                  1,                  1, "lp_afiro_structure.mtx" } ,
    {                  0,                  1, "matrix_bool.mtx" } ,
    {          -INFINITY,           INFINITY, "matrix_fp32.mtx" } ,
    {                  1,                  1, "matrix_fp32_structure.mtx" } ,
    {          -INFINITY,           INFINITY, "matrix_fp64.mtx" } ,
    {             -32768,              32767, "matrix_int16.mtx" } ,
    {      -2147483648.0,         2147483647, "matrix_int32.mtx" } ,
    {               -128,                127, "matrix_int8.mtx" } ,
    {                  0,              65535, "matrix_uint16.mtx" } ,
    {                  0,         4294967295, "matrix_uint32.mtx" } ,
    {                  0,                255, "matrix_uint8.mtx" } ,
    {                  1,                  1, "msf1.mtx" } ,
    {                  1,                  6, "msf2.mtx" } ,
    {                  1,                  2, "msf3.mtx" } ,
    {        -45777.0931,         22888.5466, "olm1000.mtx" } ,
    {                  1,                  1, "pushpull.mtx" } ,
    {                  1,                  1, "sample2.mtx" } ,
    {                  1,                  1, "sample.mtx" } ,
    {          -INFINITY,           INFINITY, "skew_fp32.mtx" } ,
    {          -INFINITY,           INFINITY, "skew_fp64.mtx" } ,
    {             -30000,              30000, "skew_int16.mtx" } ,
    {          -30000000,           30000000, "skew_int32.mtx" } ,
    {               -125,                125, "skew_int8.mtx" } ,
    {                  1,                  7, "sources_7.mtx" } ,
    {                  1,                  1, "structure.mtx" } ,
    {                  1,                  9, "test_BF.mtx" } ,
    {                  0,                457, "test_FW_1000.mtx" } ,
    {                  1,                214, "test_FW_2003.mtx" } ,
    {                  1,               5679, "test_FW_2500.mtx" } ,
    {                  1,                  1, "tree-example.mtx" } ,
    {          -1.863354,           1.863354, "west0067_jumbled.mtx" } ,
    {          -1.863354,           1.863354, "west0067.mtx" } ,
    {          -1.863354,           1.863354, "west0067_noheader.mtx" } ,
    {                  0,       1.4055985944, "zenios.mtx" } ,
    {                  0,                  0, "" }
} ;

// additional files (cast to double is not accurate):

typedef struct
{
    int64_t emin ;
    int64_t emax ;
    const char *name ;
}
matrix_info_int64 ;

const matrix_info_int64 files_int64 [ ] =
{
    { -9223372036854775800L,  9223372036854775807L, "matrix_int64.mtx" } ,
    { -9223372036854775807L,  9223372036854775807L, "skew_int64.mtx" } ,
    {                     0,                     0, "" }
} ;

typedef struct
{
    uint64_t emin ;
    uint64_t emax ;
    const char *name ;
}
matrix_info_uint64 ;

const matrix_info_uint64 files_uint64 [ ] =
{
    {                    0, 18446744073709551615UL, "matrix_uint64.mtx" } ,
    {                    0,                      0, "" }
} ;

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
// test_minmax: read a set of matrices and compute min/max
//------------------------------------------------------------------------------

void test_minmax (void)
{

    //--------------------------------------------------------------------------
    // start up the test
    //--------------------------------------------------------------------------

    setup ( ) ;

    for (int k = 0 ; ; k++)
    {

        //----------------------------------------------------------------------
        // load in the kth file and create the graph G
        //----------------------------------------------------------------------

        const char *aname = files [k].name ;
        if (strlen (aname) == 0) break;
        TEST_CASE (aname) ;
        printf ("\n============= %2d: %s\n", k, aname) ;
        snprintf (filename, LEN, LG_DATA_DIR "%s", aname) ;
        FILE *f = fopen (filename, "rb") ;
        TEST_CHECK (f != NULL) ;
        OK (LAGraph_MMRead (&A, f, msg)) ;
        OK (fclose (f)) ;
        TEST_MSG ("Failed to load %s\n", aname) ;
        GrB_Index nvals ;
        OK (GrB_Matrix_nvals (&nvals, A)) ;
        OK (LAGraph_New (&G, &A, LAGraph_ADJACENCY_DIRECTED, msg)) ;

        //----------------------------------------------------------------------
        // compute emin and emax
        //----------------------------------------------------------------------

        OK (LAGraph_Cached_EMin (G, msg)) ;
        TEST_CHECK (G->emin_state == LAGraph_VALUE) ;
        OK (LAGraph_Cached_EMax (G, msg)) ;
        TEST_CHECK (G->emax_state == LAGraph_VALUE) ;

        //----------------------------------------------------------------------
        // check the result
        //----------------------------------------------------------------------

        double emin1 = files [k].emin ;
        double emax1 = files [k].emax ;
        double emin2 = 0 ;
        double emax2 = 0 ;

        printf ("min/max as GrB_Scalars:\n") ;
        GxB_print (G->emin, 3) ;
        GxB_print (G->emax, 3) ;

        int result ;
        result = GrB_Scalar_extractElement_FP64 (&emin2, G->emin) ;
        printf ("min: %g %g err %g\n", emin1, emin2, emin1 - emin2) ;
        if (nvals == 0)
        {
            TEST_CHECK (result == GrB_NO_VALUE) ;
        }
        else
        {
            TEST_CHECK (result == GrB_SUCCESS) ;
        }

        #if 0
        if (emin1 != emin2)
        {
            // failure on MSVC, OpenMP
            // https://github.com/DrTimothyAldenDavis/SuiteSparse/actions/runs/6763376325/job/18380420493?pr=503
            // now fixed.
            printf ("Test failure, k: %d name: %s\n", k, aname) ;
            printf ("emin1: %30.20g\n", emin1) ;
            printf ("emin2: %30.20g\n", emin2) ;
            OK (LAGraph_Matrix_Print (G->A, 5, stdout, msg)) ;

            // extract as int64:
            int64_t emin2_int64 = 0 ;
            int64_t emax2_int64 = 0 ;
            GrB_Scalar_extractElement_INT64 (&emin2_int64, G->emin) ;
            GrB_Scalar_extractElement_INT64 (&emax2_int64, G->emax) ;
            printf ("emin2 int64: %" PRId64 "\n", emin2_int64) ;
            printf ("emax2 int64: %" PRId64 "\n", emax2_int64) ;

        }
        #endif

        TEST_CHECK (emin1 == emin2) ;

        result = GrB_Scalar_extractElement_FP64 (&emax2, G->emax) ;
        printf ("max: %g %g err %g\n", emax1, emax2, emax1 - emax2) ;
        if (nvals == 0)
        {
            printf ("no entries\n") ;
            TEST_CHECK (result == GrB_NO_VALUE) ;
        }
        else
        {
            TEST_CHECK (result == GrB_SUCCESS) ;
        }
        TEST_CHECK (emax1 == emax2) ;

        OK (LAGraph_Delete (&G, msg)) ;
    }

    //--------------------------------------------------------------------------
    // finish the test
    //--------------------------------------------------------------------------

    teardown ( ) ;
}

//------------------------------------------------------------------------------
// test_minmax_int64: read a set of matrices and compute min/max
//------------------------------------------------------------------------------

void test_minmax_int64 (void)
{

    //--------------------------------------------------------------------------
    // start up the test
    //--------------------------------------------------------------------------

    setup ( ) ;

    for (int k = 0 ; ; k++)
    {

        //----------------------------------------------------------------------
        // load in the kth file and create the graph G
        //----------------------------------------------------------------------

        const char *aname = files_int64 [k].name ;
        if (strlen (aname) == 0) break;
        TEST_CASE (aname) ;
        printf ("\n============= %2d: %s\n", k, aname) ;
        snprintf (filename, LEN, LG_DATA_DIR "%s", aname) ;
        FILE *f = fopen (filename, "rb") ;
        TEST_CHECK (f != NULL) ;
        OK (LAGraph_MMRead (&A, f, msg)) ;
        OK (fclose (f)) ;
        TEST_MSG ("Failed to load %s\n", aname) ;
        GrB_Index nvals ;
        OK (GrB_Matrix_nvals (&nvals, A)) ;
        OK (LAGraph_New (&G, &A, LAGraph_ADJACENCY_DIRECTED, msg)) ;

        for (int trial = 1 ; trial <= 2 ; trial++)
        {

            //------------------------------------------------------------------
            // compute emin and emax
            //------------------------------------------------------------------

            OK (LAGraph_Cached_EMin (G, msg)) ;
            TEST_CHECK (G->emin_state == LAGraph_VALUE) ;
            OK (LAGraph_Cached_EMax (G, msg)) ;
            TEST_CHECK (G->emax_state == LAGraph_VALUE) ;

            //------------------------------------------------------------------
            // check the result
            //------------------------------------------------------------------

            int64_t emin1 = files_int64 [k].emin ;
            int64_t emax1 = files_int64 [k].emax ;
            int64_t emin2 = 0 ;
            int64_t emax2 = 0 ;

            int result ;
            result = GrB_Scalar_extractElement_INT64 (&emin2, G->emin) ;
            printf ("min (int64): %" PRId64" %" PRId64 "\n", emin1, emin2) ;
            TEST_CHECK (result == GrB_SUCCESS) ;
            TEST_CHECK (emin1 == emin2) ;

            result = GrB_Scalar_extractElement_INT64 (&emax2, G->emax) ;
            printf ("max (int64): %" PRId64" %" PRId64 "\n", emax1, emax2) ;
            TEST_CHECK (result == GrB_SUCCESS) ;
            TEST_CHECK (emax1 == emax2) ;
        }

        OK (LAGraph_Delete (&G, msg)) ;
    }

    //--------------------------------------------------------------------------
    // finish the test
    //--------------------------------------------------------------------------

    teardown ( ) ;
}

//------------------------------------------------------------------------------
// test_minmax_uint64: read a set of matrices and compute min/max
//------------------------------------------------------------------------------

void test_minmax_uint64 (void)
{

    //--------------------------------------------------------------------------
    // start up the test
    //--------------------------------------------------------------------------

    setup ( ) ;

    for (int k = 0 ; ; k++)
    {

        //----------------------------------------------------------------------
        // load in the kth file and create the graph G
        //----------------------------------------------------------------------

        const char *aname = files_uint64 [k].name ;
        if (strlen (aname) == 0) break;
        TEST_CASE (aname) ;
        printf ("\n============= %2d: %s\n", k, aname) ;
        snprintf (filename, LEN, LG_DATA_DIR "%s", aname) ;
        FILE *f = fopen (filename, "rb") ;
        TEST_CHECK (f != NULL) ;
        OK (LAGraph_MMRead (&A, f, msg)) ;
        OK (fclose (f)) ;
        TEST_MSG ("Failed to load %s\n", aname) ;
        GrB_Index nvals ;
        OK (GrB_Matrix_nvals (&nvals, A)) ;
        OK (LAGraph_New (&G, &A, LAGraph_ADJACENCY_DIRECTED, msg)) ;

        //----------------------------------------------------------------------
        // compute emin and emax
        //----------------------------------------------------------------------

        OK (LAGraph_Cached_EMin (G, msg)) ;
        TEST_CHECK (G->emin_state == LAGraph_VALUE) ;
        OK (LAGraph_Cached_EMax (G, msg)) ;
        TEST_CHECK (G->emax_state == LAGraph_VALUE) ;

        //----------------------------------------------------------------------
        // check the result
        //----------------------------------------------------------------------

        uint64_t emin1 = files_uint64 [k].emin ;
        uint64_t emax1 = files_uint64 [k].emax ;
        uint64_t emin2 = 0 ;
        uint64_t emax2 = 0 ;

        int result ;
        result = GrB_Scalar_extractElement_UINT64 (&emin2, G->emin) ;
        printf ("min (uint64): %" PRIu64" %" PRIu64 "\n", emin1, emin2) ;
        TEST_CHECK (result == GrB_SUCCESS) ;
        TEST_CHECK (emin1 == emin2) ;

        result = GrB_Scalar_extractElement_UINT64 (&emax2, G->emax) ;
        printf ("max (uint64): %" PRIu64" %" PRIu64 "\n", emax1, emax2) ;
        TEST_CHECK (result == GrB_SUCCESS) ;
        TEST_CHECK (emax1 == emax2) ;

        OK (LAGraph_Delete (&G, msg)) ;
    }

    //--------------------------------------------------------------------------
    // finish the test
    //--------------------------------------------------------------------------

    teardown ( ) ;
}

//-----------------------------------------------------------------------------
// test_minmax_failures
//-----------------------------------------------------------------------------

typedef int myint ;

void test_minmax_failures (void)
{
    setup ( ) ;
    GrB_Type MyInt ;
    OK (GrB_Type_new (&MyInt, sizeof (myint))) ;
    OK (GrB_Matrix_new (&A, MyInt, 4, 4)) ;
    OK (LAGraph_New (&G, &A, LAGraph_ADJACENCY_DIRECTED, msg)) ;
    int result = LAGraph_Cached_EMax (G, msg) ;
    printf ("\nresult: %d msg: %s\n", result, msg) ;
    TEST_CHECK (result == GrB_NOT_IMPLEMENTED) ;
    result = LAGraph_Cached_EMin (G, msg) ;
    printf ("result: %d msg: %s\n", result, msg) ;
    TEST_CHECK (result == GrB_NOT_IMPLEMENTED) ;
    OK (GrB_free (&MyInt)) ;
    OK (LAGraph_Delete (&G, msg)) ;
    teardown ( ) ;
}

//-----------------------------------------------------------------------------
// TEST_LIST: the list of tasks for this entire test
//-----------------------------------------------------------------------------

TEST_LIST =
{
    { "test_minmax", test_minmax },
    { "test_minmax_int64", test_minmax_int64 },
    { "test_minmax_uint64", test_minmax_uint64 },
    { "test_minmax_failures", test_minmax_failures },
    { NULL, NULL }
} ;
