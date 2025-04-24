//------------------------------------------------------------------------------
// LAGraph/src/test/test_SampleDegree.c:  test LAGr_SampleDegree
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

//------------------------------------------------------------------------------
// global variables
//------------------------------------------------------------------------------

LAGraph_Graph G = NULL ;
char msg [LAGRAPH_MSG_LEN] ;
GrB_Matrix A = NULL ;
double mean, median ;
int ret_code ;
#define LEN 512
char filename [LEN+1] ;

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
// is_close: check whether two floats are close
//------------------------------------------------------------------------------

bool is_close (double a, double b)
{
    double abs_diff = fabs(a - b) ;
    return abs_diff < 1e-6 ;
}

//------------------------------------------------------------------------------
// test_SampleDegree:  test LAGr_SampleDegree
//------------------------------------------------------------------------------

typedef struct
{
    const char *name ;
    const double row_mean ;
    const double row_median ;
    const double col_mean ;
    const double col_median ;
    const int64_t nsamples ;
    const uint64_t seed ;
}
matrix_info ;

const matrix_info files [ ] =
{
    { "A.mtx",
        4.6, 5.0,
        4.6, 5.0,
        5, 123456 },
     { "LFAT5.mtx",
        2.2, 2.0,
        2.2, 2.0,
        5, 123456 },
     { "cover.mtx",
        1.4, 1.0,
        2.4, 3.0,
        5, 123456 },
     { "full.mtx",
        3.0, 3.0,
        3.0, 3.0,
        5, 123456 },
     { "full_symmetric.mtx",
        4.0, 4.0,
        4.0, 4.0,
        5, 123456 },
     { "karate.mtx",
        3.0, 3.0,
        3.0, 3.0,
        5, 123456 },
     // Add karate two more times to test seed and nsamples
     { "karate.mtx",
        3.46666666667, 3.0,
        3.46666666667, 3.0,
        15, 123456 },
     { "karate.mtx",
        8.4, 6.0,
        8.4, 6.0,
        5, 87654432 },
     { "ldbc-cdlp-directed-example.mtx",
        2.2, 2.0,
        1.8, 2.0,
        5, 123456 },
//   { "ldbc-directed-example-bool.mtx",
//      2.5, 3.0,
//      3.8, 3.0,
//      10, 123456 },
    { "", 0.0, 0.0, 0.0, 0.0, 1, 0 }
} ;

//-----------------------------------------------------------------------------
// test_SampleDegree
//-----------------------------------------------------------------------------

void test_SampleDegree (void)
{
    setup ( ) ;

    for (int k = 0 ; ; k++)
    {

        // load the matrix as A
        const char *aname = files [k].name ;
        if (strlen (aname) == 0) break;
        TEST_CASE (aname) ;
        printf ("\n==================== Test case: %s\n", aname) ;
        snprintf (filename, LEN, LG_DATA_DIR "%s", aname) ;
        FILE *f = fopen (filename, "r") ;
        TEST_CHECK (f != NULL) ;
        OK (LAGraph_MMRead (&A, f, msg)) ;
        OK (fclose (f)) ;
        TEST_MSG ("Loading of adjacency matrix failed") ;

        // construct the graph G with adjacency matrix A
        OK (LAGraph_New (&G, &A, LAGraph_ADJACENCY_DIRECTED, msg)) ;
        TEST_CHECK (A == NULL) ;

        // SampleDegree requires degrees to be precomputed
        ret_code = LAGr_SampleDegree (&mean, &median, G, 1,
            files [k].nsamples, files [k].seed, msg) ;
        TEST_CHECK (ret_code == LAGRAPH_NOT_CACHED) ;
        TEST_MSG ("SampleDegree without row degrees precomputed succeeded") ;

        ret_code = LAGr_SampleDegree (&mean, &median, G, 0,
            files [k].nsamples, files [k].seed, msg) ;
        TEST_CHECK (ret_code == LAGRAPH_NOT_CACHED) ;
        TEST_MSG ("SampleDegree without column degrees precomputed succeeded") ;

        // Compute and check the row samples
        OK (LAGraph_Cached_OutDegree (G, msg)) ;
        OK (LAGr_SampleDegree (&mean, &median, G, 1,
            files [k].nsamples, files [k].seed, msg)) ;

        TEST_CHECK (is_close(mean, files [k].row_mean)) ;
        TEST_MSG ("Row Mean Expected: %f", files [k].row_mean) ;
        TEST_MSG ("Row Mean Produced: %f", mean) ;

        TEST_CHECK (is_close(median, files [k].row_median)) ;
        TEST_MSG ("Row Median Expected: %f", files [k].row_median) ;
        TEST_MSG ("Row Median Produced: %f", median) ;

        // Compute the column samples
        OK (LAGraph_DeleteCached (G, msg)) ;

        OK (LAGraph_Cached_InDegree (G, msg)) ;
        OK (LAGr_SampleDegree (&mean, &median, G, 0,
            files [k].nsamples, files [k].seed, msg)) ;

        TEST_CHECK (is_close(mean, files [k].col_mean)) ;
        TEST_MSG ("Column Mean Expected: %f", files [k].col_mean) ;
        TEST_MSG ("Column Mean Produced: %f", mean) ;

        TEST_CHECK (is_close(median, files [k].col_median)) ;
        TEST_MSG ("Column Median Expected: %f", files [k].col_median) ;
        TEST_MSG ("Column Median Produced: %f", median) ;

        OK (LAGraph_Delete (&G, msg)) ;
    }

    teardown ( ) ;
}

//-----------------------------------------------------------------------------
// test_SampleDegree_brutal
//-----------------------------------------------------------------------------

#if LAGRAPH_SUITESPARSE
void test_SampleDegree_brutal (void)
{
    OK (LG_brutal_setup (msg)) ;

    for (int k = 0 ; ; k++)
    {

        // load the matrix as A
        const char *aname = files [k].name ;
        if (strlen (aname) == 0) break;
        TEST_CASE (aname) ;
        printf ("\n==================== Test case: %s\n", aname) ;
        snprintf (filename, LEN, LG_DATA_DIR "%s", aname) ;
        FILE *f = fopen (filename, "r") ;
        TEST_CHECK (f != NULL) ;
        OK (LAGraph_MMRead (&A, f, msg)) ;
        OK (fclose (f)) ;
        TEST_MSG ("Loading of adjacency matrix failed") ;

        // construct the graph G with adjacency matrix A
        OK (LAGraph_New (&G, &A, LAGraph_ADJACENCY_DIRECTED, msg)) ;
        TEST_CHECK (A == NULL) ;

        // Compute and check the row samples
        LG_BRUTAL (LAGraph_Cached_OutDegree (G, msg)) ;
        LG_BRUTAL (LAGr_SampleDegree (&mean, &median, G, 1,
            files [k].nsamples, files [k].seed, msg)) ;

        TEST_CHECK (is_close(mean, files [k].row_mean)) ;
        TEST_CHECK (is_close(median, files [k].row_median)) ;

        // Compute the column samples
        LG_BRUTAL (LAGraph_DeleteCached (G, msg)) ;

        LG_BRUTAL (LAGraph_Cached_InDegree (G, msg)) ;
        LG_BRUTAL (LAGr_SampleDegree (&mean, &median, G, 0,
            files [k].nsamples, files [k].seed, msg)) ;

        TEST_CHECK (is_close(mean, files [k].col_mean)) ;
        TEST_CHECK (is_close(median, files [k].col_median)) ;

        OK (LAGraph_Delete (&G, msg)) ;
    }

    OK (LG_brutal_teardown (msg)) ;
}
#endif

//-----------------------------------------------------------------------------
// TEST_LIST: the list of tasks for this entire test
//-----------------------------------------------------------------------------

TEST_LIST =
{
    { "SampleDegree", test_SampleDegree },
    #if LAGRAPH_SUITESPARSE
    { "SampleDegree_brutal", test_SampleDegree_brutal },
    #endif
    { NULL, NULL }
} ;
