//------------------------------------------------------------------------------
// LAGraph/src/test/test_Betweenness.c: test cases for BC (GAP method)
// -----------------------------------------------------------------------------

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

#include <LAGraph_test.h>

#define LEN 512
char msg [LAGRAPH_MSG_LEN] ;
char filename [LEN+1] ;
LAGraph_Graph G = NULL ;

//------------------------------------------------------------------------------
// difference: compare the LAGraph and GAP results
//------------------------------------------------------------------------------

float difference (GrB_Vector bc, double *gap_result) ;

float difference (GrB_Vector bc, double *gap_result)
{
    GrB_Vector diff = NULL, gap_bc = NULL ;
    GrB_Index n = 0 ;
    OK (GrB_Vector_size (&n, bc)) ;
    OK (GrB_Vector_new (&gap_bc, GrB_FP32, n)) ;
    for (int i = 0 ; i < n ; i++)
    {
        OK (GrB_Vector_setElement_FP64 (gap_bc, gap_result [i], i)) ;
    }
    // diff = max (abs (gap_bc - bc))
    OK (GrB_Vector_new (&diff, GrB_FP32, n)) ;
    OK (GrB_eWiseAdd (diff, NULL, NULL, GrB_MINUS_FP32, gap_bc, bc,
        NULL)) ;
    OK (GrB_apply (diff, NULL, NULL, GrB_ABS_FP32, diff, NULL)) ;
    float err = 0 ;
    OK (GrB_reduce (&err, NULL, GrB_MAX_MONOID_FP32, diff, NULL)) ;
    OK (GrB_free (&diff)) ;
    OK (GrB_free (&gap_bc)) ;
    return (err) ;
}

//------------------------------------------------------------------------------
// results for karate graph
//------------------------------------------------------------------------------

// Results obtained from GAP/bc.cc, but with each source node reduce by n-1
// where n is the # of nodes in the graph, and prior to normalization.
// (The GAP benchmark results are incorrect for the 4 source nodes, since the
// score includes n-1 paths of length zero, which LAGraph excludes).

//  Read Time:           0.00770
//  Build Time:          0.00021
//  Graph has 34 nodes and 156 directed edges for degree: 4
//  Read Time:           0.00163
//  Graph has 34 nodes and 156 directed edges for degree: 4
//      a                0.00001
//  source: 6
//      b                0.00143
//      p                0.00118
//  source: 29
//      b                0.00012
//      p                0.00010
//  source: 0
//      b                0.00009
//      p                0.00007
//  source: 9
//      b                0.00010
//      p                0.00008

GrB_Index karate_sources [4] = { 6, 29, 0, 9 } ;

double karate_bc [34] = {
    43.7778,
    2.83333,
    26.9143,
    0.722222,
    0.333333,
    1.83333,
    1.5,
    0,
    9.09524,
    0,
    0,
    0,
    0,
    5.19206,
    0,
    0,
    0,
    0,
    0,
    4.58095,
    0,
    0,
    0,
    2.4,
    0,
    0.422222,
    0,
    1.28889,
    0,
    0,
    0.733333,
    14.5508,
    17.1873,
    40.6349 } ;

// Trial Time:          0.00369
// Average Time:        0.00369

//------------------------------------------------------------------------------
// results for west0067 graph
//------------------------------------------------------------------------------

// Read Time:           0.00213
// Build Time:          0.00019
// Graph has 67 nodes and 292 directed edges for degree: 4
// Read Time:           0.00158
// Graph has 67 nodes and 292 directed edges for degree: 4
    // a                0.00001
// source: 13
    // b                0.00285
    // p                0.00013
// source: 58
    // b                0.00028
    // p                0.00515
// source: 1
    // b                0.00015
    // p                0.00012
// source: 18
    // b                0.00012
    // p                0.00009

GrB_Index west0067_sources [4] = { 13, 58, 1, 18 } ;

double west0067_bc [67] = {
    7.37262,
    5.3892,
    4.53788,
    3.25952,
    11.9139,
    5.73571,
    5.65336,
    1.5,
    19.2719,
    0.343137,
    0.0833333,
    0.666667,
    1.80882,
    12.4246,
    1.92647,
    22.0458,
    4.7381,
    34.8611,
    0.1,
    29.8358,
    9.52807,
    9.71836,
    17.3334,
    54.654,
    23.3118,
    7.31765,
    2.52381,
    6.96905,
    19.2291,
    6.97003,
    33.0464,
    7.20128,
    3.78571,
    7.87698,
    15.3556,
    7.43333,
    7.19091,
    9.20411,
    1.10325,
    6.38095,
    17.808,
    5.18172,
    25.8441,
    7.91581,
    1.13501,
    0,
    2.53004,
    2.48168,
    8.84857,
    3.80708,
    1.16978,
    0.0714286,
    1.76786,
    3.06661,
    12.0742,
    1.6,
    4.73908,
    2.3701,
    3.75,
    1.08571,
    1.69697,
    0,
    0.571429,
    0,
    0,
    2.22381,
    0.659341 } ;

//  Trial Time:          0.00912
//  Average Time:        0.00912

//------------------------------------------------------------------------------
// test_bc
//------------------------------------------------------------------------------

void test_bc (void)
{
    LAGraph_Init (msg) ;
    GrB_Matrix A = NULL ;
    GrB_Vector centrality = NULL ;
    int niters = 0 ;

    // create the karate graph
    snprintf (filename, LEN, LG_DATA_DIR "%s", "karate.mtx") ;
    FILE *f = fopen (filename, "r") ;
    TEST_CHECK (f != NULL) ;
    OK (LAGraph_MMRead (&A, f, msg)) ;
    OK (fclose (f)) ;
    OK (LAGraph_New (&G, &A, LAGraph_ADJACENCY_UNDIRECTED, msg)) ;
    TEST_CHECK (A == NULL) ;    // A has been moved into G->A

    // compute its betweenness centrality
    OK (LAGr_Betweenness (&centrality, G, karate_sources, 4, msg)) ;
    printf ("\nkarate bc:\n") ;
    OK (LAGraph_Delete (&G, msg)) ;

    // compare with GAP:
    float err = difference (centrality, karate_bc) ;
    printf ("karate:   err: %e\n", err) ;
    TEST_CHECK (err < 1e-4) ;
    OK (GrB_free (&centrality)) ;

    // create the west0067 graph
    snprintf (filename, LEN, LG_DATA_DIR "%s", "west0067.mtx") ;
    f = fopen (filename, "r") ;
    TEST_CHECK (f != NULL) ;
    OK (LAGraph_MMRead (&A, f, msg)) ;
    OK (fclose (f)) ;
    OK (LAGraph_New (&G, &A, LAGraph_ADJACENCY_DIRECTED, msg)) ;
    TEST_CHECK (A == NULL) ;    // A has been moved into G->A
    OK (LAGraph_Cached_AT (G, msg)) ;

    // compute its betweenness centrality
    OK (LAGr_Betweenness (&centrality, G, west0067_sources, 4, msg)) ;
    printf ("\nwest0067 bc:\n") ;
    OK (LAGraph_Delete (&G, msg)) ;

    // compare with GAP:
    err = difference (centrality, west0067_bc) ;
    printf ("west0067: err: %e\n", err) ;
    TEST_CHECK (err < 1e-4) ;
    OK (GrB_free (&centrality)) ;

    LAGraph_Finalize (msg) ;
}

//------------------------------------------------------------------------------
// test_bc_brutal: test BetweenessCentraliy with brutal malloc debugging
//------------------------------------------------------------------------------

#if LAGRAPH_SUITESPARSE
void test_bc_brutal (void)
{
    OK (LG_brutal_setup (msg)) ;

    GrB_Matrix A = NULL ;
    GrB_Vector centrality = NULL ;
    int niters = 0 ;

    // create the karate graph
    snprintf (filename, LEN, LG_DATA_DIR "%s", "karate.mtx") ;
    FILE *f = fopen (filename, "r") ;
    TEST_CHECK (f != NULL) ;
    OK (LAGraph_MMRead (&A, f, msg)) ;
    OK (fclose (f)) ;
    OK (LAGraph_New (&G, &A, LAGraph_ADJACENCY_UNDIRECTED, msg)) ;
    TEST_CHECK (A == NULL) ;    // A has been moved into G->A
    printf ("\n") ;

    // compute its betweenness centrality
    LG_BRUTAL_BURBLE (LAGr_Betweenness (&centrality, G,
            karate_sources, 4, msg)) ;

    // compare with GAP:
    float err = difference (centrality, karate_bc) ;
    printf ("karate:   err: %e\n", err) ;
    TEST_CHECK (err < 1e-4) ;
    OK (GrB_free (&centrality)) ;
    OK (LAGraph_Delete (&G, msg)) ;

    OK (LG_brutal_teardown (msg)) ;
}
#endif

//------------------------------------------------------------------------------
// list of tests
//------------------------------------------------------------------------------

TEST_LIST = {
    {"test_bc", test_bc},
    #if LAGRAPH_SUITESPARSE
    {"test_bc_brutal", test_bc_brutal },
    #endif
    {NULL, NULL}
};
