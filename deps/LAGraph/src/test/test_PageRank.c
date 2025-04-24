//------------------------------------------------------------------------------
// LAGraph/src/test/test_PageRank.c: test cases for pagerank
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
// difference: compare the LAGraph and MATLAB results
//------------------------------------------------------------------------------

float difference (GrB_Vector centrality, double *matlab_result) ;

float difference (GrB_Vector centrality, double *matlab_result)
{
    GrB_Vector diff = NULL, cmatlab = NULL ;
    GrB_Index n = 0 ;
    OK (GrB_Vector_size (&n, centrality)) ;
    OK (GrB_Vector_new (&cmatlab, GrB_FP32, n)) ;
    for (int i = 0 ; i < n ; i++)
    {
        OK (GrB_Vector_setElement_FP64 (cmatlab, matlab_result [i], i)) ;
    }
    // diff = max (abs (cmatlab - centrality))
    OK (GrB_Vector_new (&diff, GrB_FP32, n)) ;
    OK (GrB_eWiseAdd (diff, NULL, NULL, GrB_MINUS_FP32, cmatlab, centrality,
        NULL)) ;
    OK (GrB_apply (diff, NULL, NULL, GrB_ABS_FP32, diff, NULL)) ;
    float err = 0 ;
    OK (GrB_reduce (&err, NULL, GrB_MAX_MONOID_FP32, diff, NULL)) ;
    OK (GrB_free (&diff)) ;
    OK (GrB_free (&cmatlab)) ;
    return (err) ;
}

//------------------------------------------------------------------------------
// valid results for karate graph and west0067 graphs
//------------------------------------------------------------------------------

// The first two matrices have no sinks (nodes with zero outdegree) so the
// MATLAB centrality (G, 'pagerank'), LAGraph_VertextCentrality_PageRankGAP,
// and LAGr_PageRank results will be essentially the same.

// MATLAB computes in double precision, while LAGraph_*PageRank* computes in
// single precision, so the difference will be about 1e-5 or so.

double karate_rank [34] = {
    0.0970011147,
    0.0528720584,
    0.0570750515,
    0.0358615175,
    0.0219857202,
    0.0291233505,
    0.0291233505,
    0.0244945048,
    0.0297681451,
    0.0143104668,
    0.0219857202,
    0.0095668739,
    0.0146475355,
    0.0295415677,
    0.0145381625,
    0.0145381625,
    0.0167900065,
    0.0145622041,
    0.0145381625,
    0.0196092670,
    0.0145381625,
    0.0145622041,
    0.0145381625,
    0.0315206825,
    0.0210719482,
    0.0210013837,
    0.0150430281,
    0.0256382216,
    0.0195723309,
    0.0262863139,
    0.0245921424,
    0.0371606178,
    0.0716632142,
    0.1008786453 } ;

double west0067_rank [67] = {
    0.0233753869,
    0.0139102552,
    0.0123441027,
    0.0145657095,
    0.0142018541,
    0.0100791606,
    0.0128753395,
    0.0143945684,
    0.0110203141,
    0.0110525383,
    0.0119311961,
    0.0072382247,
    0.0188680398,
    0.0141596605,
    0.0174877889,
    0.0170362099,
    0.0120433909,
    0.0219844489,
    0.0195274443,
    0.0394465722,
    0.0112038726,
    0.0090174094,
    0.0140088120,
    0.0122532937,
    0.0153346283,
    0.0135241334,
    0.0158714693,
    0.0149689529,
    0.0144097230,
    0.0137583019,
    0.0314386080,
    0.0092857745,
    0.0081814168,
    0.0102137827,
    0.0096547214,
    0.0129622400,
    0.0244173417,
    0.0173963657,
    0.0127705717,
    0.0143297446,
    0.0140509341,
    0.0104117131,
    0.0173516407,
    0.0149175105,
    0.0119979624,
    0.0095043613,
    0.0153295328,
    0.0077710930,
    0.0259969472,
    0.0126926269,
    0.0088870166,
    0.0080836101,
    0.0096023576,
    0.0091000837,
    0.0246131958,
    0.0159589365,
    0.0183500031,
    0.0155811507,
    0.0157693756,
    0.0116319823,
    0.0230649292,
    0.0149070613,
    0.0157469640,
    0.0134396036,
    0.0189218603,
    0.0114528518,
    0.0223213267 } ;

// ldbc-directed-example.mtx has two sinks: nodes 3 and 9
// its pagerank must be computed with LAGr_PageRank.
double ldbc_directed_example_rank [10] = {
    0.1697481823,
    0.0361514465,
    0.1673241104,
    0.1669092572,
    0.1540948145,
    0.0361514465,
    0.0361514465,
    0.1153655134,
    0.0361514465,
    0.0819523364 } ;

//------------------------------------------------------------------------------
// tesk_ranker
//------------------------------------------------------------------------------

void test_ranker(void)
{
    LAGraph_Init (msg) ;
    GrB_Matrix A = NULL ;
    GrB_Vector centrality = NULL, cmatlab = NULL, diff = NULL ;
    int niters = 0 ;

    //--------------------------------------------------------------------------
    // karate: no sinks
    //--------------------------------------------------------------------------

    // create the karate graph
    snprintf (filename, LEN, LG_DATA_DIR "%s", "karate.mtx") ;
    FILE *f = fopen (filename, "r") ;
    TEST_CHECK (f != NULL) ;
    OK (LAGraph_MMRead (&A, f, msg)) ;
    OK (fclose (f)) ;
    OK (LAGraph_New (&G, &A, LAGraph_ADJACENCY_UNDIRECTED, msg)) ;
    TEST_CHECK (A == NULL) ;    // A has been moved into G->A
    OK (LAGraph_Cached_OutDegree (G, msg)) ;

    // compute its pagerank using the GAP method
    OK (LAGr_PageRankGAP (&centrality, &niters, G, 0.85,
        1e-4, 100, msg)) ;

    // compare with MATLAB: cmatlab = centrality (G, 'pagerank')
    float err = difference (centrality, karate_rank) ;
    float rsum = 0 ;
    OK (GrB_reduce (&rsum, NULL, GrB_PLUS_MONOID_FP32, centrality, NULL)) ;
    printf ("\nkarate:   err: %e (GAP),      sum(r): %e iters: %d\n",
        err, rsum, niters) ;
    TEST_CHECK (err < 1e-4) ;
    OK (GrB_free (&centrality)) ;

    // compute its pagerank using the standard method
    OK (LAGr_PageRank (&centrality, &niters, G, 0.85, 1e-4, 100, msg)) ;

    // compare with MATLAB: cmatlab = centrality (G, 'pagerank')
    err = difference (centrality, karate_rank) ;
    OK (GrB_reduce (&rsum, NULL, GrB_PLUS_MONOID_FP32, centrality, NULL)) ;
    printf ("karate:   err: %e (standard), sum(r): %e iters: %d\n",
        err, rsum, niters) ;
    TEST_CHECK (err < 1e-4) ;
    OK (GrB_free (&centrality)) ;

    // test for failure to converge
    int status = LAGr_PageRank (&centrality, &niters, G, 0.85, 1e-4, 2, msg) ;
    printf ("status: %d msg: %s\n", status, msg) ;
    TEST_CHECK (status == LAGRAPH_CONVERGENCE_FAILURE) ;

    OK (LAGraph_Delete (&G, msg)) ;

    //--------------------------------------------------------------------------
    // west0067: no sinks
    //--------------------------------------------------------------------------

    // create the west0067 graph
    snprintf (filename, LEN, LG_DATA_DIR "%s", "west0067.mtx") ;
    f = fopen (filename, "r") ;
    TEST_CHECK (f != NULL) ;
    OK (LAGraph_MMRead (&A, f, msg)) ;
    OK (fclose (f)) ;
    OK (LAGraph_New (&G, &A, LAGraph_ADJACENCY_DIRECTED, msg)) ;
    TEST_CHECK (A == NULL) ;    // A has been moved into G->A
    OK (LAGraph_Cached_AT (G, msg)) ;
    OK (LAGraph_Cached_OutDegree (G, msg)) ;

    // compute its pagerank using the GAP method
    OK (LAGr_PageRankGAP (&centrality, &niters, G, 0.85,
        1e-4, 100, msg)) ;

    // compare with MATLAB: cmatlab = centrality (G, 'pagerank')
    err = difference (centrality, west0067_rank) ;
    OK (GrB_reduce (&rsum, NULL, GrB_PLUS_MONOID_FP32, centrality, NULL)) ;
    printf ("west0067: err: %e (GAP),      sum(r): %e iters: %d\n",
        err, rsum, niters) ;
    TEST_CHECK (err < 1e-4) ;
    OK (GrB_free (&centrality)) ;

    // compute its pagerank using the standard method
    OK (LAGr_PageRank (&centrality, &niters, G, 0.85, 1e-4, 100, msg)) ;

    // compare with MATLAB: cmatlab = centrality (G, 'pagerank')
    err = difference (centrality, west0067_rank) ;
    printf ("west0067: err: %e (standard), sum(r): %e iters: %d\n",
        err, rsum, niters) ;
    TEST_CHECK (err < 1e-4) ;
    OK (GrB_free (&centrality)) ;

    OK (LAGraph_Delete (&G, msg)) ;

    //--------------------------------------------------------------------------
    // ldbc-directed-example: has 2 sinks
    //--------------------------------------------------------------------------

    // create the ldbc-directed-example graph
    snprintf (filename, LEN, LG_DATA_DIR "%s", "ldbc-directed-example.mtx") ;
    f = fopen (filename, "r") ;
    TEST_CHECK (f != NULL) ;
    OK (LAGraph_MMRead (&A, f, msg)) ;
    OK (fclose (f)) ;
    OK (LAGraph_New (&G, &A, LAGraph_ADJACENCY_DIRECTED, msg)) ;
    TEST_CHECK (A == NULL) ;    // A has been moved into G->A
    OK (LAGraph_Cached_AT (G, msg)) ;
    OK (LAGraph_Cached_OutDegree (G, msg)) ;

    printf ("\n=========== ldbc-directed-example, with sink nodes 3 and 9:\n") ;
    OK (LAGraph_Graph_Print (G, LAGraph_COMPLETE, stdout, msg)) ;

    // compute its pagerank using the GAP method ("bleeds" rank)
    OK (LAGr_PageRankGAP (&centrality, &niters, G, 0.85,
        1e-4, 100, msg)) ;
    err = difference (centrality, ldbc_directed_example_rank) ;
    OK (GrB_reduce (&rsum, NULL, GrB_PLUS_MONOID_FP32, centrality, NULL)) ;
    printf ("\nGAP-style page rank is expected to be wrong:\n") ;
    printf ("ldbc-directed: err: %e (GAP), sum(r): %e, niters %d\n",
        err, rsum, niters) ;
    printf ("The GAP pagerank is incorrect for this method:\n") ;
    OK (LAGraph_Vector_Print (centrality, LAGraph_COMPLETE, stdout, msg)) ;
    OK (GrB_free (&centrality)) ;

    // compute its pagerank using the standard method
    OK (LAGr_PageRank (&centrality, &niters, G, 0.85, 1e-4, 100, msg)) ;

    // compare with MATLAB: cmatlab = centrality (G, 'pagerank')
    err = difference (centrality, ldbc_directed_example_rank) ;
    OK (GrB_reduce (&rsum, NULL, GrB_PLUS_MONOID_FP32, centrality, NULL)) ;
    printf ("\nwith sinks handled properly:\n") ;
    printf ("ldbc-directed: err: %e (standard), sum(r): %e, niters %d\n",
        err, rsum, niters) ;
    TEST_CHECK (err < 1e-4) ;
    printf ("This is the correct pagerank, with sinks handled properly:\n") ;
    OK (LAGraph_Vector_Print (centrality, LAGraph_COMPLETE, stdout, msg)) ;
    OK (GrB_free (&centrality)) ;

    OK (LAGraph_Delete (&G, msg)) ;

    //--------------------------------------------------------------------------

    LAGraph_Finalize (msg) ;
}

//------------------------------------------------------------------------------
// list of tests
//------------------------------------------------------------------------------

TEST_LIST = {
    {"test_ranker", test_ranker},
    {NULL, NULL}
};
