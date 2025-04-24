//----------------------------------------------------------------------------
// LAGraph/src/test/test_cdlp.c: test cases for CDLP
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

// todo: write a simple cdlp method, as LG_check_cdlp, and compare its results
// with LAGraph_cdlp

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

double cdlp_jagmesh7 [1138] = {
    0, 0, 0, 2, 2, 2, 2, 2, 8, 8, 8, 8, 8, 8, 14, 14, 0, 0, 0, 2, 2, 2, 14,
    2, 8, 14, 8, 8, 0, 0, 29, 29, 29, 29, 29, 35, 35, 35, 2, 0, 29, 29, 29,
    2, 2, 35, 2, 35, 2, 0, 50, 50, 50, 50, 50, 50, 56, 56, 56, 29, 0, 50, 50,
    50, 29, 29, 56, 29, 56, 29, 50, 71, 71, 71, 71, 71, 71, 77, 77, 77, 29,
    56, 71, 71, 71, 56, 56, 77, 56, 77, 56, 50, 92, 92, 92, 92, 96, 92, 98,
    98, 98, 71, 50, 92, 92, 92, 71, 71, 98, 71, 98, 71, 96, 113, 113, 113,
    113, 113, 113, 119, 119, 119, 74, 98, 113, 113, 113, 98, 98, 119, 98, 119,
    98, 96, 96, 134, 134, 134, 134, 134, 140, 140, 140, 113, 96, 134, 134, 134,
    113, 113, 140, 113, 140, 113, 96, 155, 155, 155, 155, 155, 155, 161, 161,
    161, 134, 96, 155, 155, 155, 134, 134, 161, 134, 161, 134, 155, 176, 176,
    176, 176, 176, 176, 182, 182, 182, 134, 161, 176, 176, 176, 161, 161, 182,
    161, 182, 161, 155, 197, 197, 197, 197, 197, 197, 203, 203, 203, 176, 155,
    197, 197, 197, 176, 176, 203, 176, 203, 176, 197, 218, 218, 218, 218, 222,
    218, 224, 224, 224, 176, 203, 218, 218, 218, 203, 203, 224, 203, 224, 203,
    222, 239, 239, 239, 239, 239, 239, 245, 245, 180, 180, 224, 239, 239, 239,
    224, 224, 245, 224, 245, 180, 222, 222, 260, 260, 260, 260, 260, 266, 266,
    266, 239, 222, 260, 260, 260, 239, 239, 266, 239, 266, 239, 222, 281, 281,
    281, 281, 281, 281, 287, 287, 287, 260, 222, 281, 281, 281, 260, 260, 287,
    260, 287, 260, 281, 302, 302, 302, 302, 302, 302, 308, 308, 308, 260, 287,
    302, 302, 302, 287, 287, 308, 287, 308, 287, 302, 323, 323, 323, 323, 323,
    323, 329, 329, 260, 260, 260, 308, 308, 308, 329, 308, 323, 329, 323, 323,
    302, 344, 344, 344, 344, 344, 344, 350, 350, 350, 323, 302, 344, 344, 344,
    323, 323, 350, 323, 350, 323, 239, 365, 365, 365, 365, 365, 365, 371, 371,
    180, 180, 180, 245, 245, 245, 371, 245, 365, 371, 365, 365, 260, 386, 386,
    386, 386, 386, 386, 392, 392, 392, 239, 266, 386, 386, 386, 266, 266, 392,
    266, 392, 266, 239, 407, 407, 407, 407, 407, 407, 413, 413, 413, 365, 239,
    407, 407, 407, 365, 365, 413, 365, 413, 365, 386, 428, 428, 428, 407, 239,
    392, 392, 392, 407, 392, 428, 407, 428, 407, 344, 443, 443, 443, 443, 443,
    443, 449, 449, 449, 323, 350, 350, 350, 350, 449, 350, 443, 449, 443, 443,
    344, 464, 464, 464, 464, 464, 464, 470, 470, 470, 443, 344, 464, 464, 464,
    443, 443, 470, 443, 470, 443, 464, 485, 485, 485, 485, 485, 485, 491, 491,
    491, 443, 470, 470, 470, 470, 491, 470, 485, 491, 485, 485, 464, 506, 506,
    506, 506, 506, 506, 512, 512, 512, 485, 485, 485, 485, 464, 512, 485, 506,
    512, 506, 506, 506, 527, 527, 527, 527, 531, 527, 533, 533, 533, 485, 512,
    512, 512, 512, 533, 512, 527, 533, 527, 527, 547, 547, 547, 533, 533, 485,
    531, 554, 554, 547, 547, 547, 533, 533, 533, 554, 533, 533, 554, 533, 533,
    386, 569, 569, 547, 547, 547, 574, 574, 574, 407, 386, 569, 569, 547, 428,
    428, 574, 428, 574, 428, 531, 531, 589, 411, 411, 411, 574, 574, 547, 589,
    554, 554, 589, 554, 531, 531, 531, 604, 604, 604, 604, 604, 610, 610, 411,
    411, 411, 589, 589, 531, 610, 589, 604, 610, 604, 604, 531, 625, 625, 625,
    625, 625, 625, 631, 631, 631, 604, 604, 604, 604, 531, 631, 604, 625, 631,
    625, 625, 625, 646, 646, 646, 646, 646, 646, 652, 652, 652, 604, 631, 631,
    631, 631, 652, 631, 646, 652, 646, 646, 666, 666, 666, 652, 652, 604, 646,
    673, 673, 666, 666, 666, 652, 652, 652, 673, 652, 652, 673, 652, 652, 646,
    688, 688, 688, 688, 692, 688, 694, 694, 666, 666, 666, 673, 673, 646, 694,
    673, 688, 694, 688, 688, 708, 708, 708, 694, 666, 666, 692, 715, 715, 708,
    708, 708, 694, 694, 666, 715, 694, 694, 715, 694, 694, 729, 729, 729, 715,
    708, 708, 692, 692, 736, 729, 729, 729, 715, 715, 708, 736, 715, 715, 736,
    715, 692, 729, 729, 751, 751, 751, 751, 751, 757, 757, 708, 708, 729, 751,
    751, 751, 731, 731, 757, 731, 757, 708, 771, 771, 771, 773, 773, 751, 729,
    729, 778, 771, 771, 771, 751, 751, 751, 778, 751, 751, 778, 751, 729, 771,
    771, 793, 793, 793, 793, 793, 799, 799, 799, 751, 771, 793, 793, 793, 773,
    773, 799, 773, 799, 773, 113, 814, 814, 814, 793, 771, 819, 819, 113, 113,
    113, 814, 793, 793, 819, 793, 793, 819, 793, 771, 134, 834, 834, 834, 793,
    140, 140, 140, 140, 814, 140, 834, 814, 834, 814, 113, 849, 849, 849, 849,
    849, 849, 855, 855, 855, 74, 74, 119, 119, 119, 855, 119, 849, 855, 849,
    849, 849, 870, 870, 29, 29, 29, 77, 77, 77, 870, 77, 855, 870, 855, 855, 2,
    885, 885, 885, 885, 889, 885, 891, 891, 891, 8, 8, 8, 8, 8, 891, 8, 885,
    891, 885, 885, 889, 906, 906, 906, 906, 906, 906, 912, 912, 912, 8, 891,
    891, 891, 891, 912, 891, 906, 912, 906, 906, 889, 889, 927, 927, 927, 927,
    927, 933, 933, 933, 906, 889, 927, 927, 927, 906, 906, 933, 906, 933, 906,
    889, 948, 948, 948, 948, 948, 948, 954, 954, 954, 927, 889, 948, 948, 948,
    927, 927, 954, 927, 954, 927, 927, 969, 969, 969, 969, 969, 969, 975, 975,
    975, 906, 933, 933, 933, 933, 975, 933, 969, 975, 969, 969, 948, 990, 990,
    990, 990, 994, 990, 996, 996, 996, 927, 954, 990, 990, 990, 954, 954, 996,
    954, 996, 954, 927, 1011, 1011, 1011, 1011, 1011, 1011, 1017, 1017, 1017,
    969, 927, 1011, 1011, 1011, 969, 969, 1017, 969, 1017, 969, 994, 1032,
    1032, 1032, 1011, 927, 996, 996, 996, 1011, 996, 1032, 1011, 1032, 1011,
    994, 994, 1047, 1047, 1047, 1047, 1047, 1053, 1053, 1053, 1011, 1032, 1032,
    1032, 994, 1053, 1032, 1047, 1053, 1047, 1047, 994, 1068, 1068, 1068, 1068,
    1068, 1068, 1074, 1074, 1074, 1047, 994, 1068, 1068, 1068, 1047, 1047,
    1074, 1047, 1074, 1047, 1068, 1089, 1089, 729, 729, 729, 1094, 1094, 1094,
    1047, 1074, 1089, 1089, 729, 1074, 1074, 1094, 1074, 1094, 1074, 1068,
    1109, 1109, 771, 771, 1068, 1109, 1109, 771, 1089, 778, 778, 1089, 778,
    729, 692, 1124, 1124, 1047, 1047, 729, 736, 736, 692, 1094, 736, 1124,
    1094, 1124, 1047 } ;

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
void test_cdlp (void)
{
    LAGraph_Init (msg) ;

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

        // compute the communities with LAGraph_cdlp
        OK (LAGraph_cdlp (&c, G->A, symmetric, sanitize, 100, t, msg)) ;

        GrB_Index n ;
        OK (GrB_Vector_size (&n, c)) ;
        LAGraph_PrintLevel pr = (n <= 100) ? LAGraph_COMPLETE : LAGraph_SHORT ;

        // check result c for jagmesh7
        if (strcmp (aname, "jagmesh7.mtx") == 0)
        {
            GrB_Vector cgood = NULL ;
            OK (GrB_Vector_new (&cgood, GrB_UINT64, n)) ;
            for (int k = 0 ; k < n ; k++)
            {
                OK (GrB_Vector_setElement (cgood, cdlp_jagmesh7 [k], k)) ;
            }
            OK (GrB_wait (cgood, GrB_MATERIALIZE)) ;
            printf ("\ncdlp (known result):\n") ;
            OK (LAGraph_Vector_Print (cgood, pr, stdout, msg)) ;
            bool ok = false ;
            OK (LAGraph_Vector_IsEqual (&ok, c, cgood, msg)) ;
            TEST_CHECK (ok) ;
            OK (GrB_free (&cgood)) ;
        }

        printf ("\ncdlp:\n") ;
        OK (LAGraph_Vector_Print (c, pr, stdout, msg)) ;
        OK (GrB_free (&c)) ;

        OK (LAGraph_Delete (&G, msg)) ;
    }

    LAGraph_Finalize (msg) ;
}

//------------------------------------------------------------------------------
// test_errors
//------------------------------------------------------------------------------

void test_errors (void)
{
    LAGraph_Init (msg) ;

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
    int result = LAGraph_cdlp (NULL, G->A, true, false, 100, t, msg) ;
    printf ("\nresult: %d\n", result) ;
    TEST_CHECK (result == GrB_NULL_POINTER) ;

    #if LAGRAPH_SUITESPARSE
    // G->A is held by column
    OK (GxB_set (G->A, GxB_FORMAT, GxB_BY_COL)) ;
    result = LAGraph_cdlp (&c, G->A, true, true, 100, t, msg) ;
    printf ("\nresult: %d\n", result) ;
    TEST_CHECK (result == GrB_INVALID_VALUE) ;
    TEST_CHECK (c == NULL) ;
    #endif

    OK (LAGraph_Delete (&G, msg)) ;
    LAGraph_Finalize (msg) ;
}

//****************************************************************************

TEST_LIST = {
    {"cdlp", test_cdlp},
    {"cdlp_errors", test_errors},
    {NULL, NULL}
};
