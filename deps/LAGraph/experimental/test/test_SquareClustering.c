//------------------------------------------------------------------------------
// LAGraph/expirimental/test/test_SquareClustering.c: test cases for
// square clustering
// -----------------------------------------------------------------------------

// LAGraph, (c) 2019-2022 by The LAGraph Contributors, All Rights Reserved.
// SPDX-License-Identifier: BSD-2-Clause
//
// For additional details (including references to third party source code and
// other files) see the LICENSE file or contact permission@sei.cmu.edu. See
// Contributors.txt for a full list of contributors. Created, in part, with
// funding and support from the U.S. Government (see Acknowledgments.txt file).
// DM22-0790

// Contributed by Erik Welch, NVIDIA

//------------------------------------------------------------------------------

#include <stdio.h>
#include <acutest.h>

#include <LAGraphX.h>
#include <LAGraph_test.h>
#include "LG_Xtest.h"

char msg [LAGRAPH_MSG_LEN] ;

// Data from NetworkX, test_lind_square_clustering: https://github.com/networkx\
//      /networkx/blob/main/networkx/algorithms/tests/test_cluster.py
GrB_Index rows[19] = {1, 1, 1, 1, 2, 2, 3, 3, 6, 7, 6, 7, 7, 6, 6, 2, 2, 3, 3} ;
GrB_Index cols[19] = {2, 3, 6, 7, 4, 5, 4, 5, 7, 8, 8, 9, 10, 11, 12, 13, 14,
    15, 16} ;
int64_t vals[19] = {1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1} ;

GrB_Index soln_indices[8] = {1, 2, 3, 4, 5, 6, 7, 8} ;
double soln_values[8] = {
    3.0 / 43.0,
    3.0 / 17.0,
    3.0 / 17.0,
    1.0 / 3.0,
    1.0 / 3.0,
    1.0 / 27.0,
    1.0 / 27.0,
    1.0 / 5.0
};

//------------------------------------------------------------------------------
// is_close: check whether two floats are close
//------------------------------------------------------------------------------

bool is_close (double a, double b)
{
    double abs_diff = fabs(a - b) ;
    return abs_diff < 1e-6 ;
}

void test_SquareClustering (void)
{
    LAGraph_Init (msg) ;

    GrB_Matrix A = NULL ;
    LAGraph_Graph G = NULL ;

    GrB_Index n = 17;
    OK (GrB_Matrix_new (&A, GrB_INT64, n, n)) ;
    OK (GrB_Matrix_build_INT64 (A, rows, cols, vals, 19, GrB_PLUS_INT64)) ;
    // Symmetrize A
    OK (GrB_Matrix_eWiseAdd_BinaryOp(A, NULL, NULL, GrB_FIRST_INT64, A, A,
        GrB_DESC_T1));
    OK (LAGraph_New (&G, &A, LAGraph_ADJACENCY_UNDIRECTED, msg)) ;
    TEST_CHECK (A == NULL) ;

    // check for self-edges
    OK (LAGraph_Cached_NSelfEdges (G, msg)) ;
    bool sanitize = (G->nself_edges != 0) ;

    OK (LAGraph_Cached_OutDegree (G, msg)) ;

    GrB_Vector c = NULL ;
    OK (LAGraph_SquareClustering (&c, G, msg)) ;

    GrB_Index nvals;
    OK (GrB_Vector_nvals(&nvals, c)) ;
    TEST_CHECK (nvals == 8) ;

    // OK (GrB_Vector_new(&soln, GrB_FP64, n)) ;
    // OK (GrB_Vector_build_FP64(soln, soln_indices, soln_values, 8,
    //     GrB_PLUS_FP64)) ;
    double val;
    for (GrB_Index i = 0 ; i < 8 ; ++i)
    {
        GrB_Vector_extractElement_FP64(&val, c, soln_indices[i]) ;
        TEST_CHECK (is_close(val, soln_values[i])) ;
    }
    // OK (GrB_free (&soln)) ;

    OK (GrB_free (&c)) ;
    OK (LAGraph_Delete (&G, msg)) ;

    LAGraph_Finalize (msg) ;
};

TEST_LIST = {
    {"SquareClustering", test_SquareClustering},
    // {"SquareClustering_errors", test_errors},
    {NULL, NULL}
};
