//------------------------------------------------------------------------------
// LAGraph_SquareClustering: vertex square-clustering
//------------------------------------------------------------------------------

// LAGraph, (c) 2019-2022 by The LAGraph Contributors, All Rights Reserved.
// SPDX-License-Identifier: BSD-2-Clause
//
// For additional details (including references to third party source code and
// other files) see the LICENSE file or contact permission@sei.cmu.edu. See
// Contributors.txt for a full list of contributors. Created, in part, with
// funding and support from the U.S. Government (see Acknowledgments.txt file).
// DM22-0790

// Contributed by Erik Welch, NVIDIA.

//------------------------------------------------------------------------------

// Compute the square clustering coefficient for each node of an undirected
// graph, which is the fraction of possible squares that exist at each node.
// It is a clustering coefficient suitable for bipartite graphs and is fully
// described here:
//      https://arxiv.org/pdf/0710.0117v1.pdf
// which uses a different denominator than the original definition:
//      https://arxiv.org/pdf/cond-mat/0504241.pdf
// Furthermore, we count squares based on
//      https://arxiv.org/pdf/2007.11111.pdf (sigma_12, c_4)
// which is implemented in LAGraph_FastGraphletTransform.c (thanks Tim Davis
// for mentioning this to me!).

// The NetworkX implementation of square clustering was used heavily during
// development.  I used it to determine the contributions to the denominator
// and to verify correctness (including on larger graphs).
//      https://networkx.org/documentation/stable/reference/algorithms/\
//      generated/networkx.algorithms.cluster.square_clustering.html

// Pseudocode (doesn't show dropping 0s in the final result):
//
//    P2(~degrees.diag().S) = plus_pair(A @ A.T)
//    tri = first(P2 & A).reduce_rowwise()
//    squares = (P2 * (P2 - 1)).reduce_rowwise() / 2
//    uw_count = degrees * (degrees - 1)
//    uw_degrees = plus_times(A @ degrees) * (degrees - 1)
//    square_clustering = squares / (uw_degrees - uw_count - tri - squares)

// The coefficient as described in https://arxiv.org/pdf/0710.0117v1.pdf
// where m and n are different neighbors of node i.
// Note that summations over mn are implied in the numerator and denominator:
//
//    C_{4,mn}(i) = q_imn / ((k_m - eta_imn) + (k_n - eta_imn) + q_imn)
//    q_imn = # of common neighbors between m and n (i.e., squares)
//    k_m = number of neighbors of m (i.e., degrees[m])
//    eta_imn = 1 + q_imn + theta_mn
//    theta_mn = 1 if m and n are connected, otherwise 0 (i.e., triangles)

// Here are the corresponding terms between the equation and pseudocode:
//    theta_mn          <--> tri
//    q_imn             <--> squares
//    eta_imn = 1 + ... <--> uw_count
//    k_m               <--> uw_degrees

// I first implemented this in the Python library graphblas-algorithms
//      https://github.com/python-graphblas/graphblas-algorithms/\
//      blob/main/graphblas_algorithms/algorithms/cluster.py
// and I copy/pasted C code generated from the Recorder in Python-graphblas
//      https://github.com/python-graphblas/python-graphblas

// This implementation requires that `out_degree` property is already cached.
// 0 values are omitted from the result (i.e., missing values <--> zero).
// Also, it computes `P2 = A @ A.T`, which may be very large.  We could modify
// the algorithm to compute coefficients for a subset of nodes, which would
// allow expert users to compute in batches.  Also, since this algorithm only
// operates on undirected or symmetric graphs, we only need to compute the
// upper (or lower) triangle of P2, which should reduce memory by about half.
// However, this is not easy to do, and would complicate the implementation.

//------------------------------------------------------------------------------

#define LG_FREE_WORK                \
{                                   \
    GrB_free (&squares) ;           \
    GrB_free (&denom) ;             \
    GrB_free (&neg_denom) ;         \
    GrB_free (&P2) ;                \
    GrB_free (&D) ;                 \
}

#define LG_FREE_ALL                 \
{                                   \
    LG_FREE_WORK ;                  \
    GrB_free (&r) ;                 \
}

#include <LAGraph.h>
#include <LAGraphX.h>
#include <LG_internal.h>  // from src/utility

int LAGraph_SquareClustering
(
    // outputs:
    GrB_Vector *square_clustering,
    // inputs:
    LAGraph_Graph G,
    char *msg
)
{
    LG_CLEAR_MSG ;

    // The number of squares each node is part of
    GrB_Vector squares = NULL ;

    // Thought of as the total number of possible squares for each node
    GrB_Vector denom = NULL ;

    // Negative contributions to the denominator
    GrB_Vector neg_denom = NULL ;

    // Final result: the square coefficients for each node (squares / denom)
    GrB_Vector r = NULL ;

    // out_degrees assigned to diagonal matrix
    // Then used as triangles: first(P2 & A)
    GrB_Matrix D = NULL ;

    // P2 = plus_pair(A @ A.T).new(mask=~D.S)
    // Then used as a temporary workspace matrix (int64)
    GrB_Matrix P2 = NULL ;

    GrB_Vector deg = G->out_degree ;
    GrB_Matrix A = G->A ;
    GrB_Index n = 0 ;

    //--------------------------------------------------------------------------
    // check inputs
    //--------------------------------------------------------------------------

    LG_ASSERT (square_clustering != NULL, GrB_NULL_POINTER) ;
    (*square_clustering) = NULL ;

    LG_ASSERT_MSG (deg != NULL,
        LAGRAPH_NOT_CACHED, "G->out_degree is required") ;

    LG_TRY (LAGraph_CheckGraph (G, msg)) ;

    LG_ASSERT_MSG ((G->kind == LAGraph_ADJACENCY_UNDIRECTED ||
       (G->kind == LAGraph_ADJACENCY_DIRECTED &&
        G->is_symmetric_structure == LAGraph_TRUE)),
        LAGRAPH_SYMMETRIC_STRUCTURE_REQUIRED,
        "G->A must be known to be symmetric") ;

    // # of nodes
    GRB_TRY (GrB_Matrix_nrows (&n, A)) ;

    // out_degrees as a diagonal matrix.
    #if LAGRAPH_SUITESPARSE
        #if GxB_IMPLEMENTATION >= GxB_VERSION (7,0,0)
        // SuiteSparse 7.x and later:
        GRB_TRY (GrB_Matrix_diag(&D, deg, 0)) ;
        #else
        // SuiteSparse 6.x and earlier, which had the incorrect signature:
        GRB_TRY (GrB_Matrix_new(&D, GrB_INT64, n, n)) ;
        GRB_TRY (GrB_Matrix_diag(D, deg, 0)) ;
        #endif
    #else
    // standard GrB:
    GRB_TRY (GrB_Matrix_diag(&D, deg, 0)) ;
    #endif

    // We use ~D.S as a mask so P2 won't have values along the diagonal.
    //    P2(~D.S) = plus_pair(A @ A.T)
    GRB_TRY (GrB_Matrix_new (&P2, GrB_INT64, n, n)) ;
    GRB_TRY (GrB_mxm (P2, D, NULL, LAGraph_plus_one_int64, A, A, GrB_DESC_SCT1)) ;

    // Denominator is thought of as total number of squares that could exist.
    // It has four terms (indicated below), and we use the definition from:
    //      https://arxiv.org/pdf/0710.0117v1.pdf.
    //
    // (1) tri = first(P2 & A).reduce_rowwise()
    // Subtract 1 for each edge where u-w or w-u are connected.
    // In other words, triangles.  Use P2, since we already have it.
    //     D = first(P2 & A)
    //     neg_denom = D.reduce_rowwise()
    GRB_TRY (GrB_Matrix_eWiseMult_BinaryOp (D, NULL, NULL, GrB_FIRST_INT64, P2,
        A, NULL)) ;
    GRB_TRY (GrB_Vector_new (&neg_denom, GrB_INT64, n)) ;
    GRB_TRY (GrB_Matrix_reduce_Monoid (neg_denom, NULL, NULL,
        GrB_PLUS_MONOID_INT64, D, NULL)) ;
    GrB_free (&D) ;

    // squares = (P2 * (P2 - 1)).reduce_rowwise() / 2
    // Now compute the number of squares (the numerator).  We count squares
    // based on https://arxiv.org/pdf/2007.11111.pdf (sigma_12, c_4).
    //     P2 *= P2 - 1
    //     squares = P2.reduce_rowwise() / 2  (and drop zeros)
    GRB_TRY (GrB_Matrix_apply_BinaryOp2nd_INT64 (P2, NULL, GrB_TIMES_INT64,
        GrB_MINUS_INT64, P2, 1, NULL)) ;
    GRB_TRY (GrB_Vector_new (&squares, GrB_INT64, n)) ;
    GRB_TRY (GrB_Matrix_reduce_Monoid (squares, NULL, NULL,
        GrB_PLUS_MONOID_INT64, P2, NULL)) ;
    GrB_free (&P2) ;
    // Divide by 2, and use squares as value mask to drop zeros
    GRB_TRY (GrB_Vector_apply_BinaryOp2nd_INT64 (squares, squares, NULL,
        GrB_DIV_INT64, squares, 2, GrB_DESC_R)) ;

    // (2) uw_count = degrees * (degrees - 1).
    // Subtract 1 for each u and 1 for each w for all combos.
    //    denom(squares.S) = degrees - 1
    GRB_TRY (GrB_Vector_new (&denom, GrB_INT64, n)) ;
    GRB_TRY (GrB_Vector_apply_BinaryOp2nd_INT64(denom, squares, NULL,
        GrB_MINUS_INT64, deg, 1, GrB_DESC_S)) ;
    // neg_denom += degrees * (degrees - 1)
    GRB_TRY (GrB_Vector_eWiseMult_BinaryOp(neg_denom, NULL, GrB_PLUS_INT64,
        GrB_TIMES_INT64, deg, denom, NULL)) ;

    // (3) uw_degrees = plus_times(A @ degrees) * (degrees - 1).
    // The main contribution to (and only positive term of) the denominator:
    // degrees[u] + degrees[w] for each u-w combo.
    // Recall that `denom = degrees - 1` from above.
    //    denom(denom.S) *= plus_times(A @ deg)
    GRB_TRY (GrB_mxv(denom, denom, GrB_TIMES_INT64,
        GrB_PLUS_TIMES_SEMIRING_INT64, A, deg, GrB_DESC_S)) ;

    // (4) squares.  Subtract the number of squares
    //    denom -= neg_denom + squares
    GRB_TRY (GrB_Vector_eWiseMult_BinaryOp(denom, NULL, GrB_MINUS_INT64,
        GrB_PLUS_INT64, neg_denom, squares, NULL)) ;

    // square_clustering = squares / (uw_degrees - uw_count - tri - squares)
    // Almost done!  Now compute the final result:
    //    square_clustering = r = squares / denom
    GRB_TRY (GrB_Vector_new (&r, GrB_FP64, n)) ;
    GRB_TRY (GrB_Vector_eWiseMult_BinaryOp (r, NULL, NULL, GrB_DIV_FP64,
        squares, denom, NULL)) ;

    (*square_clustering) = r ;
    LG_FREE_WORK ;
    return (GrB_SUCCESS) ;
}
