//------------------------------------------------------------------------------
// LAGraph_cdlp: community detection using label propagation
//------------------------------------------------------------------------------

// LAGraph, (c) 2019-2022 by The LAGraph Contributors, All Rights Reserved.
// SPDX-License-Identifier: BSD-2-Clause
//
// For additional details (including references to third party source code and
// other files) see the LICENSE file or contact permission@sei.cmu.edu. See
// Contributors.txt for a full list of contributors. Created, in part, with
// funding and support from the U.S. Government (see Acknowledgments.txt file).
// DM22-0790

// Contributed by Gabor Szarnyas and Balint Hegyi, Budapest University of
// Technology and Economics (with accented characters: G\'{a}bor Sz\'{a}rnyas
// and B\'{a}lint Hegyi, using LaTeX syntax).
// https://inf.mit.bme.hu/en/members/szarnyasg .

//------------------------------------------------------------------------------

// ## Background
//
// This function was originally written for the LDBC Graphalytics benchmark.
//
// The community detection using label propagation (CDLP) algorithm is
// defined both for directed and undirected graphs.
//
// The definition implemented here is described in the following document:
// https://ldbc.github.io/ldbc_graphalytics_docs/graphalytics_spec.pdf
//
// The algorithm is based on the one given in the following paper:
//
// Usha Raghavan, Reka Albert, and Soundar Kumara. "Near linear time algorithm
// to detect community structures in large-scale networks". In: Physical
// Review E 76.3 (2007), p. 036106, https://arxiv.org/abs/0709.2938
//
// The key idea of the algorithm is that each vertex is assigned the label
// that is most frequent among its neighbors. To allow reproducible
// experiments, the algorithm is modified to guarantee deterministic behavior:
// it always picks the smallest label in case of a tie:
//
// min ( argmax_{l} (#neighbors with label l) )
//
// In other words, we need to compute the *minimum mode value* (minmode) for
// the labels among the neighbors.
//
// For directed graphs, a label on a neighbor that is connected through both
// an outgoing and on an incoming edge counts twice:
//
// min ( argmax_{l} (#incoming neighbors with l + #outgoing neighbors with l) )
//
// ## Example (undirected)
//
// For an example, let's assume an undirected graph where vertex 1 has four
// neighbors {2, 3, 4, 5}, and the current labels in the graph are
// L = [3, 5, 4, 5, 4].
//
// In this example, the distribution of labels among the neighbors of vertex 1
// is {4 => 2, 5 => 2}, therefore, the minimum mode value is 4.
//
// Next, we capture this operation using GraphBLAS operations and
// data structures. Notice that the neighbors of vertex 1 are encoded
// as a sparse vector in the adjacency matrix:
//
// A = | 0 1 1 1 1 |
//     | 1 . . .   |
//     | 1 .       |
//     | 1 .       |
//     | 1         |
//
// To allow propagating the labels along edges, we use a diagonal matrix
// with the elements of the diagonal set to the values of L:
//
// diag(L) = | 3 0 0 0 0 |
//           | 0 5 0 0 0 |
//           | 0 0 4 0 0 |
//           | 0 0 0 5 0 |
//           | 0 0 0 0 4 |
//
// If we multiply adjacency matrix with diag(L), we get a matrix
// containing the labels of the neighbor nodes. We use the 'sel2nd' operator
// for multiplication to avoid having to lookup the value on the left.
// The conventional plus.times semiring would also work: 1 * y = sel2nd(1, y).
// Note that we multiply with a diagonal matrix so the addition operator
// is not used. In the implementation, we use "min" so the semiring is
// "min.sel2nd" on uint64 values.
//
// In the example, this gives the following:
//
// AL = A min.sel2nd diag(L) = | 0 5 4 5 4 |
//                             | 3 . . . . |
//
// ## Selecting the minimum mode value
//
// Next, we need to compute the minimum mode value for each row. As it is
// difficult to capture this operation as a monoid, we use a sort operation
// on each row. In the undirected case, we extract tuples <I, _, X> from the
// matrix, then use <I, X> for sorting. In the directed case, we extract
// tuples <I1, _, X1> and <I2, _, X2>, then use <I1+I2, X1+X2>,
// where '+' denotes concatenation. Column indices (J) are not used.
//
// The resulting two-tuples are sorted using a parallel merge sort.
// Finally, we use the sorted arrays compute the minimum mode value for each
// row.
//
// ## Fixed point
//
// At the end of each iteration, we check whether L[i-1] == L[i] and
// terminate if we reached a fixed point.
//
// ## Further optimizations
//
// A possible optimization is that the first iteration is rather trivial:
//
// * in the undirected case, each vertex gets the minimal initial label (=id)
//   of its neighbors.
// * in the directed case, each vertex gets the minimal initial label (=id)
//   of its neighbors which are doubly-linked (on an incoming and on an
//   outgoing edge). In the absence of such a neighbor, it picks the minimal
//   label of its neighbors (connected through either an incoming or through
//   an outgoing edge).

#define LG_FREE_ALL                                                     \
{                                                                       \
    LAGraph_Free ((void *) &I, NULL) ;                                  \
    LAGraph_Free ((void *) &X, NULL) ;                                  \
    LAGraph_Free ((void *) &AI, NULL) ;                                 \
    LAGraph_Free ((void *) &AJ, NULL) ;                                 \
    LAGraph_Free ((void *) &AX, NULL) ;                                 \
    LAGraph_Free ((void *) &X, NULL) ;                                  \
    LAGraph_Free ((void *) &X, NULL) ;                                  \
    GrB_free (&L) ;                                                     \
    GrB_free (&L_prev) ;                                                \
    if (sanitize) GrB_free (&S) ;                                       \
    GrB_free (&AT) ;                                                    \
}

#include <LAGraph.h>
#include <LAGraphX.h>
#include "LG_internal.h"

//****************************************************************************
int LAGraph_cdlp
(
    GrB_Vector *CDLP_handle, // output vector
    const GrB_Matrix A,      // input matrix
    bool symmetric,          // denote whether the matrix is symmetric
    bool sanitize,           // if true, ensure A is binary
    int itermax,             // max number of iterations,
    double *t,               // t [0] = sanitize time, t [1] = cdlp time,
                             // in seconds
    char *msg
)
{
    GrB_Info info;
    LG_CLEAR_MSG ;

    // Diagonal label matrix
    GrB_Matrix L = NULL;
    GrB_Matrix L_prev = NULL;
    // Source adjacency matrix
    GrB_Matrix S = NULL;
    // Transposed matrix for the unsymmetric case
    GrB_Matrix AT = NULL;
    // Result CDLP vector
    GrB_Vector CDLP = NULL;

    // Arrays holding extracted tuples if the matrix needs to be copied
    GrB_Index *AI = NULL;
    GrB_Index *AJ = NULL;
    GrB_Index *AX = NULL;
    // Arrays holding extracted tuples during the algorithm
    GrB_Index *I = NULL;
    GrB_Index *X = NULL;

    //--------------------------------------------------------------------------
    // check inputs
    //--------------------------------------------------------------------------

    if (CDLP_handle == NULL || t == NULL)
    {
        return GrB_NULL_POINTER;
    }

    //--------------------------------------------------------------------------
    // ensure input is binary and has no self-edges
    //--------------------------------------------------------------------------

    t [0] = 0;         // sanitize time
    t [1] = 0;         // CDLP time

    // n = size of A (# of nodes in the graph)
    // nz = # of non-zero elements in the matrix
    // nnz = # of non-zero elements used in the computations
    //   (twice as many for directed graphs)
    GrB_Index n, nz, nnz;
    GRB_TRY (GrB_Matrix_nrows(&n, A))
    GRB_TRY (GrB_Matrix_nvals(&nz, A))
    if (!symmetric)
    {
        nnz = 2 * nz;
    }
    else
    {
        nnz = nz;
    }

    if (sanitize)
    {
        t [0] = LAGraph_WallClockTime ( ) ;

        LAGRAPH_TRY (LAGraph_Malloc ((void **) &AI, nz, sizeof(GrB_Index),msg));
        LAGRAPH_TRY (LAGraph_Malloc ((void **) &AJ, nz, sizeof(GrB_Index),msg));
        GRB_TRY (GrB_Matrix_extractTuples_UINT64(AI, AJ, GrB_NULL, &nz, A))

        LAGRAPH_TRY (LAGraph_Calloc ((void **) &AX, nz, sizeof(GrB_Index),msg));
        GRB_TRY (GrB_Matrix_new(&S, GrB_UINT64, n, n));
        GRB_TRY (GrB_Matrix_build(S, AI, AJ, AX, nz, GrB_PLUS_UINT64));

        t [0] = LAGraph_WallClockTime ( ) - t [0] ;
    }
    else
    {
        // Use the input as-is, and assume it is UINT64(!) with no self edges.
        // Results are undefined if this condition does not hold.
        S = A;
    }

    t [1] = LAGraph_WallClockTime ( ) ;

#ifdef LAGRAPH_SUITESPARSE
    GxB_Format_Value A_format = -1, global_format = -1 ;
    GRB_TRY (GxB_get(A, GxB_FORMAT, &A_format))
    GRB_TRY (GxB_get(GxB_FORMAT, &global_format))
    if (A_format != GxB_BY_ROW || global_format != GxB_BY_ROW)
    {
        LG_FREE_ALL;
        return (GrB_INVALID_VALUE) ;
    }
#endif

    // Initialize L with diagonal elements 1..n
    LAGRAPH_TRY (LAGraph_Malloc ((void **) &I, n, sizeof (GrB_Index), msg)) ;
    LAGRAPH_TRY (LAGraph_Malloc ((void **) &X, n, sizeof (GrB_Index), msg)) ;
    for (GrB_Index i = 0; i < n; i++) {
        I[i] = i;
        X[i] = i;
    }
    GRB_TRY (GrB_Matrix_new (&L, GrB_UINT64, n, n)) ;
    GRB_TRY (GrB_Matrix_build (L, I, I, X, n, GrB_PLUS_UINT64)) ;
    LAGraph_Free ((void **) &I, NULL) ;
    LAGraph_Free ((void **) &X, NULL) ;

    // Initialize matrix for storing previous labels
    GRB_TRY (GrB_Matrix_new(&L_prev, GrB_UINT64, n, n))

    if (!symmetric)
    {
        // compute AT for the unsymmetric case as it will be used
        // to compute A' = A' min.2nd L in each iteration
        GRB_TRY (GrB_Matrix_new (&AT, GrB_UINT64, n, n)) ;
        GRB_TRY (GrB_transpose (AT, NULL, NULL, A, NULL)) ;
    }

    for (int iteration = 0; iteration < itermax; iteration++)
    {
        // Initialize data structures for extraction from 'AL_in' and (for directed graphs) 'AL_out'
        LAGRAPH_TRY (LAGraph_Malloc((void **) &I, nnz, sizeof(GrB_Index), msg));
        LAGRAPH_TRY (LAGraph_Malloc((void **) &X, nnz, sizeof(GrB_Index), msg));

        // A = A min.2nd L
        // (using the "push" (saxpy) method)
        GRB_TRY (GrB_mxm(S, GrB_NULL, GrB_NULL,
                           GrB_MIN_SECOND_SEMIRING_UINT64, S, L, NULL));
        GRB_TRY (GrB_Matrix_extractTuples_UINT64(I, GrB_NULL, X, &nz, S));

        if (!symmetric)
        {
            // A' = A' min.2nd L
            // (using the "push" (saxpy) method)
            GRB_TRY (GrB_mxm(AT, GrB_NULL, GrB_NULL,
                               GrB_MIN_SECOND_SEMIRING_UINT64, AT, L, NULL));
            GRB_TRY (GrB_Matrix_extractTuples_UINT64(&I[nz],
                                                       GrB_NULL, &X[nz], &nz, AT));
        }

        LG_msort2((int64_t *) I, (int64_t *) X, nnz, NULL);

        // save current labels for comparison by swapping L and L_prev
        GrB_Matrix L_swap = L;
        L = L_prev;
        L_prev = L_swap;

        GrB_Index mode_value = -1;
        GrB_Index mode_length = 0;
        GrB_Index run_length = 1;

        // I[k] is the current row index
        // X[k] is the current value
        // we iterate in range 1..nnz and use the last index (nnz) to process the last row of the matrix
        for (GrB_Index k = 1; k <= nnz; k++)
        {
            // check if we have a reason to recompute the mode value
            if (k == nnz           // we surpassed the last element
                || I[k-1] != I[k]  // the row index has changed
                || X[k-1] != X[k]) // the run value has changed
            {
                if (run_length > mode_length)
                {
                    mode_value = X[k-1];
                    mode_length = run_length;
                }
                run_length = 0;
            }
            run_length++;

            // check if we passed a row
            if (k == nnz           // we surpassed the last element
                || I[k-1] != I[k]) // or the row index has changed
            {
                GrB_Matrix_setElement(L, mode_value, I[k-1], I[k-1]);
                mode_length = 0;
            }
        }

        LAGraph_Free ((void **) &I, NULL) ;
        LAGraph_Free ((void **) &X, NULL) ;

        bool isequal;
        LAGraph_Matrix_IsEqual (&isequal, L_prev, L, NULL);
        if (isequal) {
            break;
        }
    }

    //--------------------------------------------------------------------------
    // extract final labels to the result vector
    //--------------------------------------------------------------------------

    GRB_TRY (GrB_Vector_new(&CDLP, GrB_UINT64, n))
    for (GrB_Index i = 0; i < n; i++)
    {
        uint64_t x;
        GRB_TRY (GrB_Matrix_extractElement(&x, L, i, i))
        GRB_TRY (GrB_Vector_setElement(CDLP, x, i))
    }

    //--------------------------------------------------------------------------
    // free workspace and return result
    //--------------------------------------------------------------------------

    (*CDLP_handle) = CDLP;
    CDLP = NULL;            // set to NULL so LG_FREE_ALL doesn't free it
    LG_FREE_ALL;

    t [1] = LAGraph_WallClockTime ( ) - t [1] ;

    return (GrB_SUCCESS);
}
