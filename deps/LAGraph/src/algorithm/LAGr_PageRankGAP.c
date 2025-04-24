//------------------------------------------------------------------------------
// LAGr_PageRankGAP: pagerank for the GAP benchmark
//------------------------------------------------------------------------------

// LAGraph, (c) 2019-2022 by The LAGraph Contributors, All Rights Reserved.
// SPDX-License-Identifier: BSD-2-Clause
//
// For additional details (including references to third party source code and
// other files) see the LICENSE file or contact permission@sei.cmu.edu. See
// Contributors.txt for a full list of contributors. Created, in part, with
// funding and support from the U.S. Government (see Acknowledgments.txt file).
// DM22-0790

// Contributed by Timothy A. Davis and Mohsen Aznaveh, Texas A&M University

//------------------------------------------------------------------------------

// This is an Advanced algorithm (G->AT and G->out_degree are required).

// PageRank for the GAP benchmark (only).  Do not use in production.

// This algorithm follows the specification given in the GAP Benchmark Suite:
// https://arxiv.org/abs/1508.03619 which assumes that both A and A' are
// already available, as are the row and column degrees.  The GAP specification
// ignores dangling nodes (nodes with no outgoing edges, also called sinks),
// and thus shouldn't be used in production.  This method is for the GAP
// benchmark only.  See LAGr_PageRank for a method that
// handles sinks correctly.  This method does not return a centrality metric
// such that sum(centrality) is 1, if sinks are present.

// The G->AT and G->out_degree cached properties must be defined for this
// method.  If G is undirected or G->A is known to have a symmetric structure,
// then G->A is used instead of G->AT, however.

#define LG_FREE_WORK                \
{                                   \
    GrB_free (&d1) ;                \
    GrB_free (&d) ;                 \
    GrB_free (&t) ;                 \
    GrB_free (&w) ;                 \
}

#define LG_FREE_ALL                 \
{                                   \
    LG_FREE_WORK ;                  \
    GrB_free (&r) ;                 \
}

#include "LG_internal.h"

int LAGr_PageRankGAP
(
    // output:
    GrB_Vector *centrality, // centrality(i): GAP-style pagerank of node i
    int *iters,             // number of iterations taken
    // input:
    const LAGraph_Graph G,  // input graph
    float damping,          // damping factor (typically 0.85)
    float tol,              // stopping tolerance (typically 1e-4) ;
    int itermax,            // maximum number of iterations (typically 100)
    char *msg
)
{

    //--------------------------------------------------------------------------
    // check inputs
    //--------------------------------------------------------------------------

    LG_CLEAR_MSG ;
    GrB_Vector r = NULL, d = NULL, t = NULL, w = NULL, d1 = NULL ;
    LG_ASSERT (centrality != NULL, GrB_NULL_POINTER) ;
    LG_TRY (LAGraph_CheckGraph (G, msg)) ;
    GrB_Matrix AT ;
    if (G->kind == LAGraph_ADJACENCY_UNDIRECTED ||
        G->is_symmetric_structure == LAGraph_TRUE)
    {
        // A and A' have the same structure
        AT = G->A ;
    }
    else
    {
        // A and A' differ
        AT = G->AT ;
        LG_ASSERT_MSG (AT != NULL,
            LAGRAPH_NOT_CACHED, "G->AT is required") ;
    }
    GrB_Vector d_out = G->out_degree ;
    LG_ASSERT_MSG (d_out != NULL,
        LAGRAPH_NOT_CACHED, "G->out_degree is required") ;

    //--------------------------------------------------------------------------
    // initializations
    //--------------------------------------------------------------------------

    GrB_Index n ;
    (*centrality) = NULL ;
    GRB_TRY (GrB_Matrix_nrows (&n, AT)) ;

    const float scaled_damping = (1 - damping) / n ;
    const float teleport = scaled_damping ; // teleport = (1 - damping) / n
    float rdiff = 1 ;       // first iteration is always done

    // r = 1 / n
    GRB_TRY (GrB_Vector_new (&t, GrB_FP32, n)) ;
    GRB_TRY (GrB_Vector_new (&r, GrB_FP32, n)) ;
    GRB_TRY (GrB_Vector_new (&w, GrB_FP32, n)) ;
    GRB_TRY (GrB_assign (r, NULL, NULL, (float) (1.0 / n), GrB_ALL, n, NULL)) ;

    // prescale with damping factor, so it isn't done each iteration
    // d = d_out / damping ;
    GRB_TRY (GrB_Vector_new (&d, GrB_FP32, n)) ;
    GRB_TRY (GrB_apply (d, NULL, NULL, GrB_DIV_FP32, d_out, damping, NULL)) ;

    // d1 = 1 / damping
    float dmin = 1.0 / damping ;
    GRB_TRY (GrB_Vector_new (&d1, GrB_FP32, n)) ;
    GRB_TRY (GrB_assign (d1, NULL, NULL, dmin, GrB_ALL, n, NULL)) ;
    // d = max (d1, d)
    GRB_TRY (GrB_eWiseAdd (d, NULL, NULL, GrB_MAX_FP32, d1, d, NULL)) ;
    GrB_free (&d1) ;

    //--------------------------------------------------------------------------
    // pagerank iterations
    //--------------------------------------------------------------------------

    for ((*iters) = 0 ; (*iters) < itermax && rdiff > tol ; (*iters)++)
    {
        // swap t and r ; now t is the old score
        GrB_Vector temp = t ; t = r ; r = temp ;
        // w = t ./ d
        GRB_TRY (GrB_eWiseMult (w, NULL, NULL, GrB_DIV_FP32, t, d, NULL)) ;
        // r = teleport
        GRB_TRY (GrB_assign (r, NULL, NULL, teleport, GrB_ALL, n, NULL)) ;
        // r += A'*w
        GRB_TRY (GrB_mxv (r, NULL, GrB_PLUS_FP32, LAGraph_plus_second_fp32,
            AT, w, NULL)) ;
        // t -= r
        GRB_TRY (GrB_assign (t, NULL, GrB_MINUS_FP32, r, GrB_ALL, n, NULL)) ;
        // t = abs (t)
        GRB_TRY (GrB_apply (t, NULL, NULL, GrB_ABS_FP32, t, NULL)) ;
        // rdiff = sum (t)
        GRB_TRY (GrB_reduce (&rdiff, NULL, GrB_PLUS_MONOID_FP32, t, NULL)) ;
    }

    //--------------------------------------------------------------------------
    // free workspace and return result
    //--------------------------------------------------------------------------

    (*centrality) = r ;
    LG_FREE_WORK ;
    return (GrB_SUCCESS) ;
}
