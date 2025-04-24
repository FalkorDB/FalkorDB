//------------------------------------------------------------------------------
// LAGraph_MaximalIndependentSet: maximal independent set, with constraints
//------------------------------------------------------------------------------

// LAGraph, (c) 2019-2022 by The LAGraph Contributors, All Rights Reserved.
// SPDX-License-Identifier: BSD-2-Clause
//
// For additional details (including references to third party source code and
// other files) see the LICENSE file or contact permission@sei.cmu.edu. See
// Contributors.txt for a full list of contributors. Created, in part, with
// funding and support from the U.S. Government (see Acknowledgments.txt file).
// DM22-0790

// Modified from the GraphBLAS C API Specification, by Aydin Buluc, Timothy
// Mattson, Scott McMillan, Jose' Moreira, Carl Yang.  Based on "GraphBLAS
// Mathematics" by Jeremy Kepner.  Revised by Timothy A. Davis, Texas A&M
// University.

//------------------------------------------------------------------------------

#define LG_FREE_WORK                \
{                                   \
    GrB_free (&neighbor_max) ;      \
    GrB_free (&new_members) ;       \
    GrB_free (&new_neighbors) ;     \
    GrB_free (&candidates) ;        \
    GrB_free (&empty) ;             \
    GrB_free (&Seed) ;              \
    GrB_free (&score) ;             \
    GrB_free (&degree) ;            \
}

#define LG_FREE_ALL                 \
{                                   \
    LG_FREE_WORK ;                  \
    GrB_free (&iset) ;              \
}

#include "LG_internal.h"
#include "LAGraphX.h"

// A variant of Luby's randomized algorithm [Luby 1985].

// Given a numeric n x n adjacency matrix A of an unweighted and undirected
// graph (where the value true represents an edge), compute a maximal set of
// independent nodes and return it in a boolean n-vector, mis where
// mis[i] == true implies node i is a member of the set.

// The graph cannot have any self edges, and it must be symmetric.  Self-edges
// (diagonal entries) will cause the method to stall, and thus G->nself_edges
// must be zero on input.  G->out_degree must be present on input.  It must not
// contain any explicit zeros (this is handled by LAGraph_Cached_OutDegree).

// Singletons require special treatment.  Since they have no neighbors, their
// score is never greater than the max of their neighbors, so they never get
// selected and cause the method to stall.  To avoid this case they are removed
// from the candidate set at the begining, and added to the independent set.

int LAGraph_MaximalIndependentSet       // maximal independent set
(
    // outputs:
    GrB_Vector *mis,            // mis(i) = true if i is in the set
    // inputs:
    LAGraph_Graph G,            // input graph
    uint64_t seed,              // random number seed
    GrB_Vector ignore_node,     // if NULL, no nodes are ignored.  Otherwise
                                // ignore_node(i) = true if node i is to be
                                // ignored, and not treated as a candidate
                                // added to maximal independent set.
    char *msg
)
{

    //--------------------------------------------------------------------------
    // check inputs
    //--------------------------------------------------------------------------

    LG_CLEAR_MSG ;
    GrB_Vector iset = NULL ;            // independent set (output vector)
    GrB_Vector score = NULL ;           // random score for each node
    GrB_Vector neighbor_max = NULL ;    // value of max neighbor score
    GrB_Vector new_members = NULL ;     // set of new members to add to iset
    GrB_Vector new_neighbors = NULL ;   // new neighbors to new iset members
    GrB_Vector candidates = NULL ;      // candidate nodes
    GrB_Vector empty = NULL ;           // an empty vector
    GrB_Vector Seed = NULL ;            // random number seed vector
    GrB_Vector degree = NULL ;          // (float) G->out_degree
    GrB_Matrix A ;                      // G->A, the adjacency matrix
    GrB_Index n ;                       // # of nodes

    LG_TRY (LAGraph_CheckGraph (G, msg)) ;
    LG_ASSERT (mis != NULL, GrB_NULL_POINTER) ;

    if (G->kind == LAGraph_ADJACENCY_UNDIRECTED ||
       (G->kind == LAGraph_ADJACENCY_DIRECTED &&
        G->is_symmetric_structure == LAGraph_TRUE))
    {
        // the structure of A is known to be symmetric
        A = G->A ;
    }
    else
    {
        // A is not known to be symmetric
        LG_ASSERT_MSG (false, -105, "G->A must be symmetric") ;
    }

    LG_ASSERT_MSG (G->out_degree != NULL, -106,
        "G->out_degree must be defined") ;
    LG_ASSERT_MSG (G->nself_edges == 0, -107, "G->nself_edges must be zero") ;

    //--------------------------------------------------------------------------
    // initializations
    //--------------------------------------------------------------------------

    GRB_TRY (GrB_Matrix_nrows (&n, A)) ;
    GRB_TRY (GrB_Vector_new (&neighbor_max, GrB_FP32, n)) ;
    GRB_TRY (GrB_Vector_new (&degree, GrB_FP32, n)) ;
    GRB_TRY (GrB_Vector_new (&new_members, GrB_BOOL, n)) ;
    GRB_TRY (GrB_Vector_new (&new_neighbors, GrB_BOOL, n)) ;
    GRB_TRY (GrB_Vector_new (&candidates, GrB_BOOL, n)) ;
    GRB_TRY (GrB_Vector_new (&empty, GrB_BOOL, n)) ;
    GRB_TRY (GrB_Vector_new (&Seed, GrB_UINT64, n)) ;
    GRB_TRY (GrB_Vector_new (&score, GrB_FP32, n)) ;
    GRB_TRY (GrB_Vector_new (&iset, GrB_BOOL, n)) ;

    // degree = (float) G->out_degree
    GRB_TRY (GrB_assign (degree, NULL, NULL, G->out_degree, GrB_ALL, n, NULL)) ;

    //--------------------------------------------------------------------------
    // remove singletons (nodes of degree zero) and handle ignore_node
    //--------------------------------------------------------------------------

    GrB_Index nonsingletons = 0 ;
    GRB_TRY (GrB_Vector_nvals (&nonsingletons, degree)) ;
    if (nonsingletons == n)
    {
        if (ignore_node == NULL)
        {
            // all nodes have degree 1 or more; all nodes are candidates
            // candidates (0:n-1) = true
            GRB_TRY (GrB_assign (candidates, NULL, NULL, (bool) true, GrB_ALL,
                n, NULL)) ;
            // Seed vector starts out dense
            // Seed (0:n-1) = 0
            GRB_TRY (GrB_assign (Seed, NULL, NULL, 0, GrB_ALL, n, NULL)) ;
        }
        else
        {
            // all nodes have degree 1 or more, but some nodes are to be
            // ignored.  Use ignore_node as a valued mask.
            // candidates<!ignore_node> = true
            GRB_TRY (GrB_assign (candidates, ignore_node, NULL, (bool) true,
                GrB_ALL, n, GrB_DESC_C)) ;
            // Seed vector starts out sparse
            // Seed{candidates} = 0
            GRB_TRY (GrB_assign (Seed, candidates, NULL, 0, GrB_ALL, n,
                GrB_DESC_S)) ;
        }
    }
    else
    {
        // one or more singleton is present.
        // candidates{degree} = 1
        GRB_TRY (GrB_assign (candidates, degree, NULL, (bool) true,
            GrB_ALL, n, GrB_DESC_S)) ;
        // add all singletons to iset
        // iset{!degree} = 1
        GRB_TRY (GrB_assign (iset, degree, NULL, (bool) true, GrB_ALL, n,
            GrB_DESC_SC)) ;
        if (ignore_node != NULL)
        {
            // one or more singletons are present, and some nodes are to be
            // ignored.  The candidates are all those nodes with degree > 0
            // for which ignore_node(i) is false (or not present).  Delete
            // any candidate i for which ignore_node(i) is true.  Use
            // ignore_node as a valued mask.
            // candidates<ignore_node> = empty
            GRB_TRY (GrB_assign (candidates, ignore_node, NULL, empty,
                GrB_ALL, n, NULL)) ;
            // Delete any ignored nodes from iset
            // iset<ignore_node> = empty
            GRB_TRY (GrB_assign (iset, ignore_node, NULL, empty,
                GrB_ALL, n, NULL)) ;
        }
        // Seed vector starts out sparse
        // Seed{candidates} = 0
        GRB_TRY (GrB_assign (Seed, candidates, NULL, 0, GrB_ALL, n,
            GrB_DESC_S)) ;
    }

    // create the random number seeds
    LG_TRY (LAGraph_Random_Seed (Seed, seed, msg)) ;

    //--------------------------------------------------------------------------
    // iterate while there are candidates to check
    //--------------------------------------------------------------------------

    int nstall = 0 ;
    GrB_Index ncandidates ;
    GRB_TRY (GrB_Vector_nvals (&ncandidates, candidates)) ;
    GrB_Index last_ncandidates = ncandidates ;
    GrB_Index n1 = (GrB_Index) (0.04 * (double) n) ;
    GrB_Index n2 = (GrB_Index) (0.10 * (double) n) ;

    while (ncandidates > 0)
    {
        // compute the score for each node; scale the Seed by degree
        // score = (float) Seed
        GRB_TRY (GrB_assign (score, NULL, NULL, Seed, GrB_ALL, n, NULL)) ;
        // score = score / degree
        GRB_TRY (GrB_eWiseMult (score, NULL, NULL, GrB_DIV_FP32, score, degree,
            NULL)) ;

        // compute the max score of all candidate neighbors (only candidates
        // have a score, so non-candidate neighbors are excluded)
        // neighbor_max{candidates,replace} = score * A
        GRB_TRY (GrB_Vector_nvals (&ncandidates, candidates)) ;
        if (ncandidates < n1)
        {
            // push
            // neighbor_max'{candidates,replace} = score' * A
            GRB_TRY (GrB_vxm (neighbor_max, candidates, NULL,
                GrB_MAX_FIRST_SEMIRING_FP32, score, A, GrB_DESC_RS)) ;
        }
        else
        {
            // pull
            // neighbor_max{candidates,replace} = A * score
            GRB_TRY (GrB_mxv (neighbor_max, candidates, NULL,
                GrB_MAX_SECOND_SEMIRING_FP32, A, score, GrB_DESC_RS)) ;
        }

        // select node if its score is > than all its active neighbors
        // new_members = (score > neighbor_max) using set union so that nodes
        // with no neighbors fall through to the output, as true (since no
        // score is equal to zero).
        GRB_TRY (GrB_eWiseAdd (new_members, NULL, NULL, GrB_GT_FP32,
            score, neighbor_max, NULL)) ;

        // drop explicit zeros from new_members
        GRB_TRY (GrB_select (new_members, NULL, NULL, GrB_VALUEEQ_BOOL,
            new_members, (bool) true, NULL)) ;

        // add new members to independent set
        // iset{new_members} = true
        GRB_TRY (GrB_assign (iset, new_members, NULL, (bool) true,
            GrB_ALL, n, GrB_DESC_S)) ;

        // remove new members from set of candidates
        // candidates{new_members} = empty
        GRB_TRY (GrB_assign (candidates, new_members, NULL, empty,
            GrB_ALL, n, GrB_DESC_S)) ;

        // early exit if candidates is empty
        GRB_TRY (GrB_Vector_nvals (&ncandidates, candidates)) ;
        if (ncandidates == 0) { break ; }

        // Neighbors of new members can also be removed from candidates
        // new_neighbors{candidates,replace} = new_members * A
        GrB_Index n_new_members ;
        GRB_TRY (GrB_Vector_nvals (&n_new_members, new_members)) ;
        if (n_new_members < n2)
        {
            // push
            // new_neighbors{candidates,replace} = new_members' * A
            GRB_TRY (GrB_vxm (new_neighbors, candidates, NULL,
                LAGraph_any_one_bool, new_members, A, GrB_DESC_RS)) ;
        }
        else
        {
            // pull
            // new_neighbors{candidates,replace} = A * new_members
            GRB_TRY (GrB_mxv (new_neighbors, candidates, NULL,
                LAGraph_any_one_bool, A, new_members, GrB_DESC_RS)) ;
        }

        // remove new neighbors of new members from set of candidates
        // candidates{new_neighbors} = empty
        GRB_TRY (GrB_assign (candidates, new_neighbors, NULL, empty,
            GrB_ALL, n, GrB_DESC_S)) ;

        // sparsify the random number seeds (just keep it for each candidate)
        // Seed{candidates,replace} = Seed
        GRB_TRY (GrB_assign (Seed, candidates, NULL, Seed, GrB_ALL, n,
            GrB_DESC_RS)) ;

        // Check for stall (can only occur if the matrix has self-edges, or in
        // the exceedingly rare case that 2 nodes have the exact same score).
        // If the method happens to stall, with no nodes selected because
        // the scores happen to tie, try again with another random score.
        GRB_TRY (GrB_Vector_nvals (&ncandidates, candidates)) ;
        if (last_ncandidates == ncandidates)
        {
            // This case is nearly untestable since it can almost never occur.
            nstall++ ;
            // terminate if the method has stalled too many times
            LG_ASSERT_MSG (nstall <= 32, -111, "stall") ;
            // recreate the random number seeds with a new starting seed
            LG_TRY (LAGraph_Random_Seed (Seed, seed + nstall, msg)) ;
        }
        last_ncandidates = ncandidates ;

        // get the next random Seed vector
        LG_TRY (LAGraph_Random_Next (Seed, msg)) ;
    }

    //--------------------------------------------------------------------------
    // free workspace and return result
    //--------------------------------------------------------------------------

    GRB_TRY (GrB_wait (iset, GrB_MATERIALIZE)) ;
    (*mis) = iset ;
    LG_FREE_WORK ;
    return (GrB_SUCCESS) ;
}
