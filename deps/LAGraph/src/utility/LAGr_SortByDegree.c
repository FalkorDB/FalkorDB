//------------------------------------------------------------------------------
// LAGr_SortByDegree: sort a graph by its row or column degree
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

// LAGr_SortByDegree computes a permutation vector P that sorts a graph
// by degree (either row or column degree of its adjacency matrix A).
// If G is undirected, or if G is directed but is known to have a symmetric
// adjacency matrix, then G->out_degree is used (and byout is ignored).
// Otherwise, if G->out_degree is used if byout is true, and G->in_degree is
// used if byout is false.

// G->out_degree or G->in_degree must first be computed.  An error is returned
// if the required degree vector has not yet been computed.  See
// LAGraph_Cached_OutDegree and LAGraph_Cached_InDegree.

// The permutation is in ascending order of degree if ascending is true, and
// in descending order otherwise.

// Ties are broken by the node id, so the sort is always predicable.  Lower
// numbered rows/columns always appear before higher ones, if they have the
// same degree.

// The output is a permutation P where P [k] = i if row i is the kth row in
// the permutation (or P [k] = j if column j is the kth column in the
// permutation, with byout false).

#define LG_FREE_WORK                    \
{                                       \
    LAGraph_Free ((void **) &W, NULL) ; \
    LAGraph_Free ((void **) &D, NULL) ; \
}

#define LG_FREE_ALL                     \
{                                       \
    LG_FREE_WORK ;                      \
    LAGraph_Free ((void **) &P, NULL) ; \
}

#include "LG_internal.h"

int LAGr_SortByDegree
(
    // output:
    int64_t **P_handle,     // permutation vector of size n
    // input:
    const LAGraph_Graph G,  // graph of n nodes
    bool byout,             // if true, sort G->out_degree, else G->in_degree
    bool ascending,         // sort in ascending or descending order
    char *msg
)
{

    //--------------------------------------------------------------------------
    // check inputs
    //--------------------------------------------------------------------------

    LG_CLEAR_MSG ;
    int64_t *P = NULL ;
    int64_t *W = NULL ;
    int64_t *D = NULL ;
    LG_ASSERT_MSG (P_handle != NULL, GrB_NULL_POINTER, "&P != NULL") ;
    (*P_handle) = NULL ;
    LG_TRY (LAGraph_CheckGraph (G, msg)) ;

    GrB_Vector Degree ;

    if (G->kind == LAGraph_ADJACENCY_UNDIRECTED ||
       (G->kind == LAGraph_ADJACENCY_DIRECTED &&
        G->is_symmetric_structure == LAGraph_TRUE))
    {
        // the structure of A is known to be symmetric
        Degree = G->out_degree ;
    }
    else
    {
        // A is not known to be symmetric
        Degree = (byout) ? G->out_degree : G->in_degree ;
    }

    LG_ASSERT_MSG (Degree != NULL, LAGRAPH_NOT_CACHED, "degree unknown") ;

    //--------------------------------------------------------------------------
    // decide how many threads to use
    //--------------------------------------------------------------------------

    GrB_Index n ;
    GRB_TRY (GrB_Vector_size (&n, Degree)) ;

    #define CHUNK (64*1024)
    int nthreads = LG_nthreads_outer * LG_nthreads_inner ;
    nthreads = LAGRAPH_MIN (nthreads, n/CHUNK) ;
    nthreads = LAGRAPH_MAX (nthreads, 1) ;

    //--------------------------------------------------------------------------
    // allocate result and workspace
    //--------------------------------------------------------------------------

    LG_TRY (LAGraph_Malloc ((void **) &P, n, sizeof (int64_t), msg)) ;
    LG_TRY (LAGraph_Malloc ((void **) &D, n, sizeof (int64_t), msg)) ;
    LG_TRY (LAGraph_Malloc ((void **) &W, 2*n, sizeof (int64_t), msg)) ;
    int64_t *W0 = W ;
    int64_t *W1 = W + n ;

    //--------------------------------------------------------------------------
    // construct the pair [D,P] to sort
    //--------------------------------------------------------------------------

    int64_t k;
    #pragma omp parallel for num_threads(nthreads) schedule(static)
    for (k = 0 ; k < n ; k++)
    {
        D [k] = 0 ;
        P [k] = k ;
    }

    // extract the degrees
    GrB_Index nvals = n ;
    GRB_TRY (GrB_Vector_extractTuples ((GrB_Index *) W0, W1, &nvals, Degree)) ;

    if (ascending)
    {
        // sort [D,P] in ascending order of degree, tie-breaking on P
        #pragma omp parallel for num_threads(nthreads) schedule(static)
        for (k = 0 ; k < nvals ; k++)
        {
            D [W0 [k]] = W1 [k] ;
        }
    }
    else
    {
        // sort [D,P] in descending order of degree, tie-breaking on P
        #pragma omp parallel for num_threads(nthreads) schedule(static)
        for (k = 0 ; k < nvals ; k++)
        {
            D [W0 [k]] = -W1 [k] ;
        }
    }

    LG_TRY (LAGraph_Free ((void **) &W, NULL)) ;

    //--------------------------------------------------------------------------
    // sort by degrees, with ties by node id
    //--------------------------------------------------------------------------

    LG_TRY (LG_msort2 (D, P, n, msg)) ;

    //--------------------------------------------------------------------------
    // free workspace and return result
    //--------------------------------------------------------------------------

    LG_FREE_WORK ;
    (*P_handle) = P ;
    return (GrB_SUCCESS) ;
}
