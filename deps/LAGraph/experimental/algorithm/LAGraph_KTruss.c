//------------------------------------------------------------------------------
// LAGraph_KTruss: k-truss subgraph
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

// LAGraph_KTruss: k-truss subgraph.

// Given a symmetric graph A with no-self edges, LAGraph_KTruss finds the
// k-truss subgraph of A.

// The graph G must be undirected, or have an adjacency matrix with symmetric
// structure.  Only the structure of G->A is considered.  Its values are
// ignored.  G must not have any self-edges.

// The output matrix C is the k-truss subgraph of A.  Its edges are a subset of
// G->A.  Each edge in C is part of at least k-2 triangles in C.  The structure
// of C is the adjacency matrix of the k-truss subgraph of A.  The edge weights
// of C are the support of each edge.  That is, C(i,j)=nt if the edge (i,j) is
// part of nt triangles in C.  All edges in C have support of at least k-2.
// The total number of triangles in C is sum(C)/6.  C is returned as symmetric
// with a zero-free diagonal.

#define LG_FREE_ALL GrB_free (&C) ;
#include "LG_internal.h"
#include "LAGraphX.h"

//------------------------------------------------------------------------------
// LAGraph_KTruss: find the k-truss subgraph of a graph
//------------------------------------------------------------------------------

int LAGraph_KTruss              // compute the k-truss of a graph
(
    // outputs:
    GrB_Matrix *C_handle,       // output k-truss subgraph, C
    // inputs:
    LAGraph_Graph G,            // input graph
    uint32_t k,                 // find the k-truss, where k >= 3
    char *msg
)
{

    //--------------------------------------------------------------------------
    // check inputs
    //--------------------------------------------------------------------------

    LG_CLEAR_MSG ;
    GrB_Matrix C = NULL ;
    LG_ASSERT (C_handle != NULL, GrB_NULL_POINTER) ;
    (*C_handle) = NULL ;
    LG_ASSERT_MSG (k >= 3, GrB_INVALID_VALUE, "k invalid") ;
    LG_TRY (LAGraph_CheckGraph (G, msg)) ;

    if (G->kind == LAGraph_ADJACENCY_UNDIRECTED ||
       (G->kind == LAGraph_ADJACENCY_DIRECTED &&
        G->is_symmetric_structure == LAGraph_TRUE))
    {
        // the structure of A is known to be symmetric
        ;
    }
    else
    {
        // A is not known to be symmetric
        LG_ASSERT_MSG (false, -1005, "G->A must be symmetric") ;
    }

    // no self edges can be present
    LG_ASSERT_MSG (G->nself_edges == 0, -1004, "G->nself_edges must be zero") ;

    //--------------------------------------------------------------------------
    // initializations
    //--------------------------------------------------------------------------

    GrB_Index n ;
    GrB_Matrix S = G->A ;
    GRB_TRY (GrB_Matrix_nrows (&n, S)) ;
    GRB_TRY (GrB_Matrix_new (&C, GrB_UINT32, n, n)) ;
    GrB_Index nvals, nvals_last ;
    GRB_TRY (GrB_Matrix_nvals (&nvals_last, S)) ;

    //--------------------------------------------------------------------------
    // find the k-truss of G->A
    //--------------------------------------------------------------------------

    while (true)
    {
        // C{S} = S*S'
        GRB_TRY (GrB_mxm (C, S, NULL, LAGraph_plus_one_uint32, S, S,
            GrB_DESC_RST1)) ;
        // keep entries in C that are >= k-2
        GRB_TRY (GrB_select (C, NULL, NULL, GrB_VALUEGE_UINT32, C, k-2, NULL)) ;
        // return if the k-truss has been found
        GRB_TRY (GrB_Matrix_nvals (&nvals, C)) ;
        if (nvals == nvals_last)
        {
            (*C_handle) = C ;
            return (GrB_SUCCESS) ;
        }
        // advance to the next step
        nvals_last = nvals ;
        S = C ;
    }
}
