//------------------------------------------------------------------------------
// LAGraph_DeleteCached: deletes the cached properties of a graph
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

#include "LG_internal.h"

int LAGraph_DeleteCached
(
    // input/output:
    LAGraph_Graph G,    // G stays valid, only cached properties are freed
    char *msg
)
{

    //--------------------------------------------------------------------------
    // check inputs
    //--------------------------------------------------------------------------

    LG_CLEAR_MSG ;
    if (G == NULL)
    {
        // success: nothing to do
        return (GrB_SUCCESS) ;
    }

    //--------------------------------------------------------------------------
    // free all cached properties of the graph
    //--------------------------------------------------------------------------

    GRB_TRY (GrB_free (&(G->AT))) ;
    GRB_TRY (GrB_free (&(G->out_degree))) ;
    GRB_TRY (GrB_free (&(G->in_degree))) ;
    GRB_TRY (GrB_free (&(G->emin))) ;
    GRB_TRY (GrB_free (&(G->emax))) ;

    //--------------------------------------------------------------------------
    // clear the cached scalar properties of the graph
    //--------------------------------------------------------------------------

    G->is_symmetric_structure =
        (G->kind == LAGraph_ADJACENCY_UNDIRECTED)
        ? LAGraph_TRUE
        : LAGRAPH_UNKNOWN ;
    G->emin_state = LAGRAPH_UNKNOWN ;
    G->emax_state = LAGRAPH_UNKNOWN ;
    G->nself_edges = LAGRAPH_UNKNOWN ;
    return (GrB_SUCCESS) ;
}
