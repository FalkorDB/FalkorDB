//------------------------------------------------------------------------------
// LAGraph_TriangleCount:  triangle counting dispatch, basic API
//------------------------------------------------------------------------------

// LAGraph, (c) 2019-2022 by The LAGraph Contributors, All Rights Reserved.
// SPDX-License-Identifier: BSD-2-Clause
//
// For additional details (including references to third party source code and
// other files) see the LICENSE file or contact permission@sei.cmu.edu. See
// Contributors.txt for a full list of contributors. Created, in part, with
// funding and support from the U.S. Government (see Acknowledgments.txt file).
// DM22-0790

// Contributed by Scott McMillan, SEI Carnegie Mellon University

//------------------------------------------------------------------------------

// This is a Basic algorithm (G->nself_edges, G->out_degree,
// G->is_symmetric_structure are computed, if not present).

#define LG_FREE_ALL ;

#include <LAGraph.h>
#include "LG_internal.h"
#include "LG_alg_internal.h"

// Pick the default method with auto presort.  Compute G->nself_edges, and
// G->out_degree if needed.  Determine if G->A is symmetric, if not known.

int LAGraph_TriangleCount
(
    // output:
    uint64_t *ntriangles,   // number of triangles in G.
    // input/output:
    LAGraph_Graph G,        // graph to examine; cached properties computed.
    char *msg
)
{
    // find out if graph is symmetric, compute G->out_degree, and G->nself_edges
    LG_TRY (LAGraph_Cached_IsSymmetricStructure (G, msg)) ;
    LG_TRY (LAGraph_Cached_OutDegree (G, msg)) ;
    LG_TRY (LAGraph_Cached_NSelfEdges (G, msg)) ;

    // auto method and auto sort
    return (LAGr_TriangleCount (ntriangles, G, NULL, NULL, msg)) ;
}
