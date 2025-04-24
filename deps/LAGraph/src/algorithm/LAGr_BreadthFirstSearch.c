//------------------------------------------------------------------------------
// LAGr_BreadthFirstSearch:  breadth-first search dispatch
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

// Breadth-first-search via push/pull method if using SuiteSparse:GraphBLAS
// and its GxB extensions, or a push-only method otherwise.  The former is
// much faster.

// This is an Advanced algorithm.  SuiteSparse can use a push/pull method if
// G->AT and G->out_degree are provided.  G->AT is not required if G is
// undirected.  The vanilla method is always push-only.

#include "LG_alg_internal.h"

int LAGr_BreadthFirstSearch
(
    // output:
    GrB_Vector *level,
    GrB_Vector *parent,
    // input:
    const LAGraph_Graph G,
    GrB_Index src,
    char *msg
)
{

#if LAGRAPH_SUITESPARSE
    return LG_BreadthFirstSearch_SSGrB   (level, parent, G, src, msg) ;
#else
    return LG_BreadthFirstSearch_vanilla (level, parent, G, src, msg) ;
#endif
}
