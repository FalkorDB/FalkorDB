//------------------------------------------------------------------------------
// LAGr_ConnectedComponents:  connected components of an undirected graph
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

// This is an Advanced algorithm (G->is_symmetric_structure must be known).

// Connected Components via LG_CC_FastSV6 if using SuiteSparse:GraphBLAS and
// its GxB extensions, or LG_CC_Boruvka otherwise.  The former is much faster.

#include "LG_alg_internal.h"

int LAGr_ConnectedComponents
(
    // output:
    GrB_Vector *component,  // component(i)=s if node i is in the component
                            // whose representative node is s
    // input:
    const LAGraph_Graph G,  // input graph
    char *msg
)
{

    #if LAGRAPH_SUITESPARSE
    return (LG_CC_FastSV6 (component, G, msg)) ;
    #else
    return (LG_CC_Boruvka (component, G, msg)) ;
    #endif
}
