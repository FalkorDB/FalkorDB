//------------------------------------------------------------------------------
// LAGraph_Graph_Print: print the contents of a graph
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

int LAGraph_Graph_Print
(
    // input:
    const LAGraph_Graph G,  // graph to display
    LAGraph_PrintLevel pr,  // print level (0 to 5)
    FILE *f,                // file to write to, must already be open
    char *msg
)
{

    //--------------------------------------------------------------------------
    // clear the msg and check the graph
    //--------------------------------------------------------------------------

    LG_CLEAR_MSG ;
    LG_ASSERT (f != NULL, GrB_NULL_POINTER) ;
    LG_TRY (LAGraph_CheckGraph (G, msg)) ;
    int prl = (int) pr ;
    prl = LAGRAPH_MAX (prl, 0) ;
    prl = LAGRAPH_MIN (prl, 5) ;
    if (prl == 0) return (GrB_SUCCESS) ;

    //--------------------------------------------------------------------------
    // display the primary graph components
    //--------------------------------------------------------------------------

    GrB_Matrix A = G->A ;
    LAGraph_Kind kind = G->kind ;

    GrB_Index n, nvals ;
    GRB_TRY (GrB_Matrix_nrows (&n, A)) ;
    GRB_TRY (GrB_Matrix_nvals (&nvals, A)) ;
    char typename [LAGRAPH_MAX_NAME_LEN] ;
    char kindname [LAGRAPH_MAX_NAME_LEN] ;
    LG_TRY (LAGraph_Matrix_TypeName (typename, A, msg)) ;
    LG_TRY (LG_KindName (kindname, kind, msg)) ;

    // print the basic cached scalar properties
    FPRINTF (f, "Graph: kind: %s, nodes: %g entries: %g type: %s\n",
        kindname, (double)n, (double)nvals, typename) ;

    // print the scalar cached properties
    FPRINTF (f, "  structural symmetry: ") ;
    switch (G->is_symmetric_structure)
    {
        case LAGraph_FALSE : FPRINTF (f, "unsymmetric") ; break ;
        case LAGraph_TRUE  : FPRINTF (f, "symmetric")   ; break ;
        default            : FPRINTF (f, "unknown")     ; break ;
    }
    if (G->nself_edges >= 0)
    {
        FPRINTF (f, "  self-edges: %g", (double) G->nself_edges) ;
    }
    FPRINTF (f, "\n") ;

    FPRINTF (f, "  adjacency matrix: ") ;

    LAGraph_PrintLevel pr2 = (LAGraph_PrintLevel) prl ;
    LG_TRY (LAGraph_Matrix_Print (A, pr2, stdout, msg)) ;

    //--------------------------------------------------------------------------
    // display the cached properties
    //--------------------------------------------------------------------------

    GrB_Matrix AT = G->AT ;
    if (AT != NULL)
    {
        FPRINTF (f, "  adjacency matrix transposed: ") ;
        LG_TRY (LAGraph_Matrix_Print (AT, pr2, stdout, msg)) ;
    }

    GrB_Vector out_degree = G->out_degree ;
    if (out_degree != NULL)
    {
        FPRINTF (f, "  out degree: ") ;
        LG_TRY (LAGraph_Vector_Print (out_degree, pr2, stdout, msg)) ;
    }

    GrB_Vector in_degree = G->in_degree ;
    if (in_degree != NULL)
    {
        FPRINTF (f, "  in degree: ") ;
        LG_TRY (LAGraph_Vector_Print (in_degree, pr2, stdout, msg)) ;
    }

    return (GrB_SUCCESS) ;
}
