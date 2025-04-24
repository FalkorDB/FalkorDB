//------------------------------------------------------------------------------
// LAGraph_New:  create a new graph
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

// If succesful, the matrix A is moved into G->A, and the caller's A is set
// to NULL.

#include "LG_internal.h"

int LAGraph_New
(
    // output:
    LAGraph_Graph *G,   // the graph to create, NULL if failure
    // input/output:
    GrB_Matrix    *A,   // the adjacency matrix of the graph, may be NULL.
                        // A is moved into G as G->A, and A itself is set
                        // to NULL to denote that is now a part of G.
                        // That is, { G->A = A ; A = NULL ; } is performed.
                        // When G is deleted, G->A is freed.  If A is NULL,
                        // the graph is invalid until G->A is set.
    // input:
    LAGraph_Kind kind,  // the kind of graph. This may be LAGRAPH_UNKNOWN,
                        // which must then be revised later before the
                        // graph can be used.
    char *msg
)
{

    //--------------------------------------------------------------------------
    // check inputs
    //--------------------------------------------------------------------------

    LG_CLEAR_MSG ;
    LG_ASSERT (G != NULL, GrB_NULL_POINTER) ;

    //--------------------------------------------------------------------------
    // allocate the graph
    //--------------------------------------------------------------------------

    LG_TRY (LAGraph_Malloc ((void **) G, 1,
        sizeof (struct LAGraph_Graph_struct), msg)) ;

    //--------------------------------------------------------------------------
    // initialize its members
    //--------------------------------------------------------------------------

    (*G)->A = NULL ;
    (*G)->kind = LAGraph_KIND_UNKNOWN ;
    (*G)->AT = NULL ;
    (*G)->out_degree = NULL ;
    (*G)->in_degree = NULL ;
    (*G)->is_symmetric_structure = LAGRAPH_UNKNOWN ;
    (*G)->nself_edges = LAGRAPH_UNKNOWN ;
    (*G)->emin = NULL ;
    (*G)->emin_state = LAGRAPH_UNKNOWN ;
    (*G)->emax = NULL ;
    (*G)->emax_state = LAGRAPH_UNKNOWN ;

    //--------------------------------------------------------------------------
    // assign its primary components
    //--------------------------------------------------------------------------

    if ((A != NULL) && (*A != NULL))
    {
        // move &A into the graph and set &A to NULL to denote to the caller
        // that it is now a component of G.  The graph G is not opaque, so the
        // caller can get A back with A = G->A, but this helps with memory
        // management, since LAGraph_Delete (&G,msg) frees G->A, and if the
        // caller also does GrB_free (&A), a double-free would occur if this
        // move does not set A to NULL.
        (*G)->A = (*A) ;
        (*A) = NULL ;

        (*G)->kind = kind ;
        (*G)->is_symmetric_structure =
            (kind == LAGraph_ADJACENCY_UNDIRECTED)
            ? LAGraph_TRUE
            : LAGRAPH_UNKNOWN ;
    }

    return (GrB_SUCCESS) ;
}
