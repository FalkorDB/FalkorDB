//------------------------------------------------------------------------------
// LG_BreadthFirstSearch_vanilla:  BFS using only GraphBLAS API
//------------------------------------------------------------------------------

// LAGraph, (c) 2019-2022 by The LAGraph Contributors, All Rights Reserved.
// SPDX-License-Identifier: BSD-2-Clause
//
// For additional details (including references to third party source code and
// other files) see the LICENSE file or contact permission@sei.cmu.edu. See
// Contributors.txt for a full list of contributors. Created, in part, with
// funding and support from the U.S. Government (see Acknowledgments.txt file).
// DM22-0790

// Contributed by Scott McMillan, derived from examples in the appendix of
// The GraphBLAS C API Specification, v1.3.0

//------------------------------------------------------------------------------

// This is a Basic algorithm (no extra cached properties are required),
// but it is not user-callable (see LAGr_BreadthFirstSearch instead).

#define LG_FREE_WORK        \
{                           \
    GrB_free (&frontier);   \
}

#define LG_FREE_ALL         \
{                           \
    LG_FREE_WORK ;          \
    GrB_free (&l_parent);   \
    GrB_free (&l_level);    \
}

#include "LG_internal.h"

int LG_BreadthFirstSearch_vanilla
(
    GrB_Vector    *level,
    GrB_Vector    *parent,
    const LAGraph_Graph G,
    GrB_Index      src,
    char          *msg
)
{

    //--------------------------------------------------------------------------
    // check inputs
    //--------------------------------------------------------------------------

    LG_CLEAR_MSG ;
    GrB_Vector frontier = NULL;     // the current frontier
    GrB_Vector l_parent = NULL;     // parent vector
    GrB_Vector l_level = NULL;      // level vector

    bool compute_level  = (level != NULL);
    bool compute_parent = (parent != NULL);
    if (compute_level ) (*level ) = NULL;
    if (compute_parent) (*parent) = NULL;
    LG_ASSERT_MSG (compute_level || compute_parent, GrB_NULL_POINTER,
        "either level or parent must be non-NULL") ;

    LG_TRY (LAGraph_CheckGraph (G, msg)) ;

    //--------------------------------------------------------------------------
    // get the problem size
    //--------------------------------------------------------------------------

    GrB_Matrix A = G->A ;

    GrB_Index n;
    GRB_TRY( GrB_Matrix_nrows (&n, A) );
    LG_ASSERT_MSG (src < n, GrB_INVALID_INDEX, "invalid source node") ;

    // determine the semiring type
    GrB_Type     int_type  = (n > INT32_MAX) ? GrB_INT64 : GrB_INT32 ;
    GrB_BinaryOp
        second_op = (n > INT32_MAX) ? GrB_SECOND_INT64 : GrB_SECOND_INT32 ;
    GrB_Semiring semiring  = NULL;
    GrB_IndexUnaryOp ramp = NULL ;

    if (compute_parent)
    {
        // create the parent vector.  l_parent(i) is the parent id of node i
        GRB_TRY (GrB_Vector_new(&l_parent, int_type, n)) ;

        semiring = (n > INT32_MAX) ?
            GrB_MIN_FIRST_SEMIRING_INT64 : GrB_MIN_FIRST_SEMIRING_INT32;

        // create a sparse integer vector frontier, and set frontier(src) = src
        GRB_TRY (GrB_Vector_new(&frontier, int_type, n)) ;
        GRB_TRY (GrB_Vector_setElement(frontier, src, src)) ;

        // pick the ramp operator
        ramp = (n > INT32_MAX) ? GrB_ROWINDEX_INT64 : GrB_ROWINDEX_INT32 ;
    }
    else
    {
        // only the level is needed
        semiring = LAGraph_any_one_bool ;

        // create a sparse boolean vector frontier, and set frontier(src) = true
        GRB_TRY (GrB_Vector_new(&frontier, GrB_BOOL, n)) ;
        GRB_TRY (GrB_Vector_setElement(frontier, true, src)) ;
    }

    if (compute_level)
    {
        // create the level vector. v(i) is the level of node i
        // v (src) = 0 denotes the source node
        GRB_TRY (GrB_Vector_new(&l_level, int_type, n)) ;
    }

    //--------------------------------------------------------------------------
    // BFS traversal and label the nodes
    //--------------------------------------------------------------------------

    GrB_Index nq = 1 ;          // number of nodes in the current level
    GrB_Index last_nq = 0 ;
    GrB_Index current_level = 0;
    GrB_Index nvals = 1;

    // {!mask} is the set of unvisited nodes
    GrB_Vector mask = (compute_parent) ? l_parent : l_level ;

    // parent BFS
    do
    {
        if (compute_level)
        {
            // assign levels: l_level<s(frontier)> = current_level
            GRB_TRY( GrB_assign(l_level, frontier, GrB_NULL,
                                current_level, GrB_ALL, n, GrB_DESC_S) );
            ++current_level;
        }

        if (compute_parent)
        {
            // frontier(i) currently contains the parent id of node i in tree.
            // l_parent<s(frontier)> = frontier
            GRB_TRY( GrB_assign(l_parent, frontier, GrB_NULL,
                                frontier, GrB_ALL, n, GrB_DESC_S) );

            // convert all stored values in frontier to their indices
            GRB_TRY (GrB_apply (frontier, GrB_NULL, GrB_NULL, ramp,
                frontier, 0, GrB_NULL)) ;
        }

        // frontier = kth level of the BFS
        // mask is l_parent if computing parent, l_level if computing just level
        GRB_TRY( GrB_vxm(frontier, mask, GrB_NULL, semiring,
                         frontier, A, GrB_DESC_RSC) );

        // done if frontier is empty
        GRB_TRY( GrB_Vector_nvals(&nvals, frontier) );
    } while (nvals > 0);

    //--------------------------------------------------------------------------
    // free workspace and return result
    //--------------------------------------------------------------------------

    if (compute_parent) (*parent) = l_parent ;
    if (compute_level ) (*level ) = l_level ;
    LG_FREE_WORK ;
    return (GrB_SUCCESS) ;
}
