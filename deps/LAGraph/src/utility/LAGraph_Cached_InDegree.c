//------------------------------------------------------------------------------
// LAGraph_Cached_InDegree: determine G->in_degree
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

// LAGraph_Cached_InDegree computes G->in_degree, where G->in_degree(j) is
// the number of entries in G->A (:,j).  If there are no entries in G->A (:,j),
// G->coldgree(j) is not present in the structure of G->in_degree.  That is,
// G->in_degree contains no explicit zero entries.

// G->in_degree is not computed if the graph is undirected.  Use G->out_degree
// instead, and LAGraph_Cached_OutDegree.

#define LG_FREE_WORK            \
{                               \
    GrB_free (&S) ;             \
    GrB_free (&x) ;             \
}

#define LG_FREE_ALL             \
{                               \
    LG_FREE_WORK ;              \
    GrB_free (&in_degree) ;    \
}

#include "LG_internal.h"

int LAGraph_Cached_InDegree
(
    // input/output:
    LAGraph_Graph G,    // graph to determine G->in_degree
    char *msg
)
{

    //--------------------------------------------------------------------------
    // clear msg and check G
    //--------------------------------------------------------------------------

    GrB_Matrix S = NULL ;
    GrB_Vector in_degree = NULL, x = NULL ;
    LG_CLEAR_MSG_AND_BASIC_ASSERT (G, msg) ;

    if (G->in_degree != NULL)
    {
        // G->in_degree already computed
        return (GrB_SUCCESS) ;
    }

    if (G->kind == LAGraph_ADJACENCY_UNDIRECTED)
    {
        // G->in_degree is not computed since A is symmetric (warning only)
        return (LAGRAPH_CACHE_NOT_NEEDED) ;
    }

    //--------------------------------------------------------------------------
    // determine the size of A
    //--------------------------------------------------------------------------

    GrB_Matrix A = G->A ;
    GrB_Matrix AT = G->AT ;
    GrB_Index nrows, ncols ;
    GRB_TRY (GrB_Matrix_nrows (&nrows, A)) ;
    GRB_TRY (GrB_Matrix_ncols (&ncols, A)) ;

    //--------------------------------------------------------------------------
    // compute the in_degree
    //--------------------------------------------------------------------------

    GRB_TRY (GrB_Vector_new (&in_degree, GrB_INT64, ncols)) ;
    // x = zeros (nrows,1)
    GRB_TRY (GrB_Vector_new (&x, GrB_INT64, nrows)) ;
    GRB_TRY (GrB_assign (x, NULL, NULL, 0, GrB_ALL, nrows, NULL)) ;

    if (AT != NULL)
    {
        // G->in_degree = row degree of AT; this will be faster assuming
        // AT is held in a row-oriented format.
        GRB_TRY (GrB_mxv (in_degree, NULL, NULL, LAGraph_plus_one_int64,
            AT, x, NULL)) ;
    }
    else
    {
        // G->in_degree = column degree of A
        GRB_TRY (GrB_mxv (in_degree, NULL, NULL, LAGraph_plus_one_int64,
            A, x, GrB_DESC_T0)) ;
    }

    G->in_degree = in_degree ;

    LG_FREE_WORK ;
    return (GrB_SUCCESS) ;
}
