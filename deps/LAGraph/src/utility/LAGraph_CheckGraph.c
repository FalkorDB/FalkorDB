//------------------------------------------------------------------------------
// LAGraph_CheckGraph: check if a graph is valid
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

int LAGraph_CheckGraph
(
    // input/output:
    LAGraph_Graph G,    // graph to check
    char *msg
)
{

    //--------------------------------------------------------------------------
    // clear the msg and check basic components
    //--------------------------------------------------------------------------

    LG_CLEAR_MSG_AND_BASIC_ASSERT (G, msg) ;
    GrB_Matrix A = G->A ;
    LAGraph_Kind kind = G->kind ;

    //--------------------------------------------------------------------------
    // ensure the matrix is square for directed or undirected graphs
    //--------------------------------------------------------------------------

    GrB_Index nrows, ncols ;
    if (kind == LAGraph_ADJACENCY_UNDIRECTED ||
        kind == LAGraph_ADJACENCY_DIRECTED)
    {
        GRB_TRY (GrB_Matrix_nrows (&nrows, A)) ;
        GRB_TRY (GrB_Matrix_ncols (&ncols, A)) ;
        LG_ASSERT_MSG (nrows == ncols, LAGRAPH_INVALID_GRAPH,
            "adjacency matrix must be square") ;
    }

    #if LAGRAPH_SUITESPARSE
        // only by-row format is supported when using SuiteSparse
        GxB_Format_Value fmt ;
        GRB_TRY (GxB_get (A, GxB_FORMAT, &fmt)) ;
        LG_ASSERT_MSG (fmt == GxB_BY_ROW, LAGRAPH_INVALID_GRAPH,
            "only by-row format supported") ;
    #endif

    //--------------------------------------------------------------------------
    // check the cached properties
    //--------------------------------------------------------------------------

    GrB_Matrix AT = G->AT ;
    if (AT != NULL)
    {
        GrB_Index nrows2, ncols2;
        GRB_TRY (GrB_Matrix_nrows (&nrows2, AT)) ;
        GRB_TRY (GrB_Matrix_ncols (&ncols2, AT)) ;
        LG_ASSERT_MSG (nrows == ncols2 && ncols == nrows2,
            LAGRAPH_INVALID_GRAPH, "G->AT matrix has the wrong dimensions") ;

        #if LAGRAPH_SUITESPARSE
            // only by-row format is supported when using SuiteSparse
            GxB_Format_Value fmt ;
            GRB_TRY (GxB_get (AT, GxB_FORMAT, &fmt)) ;
            LG_ASSERT_MSG (fmt == GxB_BY_ROW,
                LAGRAPH_INVALID_GRAPH, "only by-row format supported") ;
        #endif

        // ensure the types of A and AT are the same
        char atype [LAGRAPH_MAX_NAME_LEN] ;
        char ttype [LAGRAPH_MAX_NAME_LEN] ;
        LG_TRY (LAGraph_Matrix_TypeName (atype, A, msg)) ;
        LG_TRY (LAGraph_Matrix_TypeName (ttype, AT, msg)) ;
        LG_ASSERT_MSG (MATCHNAME (atype, ttype),
            LAGRAPH_INVALID_GRAPH, "A and AT must have the same type") ;
    }

    GrB_Vector out_degree = G->out_degree ;
    if (out_degree != NULL)
    {
        GrB_Index m ;
        GRB_TRY (GrB_Vector_size (&m, out_degree)) ;
        LG_ASSERT_MSG (m == nrows, LAGRAPH_INVALID_GRAPH,
            "out_degree invalid size") ;
        char rtype [LAGRAPH_MAX_NAME_LEN] ;
        LG_TRY (LAGraph_Vector_TypeName (rtype, out_degree, msg)) ;
        LG_ASSERT_MSG (MATCHNAME (rtype, "int64_t"),
            LAGRAPH_INVALID_GRAPH,
            "out_degree has wrong type; must be GrB_INT64") ;
    }

    GrB_Vector in_degree = G->in_degree ;
    if (in_degree != NULL)
    {
        GrB_Index n ;
        GRB_TRY (GrB_Vector_size (&n, in_degree)) ;
        LG_ASSERT_MSG (n == ncols, LAGRAPH_INVALID_GRAPH,
            "in_degree invalid size") ;
        char ctype [LAGRAPH_MAX_NAME_LEN] ;
        LG_TRY (LAGraph_Vector_TypeName (ctype, in_degree, msg)) ;
        LG_ASSERT_MSG (MATCHNAME (ctype, "int64_t"),
            LAGRAPH_INVALID_GRAPH,
            "in_degree has wrong type; must be GrB_INT64") ;
    }

    return (GrB_SUCCESS) ;
}
