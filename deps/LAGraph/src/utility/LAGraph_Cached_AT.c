//------------------------------------------------------------------------------
// LAGraph_Cached_AT: construct G->AT for a graph
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

#define LG_FREE_ALL GrB_free (&AT) ;

#include "LG_internal.h"

int LAGraph_Cached_AT
(
    // input/output:
    LAGraph_Graph G,    // graph for which to compute G->AT
    char *msg
)
{

    //--------------------------------------------------------------------------
    // clear msg and check G
    //--------------------------------------------------------------------------

    GrB_Matrix AT = NULL ;
    LG_CLEAR_MSG_AND_BASIC_ASSERT (G, msg) ;
    GrB_Matrix A = G->A ;

    if (G->AT != NULL)
    {
        // G->AT already computed
        return (GrB_SUCCESS) ;
    }

    if (G->kind == LAGraph_ADJACENCY_UNDIRECTED)
    {
        // G->AT not needed since A is symmetric (warning only, not an error)
        return (LAGRAPH_CACHE_NOT_NEEDED) ;
    }

    //--------------------------------------------------------------------------
    // G->AT = (G->A)'
    //--------------------------------------------------------------------------

    GrB_Index nrows, ncols ;
    GRB_TRY (GrB_Matrix_nrows (&nrows, A)) ;
    GRB_TRY (GrB_Matrix_ncols (&ncols, A)) ;
    GrB_Type atype ;
    char atype_name [LAGRAPH_MAX_NAME_LEN] ;
    LG_TRY (LAGraph_Matrix_TypeName (atype_name, A, msg)) ;
    LG_TRY (LAGraph_TypeFromName (&atype, atype_name, msg)) ;
    GRB_TRY (GrB_Matrix_new (&AT, atype, ncols, nrows)) ;
    GRB_TRY (GrB_transpose (AT, NULL, NULL, A, NULL)) ;
    G->AT = AT ;

    return (GrB_SUCCESS) ;
}
