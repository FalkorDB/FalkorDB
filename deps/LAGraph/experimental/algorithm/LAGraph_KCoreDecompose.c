//------------------------------------------------------------------------------
// LAGraph_KCoreDecompose: Helper method to LAGraph_KCore and LAGraph_AllKCore
// that performs graph decomposition given a specified value k.
//------------------------------------------------------------------------------

// LAGraph, (c) 2019-2022 by The LAGraph Contributors, All Rights Reserved.
// SPDX-License-Identifier: BSD-2-Clause
//
// For additional details (including references to third party source code and
// other files) see the LICENSE file or contact permission@sei.cmu.edu. See
// Contributors.txt for a full list of contributors. Created, in part, with
// funding and support from the U.S. Government (see Acknowledgments.txt file).
// DM22-0790

// Contributed by Pranav Konduri, Texas A&M University

//------------------------------------------------------------------------------

#define LG_FREE_WORK                \
{                                   \
    GrB_free (&C) ;                 \
    GrB_free (&deg) ;               \
}

#define LG_FREE_ALL                 \
{                                   \
    LG_FREE_WORK                    \
    GrB_free (D) ;                  \
}

#include "LG_internal.h"


int LAGraph_KCore_Decompose
(
    // outputs:
    GrB_Matrix *D,              // kcore decomposition
    // inputs:
    LAGraph_Graph G,            // input graph
    GrB_Vector decomp,         // input decomposition matrix
    uint64_t k,
    char *msg
)
{
    LG_CLEAR_MSG ;

    // declare items
    GrB_Matrix A = NULL, C = NULL;
    GrB_Vector deg = NULL;


    LG_ASSERT (D != NULL, GrB_NULL_POINTER) ;
    (*D) = NULL ;

#if !LAGRAPH_SUITESPARSE
    LG_ASSERT (false, GrB_NOT_IMPLEMENTED) ;
#else

    LG_TRY (LAGraph_CheckGraph (G, msg)) ;

    if (G->kind == LAGraph_ADJACENCY_UNDIRECTED ||
       (G->kind == LAGraph_ADJACENCY_DIRECTED &&
        G->is_symmetric_structure == LAGraph_TRUE))
    {
         // the structure of A is known to be symmetric
        A = G->A ;
    }
    else
    {
        // A is not known to be symmetric
        LG_ASSERT_MSG (false, -1005, "G->A must be symmetric") ;
    }

    // no self edges can be present
    // todo: what would happen if there are self edges?
    LG_ASSERT_MSG (G->nself_edges == 0, -1004, "G->nself_edges must be zero") ;

    //create work scalars
    GrB_Index nrows, n;
    GRB_TRY (GrB_Matrix_nrows(&nrows, A)) ;
    GRB_TRY (GrB_Vector_size(&n, decomp)) ;
    LG_ASSERT_MSG (nrows == n, -1003, "Size of vector and rows of matrix must be same") ;

    //Create Vectors and Matrices
    GRB_TRY (GrB_Vector_new(&deg, GrB_INT64, n)) ;
    GRB_TRY (GrB_Matrix_new(D, GrB_INT64, n, n)) ;

    //create deg vector using select
    GRB_TRY (GrB_select (deg, GrB_NULL, GrB_NULL, GrB_VALUEGE_INT64, decomp, k, GrB_NULL)) ;

    //create decomposition matrix (C * A * C)

    #if LAGRAPH_SUITESPARSE
        #if GxB_IMPLEMENTATION >= GxB_VERSION (7,0,0)
        // SuiteSparse 7.x and later:
        GRB_TRY (GrB_Matrix_diag(&C, deg, 0)) ;
        #else
        // SuiteSparse 6.x and earlier, which had the incorrect signature:
        GRB_TRY (GrB_Matrix_new(&C, GrB_INT64, n, n)) ;
        GRB_TRY (GrB_Matrix_diag(C, deg, 0)) ;
        #endif
    #else
    // standard GrB:
    GRB_TRY (GrB_Matrix_diag(&C, deg, 0)) ;
    #endif

    GRB_TRY (GrB_mxm (*D, NULL, NULL, GxB_ANY_SECONDI_INT64, C, A, GrB_NULL)) ;
    GRB_TRY (GrB_mxm (*D, NULL, NULL, GxB_MIN_SECONDI_INT64, *D, C, GrB_NULL)) ;

    //Assigns all values as 1 (todo: change to something cleaner)
    GRB_TRY (GrB_assign (*D, *D, NULL, (int64_t) 1, GrB_ALL, n, GrB_ALL, n, GrB_DESC_S)) ;

    LG_FREE_WORK ;
    return (GrB_SUCCESS) ;
#endif
}
