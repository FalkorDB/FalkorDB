//------------------------------------------------------------------------------
// LG_nself_edges: count the # of diagonal entries in a matrix
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

#define LG_FREE_ALL         \
{                           \
    GrB_free (&M) ;         \
    GrB_free (&D) ;         \
    GrB_free (&d) ;         \
}

#include "LG_internal.h"

int LG_nself_edges
(
    // output:
    int64_t *nself_edges,         // # of entries
    // input:
    GrB_Matrix A,           // matrix to count
    char *msg               // error message
)
{

    //--------------------------------------------------------------------------
    // extract the diagonal and count its entries
    //--------------------------------------------------------------------------

    GrB_Matrix D = NULL, M = NULL ;
    GrB_Vector d = NULL ;
    LG_ASSERT (nself_edges != NULL, GrB_NULL_POINTER) ;
    (*nself_edges) = LAGRAPH_UNKNOWN ;

    GrB_Index nrows, ncols ;
    GRB_TRY (GrB_Matrix_nrows (&nrows, A)) ;
    GRB_TRY (GrB_Matrix_ncols (&ncols, A)) ;
    GrB_Index n = LAGRAPH_MIN (nrows, ncols) ;

    // FUTURE: use a method that does not require atype

    GrB_Type atype ;
    char atype_name [LAGRAPH_MAX_NAME_LEN] ;
    LG_TRY (LAGraph_Matrix_TypeName (atype_name, A, msg)) ;
    LG_TRY (LAGraph_TypeFromName (&atype, atype_name, msg)) ;

    #if LAGRAPH_SUITESPARSE

        //----------------------------------------------------------------------
        // SuiteSparse:GraphBLAS v5.0.2: use GxB_Vector_diag
        //----------------------------------------------------------------------

        GRB_TRY (GrB_Vector_new (&d, atype, n)) ;
        GRB_TRY (GxB_Vector_diag (d, A, 0, NULL)) ;
        GRB_TRY (GrB_Vector_nvals ((GrB_Index *) nself_edges, d)) ;

    #else

        //----------------------------------------------------------------------
        // pure GrB version with no GxB extensions
        //----------------------------------------------------------------------

        GRB_TRY (GrB_Matrix_new (&M, GrB_BOOL, nrows, ncols)) ;
        GRB_TRY (GrB_Matrix_new (&D, atype, nrows, ncols)) ;
        for (int64_t i = 0 ; i < n ; i++)
        {
            // M (i,i) = true
            GRB_TRY (GrB_Matrix_setElement (M, (bool) true, i, i)) ;
        }

        // D<M,struct> = A
        GRB_TRY (GrB_assign (D, M, NULL, A, GrB_ALL, nrows, GrB_ALL, ncols,
            GrB_DESC_S)) ;
        GRB_TRY (GrB_Matrix_nvals ((GrB_Index *) nself_edges, D)) ;

    #endif

    LG_FREE_ALL ;
    return (GrB_SUCCESS) ;
}
