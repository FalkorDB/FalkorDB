//------------------------------------------------------------------------------
// gb_is_dense: determine if a GrB_matrix is dense
//------------------------------------------------------------------------------

// SuiteSparse:GraphBLAS, Timothy A. Davis, (c) 2017-2025, All Rights Reserved.
// SPDX-License-Identifier: Apache-2.0

//------------------------------------------------------------------------------

#include "gb_interface.h"

// A is dense if it is in the full format, or if all entries are present.
// If A is NULL, it is not dense; this is not an error condition.

bool gb_is_dense                // true if A is dense
(
    GrB_Matrix A                // GrB_Matrix to query
)
{
    if (A == NULL) return (false) ;
    int sparsity ;
    OK (GrB_Matrix_get_INT32 (A, &sparsity, GxB_SPARSITY_STATUS)) ;
    if (sparsity == GxB_FULL)
    { 
        return (true) ;
    }
    uint64_t nrows, ncols, nvals ;
    OK (GrB_Matrix_nrows (&nrows, A)) ;
    OK (GrB_Matrix_ncols (&ncols, A)) ;
    OK (GrB_Matrix_nvals (&nvals, A)) ;
    return ((((double) nrows) * ((double) ncols) < ((double) INT64_MAX)) 
        && (nvals == nrows * ncols)) ;
}

