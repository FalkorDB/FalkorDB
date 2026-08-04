//------------------------------------------------------------------------------
// gb_is_dense: determine if a GrB_matrix is dense
//------------------------------------------------------------------------------

// SuiteSparse:GraphBLAS, Timothy A. Davis, (c) 2017-2026, All Rights Reserved.
// SPDX-License-Identifier: Apache-2.0

//------------------------------------------------------------------------------

// A is dense if it is in the full format, or if all entries are present.

GrB_Info gb_is_dense            // determine if A is dense
(
    // output:
    bool *is_dense,
    // input:
    GrB_Matrix A,               // GrB_Matrix to query
    char err [ERRLEN]
)
{ 

    int sparsity ;
    OK (GrB_Matrix_get_INT32 (A, &sparsity, GxB_SPARSITY_STATUS)) ;
    if (sparsity == GxB_FULL)
    { 
        (*is_dense) = true ;
        return (GrB_SUCCESS) ;
    }

    uint64_t nrows, ncols, nvals ;
    OK (GrB_Matrix_nrows (&nrows, A)) ;
    OK (GrB_Matrix_ncols (&ncols, A)) ;
    OK (GrB_Matrix_nvals (&nvals, A)) ;

    (*is_dense) = 
        ((((double) nrows) * ((double) ncols) < ((double) INT64_MAX)) 
        && (nvals == nrows * ncols)) ;

    return (GrB_SUCCESS) ;
}

