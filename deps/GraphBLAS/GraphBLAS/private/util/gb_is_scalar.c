//------------------------------------------------------------------------------
// gb_is_scalar: check if a GrB_Matrix is a scalar with a single entry
//------------------------------------------------------------------------------

// SuiteSparse:GraphBLAS, Timothy A. Davis, (c) 2017-2026, All Rights Reserved.
// SPDX-License-Identifier: Apache-2.0

//------------------------------------------------------------------------------

GrB_Info gb_is_scalar
(
    // output:
    bool *is_scalar,    // true if A is a 1-by-1 GrB_Matrix with 1 entry
    // input
    GrB_Matrix A,
    char err [ERRLEN]
)
{ 
    uint64_t nrows, ncols, nvals ;
    OK (GrB_Matrix_nrows (&nrows, A)) ;
    OK (GrB_Matrix_ncols (&ncols, A)) ;
    OK (GrB_Matrix_nvals (&nvals, A)) ;
    (*is_scalar) = (nrows == 1 && ncols == 1 && nvals == 1) ;
    return (GrB_SUCCESS) ;
}

