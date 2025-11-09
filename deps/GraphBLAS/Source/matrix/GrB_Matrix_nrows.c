//------------------------------------------------------------------------------
// GrB_Matrix_nrows: number of rows of a sparse matrix
//------------------------------------------------------------------------------

// SuiteSparse:GraphBLAS, Timothy A. Davis, (c) 2017-2025, All Rights Reserved.
// SPDX-License-Identifier: Apache-2.0

//------------------------------------------------------------------------------

#include "GB.h"

GrB_Info GrB_Matrix_nrows   // get the number of rows of a matrix
(
    uint64_t *nrows,        // matrix has nrows rows
    const GrB_Matrix A      // matrix to query
)
{ 

    //--------------------------------------------------------------------------
    // check inputs
    //--------------------------------------------------------------------------

    GrB_Info info ;
    GB_CHECK_INIT ;
    GB_RETURN_IF_NULL (nrows) ;
    GB_RETURN_IF_NULL_OR_INVALID (A) ;

    //--------------------------------------------------------------------------
    // get the number of rows
    //--------------------------------------------------------------------------

    (*nrows) = GB_NROWS (A) ;
    #pragma omp flush
    return (GrB_SUCCESS) ;
}

