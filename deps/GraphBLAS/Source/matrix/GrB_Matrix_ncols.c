//------------------------------------------------------------------------------
// GrB_Matrix_ncols: number of columns of a sparse matrix
//------------------------------------------------------------------------------

// SuiteSparse:GraphBLAS, Timothy A. Davis, (c) 2017-2025, All Rights Reserved.
// SPDX-License-Identifier: Apache-2.0

//------------------------------------------------------------------------------

#include "GB.h"

GrB_Info GrB_Matrix_ncols   // get the number of columns of a matrix
(
    uint64_t *ncols,        // matrix has ncols columns
    const GrB_Matrix A      // matrix to query
)
{ 

    //--------------------------------------------------------------------------
    // check inputs
    //--------------------------------------------------------------------------

    GrB_Info info ;
    GB_CHECK_INIT ;
    GB_RETURN_IF_NULL (ncols) ;
    GB_RETURN_IF_NULL_OR_INVALID (A) ;

    //--------------------------------------------------------------------------
    // return the number of columns
    //--------------------------------------------------------------------------

    (*ncols) = GB_NCOLS (A) ;
    #pragma omp flush
    return (GrB_SUCCESS) ;
}

