//------------------------------------------------------------------------------
// GrB_Matrix_ncols: number of columns of a sparse matrix
//------------------------------------------------------------------------------

// SuiteSparse:GraphBLAS, Timothy A. Davis, (c) 2017-2023, All Rights Reserved.
// SPDX-License-Identifier: Apache-2.0

//------------------------------------------------------------------------------

#include "GB.h"

GrB_Info GrB_Matrix_ncols   // get the number of columns of a matrix
(
    GrB_Index *ncols,       // matrix has ncols columns
    const GrB_Matrix A      // matrix to query
)
{ 

    //--------------------------------------------------------------------------
    // check inputs
    //--------------------------------------------------------------------------

    GB_WHERE1 ("GrB_Matrix_ncols (&ncols, A)") ;
    GB_RETURN_IF_NULL (ncols) ;
    GB_RETURN_IF_NULL_OR_FAULTY (A) ;

    //--------------------------------------------------------------------------
    // return the number of columns
    //--------------------------------------------------------------------------

    (*ncols) = GB_NCOLS (A) ;
    #pragma omp flush
    return (GrB_SUCCESS) ;
}

