//------------------------------------------------------------------------------
// GxB_Matrix_iso: report if a matrix is iso-valued or not
//------------------------------------------------------------------------------

// SuiteSparse:GraphBLAS, Timothy A. Davis, (c) 2017-2023, All Rights Reserved.
// SPDX-License-Identifier: Apache-2.0

//------------------------------------------------------------------------------

#include "GB.h"

GrB_Info GxB_Matrix_iso     // return iso status of a matrix
(
    bool *iso,              // true if the matrix is iso-valued
    const GrB_Matrix A      // matrix to query
)
{ 

    //--------------------------------------------------------------------------
    // check inputs
    //--------------------------------------------------------------------------

    GB_WHERE1 ("GxB_Matrix_iso (&iso, A)") ;
    GB_RETURN_IF_NULL (iso) ;
    GB_RETURN_IF_NULL_OR_FAULTY (A) ;

    //--------------------------------------------------------------------------
    // return the iso status of a matrix
    //--------------------------------------------------------------------------

    (*iso) = A->iso ;
    #pragma omp flush
    return (GrB_SUCCESS) ;
}

