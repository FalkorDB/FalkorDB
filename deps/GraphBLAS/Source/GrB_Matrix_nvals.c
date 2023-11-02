//------------------------------------------------------------------------------
// GrB_Matrix_nvals: number of entries in a sparse matrix
//------------------------------------------------------------------------------

// SuiteSparse:GraphBLAS, Timothy A. Davis, (c) 2017-2023, All Rights Reserved.
// SPDX-License-Identifier: Apache-2.0

//------------------------------------------------------------------------------

#include "GB.h"

GrB_Info GrB_Matrix_nvals   // get the number of entries in a matrix
(
    GrB_Index *nvals,       // matrix has nvals entries
    const GrB_Matrix A      // matrix to query
)
{ 

    //--------------------------------------------------------------------------
    // check inputs
    //--------------------------------------------------------------------------

    GB_WHERE1 ("GrB_Matrix_nvals (&nvals, A)") ;
    GB_BURBLE_START ("GrB_Matrix_nvals") ;
    GB_RETURN_IF_NULL_OR_FAULTY (A) ;

    //--------------------------------------------------------------------------
    // get the number of entries
    //--------------------------------------------------------------------------

    GrB_Info info = GB_nvals (nvals, A, Werk) ;
    GB_BURBLE_END ;
    #pragma omp flush
    return (info) ;
}

