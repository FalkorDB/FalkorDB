//------------------------------------------------------------------------------
// GrB_Vector_nvals: number of entries in a sparse vector
//------------------------------------------------------------------------------

// SuiteSparse:GraphBLAS, Timothy A. Davis, (c) 2017-2025, All Rights Reserved.
// SPDX-License-Identifier: Apache-2.0

//------------------------------------------------------------------------------

#include "GB.h"

GrB_Info GrB_Vector_nvals   // get the number of entries in a vector
(
    uint64_t *nvals,        // number of entries
    const GrB_Vector v      // vector to query
)
{ 

    //--------------------------------------------------------------------------
    // check inputs
    //--------------------------------------------------------------------------

    GB_RETURN_IF_NULL (v) ;
    GB_WHERE_1 (v, "GrB_Vector_nvals (&nvals, v)") ;

    GB_BURBLE_START ("GrB_Vector_nvals") ;
    ASSERT (GB_VECTOR_OK (v)) ;

    //--------------------------------------------------------------------------
    // get the number of entries
    //--------------------------------------------------------------------------

    info = GB_nvals (nvals, (GrB_Matrix) v, Werk) ;
    GB_BURBLE_END ;
    #pragma omp flush
    return (info) ;
}

