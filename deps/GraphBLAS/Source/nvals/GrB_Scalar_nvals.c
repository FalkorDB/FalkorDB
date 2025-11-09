//------------------------------------------------------------------------------
// GrB_Scalar_nvals: number of entries in a sparse GrB_Scalar
//------------------------------------------------------------------------------

// SuiteSparse:GraphBLAS, Timothy A. Davis, (c) 2017-2025, All Rights Reserved.
// SPDX-License-Identifier: Apache-2.0

//------------------------------------------------------------------------------

#include "GB.h"

GrB_Info GrB_Scalar_nvals   // get the number of entries in a GrB_Scalar
(
    uint64_t *nvals,        // number of entries (1 or 0)
    const GrB_Scalar s      // GrB_Scalar to query
)
{ 

    //--------------------------------------------------------------------------
    // check inputs
    //--------------------------------------------------------------------------

    GB_RETURN_IF_NULL (s) ;
    GB_WHERE_1 (s, "GrB_Scalar_nvals (&nvals, s)") ;

    ASSERT (GB_SCALAR_OK (s)) ;

    //--------------------------------------------------------------------------
    // get the number of entries
    //--------------------------------------------------------------------------

    info = GB_nvals (nvals, (GrB_Matrix) s, Werk) ;
    #pragma omp flush
    return (info) ;
}

//------------------------------------------------------------------------------
// GxB_Scalar_nvals: number of entries in a sparse GrB_Scalar (historical)
//------------------------------------------------------------------------------

GrB_Info GxB_Scalar_nvals   // get the number of entries in a GrB_Scalar
(
    uint64_t *nvals,        // number of entries (1 or 0)
    const GrB_Scalar s      // GrB_Scalar to query
)
{
    return (GrB_Scalar_nvals (nvals, s)) ;
}

