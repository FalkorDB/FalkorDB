//------------------------------------------------------------------------------
// GrB_Scalar_nvals: number of entries in a sparse GrB_Scalar
//------------------------------------------------------------------------------

// SuiteSparse:GraphBLAS, Timothy A. Davis, (c) 2017-2023, All Rights Reserved.
// SPDX-License-Identifier: Apache-2.0

//------------------------------------------------------------------------------

#include "GB.h"

GrB_Info GrB_Scalar_nvals   // get the number of entries in a GrB_Scalar
(
    GrB_Index *nvals,       // number of entries (1 or 0)
    const GrB_Scalar s      // GrB_Scalar to query
)
{ 

    //--------------------------------------------------------------------------
    // check inputs
    //--------------------------------------------------------------------------

    GB_WHERE1 ("GrB_Scalar_nvals (&nvals, s)") ;
    GB_RETURN_IF_NULL_OR_FAULTY (s) ;
    ASSERT (GB_SCALAR_OK (s)) ;

    //--------------------------------------------------------------------------
    // get the number of entries
    //--------------------------------------------------------------------------

    GrB_Info info = GB_nvals (nvals, (GrB_Matrix) s, Werk) ;
    #pragma omp flush
    return (info) ;
}

//------------------------------------------------------------------------------
// GxB_Scalar_nvals: number of entries in a sparse GrB_Scalar (historical)
//------------------------------------------------------------------------------

GrB_Info GxB_Scalar_nvals   // get the number of entries in a GrB_Scalar
(
    GrB_Index *nvals,       // number of entries (1 or 0)
    const GrB_Scalar s      // GrB_Scalar to query
)
{
    return (GrB_Scalar_nvals (nvals, s)) ;
}

