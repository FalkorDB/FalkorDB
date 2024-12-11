//------------------------------------------------------------------------------
// GxB_Vector_iso: report if a vector is iso-valued or not
//------------------------------------------------------------------------------

// SuiteSparse:GraphBLAS, Timothy A. Davis, (c) 2017-2023, All Rights Reserved.
// SPDX-License-Identifier: Apache-2.0

//------------------------------------------------------------------------------

#include "GB.h"

GrB_Info GxB_Vector_iso     // return iso status of a vector
(
    bool *iso,              // true if the vector is iso-valued
    const GrB_Vector v      // vector to query
)
{ 

    //--------------------------------------------------------------------------
    // check inputs
    //--------------------------------------------------------------------------

    GB_WHERE1 ("GxB_Vector_iso (&iso, v)") ;
    GB_RETURN_IF_NULL (iso) ;
    GB_RETURN_IF_NULL_OR_FAULTY (v) ;
    ASSERT (GB_VECTOR_OK (v)) ;

    //--------------------------------------------------------------------------
    // return the iso status of a vector
    //--------------------------------------------------------------------------

    (*iso) = v->iso ;
    #pragma omp flush
    return (GrB_SUCCESS) ;
}

