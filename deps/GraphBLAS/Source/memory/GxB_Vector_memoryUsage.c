//------------------------------------------------------------------------------
// GxB_Vector_memoryUsage: # of bytes used for a vector
//------------------------------------------------------------------------------

// SuiteSparse:GraphBLAS, Timothy A. Davis, (c) 2017-2023, All Rights Reserved.
// SPDX-License-Identifier: Apache-2.0

//------------------------------------------------------------------------------

#include "GB.h"

GrB_Info GxB_Vector_memoryUsage  // return # of bytes used for a vector
(
    size_t *size,           // # of bytes used by the vector v
    const GrB_Vector v      // vector to query
)
{ 

    //--------------------------------------------------------------------------
    // check inputs
    //--------------------------------------------------------------------------

    GB_WHERE1 ("GxB_Vector_memoryUsage (&size, v)") ;
    GB_RETURN_IF_NULL (size) ;
    GB_RETURN_IF_NULL_OR_FAULTY (v) ;
    ASSERT (GB_VECTOR_OK (v)) ;

    //--------------------------------------------------------------------------
    // get the memory size taken by the vector
    //--------------------------------------------------------------------------

    int64_t nallocs ;
    size_t mem_shallow ;
    GB_memoryUsage (&nallocs, size, &mem_shallow, (GrB_Matrix) v, false) ;
    return (GrB_SUCCESS) ;
}

