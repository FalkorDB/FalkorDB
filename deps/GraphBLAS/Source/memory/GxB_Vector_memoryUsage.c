//------------------------------------------------------------------------------
// GxB_Vector_memoryUsage: # of bytes used for a vector
//------------------------------------------------------------------------------

// SuiteSparse:GraphBLAS, Timothy A. Davis, (c) 2017-2025, All Rights Reserved.
// SPDX-License-Identifier: Apache-2.0

//------------------------------------------------------------------------------

#include "GB.h"

GrB_Info GxB_Vector_memoryUsage  // return # of bytes used for a vector
(
    size_t *memsize,        // # of bytes used by the vector v
    const GrB_Vector v      // vector to query
)
{ 

    //--------------------------------------------------------------------------
    // check inputs
    //--------------------------------------------------------------------------

    GrB_Info info ;
    GB_CHECK_INIT ;
    GB_RETURN_IF_NULL (memsize) ;
    GB_RETURN_IF_NULL_OR_INVALID (v) ;
    ASSERT (GB_VECTOR_OK (v)) ;

    //--------------------------------------------------------------------------
    // get the memory size taken by the vector
    //--------------------------------------------------------------------------

    int64_t nallocs ;
    uint64_t mem_deep, mem_shallow ;
    GB_memoryUsage (&nallocs, &mem_deep, &mem_shallow, (GrB_Matrix) v, false) ;
    (*memsize) = (size_t) mem_deep ;
    if (GB_Global_stats_mem_shallow_get ( ))
    { 
        (*memsize) += (size_t) mem_shallow ;
    }
    return (GrB_SUCCESS) ;
}
