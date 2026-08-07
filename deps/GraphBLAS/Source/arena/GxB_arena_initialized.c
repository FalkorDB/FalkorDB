//------------------------------------------------------------------------------
// GxB_arena_initialized:  determine if an arena has been initialized
//------------------------------------------------------------------------------

// SuiteSparse:GraphBLAS, Timothy A. Davis, (c) 2017-2026, All Rights Reserved.
// SPDX-License-Identifier: Apache-2.0

//------------------------------------------------------------------------------

#include "GB.h"

GrB_Info GxB_arena_initialized
(
    // output
    int *flag,              // returns true if the arena has been initialized
    // input
    int arena               // 1 to GxB_NARENAS-1
)
{ 

    //--------------------------------------------------------------------------
    // check inputs
    //--------------------------------------------------------------------------

    GB_CHECK_INIT ;
    if (flag == NULL)
    { 
        return (GrB_NULL_POINTER) ;
    }

    (*flag) = false ;

    if (arena < 0 || arena >= GxB_NARENAS)
    { 
        // arena out of range
        return (GrB_INVALID_VALUE) ;

    }

    //--------------------------------------------------------------------------
    // determine if the arena has been initialized
    //--------------------------------------------------------------------------

    (*flag) =
        GB_Global_malloc_function_get  (arena) != NULL ||
        GB_Global_calloc_function_get  (arena) != NULL ||
        GB_Global_realloc_function_get (arena) != NULL ||
        GB_Global_free_function_get    (arena) != NULL ;

    return (GrB_SUCCESS) ;
}

