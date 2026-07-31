//------------------------------------------------------------------------------
// GxB_arena_init: define the malloc/calloc/realloc/free methods for an arena
//------------------------------------------------------------------------------

// SuiteSparse:GraphBLAS, Timothy A. Davis, (c) 2017-2026, All Rights Reserved.
// SPDX-License-Identifier: Apache-2.0

//------------------------------------------------------------------------------

// The arena must currently be uninitialized.  GrB_finalize is the only method
// that de-initializes the arenas.

#include "GB.h"

GrB_Info GxB_arena_init
(
    // input
    int arena,              // 2 to GxB_NARENAS-1
    // pointers to memory management functions
    void * (* user_malloc_function  ) (size_t),         // required
    void * (* user_calloc_function  ) (size_t, size_t), // not used
    void * (* user_realloc_function ) (void *, size_t), // optional, can be NULL
    void   (* user_free_function    ) (void *)          // required
)
{ 

    //--------------------------------------------------------------------------
    // check inputs
    //--------------------------------------------------------------------------

    GB_CHECK_INIT ;
    if (user_malloc_function == NULL || user_free_function == NULL)
    { 
        return (GrB_NULL_POINTER) ;
    }

    if (arena < 0 || arena >= GxB_NARENAS ||
        GB_Global_malloc_function_get  (arena) != NULL ||
        GB_Global_calloc_function_get  (arena) != NULL ||
        GB_Global_realloc_function_get (arena) != NULL ||
        GB_Global_free_function_get    (arena) != NULL)
    { 
        // arena out of range or already initialized
        return (GrB_INVALID_VALUE) ;
    }

    //--------------------------------------------------------------------------
    // set the malloc/calloc/realloc/free methods for the arena
    //--------------------------------------------------------------------------

    GB_Global_malloc_function_set (user_malloc_function, arena) ;
    GB_Global_calloc_function_set (user_calloc_function, arena) ;
    GB_Global_realloc_function_set (user_realloc_function, arena) ;
    GB_Global_free_function_set (user_free_function, arena) ;

    return (GrB_SUCCESS) ;
}

