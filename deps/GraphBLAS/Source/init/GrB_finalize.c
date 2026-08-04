//------------------------------------------------------------------------------
// GrB_finalize: finalize GraphBLAS
//------------------------------------------------------------------------------

// SuiteSparse:GraphBLAS, Timothy A. Davis, (c) 2017-2025, All Rights Reserved.
// SPDX-License-Identifier: Apache-2.0

//------------------------------------------------------------------------------

// GrB_finalize must be called as the last GraphBLAS function, per the
// GraphBLAS C API Specification.  Only one user thread can call this function.
// Results are undefined if more than one thread calls this function at the
// same time.

// This method always succeeds and returns GrB_SUCCESS.

#define GB_FREE_ALL ;
#include "GB.h"
#include "jitifyer/GB_jitifyer.h"

GrB_Info GrB_finalize ( )
{ 

    //--------------------------------------------------------------------------
    // finalize GraphBLAS
    //--------------------------------------------------------------------------

    GB_jitifyer_finalize ( ) ;

    #if defined ( GRAPHBLAS_HAS_CUDA )
    {
        // finalize the GPUs
        GB_cuda_finalize ( ) ;
    }
    #endif

    GB_Global_lock_destroy ( ) ;

    // clear all arenas
    for (int arena = 0 ; arena < GxB_NARENAS ; arena++)
    { 
        GB_Global_malloc_function_set (NULL, arena) ;
        GB_Global_calloc_function_set (NULL, arena) ;
        GB_Global_realloc_function_set (NULL, arena) ;
        GB_Global_free_function_set (NULL, arena) ;
    }

    // arena 0 default allocators:
    GB_Global_malloc_function_set (malloc, GrB_DEFAULT) ;
    GB_Global_calloc_function_set (calloc, GrB_DEFAULT) ;
    GB_Global_realloc_function_set (realloc, GrB_DEFAULT) ;
    GB_Global_free_function_set (free, GrB_DEFAULT) ;

    // arena 1 default allocators:
    #ifdef GRAPHBLAS_HAS_CUDA
    GB_Global_malloc_function_set (GB_rmm_malloc, GxB_ARENA_RMM) ;
    GB_Global_free_function_set (GB_rmm_free, GxB_ARENA_RMM) ;
    #else
    GB_Global_malloc_function_set (malloc, GxB_ARENA_RMM) ;
    GB_Global_free_function_set (free, GxB_ARENA_RMM) ;
    #endif

    //--------------------------------------------------------------------------
    // GraphBLAS has now been finalized
    //--------------------------------------------------------------------------

    GB_Global_GrB_init_called_set (false) ;
    return (GrB_SUCCESS) ;
}

