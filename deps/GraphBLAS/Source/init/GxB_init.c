//------------------------------------------------------------------------------
// GxB_init: initialize GraphBLAS and declare malloc/calloc/realloc/free to use
//------------------------------------------------------------------------------

// SuiteSparse:GraphBLAS, Timothy A. Davis, (c) 2017-2025, All Rights Reserved.
// SPDX-License-Identifier: Apache-2.0

//------------------------------------------------------------------------------

// GrB_init (or GxB_init) must called before any other GraphBLAS operation.
// GrB_finalize must be called as the last GraphBLAS operation.  GxB_init is
// identical to GrB_init, except that it allows the user application to define
// the malloc/calloc/realloc/free functions that SuiteSparse:GraphBLAS will
// use.  The functions cannot be modified once GraphBLAS starts.

// The calloc and realloc function pointers are optional and can be NULL.  If
// calloc is NULL, it is not used, and malloc/memset are used instead.  If
// realloc is NULL, it is not used, and malloc/memcpy/free are used instead.

// Examples:
//
// To use GraphBLAS with the C11 functions (or to another library
// linked in that replaces them): 
//
//      // either use:
//      GrB_init (mode) ;
//      // or use this (but not both):
//      GxB_init (mode, malloc, calloc, realloc, free) ;
//
// To use GraphBLAS from within a mexFunction:
//
//      #include "mex.h"
//      GxB_init (mode, mxMalloc, mxCalloc, mxRealloc, mxFree) ;
//
// To use the C interface to the Intel TBB scalable allocators:
//
//      #include "tbb/scalable_allocator.h"
//      GxB_init (mode, scalable_malloc, scalable_calloc, scalable_realloc,
//          scalable_free) ;
//
// To use CUDA and its RMM memory manager:
//
//      GxB_init (mode, rmm_malloc, rmm_calloc, rmm_realloc, rmm_free) ;
//
//          where mode is GxB_BLOCKING_GPU or GxB_NONBLOCKING_GPU
//
// To use user-provided malloc and free functions, but not calloc/realloc:
//
//      GxB_init (mode, my_malloc, NULL, NULL, my_free) ;

#include "GB.h"
#include "init/GB_init.h"

GrB_Info GxB_init           // start up GraphBLAS and also define malloc, etc
(
    int mode,               // blocking or non-blocking mode

    // pointers to memory management functions
    void * (* user_malloc_function  ) (size_t),         // required
    void * (* user_calloc_function  ) (size_t, size_t), // optional, can be NULL
    void * (* user_realloc_function ) (void *, size_t), // optional, can be NULL
    void   (* user_free_function    ) (void *)          // required
)
{ 

    //--------------------------------------------------------------------------
    // check inputs
    //--------------------------------------------------------------------------

    GB_WERK ("GxB_init (mode, malloc, calloc, realloc, free)") ;

    //--------------------------------------------------------------------------
    // initialize GraphBLAS
    //--------------------------------------------------------------------------

#if defined ( GRAPHBLAS_HAS_CUDA )
    if (mode == GxB_BLOCKING_GPU || mode == GxB_NONBLOCKING_GPU)
    {
        return (GB_init (mode,              // blocking or non-blocking mode
            // RMM C memory management functions
            rmm_wrap_malloc, rmm_wrap_calloc, rmm_wrap_realloc, rmm_wrap_free,
            Werk)) ;
    }
#endif

    return (GB_init
        (mode,                          // blocking or non-blocking mode
        user_malloc_function,           // user-defined malloc, required
        user_calloc_function,           // user-defined malloc, may be NULL
        user_realloc_function,          // user-defined realloc, may be NULL
        user_free_function,             // user-defined free, required
        Werk)) ;
}

