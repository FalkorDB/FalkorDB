//------------------------------------------------------------------------------
// GB_init: initialize GraphBLAS
//------------------------------------------------------------------------------

// SuiteSparse:GraphBLAS, Timothy A. Davis, (c) 2017-2025, All Rights Reserved.
// SPDX-License-Identifier: Apache-2.0

//------------------------------------------------------------------------------

// GrB_init or GxB_init must called before any other GraphBLAS
// operation; all three rely on this internal function.  If GraphBLAS is used
// by multiple user threads, only one can call GrB_init or GxB_init.

// Result are undefined if multiple user threads simultaneously call GrB_init
// or GxB_init.

// GrB_finalize must be called as the last GraphBLAS operation.
// Not even GrB_Matrix_free can be safely called after GrB_finalize.

// GrB_init or GxB_init define the mode that GraphBLAS will use:  blocking or
// non-blocking.  With blocking mode, all operations finish before returning to
// the user application.  With non-blocking mode, operations can be left
// pending, and are computed only when needed.

// GxB_init is the same as GrB_init except that it also defines the
// malloc/calloc/realloc/free functions to use.

// The realloc function pointer is optional and can be NULL.  If realloc is
// NULL, it is not used, and malloc/memcpy/free are used instead.

// The calloc function pointer is also optional and can be NULL.

// If the mode is GxB_BLOCKING_GPU or GxB_NONBLOCKING_GPU, the 4 function
// pointers are ignored, and rmm_wrap_malloc/.../rmm_wrap_free are used
// instead.

#define GB_FREE_ALL ;
#include "GB.h"
#include "init/GB_init.h"
#include "jitifyer/GB_stringify.h"

//------------------------------------------------------------------------------
// GB_init
//------------------------------------------------------------------------------

GrB_Info GB_init            // start up GraphBLAS
(
    int mode,               // blocking or non-blocking mode

    // pointers to memory management functions.
    void * (* malloc_function  ) (size_t),          // required
    void * (* calloc_function  ) (size_t, size_t),  // optional, can be NULL
    void * (* realloc_function ) (void *, size_t),  // optional, can be NULL
    void   (* free_function    ) (void *),          // required

    GB_Werk Werk      // from GrB_init or GxB_init
)
{

    //--------------------------------------------------------------------------
    // check inputs
    //--------------------------------------------------------------------------

    GrB_Info info ;
    if (GB_Global_GrB_init_called_get ( ))
    { 
        // GrB_init can only be called once
        return (GrB_INVALID_VALUE) ;
    }

    if (!(mode == GrB_NONBLOCKING || mode == GrB_BLOCKING ||
          mode == GxB_NONBLOCKING_GPU || mode == GxB_BLOCKING_GPU))
    { 
        // invalid mode
        return (GrB_INVALID_VALUE) ;
    }

    //--------------------------------------------------------------------------
    // initialize OpenMP locks
    //--------------------------------------------------------------------------

    GB_Global_lock_init ( ) ;

    //--------------------------------------------------------------------------
    // establish malloc/calloc/realloc/free
    //--------------------------------------------------------------------------

    bool malloc_is_thread_safe = true ;

    #if defined ( GRAPHBLAS_HAS_CUDA )
    mode = GxB_NONBLOCKING_GPU ;    // HACK FIXME for CUDA: force GPU to be used
    if (mode == GxB_NONBLOCKING_GPU || mode == GxB_BLOCKING_GPU)
    {
        // ignore the memory management function pointers and use rmm_wrap_*
        malloc_function  = rmm_wrap_malloc ;
        calloc_function  = rmm_wrap_calloc ;
        realloc_function = rmm_wrap_realloc ;
        free_function    = rmm_wrap_free ;
        // the rmm_wrap methods are not thread-safe
        malloc_is_thread_safe = false ;
    }
    #endif

    if (malloc_function == NULL || free_function == NULL)
    { 
        // only malloc and free required.  calloc and/or realloc may be NULL
        return (GrB_NULL_POINTER) ;
    }

    GB_Global_GrB_init_called_set (true) ;

    // GrB_init passes in the C11 malloc/calloc/realloc/free.

    GB_Global_malloc_function_set  (malloc_function ) ; // cannot be NULL
    GB_Global_calloc_function_set  (calloc_function ) ; // ok if NULL
    GB_Global_realloc_function_set (realloc_function) ; // ok if NULL
    GB_Global_free_function_set    (free_function   ) ; // cannot be NULL

    GB_Global_malloc_is_thread_safe_set (malloc_is_thread_safe) ;
    GB_Global_memtable_clear ( ) ;

    GB_Global_malloc_tracking_set (false) ;
    GB_Global_nmalloc_clear ( ) ;
    GB_Global_malloc_debug_set (false) ;
    GB_Global_malloc_debug_count_set (0) ;

    //--------------------------------------------------------------------------
    // query hardware features for future use
    //--------------------------------------------------------------------------

    GB_Global_cpu_features_query ( ) ;

    //--------------------------------------------------------------------------
    // max number of threads
    //--------------------------------------------------------------------------

    // Maximum number of threads for internal parallelization.
    // SuiteSparse:GraphBLAS requires OpenMP to use parallelization within
    // calls to GraphBLAS.  The user application may also call GraphBLAS in
    // parallel, from multiple user threads.  The user threads can use
    // any threading library; this has no effect on GraphBLAS.

    GB_Context_nthreads_max_set (NULL, GB_omp_get_max_threads ( )) ;
    GB_Context_chunk_set        (NULL, GB_CHUNK_DEFAULT) ;
    GB_Context_gpu_ids_set      (NULL, NULL, 0) ;

    //--------------------------------------------------------------------------
    // initialize the blocking/nonblocking mode
    //--------------------------------------------------------------------------

    // set the mode: blocking or nonblocking
    GB_Global_mode_set (mode) ;

    //--------------------------------------------------------------------------
    // initialize the GPUs, if present
    //--------------------------------------------------------------------------

    #if defined ( GRAPHBLAS_HAS_CUDA )
    if (mode == GxB_BLOCKING_GPU || mode == GxB_NONBLOCKING_GPU)
    {
        // initialize the GPUs
        GB_OK (GB_cuda_init ( )) ;
    }
    else
    #endif
    { 
        // CUDA not available at compile-time, or not requested at run time
        GB_Global_gpu_count_set (0) ;
    }

    //--------------------------------------------------------------------------
    // set the global default format
    //--------------------------------------------------------------------------

    // set the default hyper_switch and the default format (by-row);  any thread
    // can do this later as well, so there is no race condition danger.

    GB_Global_hyper_switch_set (GB_HYPER_SWITCH_DEFAULT) ;
    GB_Global_bitmap_switch_default ( ) ;
    GB_Global_is_csc_set (false) ;

    //--------------------------------------------------------------------------
    // diagnostic output
    //--------------------------------------------------------------------------

    GB_Global_burble_set (false) ;
    GB_Global_printf_set (NULL) ;
    GB_Global_flush_set (NULL) ;

    //--------------------------------------------------------------------------
    // development use only
    //--------------------------------------------------------------------------

    GB_Global_timing_clear_all ( ) ;

    //--------------------------------------------------------------------------
    // set up the JIT setting and emit the source to the cache folder
    //--------------------------------------------------------------------------

    GB_OK (GB_jitifyer_init ( )) ;

    //--------------------------------------------------------------------------
    // return result
    //--------------------------------------------------------------------------

    #pragma omp flush
    #if defined ( GRAPHBLAS_HAS_CUDA )
//  this hack_get setting is used by GB_ngpus_to_use:
//  GB_Global_hack_set (2,0) ;  // HACK FIXME for CUDA: default: GPU for big enough probs
    GB_Global_hack_set (2,1) ;  // HACK FIXME for CUDA: force the GPU always to be used
//  GB_Global_hack_set (2,2) ;  // HACK FIXME for CUDA: force the GPU never to be used
    #endif

    return (GrB_SUCCESS) ;
}

