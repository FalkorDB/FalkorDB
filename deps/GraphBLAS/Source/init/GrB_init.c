//------------------------------------------------------------------------------
// GrB_init: initialize GraphBLAS
//------------------------------------------------------------------------------

// SuiteSparse:GraphBLAS, Timothy A. Davis, (c) 2017-2025, All Rights Reserved.
// SPDX-License-Identifier: Apache-2.0

//------------------------------------------------------------------------------

// GrB_init (or GxB_init) must called before any other GraphBLAS operation.
// GrB_finalize must be called as the last GraphBLAS operation.  To use CUDA
// and its RMM memory manager: use a mode of GxB_BLOCKING_GPU or
// GxB_NONBLOCKING_GPU.

// FIXME for CUDA: rename GxB_*BLOCKING_GPU to GxB_*BLOCKING_CUDA.

#include "GB.h"
#include "init/GB_init.h"

GrB_Info GrB_init           // start up GraphBLAS
(
    int mode                // blocking or non-blocking mode
)
{ 

    //--------------------------------------------------------------------------
    // check inputs
    //--------------------------------------------------------------------------

    GB_WERK ("GrB_init (mode)") ;

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

    // default:  use the C11 malloc memory manager, which is thread-safe
    return (GB_init (mode,              // blocking or non-blocking mode
        malloc, calloc, realloc, free,  // ANSI C memory management functions
        Werk)) ;
}

