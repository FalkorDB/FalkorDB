//------------------------------------------------------------------------------
// GraphBLAS/CUDA/GB_cuda_init: initialize the GPUs for use by GraphBLAS
//------------------------------------------------------------------------------

// SuiteSparse:GraphBLAS, Timothy A. Davis, (c) 2017-2025, All Rights Reserved.
// This file: Copyright (c) 2024-2025, NVIDIA CORPORATION. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//------------------------------------------------------------------------------

// GB_cuda_init queries the system for properties of the GPUs: their memory
// sizes, SM counts, and other capabilities.  Then each GPU is "warmed up" by
// allocating a small amount of memory.

#include "GB.h"
#include <cuda.h>

GrB_Info GB_cuda_init (void)
{
    GrB_Info info ;
    // get the GPU properties
//  printf ("cuInit:\n") ;
//  CUresult cu = cuInit (0) ;
//  char *cu_string = NULL ;
//  cuGetErrorString (cu, &cu_string) ;
//  printf ("cuInit result: %d [%s]\n", cu, cu_string) ;

//  printf ("GB_cuda_init, getting gpu count:\n") ;
    GB_Global_gpu_count_set ( ) ;
    int gpu_count = GB_Global_gpu_count_get ( ) ;

//  int result = system ("/usr/bin/nvidia-smi") ;
//  printf ("system ('/usr/bin/nvidia-smi') result: %d\n", result) ;

    if (gpu_count == 0)
    { 
        // abort for now; remove this test in production
        printf ("NO GPUS!\n") ;
        fprintf (stderr, "NO GPUS!\n") ;
        fflush (stdout) ;
        fflush (stderr) ;
        abort ( ) ;
        return (GrB_SUCCESS) ;
    }
    printf ("GPU count: %d\n", gpu_count) ;

    for (int device = 0 ; device < gpu_count ; device++)
    {
        // query the GPU
        if (!GB_Global_gpu_device_properties_get (device))
        {
            return (GxB_GPU_ERROR) ;
        }
    }

    // initialize RMM if necessary
    if (!rmm_wrap_is_initialized ())
    {
        rmm_wrap_initialize_all_same ( // rmm_wrap_managed,
            // Fixme: ask the GPU(s) for good default values.  This might be
            // found by GB_cuda_init.  Perhaps GB_cuda_init needs to be split
            // into 2 methods: one to query the sizes(s) of the GPU(s) then
            // call rmm_wrap_initialize_all_same, and the other for the rest
            // of the work.  Alternatively, move GB_cuda_init here (if so,
            // ensure that it doesn't depend on any other initializations
            // below).
            #define GBYTE ((1024L)*(1024L)*(1024L))
            // 8*GBYTE, 80*GBYTE // Fixme: memory sizes are not used
            ) ;
    }

    // warm up the GPUs
    for (int device = 0 ; device < gpu_count ; device++)
    {
        if (!GB_cuda_warmup (device))
        {
            return (GxB_GPU_ERROR) ;
        }
    }

    info = GB_cuda_stream_pool_init ( ) ;
    if (info != GrB_SUCCESS)
    {
        return info ;
    }

    GB_cuda_set_device (0) ;            // make GPU 0 the default device

//  GB_cuda_set_device (1) ;            // make GPU 1 the default device
    GB_Context_gpu_ids_set (NULL, NULL, -1) ; // set global default to GPU 0

    // also check for jit cache, pre-load library of common kernels ...
    return (GrB_SUCCESS) ;
}

