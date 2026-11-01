//------------------------------------------------------------------------------
// GraphBLAS/CUDA/GB_cuda_warmup.cu: warmup the GPU
//------------------------------------------------------------------------------

// SuiteSparse:GraphBLAS, Timothy A. Davis, (c) 2017-2025, All Rights Reserved.
// This file: Copyright (c) 2024-2025, NVIDIA CORPORATION. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//------------------------------------------------------------------------------

#include "GB_cuda.hpp"

bool GB_cuda_warmup (int device)
{
    printf ("cuda warmup %d\n", device) ;
    //--------------------------------------------------------------------------
    // set the device
    //--------------------------------------------------------------------------

    if (!GB_cuda_set_device (device))
    {
        // invalid device
        return (false) ;
    }

    //--------------------------------------------------------------------------
    // allocate two small blocks just to load the drivers
    //--------------------------------------------------------------------------

    size_t size = 0 ;
    void *p = GB_malloc_memory (1, 1, &size) ;
    if (p == NULL)
    {
        // no memory on the device
        return (false) ;
    }
    GB_free_memory (&p, size) ;

    cudaMalloc (&p, size ) ;
    if (p == NULL)
    {
        // no memory on the device
        return (false) ;
    }
    cudaFree (p) ;

    //--------------------------------------------------------------------------
    // return result
    //--------------------------------------------------------------------------

    return (true) ;
}

