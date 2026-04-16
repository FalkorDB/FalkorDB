//------------------------------------------------------------------------------
// GraphBLAS/CUDA/GB_cuda_get_device_count.cu: find out how many GPUs exist
//------------------------------------------------------------------------------

// SuiteSparse:GraphBLAS, Timothy A. Davis, (c) 2017-2025, All Rights Reserved.
// This file: Copyright (c) 2024-2025, NVIDIA CORPORATION. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//------------------------------------------------------------------------------

// If no devices are found or cudaGetDeviceCount returns an error, then
// the GPU count is returned as zero.

// FIXME: remove printf

#include "GB_cuda.hpp"

int GB_cuda_get_device_count (void) // return # of GPUs in the system
{
    int gpu_count = 0 ;
    cudaError_t err = cudaGetDeviceCount (&gpu_count) ;
    printf ("GB_cuda_get_device_count: %d, cudaError_t: %d (%s)\n",
        gpu_count, err, cudaGetErrorString (err)) ;
    return ((err == cudaSuccess) ? gpu_count : 0) ;
}

