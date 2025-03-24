//------------------------------------------------------------------------------
// GraphBLAS/CUDA/GB_cuda_get_device_count.cu: find out how many GPUs exist
//------------------------------------------------------------------------------

// SuiteSparse:GraphBLAS, Timothy A. Davis, (c) 2017-2025, All Rights Reserved.
// This file: Copyright (c) 2024-2025, NVIDIA CORPORATION. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//------------------------------------------------------------------------------

// FIXME: remove printf

#include "GB_cuda.hpp"

bool GB_cuda_get_device_count   // true if OK, false if failure
(
    int *gpu_count              // return # of GPUs in the system
)
{
    (*gpu_count) = 0 ;
    cudaError_t err = cudaGetDeviceCount (gpu_count) ;
    printf ("GB_cuda_get_device_count: %d, cudaError_t: %d\n",
        *gpu_count, err) ;
    return (err == cudaSuccess) ;
}

