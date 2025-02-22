//------------------------------------------------------------------------------
// GraphBLAS/CUDA/GB_cuda_error.hpp: call a cuda method and check its result
//------------------------------------------------------------------------------

// SuiteSparse:GraphBLAS, Timothy A. Davis, (c) 2017-2025, All Rights Reserved.
// SPDX-License-Identifier: Apache-2.0

//------------------------------------------------------------------------------

#ifndef GB_CUDA_ERROR_HPP
#define GB_CUDA_ERROR_HPP

//------------------------------------------------------------------------------
// CUDA_OK: like GB_OK but for calls to cuda* methods
//------------------------------------------------------------------------------

#define CUDA_OK(cudaMethod)                                                 \
{                                                                           \
    cudaError_t cuda_error = cudaMethod ;                                   \
    if (cuda_error != cudaSuccess)                                          \
    {                                                                       \
        GrB_Info info = (cuda_error == cudaErrorMemoryAllocation) ?         \
            GrB_OUT_OF_MEMORY : GxB_GPU_ERROR ;                             \
        GBURBLE ("(cuda failed: %d:%s file:%s line:%d) ", (int) cuda_error, \
            cudaGetErrorString (cuda_error), __FILE__, __LINE__) ;          \
        GB_FREE_ALL ;                                                       \
        return (info) ;                                                     \
    }                                                                       \
}

#endif

