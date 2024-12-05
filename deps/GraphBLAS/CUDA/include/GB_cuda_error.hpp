//------------------------------------------------------------------------------
// GraphBLAS/CUDA/GB_cuda_error.hpp: call a cuda method and check its result
//------------------------------------------------------------------------------

// SuiteSparse:GraphBLAS, Timothy A. Davis, (c) 2017-2024, All Rights Reserved.
// SPDX-License-Identifier: Apache-2.0

//------------------------------------------------------------------------------

#ifndef GB_CUDA_ERROR_HPP
#define GB_CUDA_ERROR_HPP

//------------------------------------------------------------------------------
// CUDA_OK: like GB_OK but for calls to cuda* methods
//------------------------------------------------------------------------------

// FIXME: GrB_NO_VALUE means something in CUDA failed, and the caller will then
// do the computation on the CPU.  Need to turn off the JIT for CUDA kernels
// (but not CPU kernels) if some CUDA error occurred.  Current JIT control does
// not distinguish between CPU and CUDA failures.

#define CUDA_OK(cudaMethod)                                                 \
{                                                                           \
    cudaError_t cuda_error = cudaMethod ;                                   \
    if (cuda_error != cudaSuccess)                                          \
    {                                                                       \
        GrB_Info info = (cuda_error == cudaErrorMemoryAllocation) ?         \
            GrB_OUT_OF_MEMORY : GrB_NO_VALUE ;                              \
        GBURBLE ("(cuda failed: %d:%s file:%s line:%d) ", (int) cuda_error, \
            cudaGetErrorString (cuda_error), __FILE__, __LINE__) ;          \
        GB_FREE_ALL ;                                                       \
        return (info) ;                                                     \
    }                                                                       \
}

#endif

