//------------------------------------------------------------------------------
// GraphBLAS/CUDA/GB_cuda_matrix_prefetch: prefetch a matrix to a GPU or the CPU
//------------------------------------------------------------------------------

// SuiteSparse:GraphBLAS, Timothy A. Davis, (c) 2017-2025, All Rights Reserved.
// This file: Copyright (c) 2024-2025, NVIDIA CORPORATION. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//------------------------------------------------------------------------------

#include "GB_cuda.hpp"

#define GB_FREE_ALL ;

GrB_Info GB_cuda_matrix_prefetch
(
    GrB_Matrix A,
    int which,              // which compents to prefetch (phybix control)
    int device,             // GPU device or cudaCpuDeviceId
    cudaStream_t stream
)
{

    GrB_Info info ;
    const int64_t anvec = A->nvec ;
    const int64_t anz = GB_nnz_held (A) ;

    size_t psize = A->p_is_32 ? sizeof (uint32_t) : sizeof (uint64_t) ;
    size_t jsize = A->j_is_32 ? sizeof (uint32_t) : sizeof (uint64_t) ;
    size_t isize = A->i_is_32 ? sizeof (uint32_t) : sizeof (uint64_t) ;

    if (A->p != NULL && (which & GB_PREFETCH_P))
    {
        CUDA_OK (cudaMemPrefetchAsync (A->p, (anvec+1) * psize,
            device, stream)) ;
    }

    if (A->h != NULL && (which & GB_PREFETCH_H))
    {
        CUDA_OK (cudaMemPrefetchAsync (A->h, anvec * jsize,
            device, stream)) ;
    }

    if (A->Y != NULL && (which & GB_PREFETCH_Y))
    {
        // prefetch the hyper_hash: A->Y->p, A->Y->i, and A->Y->x
        GB_OK (GB_cuda_matrix_prefetch (A->Y, GB_PREFETCH_PIX,
            device, stream)) ;
    }

    if (A->b != NULL && (which & GB_PREFETCH_B))
    {
        CUDA_OK (cudaMemPrefetchAsync (A->b, anz * sizeof (int8_t),
            device, stream)) ;
    }

    if (A->i != NULL && (which & GB_PREFETCH_I))
    {
        CUDA_OK (cudaMemPrefetchAsync (A->i, anz * isize,
            device, stream)) ;
    }

    if (A->x != NULL && (which & GB_PREFETCH_X))
    {
        CUDA_OK (cudaMemPrefetchAsync (A->x, (A->iso ? 1:anz) * A->type->size,
            device, stream)) ;
    }

    return (GrB_SUCCESS) ;
}

