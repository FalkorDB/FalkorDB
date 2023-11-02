//------------------------------------------------------------------------------
// GB_cuda_matrix_prefetch: prefetch a matrix to a GPU or the CPU
//------------------------------------------------------------------------------

// SPDX-License-Identifier: Apache-2.0
// SuiteSparse:GraphBLAS, Timothy A. Davis, (c) 2017-2019, All Rights Reserved.
// http://suitesparse.com   See GraphBLAS/Doc/License.txt for license.

//------------------------------------------------------------------------------

#include "GB_cuda.h"
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

    if (A->p != NULL && (which & GB_PREFETCH_P))
    {
        CU_OK (cudaMemPrefetchAsync (A->p, (anvec+1) * sizeof (int64_t), device, stream)) ;
    }

    if (A->h != NULL && (which & GB_PREFETCH_H))
    {
        CU_OK (cudaMemPrefetchAsync (A->h, anvec * sizeof (int64_t), device, stream)) ;
    }

    if (A->Y != NULL && (which & GB_PREFETCH_Y))
    {
        // prefetch the hyper_hash: A->Y->p, A->Y->i, and A->Y->x
        GB_OK (GB_cuda_matrix_prefetch (A->Y, GB_PREFETCH_PIX, device, stream)) ;
    }

    if (A->b != NULL && (which & GB_PREFETCH_B))
    {
        CU_OK (cudaMemPrefetchAsync (A->b, anz * sizeof (int8_t), device, stream)) ;
    }

    if (A->i != NULL && (which & GB_PREFETCH_I))
    {
        CU_OK (cudaMemPrefetchAsync (A->i, anz * sizeof (int64_t), device, stream)) ;
    }

    if (A->x != NULL && (which & GB_PREFETCH_X))
    {
        CU_OK (cudaMemPrefetchAsync (A->x, (A->iso ? 1:anz) * A->type->size, device, stream)) ;
    }

    return (GrB_SUCCESS) ;
}

