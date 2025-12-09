//------------------------------------------------------------------------------
// GraphBLAS/CUDA/template/GB_cuda_cumsum: cumlative sum of array on the GPU(s)
//------------------------------------------------------------------------------

// SuiteSparse:GraphBLAS, Timothy A. Davis, (c) 2017-2025, All Rights Reserved.
// SPDX-License-Identifier: Apache-2.0

//------------------------------------------------------------------------------

#ifndef GB_CUDA_CUMSUM
#define GB_CUDA_CUMSUM

#include <cub/cub.cuh>

#define GB_FREE_ALL             \
{                               \
    cudaFree (d_temp_storage) ; \
}

typedef enum GB_cuda_cumsum_type
{
    GB_CUDA_CUMSUM_EXCLUSIVE,
    GB_CUDA_CUMSUM_INCLUSIVE
} GB_cuda_cumsum_type ;

__host__ GrB_Info GB_cuda_cumsum    // compute the cumulative sum of an array
(
    int64_t *__restrict__ out,   // size n or n+1, output.
    int64_t *__restrict__ in,    // size n or n+1, input
    // to do an in-place cumsum, pass out == in

    const int64_t n,
    cudaStream_t stream,
    GB_cuda_cumsum_type type,

    const GB_callback_struct *my_callback
)
{

    //--------------------------------------------------------------------------
    // check inputs
    //--------------------------------------------------------------------------

    ASSERT (in != NULL) ;
    ASSERT (out != NULL) ;
    ASSERT (n >= 0) ;

    GB_malloc_memory_f GB_malloc_memory = my_callback->GB_malloc_memory_func ;
    GB_free_memory_f GB_free_memory = my_callback->GB_free_memory_func ;

    void *d_temp_storage = NULL;
    size_t temp_storage_bytes = 0, tsize = 0 ;

    switch (type)
    {
        case GB_CUDA_CUMSUM_INCLUSIVE:
            cub::DeviceScan::InclusiveSum (d_temp_storage, temp_storage_bytes,
                in, out, n, stream) ;
            break;
        default:
            cub::DeviceScan::ExclusiveSum (d_temp_storage, temp_storage_bytes,
                in, out, n, stream) ;
    }

//  CUDA_OK (cudaMalloc (&d_temp_storage, temp_storage_bytes)) ;
    printf ("sizeof (size_t): %d\n", (int) sizeof (size_t)) ;
    printf ("SIZE_MAX: %zu\n", SIZE_MAX) ;
    printf ("cumsum temp space: (%zu) %g GB for n %g\n", temp_storage_bytes,
            ((double) temp_storage_bytes) / 1e9, (double) n) ;
    d_temp_storage = GB_MALLOC_MEMORY (temp_storage_bytes, 1, &tsize) ;
    if (d_temp_storage == NULL)
    {
        printf ("failed to allocate %g GB\n",
            ((double) temp_storage_bytes) / 1e9) ;
        return (GrB_OUT_OF_MEMORY) ;
    }

    // Run
    switch (type)
    {
        case GB_CUDA_CUMSUM_INCLUSIVE:
            cub::DeviceScan::InclusiveSum (d_temp_storage, temp_storage_bytes,
                in, out, n, stream) ;
            break;
        default:
            cub::DeviceScan::ExclusiveSum (d_temp_storage, temp_storage_bytes,
                in, out, n, stream) ;
    }

//  cudaFree (d_temp_storage) ;
    GB_FREE_MEMORY (&d_temp_storage, tsize) ;
    
    return GrB_SUCCESS;
}
#endif
