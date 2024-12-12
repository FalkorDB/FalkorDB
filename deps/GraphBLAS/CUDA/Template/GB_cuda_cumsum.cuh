//------------------------------------------------------------------------------
// GraphBLAS/CUDA/template/GB_cuda_cumsum: cumlative sum of an array on the GPU(s)
//------------------------------------------------------------------------------

// SuiteSparse:GraphBLAS, Timothy A. Davis, (c) 2017-2023, All Rights Reserved.
// SPDX-License-Identifier: Apache-2.0

//------------------------------------------------------------------------------

#ifndef GB_CUDA_CUMSUM
#define GB_CUDA_CUMSUM

#include <cub/cub.h>

typedef enum GB_cuda_cumsum_type
{
    GB_CUDA_CUMSUM_EXCLUSIVE,
    GB_CUDA_CUMSUM_INCLUSIVE
} GB_cuda_cumsum_type ;

__host__ GrB_Info GB_cuda_cumsum             // compute the cumulative sum of an array
(
    int64_t *restrict in,    // size n or n+1, input
    int64_t *restrict out,   // size n or n+1, output.
    // to do an in-place cumsum, pass out == in

    const int64_t n,
    cudaStream_t stream,
    GB_cuda_cumsum_type type
)
{

    //--------------------------------------------------------------------------
    // check inputs
    //--------------------------------------------------------------------------

    ASSERT (in != NULL) ;
    ASSERT (out != NULL) ;
    ASSERT (n >= 0) ;

    void *d_temp_storage = NULL;
    size_t temp_storage_bytes = 0 ;

    switch (type)
    {
        case GB_CUDA_CUMSUM_INCLUSIVE:
            cub::DeviceScan::InclusiveSum(d_temp_storage, temp_storage_bytes, in, out, n, stream) ;
            break;
        default:
            cub::DeviceScan::ExclusiveSum(d_temp_storage, temp_storage_bytes, in, out, n, stream) ;
    }

    CUDA_OK (cudaMalloc(&d_temp_storage, temp_storage_bytes)) ;

    // Run
    switch (type)
    {
        case GB_CUDA_CUMSUM_INCLUSIVE:
            cub::DeviceScan::InclusiveSum(d_temp_storage, temp_storage_bytes, in, out, n, stream) ;
            break;
        default:
            cub::DeviceScan::ExclusiveSum(d_temp_storage, temp_storage_bytes, in, out, n, stream) ;
    }

    CUDA_OK (cudaFree(d_temp_storage)) ;

    return GrB_SUCCESS;
}
#endif
