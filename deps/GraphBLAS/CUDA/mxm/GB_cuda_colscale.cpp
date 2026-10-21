//------------------------------------------------------------------------------
// GB_cuda_colscale: C = A*D
//------------------------------------------------------------------------------

// SuiteSparse:GraphBLAS, Timothy A. Davis, (c) 2017-2026, All Rights Reserved.
// SPDX-License-Identifier: Apache-2.0

//------------------------------------------------------------------------------

#include "mxm/GB_cuda_ewise.hpp"

#undef  GB_FREE_ALL
#define GB_FREE_ALL                                         \
{                                                           \
    GB_cuda_stream_pool_release (&stream) ;                 \
}

GrB_Info GB_cuda_colscale
(
    GrB_Matrix C,
    const GrB_Matrix A,
    const GrB_Matrix D,
    const GrB_Semiring semiring,
    const bool flipxy
)
{
    GrB_Info info ;
    cudaStream_t stream = nullptr ;

    int data_arena = C->data_arena ;
    int device = data_arena - GxB_NARENAS ;

    GB_OK (GB_cuda_stream_pool_acquire (device, &stream)) ;

    GB_OK (GB_wait_arenas (C)) ;
    GB_OK (GB_wait_arenas (A)) ;
    GB_OK (GB_wait_arenas (D)) ;

    GrB_Index anz = GB_nnz_held (A) ;

    // determine the geometry of the CUDA kernel launches
    int32_t number_of_sms = GB_Global_gpu_sm_get (device) ;
    int64_t raw_gridsz = GB_ICEIL (anz, GB_CUDA_SCALE_CHUNKSIZE) ;
    int32_t gridsz = std::min (raw_gridsz, (int64_t) (number_of_sms * 256)) ;
    gridsz = std::max (gridsz, 1) ;

    GBURBLE ("(cuda colscale, device %d, sms: %d, gridsz: %d) ",
        device, number_of_sms, gridsz) ;

    GB_OK (GB_cuda_colscale_jit (C, A, D, semiring->multiply,
        flipxy, device, stream, gridsz)) ;
    
    GB_OK (GB_cuda_stream_pool_release (&stream)) ;
    return GrB_SUCCESS ; 
}

