#include "mxm/GB_cuda_ewise.hpp"

#undef  GB_FREE_ALL
#define GB_FREE_ALL                                         \
{                                                           \
    GB_cuda_stream_pool_release (&stream) ;                 \
}

#define BLOCK_SIZE 128
#define LOG2_BLOCK_SIZE 7

GrB_Info GB_cuda_rowscale
(
    GrB_Matrix C,
    const GrB_Matrix D,
    const GrB_Matrix B,
    const GrB_Semiring semiring,
    const bool flipxy
)
{
    GrB_Info info ;
    cudaStream_t stream = nullptr ;
    GB_OK (GB_cuda_stream_pool_acquire (&stream)) ;

    // compute gridsz, blocksz, call GB_cuda_rowscale_jit
    GrB_Index bnz = GB_nnz_held (B) ;

    int32_t gridsz = 1 + (bnz >> LOG2_BLOCK_SIZE) ;

    GB_OK (GB_cuda_rowscale_jit ( C, D, B,
        semiring->multiply, flipxy, stream, gridsz, BLOCK_SIZE)) ;

    GB_OK (GB_cuda_stream_pool_release (&stream)) ;
    return GrB_SUCCESS ;
}

