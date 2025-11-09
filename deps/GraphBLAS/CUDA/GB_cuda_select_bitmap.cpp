#include "GB_cuda_select.hpp"

#undef  GB_FREE_WORKSPACE
#define GB_FREE_WORKSPACE                                   \
{                                                           \
    if (stream != nullptr)                                  \
    {                                                       \
        cudaStreamSynchronize (stream) ;                    \
        cudaStreamDestroy (stream) ;                        \
    }                                                       \
    stream = nullptr ;                                      \
}

#undef  GB_FREE_ALL
#define GB_FREE_ALL GB_FREE_WORKSPACE

#define BLOCK_SIZE 512
#define LOG2_BLOCK_SIZE 9

GrB_Info GB_cuda_select_bitmap
(
    GrB_Matrix C,
    const GrB_Matrix A,
    const bool flipij,
    const GB_void *ythunk,
    const GrB_IndexUnaryOp op
)
{
    GrB_Info info ;

    // FIXME: use the stream pool
    cudaStream_t stream = nullptr ;
    CUDA_OK (cudaStreamCreate (&stream)) ;

    GrB_Index anz = GB_nnz_held (A) ;

    int32_t number_of_sms = GB_Global_gpu_sm_get (0) ;
    int64_t raw_gridsz = GB_ICEIL (anz, BLOCK_SIZE) ;
    int32_t gridsz = std::min (raw_gridsz, (int64_t) (number_of_sms * 256)) ;

    GB_OK (GB_cuda_select_bitmap_jit (C, A,
        flipij, ythunk, op, stream, gridsz, BLOCK_SIZE)) ;

    GB_FREE_WORKSPACE ;
    return GrB_SUCCESS ;
}
