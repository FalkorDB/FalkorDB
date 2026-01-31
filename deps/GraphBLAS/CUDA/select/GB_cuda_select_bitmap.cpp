#include "select/GB_cuda_select.hpp"

#undef  GB_FREE_ALL
#define GB_FREE_ALL                                         \
{                                                           \
    GB_cuda_stream_pool_release (&stream) ;                 \
}

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

    GBURBLE (" (select bitmap on cuda)") ;

    cudaStream_t stream = nullptr ;
    GB_OK (GB_cuda_stream_pool_acquire (&stream)) ;

    int64_t anz = GB_nnz_held (A) ;

    int32_t number_of_sms = GB_Global_gpu_sm_get (0) ;
    int64_t raw_gridsz = GB_ICEIL (anz, GB_CUDA_SELECT_BITMAP_BLOCKDIM) ;
    int32_t gridsz = std::min (raw_gridsz, (int64_t) (number_of_sms * 256)) ;
    gridsz = std::max (gridsz, 1) ;

    GB_OK (GB_cuda_select_bitmap_jit (C, A,
        flipij, ythunk, op, stream, gridsz)) ;

    GB_OK (GB_cuda_stream_pool_release (&stream)) ;
    return GrB_SUCCESS ;
}
