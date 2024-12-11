#include "GB_cuda_select.hpp"

#undef GB_FREE_WORKSPACE
#define GB_FREE_WORKSPACE                           \
{                                                   \
    GB_FREE_WORK (&ythunk_cuda, ythunk_cuda_size) ; \
}

#undef GB_FREE_ALL
#define GB_FREE_ALL ;

#define BLOCK_SIZE 512
#define LOG2_BLOCK_SIZE 9

GrB_Info GB_cuda_select_sparse
(
    GrB_Matrix C,
    const bool C_iso,
    const GrB_IndexUnaryOp op,
    const bool flipij,
    const GrB_Matrix A,
    const GB_void *ythunk
)
{
    // in progress
    return (GrB_NOT_IMPLEMENTED) ;

#if 0

    // check inputs
    ASSERT (C != NULL && !(C->static_header)) ;
    ASSERT (A != NULL && !(A->static_header)) ;

    GB_void *ythunk_cuda = NULL ;
    size_t ythunk_cuda_size = 0 ;
    if (ythunk != NULL && op != NULL && op->ytype != NULL)
    {
        // make a copy of ythunk, since ythunk might be allocated on
        // the CPU stack and thus not accessible to the CUDA kernel.
        ythunk_cuda = GB_MALLOC_WORK (op->ytype->size, GB_void,
            &ythunk_cuda_size) ;

        if (ythunk_cuda == NULL)
        {
            return (GrB_OUT_OF_MEMORY) ;
        }
        memcpy (ythunk_cuda, ythunk, op->ytype->size) ;
    }
    // FIXME: use the stream pool
    cudaStream_t stream ;
    CUDA_OK (cudaStreamCreate (&stream)) ;

    GrB_Index anz = GB_nnz_held (A) ;

    int32_t number_of_sms = GB_Global_gpu_sm_get (0) ;
    int64_t raw_gridsz = GB_ICEIL (anz, BLOCK_SIZE) ;
    int32_t gridsz = std::min (raw_gridsz, (int64_t) (number_of_sms * 256)) ;

    GrB_Info info = GrB_NO_VALUE ;

    info = GB_cuda_select_sparse_jit (C, C_iso, A,
        flipij, ythunk_cuda, op, stream, gridsz, BLOCK_SIZE) ;
    if (info == GrB_NO_VALUE) info = GrB_PANIC ;
    GB_OK (info) ;

    CUDA_OK (cudaStreamSynchronize (stream)) ;
    CUDA_OK (cudaStreamDestroy (stream)) ;

    GB_FREE_WORKSPACE ;
    return info ;
#endif
}
