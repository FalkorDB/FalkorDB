#include "GB_cuda_apply.hpp"

#undef GB_FREE_WORKSPACE
#define GB_FREE_WORKSPACE                               \
{                                                       \
    GB_FREE_WORK (&scalarx_cuda, scalarx_cuda_size) ;   \
}

#undef GB_FREE_ALL
#define GB_FREE_ALL ;

#define BLOCK_SIZE 512
#define LOG2_BLOCK_SIZE 9

GrB_Info GB_cuda_apply_binop
(
    GB_void *Cx,
    const GrB_Type ctype,
    const GrB_BinaryOp op,
    const GrB_Matrix A,
    const GB_void *scalarx,
    const bool bind1st
)
{
    ASSERT (scalarx != NULL) ;
    // make a copy of scalarx to ensure it's not on the CPU stack
    GB_void *scalarx_cuda = NULL ;
    size_t scalarx_cuda_size = 0 ;
    if (bind1st)
    {
        ASSERT (op->xtype != NULL) ;
        scalarx_cuda = GB_MALLOC_WORK (op->xtype->size, GB_void, &scalarx_cuda_size) ;
    }
    else
    {
        ASSERT (op->ytype != NULL) ;
        scalarx_cuda = GB_MALLOC_WORK (op->ytype->size, GB_void, &scalarx_cuda_size) ;
    }
    if (scalarx_cuda == NULL)
    {
        return (GrB_OUT_OF_MEMORY) ;
    }
    memcpy (scalarx_cuda, scalarx, scalarx_cuda_size) ;
    
    // FIXME: use the stream pool
    cudaStream_t stream ;
    CUDA_OK (cudaStreamCreate (&stream)) ;

    GrB_Index anz = GB_nnz_held (A) ;

    int32_t number_of_sms = GB_Global_gpu_sm_get (0) ;
    int64_t raw_gridsz = GB_ICEIL (anz, BLOCK_SIZE) ;
    // cap #of blocks to 256 * #of sms
    int32_t gridsz = std::min (raw_gridsz, (int64_t) (number_of_sms * 256)) ;
    
    GrB_Info info ;
    if (bind1st) {
        info = GB_cuda_apply_bind1st_jit (Cx, ctype, op, A, 
            scalarx_cuda, stream, gridsz, BLOCK_SIZE) ;
    } else {
        info = GB_cuda_apply_bind2nd_jit (Cx, ctype, op, A,
            scalarx_cuda, stream, gridsz, BLOCK_SIZE) ;
    }

    if (info == GrB_NO_VALUE) info = GrB_PANIC ;
    GB_OK (info) ;

    CUDA_OK (cudaStreamSynchronize (stream)) ;
    CUDA_OK (cudaStreamDestroy (stream)) ;

    GB_FREE_WORKSPACE ;
    return GrB_SUCCESS ; 

}
