#include "apply/GB_cuda_apply.hpp"

#undef  GB_FREE_WORKSPACE
#define GB_FREE_WORKSPACE                                   \
{                                                           \
    GB_FREE_MEMORY (&scalarx_cuda, scalarx_cuda_size) ;     \
}

#undef  GB_FREE_ALL
#define GB_FREE_ALL                                         \
{                                                           \
    GB_FREE_WORKSPACE ;                                     \
    GB_cuda_stream_pool_release (&stream) ;                 \
}

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
    GrB_Info info ;
    GB_void *scalarx_cuda = NULL ;
    size_t scalarx_cuda_size = 0 ;

    cudaStream_t stream = nullptr ;
    GB_OK (GB_cuda_stream_pool_acquire (&stream)) ;

    ASSERT (scalarx != NULL) ;
    // make a copy of scalarx to ensure it's not on the CPU stack

    if (bind1st)
    {
        ASSERT (op->xtype != NULL) ;
        scalarx_cuda = (GB_void *) GB_MALLOC_MEMORY (1, op->xtype->size,
            &scalarx_cuda_size) ;
    }
    else
    {
        ASSERT (op->ytype != NULL) ;
        scalarx_cuda = (GB_void *) GB_MALLOC_MEMORY (1, op->ytype->size,
            &scalarx_cuda_size) ;
    }
    if (scalarx_cuda == NULL)
    {
        GB_FREE_ALL ;
        return (GrB_OUT_OF_MEMORY) ;
    }
    memcpy (scalarx_cuda, scalarx, scalarx_cuda_size) ;

    GrB_Index anz = GB_nnz_held (A) ;

    int32_t number_of_sms = GB_Global_gpu_sm_get (0) ;
    int64_t raw_gridsz = GB_ICEIL (anz, BLOCK_SIZE) ;
    // cap #of blocks to 256 * #of sms
    int32_t gridsz = std::min (raw_gridsz, (int64_t) (number_of_sms * 256)) ;

    if (bind1st)
    {
        GB_OK (GB_cuda_apply_bind1st_jit (Cx, ctype, op, A, 
            scalarx_cuda, stream, gridsz, BLOCK_SIZE)) ;
    }
    else
    {
        GB_OK (GB_cuda_apply_bind2nd_jit (Cx, ctype, op, A,
            scalarx_cuda, stream, gridsz, BLOCK_SIZE)) ;
    }

    GB_FREE_WORKSPACE ;
    GB_OK (GB_cuda_stream_pool_release (&stream)) ;
    return GrB_SUCCESS ; 
}
