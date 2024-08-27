#include "GB_cuda_ewise.hpp"

#undef GB_FREE_WORKSPACE
#define GB_FREE_WORKSPACE ;

#undef GB_FREE_ALL
#define GB_FREE_ALL ;

#define BLOCK_SIZE 128
#define LOG2_BLOCK_SIZE 7

GrB_Info GB_cuda_colscale
(
    GrB_Matrix C,
    const GrB_Matrix A,
    const GrB_Matrix D,
    const GrB_Semiring semiring,
    const bool flipxy
)
{
    // FIXME: use the stream pool
    cudaStream_t stream ;
    CUDA_OK (cudaStreamCreate (&stream)) ;

    // compute gridsz, blocksz, call GB_cuda_rowscale_jit
    GrB_Index anz = GB_nnz_held (A) ;
    
    int32_t gridsz = 1 + (anz >> LOG2_BLOCK_SIZE) ;

    GrB_Info info = GB_cuda_colscale_jit ( C, A, D, 
        semiring->multiply, flipxy, stream, gridsz, BLOCK_SIZE) ;
    
    if (info == GrB_NO_VALUE) info = GrB_PANIC ;
    GB_OK (info) ;

    CUDA_OK (cudaStreamSynchronize (stream)) ;
    CUDA_OK (cudaStreamDestroy (stream)) ;
    return GrB_SUCCESS ; 

}
