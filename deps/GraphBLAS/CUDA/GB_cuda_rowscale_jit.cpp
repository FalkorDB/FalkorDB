#include "GB_cuda_ewise.hpp"

extern "C"
{
    typedef GB_JIT_CUDA_KERNEL_ROWSCALE_PROTO ((*GB_jit_dl_function)) ;
}

GrB_Info GB_cuda_rowscale_jit
(
    // output:
    GrB_Matrix C,
    // input:
    GrB_Matrix D,
    GrB_Matrix B,
    GrB_BinaryOp binaryop,
    bool flipxy,
    // CUDA stream and launch parameters:
    cudaStream_t stream,
    int32_t gridsz,
    int32_t blocksz
)
{ 

    //--------------------------------------------------------------------------
    // encodify the problem
    //--------------------------------------------------------------------------

    GB_jit_encoding encoding ;
    char *suffix ;
    uint64_t hash = GB_encodify_ewise (&encoding, &suffix,
        GB_JIT_CUDA_KERNEL_ROWSCALE, false,
        false, false, GB_sparsity (C), C->type, NULL, false, false,
        binaryop, false, flipxy, D, B) ;

    //--------------------------------------------------------------------------
    // get the kernel function pointer, loading or compiling it if needed
    //--------------------------------------------------------------------------

    void *dl_function ;
    GrB_Info info = GB_jitifyer_load (&dl_function,
        GB_jit_ewise_family, "cuda_rowscale",
        hash, &encoding, suffix, NULL, NULL,
        (GB_Operator) binaryop, C->type, D->type, B->type) ;
    if (info != GrB_SUCCESS) return (info) ;

    //--------------------------------------------------------------------------
    // call the jit kernel and return result
    //--------------------------------------------------------------------------

    GB_jit_dl_function GB_jit_kernel = (GB_jit_dl_function) dl_function ;
    return (GB_jit_kernel (C, D, B, stream, gridsz, blocksz)) ;
}
