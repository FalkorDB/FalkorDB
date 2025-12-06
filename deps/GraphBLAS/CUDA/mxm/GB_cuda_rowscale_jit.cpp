#include "mxm/GB_cuda_ewise.hpp"

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
        /* C_iso: */ false, /* C_in_iso: */ false, GB_sparsity (C), C->type,
        C->p_is_32, C->j_is_32, C->i_is_32,
        /* M: */ NULL, /* Mask_comp: */ false, /* Mask_struct: */ false,
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
    return (GB_jit_kernel (C, D, B, stream, gridsz, blocksz, &GB_callback)) ;
}
