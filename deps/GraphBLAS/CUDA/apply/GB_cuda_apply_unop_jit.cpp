#include "apply/GB_cuda_apply.hpp"

extern "C"
{
    typedef GB_JIT_CUDA_KERNEL_APPLY_UNOP_PROTO ((*GB_jit_dl_function)) ;
}


GrB_Info GB_cuda_apply_unop_jit
(
    // output:
    GB_void *Cx,
    // input:
    const GrB_Type ctype,
    const GB_Operator op,
    const bool flipij,
    const GrB_Matrix A,
    const GB_void *ythunk,
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
    uint64_t hash = GB_encodify_apply (&encoding, &suffix,
        GB_JIT_CUDA_KERNEL_APPLYUNOP, GxB_FULL, false, ctype,
        /* pji_is_32: ignored; no matrix C: */ false, false, false,
        op, flipij, GB_sparsity (A), true, A->type,
        A->p_is_32, A->j_is_32, A->i_is_32,
        A->iso, A->nzombies) ;

    //--------------------------------------------------------------------------
    // get the kernel function pointer, loading or compiling it if needed
    //--------------------------------------------------------------------------

    void *dl_function ;
    GrB_Info info = GB_jitifyer_load (&dl_function,
        GB_jit_apply_family, "cuda_apply_unop",
        hash, &encoding, suffix, NULL, NULL,
        op, ctype, A->type, NULL) ;
    if (info != GrB_SUCCESS) return (info) ;

    //--------------------------------------------------------------------------
    // call the jit kernel and return result
    //--------------------------------------------------------------------------

    GB_jit_dl_function GB_jit_kernel = (GB_jit_dl_function) dl_function ;
    return (GB_jit_kernel (Cx, A, ythunk, stream, gridsz, blocksz,
        &GB_callback)) ;
}
