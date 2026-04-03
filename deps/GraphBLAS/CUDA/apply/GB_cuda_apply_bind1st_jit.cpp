#include "apply/GB_cuda_apply.hpp"

extern "C"
{
    typedef GB_JIT_CUDA_KERNEL_APPLY_BIND1ST_PROTO ((*GB_jit_dl_function)) ;
}


GrB_Info GB_cuda_apply_bind1st_jit
(
    // output:
    GB_void *Cx,
    // input:
    const GrB_Type ctype,
    const GrB_BinaryOp binaryop,
    const GrB_Matrix B,
    const GB_void *scalarx,
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
        GB_JIT_CUDA_KERNEL_APPLYBIND1, /* is_eWiseMult: */ false,
        /* C_iso: */ false, /* C_in_iso: */ false, GxB_FULL, ctype,
        /* pji is_32: ignored; there is no C matrix: */ false, false, false,
        /* M: */ NULL, /* Mask_struct: */ false, /* Mask_comp: */ false,
        binaryop, /* flipij: */ false, /* flipxy: */ false, /* A: */ NULL, B) ;

    //--------------------------------------------------------------------------
    // get the kernel function pointer, loading or compiling it if needed
    //--------------------------------------------------------------------------

    void *dl_function ;
    GrB_Info info = GB_jitifyer_load (&dl_function,
        GB_jit_ewise_family, "cuda_apply_bind1st",
        hash, &encoding, suffix, NULL, NULL,
        (GB_Operator) binaryop, ctype, NULL, B->type) ;
    if (info != GrB_SUCCESS) return (info) ;

    //--------------------------------------------------------------------------
    // call the jit kernel and return result
    //--------------------------------------------------------------------------

    GB_jit_dl_function GB_jit_kernel = (GB_jit_dl_function) dl_function ;
    return (GB_jit_kernel (Cx, scalarx, B, stream, gridsz, blocksz,
        &GB_callback)) ;
}
