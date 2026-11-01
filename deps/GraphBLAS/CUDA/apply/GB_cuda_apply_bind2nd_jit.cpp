//------------------------------------------------------------------------------
// GB_cuda_apply_bind2nd_jit: JIT kernel for apply, bind2nd case
//------------------------------------------------------------------------------

// SuiteSparse:GraphBLAS, Timothy A. Davis, (c) 2017-2026, All Rights Reserved.
// SPDX-License-Identifier: Apache-2.0

//------------------------------------------------------------------------------

#include "apply/GB_cuda_apply.hpp"

extern "C"
{
    typedef GB_JIT_CUDA_KERNEL_APPLY_BIND2ND_PROTO ((*GB_jit_dl_function)) ;
}


GrB_Info GB_cuda_apply_bind2nd_jit
(
    // output:
    GB_void *Cx,
    // input:
    const GrB_Type ctype,
    const GrB_BinaryOp binaryop,
    const GrB_Matrix A,
    const GB_void *scalarx,
    // CUDA stream and launch parameters:
    int device,
    cudaStream_t stream,
    int32_t gridsz
)
{
    //--------------------------------------------------------------------------
    // encodify the problem
    //--------------------------------------------------------------------------

    GB_jit_encoding encoding ;
    char *suffix ;
    uint64_t hash = GB_encodify_ewise (&encoding, &suffix,
        GB_JIT_CUDA_KERNEL_APPLYBIND2,
        /* C_iso: */ false, /* C_in_iso: */ false, GxB_FULL, ctype,
        /* pji is_32: ignored; there is no C matrix: */ false, false, false,
        /* M: */ NULL, /* Mask_struct: */ false, /* Mask_comp: */ false,
        binaryop, /* flipij: */ false, /* flipxy: */ false, A, /* B: */ NULL) ;

    //--------------------------------------------------------------------------
    // get the kernel function pointer, loading or compiling it if needed
    //--------------------------------------------------------------------------

    void *dl_function ;
    GrB_Info info = GB_jitifyer_load (&dl_function,
        GB_jit_ewise_family, "cuda_apply_bind2nd",
        hash, &encoding, suffix, NULL, NULL,
        (GB_Operator) binaryop, ctype, A->type, NULL) ;
    if (info != GrB_SUCCESS) return (info) ;

    //--------------------------------------------------------------------------
    // call the jit kernel and return result
    //--------------------------------------------------------------------------

    GB_jit_dl_function GB_jit_kernel = (GB_jit_dl_function) dl_function ;
    return (GB_jit_kernel (Cx, A, scalarx, device, stream, gridsz,
        &GB_callback)) ;
}

