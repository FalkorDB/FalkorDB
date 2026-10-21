//------------------------------------------------------------------------------
// GB_cuda_mask_assign_jit: JIT kernel for C<M>=A or C<!M>=A
//------------------------------------------------------------------------------

// SuiteSparse:GraphBLAS, Timothy A. Davis, (c) 2017-2026, All Rights Reserved.
// SPDX-License-Identifier: Apache-2.0

//------------------------------------------------------------------------------

#include "mask/GB_cuda_mask.hpp"

extern "C"
{
    typedef GB_JIT_CUDA_KERNEL_EWISE_SPARSE_PROTO ((*GB_jit_dl_function)) ;
}

GrB_Info GB_cuda_mask_assign_jit
(
    // output:
    GrB_Matrix C,
    // input:
    const GrB_Matrix A,
    const GrB_Matrix B,                     // B == M is the mask
    const bool Mask_struct,
    const bool Mask_comp,
    const GrB_BinaryOp binaryop,            // FIRST_atype
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
        GB_JIT_CUDA_KERNEL_MASK_ASSIGN_SPARSE,
        /* C_iso: */ A->iso, /* C_in_iso: */ false,
        /* C_sparsity: */ GxB_HYPERSPARSE,
        A->type, A->p_is_32, A->j_is_32, A->i_is_32,
        /* M: */ B, /* Mask_struct: */ Mask_struct, /* Mask_comp: */ Mask_comp,
        binaryop, /* flipij: */ false, /* flipxy: */ false, A, B) ;

    //--------------------------------------------------------------------------
    // get the kernel function pointer, loading or compiling it if needed
    //--------------------------------------------------------------------------

    void *dl_function ;
    GrB_Info info = GB_jitifyer_load (&dl_function,
        GB_jit_ewise_family, "cuda_ewise_sparse",
        hash, &encoding, suffix, NULL, NULL,
        (GB_Operator) binaryop, C->type, A->type, B->type) ;
    if (info != GrB_SUCCESS) return (info) ;

    //--------------------------------------------------------------------------
    // call the jit kernel and return result
    //--------------------------------------------------------------------------

    GB_jit_dl_function GB_jit_kernel = (GB_jit_dl_function) dl_function ;
    return (GB_jit_kernel (C, A, B, NULL, NULL, NULL,
        device, stream, gridsz, &GB_callback)) ;
}

