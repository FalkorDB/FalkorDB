//------------------------------------------------------------------------------
// GB_cuda_add_jit: JIT kernel for eWiseAdd and eWiseUnion: C=A+B
//------------------------------------------------------------------------------

// SuiteSparse:GraphBLAS, Timothy A. Davis, (c) 2017-2026, All Rights Reserved.
// SPDX-License-Identifier: Apache-2.0

//------------------------------------------------------------------------------

#include "add/GB_cuda_add.hpp"

extern "C"
{
    typedef GB_JIT_CUDA_KERNEL_EWISE_SPARSE_PROTO ((*GB_jit_dl_function)) ;
}

GrB_Info GB_cuda_add_jit
(
    // output:
    GrB_Matrix C,
    // input:
    const bool C_iso,
    const GrB_Matrix A,
    const GrB_Matrix B,
    const bool is_eWiseUnion,
    const GB_void *alpha_scalar,
    const GB_void *beta_scalar,
    const void *theta,
    const GrB_BinaryOp binaryop,
    const bool flipij,
    const bool A_and_B_are_disjoint,        // FIXME: add to encoding
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
        GB_JIT_CUDA_KERNEL_ADD_SPARSE,
        /* C_iso: */ C_iso, /* C_in_iso: */ false,
        /* C_sparsity: */ GxB_HYPERSPARSE,
        C->type, C->p_is_32, C->j_is_32, C->i_is_32,
        /* M: */ NULL, /* Mask_struct: */ false, /* Mask_comp: */ false,
        binaryop, /* flipij: */ flipij, /* flipxy: */ false, A, B) ;

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
    return (GB_jit_kernel (C, A, B, alpha_scalar, beta_scalar, theta,
        device, stream, gridsz, &GB_callback)) ;
}

