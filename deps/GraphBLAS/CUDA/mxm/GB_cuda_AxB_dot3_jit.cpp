//------------------------------------------------------------------------------
// GB_cuda_AxB_dot3_jit: reduce a matrix to a scalar, via the CUDA JIT
//------------------------------------------------------------------------------

// SuiteSparse:GraphBLAS, Timothy A. Davis, (c) 2017-2025, All Rights Reserved.
// SPDX-License-Identifier: Apache-2.0

//------------------------------------------------------------------------------

#include "mxm/GB_cuda_AxB.hpp"

extern "C"
{
    typedef GB_JIT_CUDA_KERNEL_DOT3_PROTO ((*GB_jit_dl_function)) ;
}

GrB_Info GB_cuda_AxB_dot3_jit
(
    // input/output:
    GrB_Matrix C,               // FIXME: allow iso for this kernel
    // input:
    const GrB_Matrix M, const bool Mask_struct,
    const GrB_Matrix A,
    const GrB_Matrix B,
    const GrB_Semiring semiring,
    const bool flipxy,
    // CUDA stream, device, and # of ms
    cudaStream_t stream,
    int device,
    int number_of_sms
)
{ 

    //--------------------------------------------------------------------------
    // encodify the problem
    //--------------------------------------------------------------------------

    GB_jit_encoding encoding ;
    char *suffix ;
    uint64_t hash = GB_encodify_mxm (&encoding, &suffix,
        GB_JIT_CUDA_KERNEL_AXB_DOT3,
        // FIXME: allow C to be iso:
        /* C->iso: */ false, /* C_in_iso: */ false,
        GB_sparsity (C), C->type, C->p_is_32, C->j_is_32, C->i_is_32,
        M, Mask_struct, /* Mask_comp: */ false, semiring, flipxy, A, B) ;

    //--------------------------------------------------------------------------
    // get the kernel function pointer, loading or compiling it if needed
    //--------------------------------------------------------------------------

    void *dl_function ;
    GrB_Info info = GB_jitifyer_load (&dl_function,
        GB_jit_mxm_family, "cuda_AxB_dot3",
        hash, &encoding, suffix, semiring, NULL,
        NULL, C->type, A->type, B->type) ;
    if (info != GrB_SUCCESS) { printf ("cuda_AxB_dot3 failed %d\n", info) ; return (info) ; }

    //--------------------------------------------------------------------------
    // call the jit kernel and return result
    //--------------------------------------------------------------------------

    GB_jit_dl_function GB_jit_kernel = (GB_jit_dl_function) dl_function ;
    return (GB_jit_kernel (C, M, A, B, stream, device, number_of_sms,
        semiring->multiply->theta, &GB_callback)) ;
}

