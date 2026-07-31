//------------------------------------------------------------------------------
// GB_cuda_transpose_prep_jit: construct Key_input from A for CUDA transpose
//------------------------------------------------------------------------------

// SuiteSparse:GraphBLAS, Timothy A. Davis, (c) 2017-2026, All Rights Reserved.
// SPDX-License-Identifier: Apache-2.0

//------------------------------------------------------------------------------

#include "transpose/GB_cuda_transpose.hpp"

typedef GB_JIT_CUDA_KERNEL_TRANSPOSE_PREP_PROTO ((*GB_jit_dl_function)) ;

GrB_Info GB_cuda_transpose_prep_jit
(
    // output:
    GB_void *Key_input,
    // input:
    bool Key_is_32,
    const GrB_Matrix A,
    cudaStream_t stream,
    int32_t gridsz
)
{ 

    //--------------------------------------------------------------------------
    // encodify the problem
    //--------------------------------------------------------------------------

    GB_jit_encoding encoding ;
    char *suffix ;
    GB_Operator op = (GB_Operator) GrB_IDENTITY_BOOL ;
    uint64_t hash = GB_encodify_apply (&encoding, &suffix,
        GB_JIT_CUDA_KERNEL_TRANSPOSE_PREP,
        /* C not present: */ GxB_FULL, false,
        /* ctype: unused: */ GrB_BOOL,
        /* Cp_is_32, unused: */ false,
        /* Cj_is_32: unused: */ false,
        /* Ci_is_32, used for Key_is_32: */ Key_is_32,
        op, /* flipij: */ false, GB_sparsity (A), true, GrB_BOOL,
        A->p_is_32, A->j_is_32, A->i_is_32,
        /* A iso (values not used): */ true, A->nzombies) ;

    //--------------------------------------------------------------------------
    // get the kernel function pointer, loading or compiling it if needed
    //--------------------------------------------------------------------------

    void *dl_function ;
    GrB_Info info = GB_jitifyer_load (&dl_function,
        GB_jit_apply_family, "cuda_transpose_prep",
        hash, &encoding, suffix, /* semiring: */ NULL, /* monoid: */ NULL,
        /* op, not used: */ op,
        /* types (not used): */ GrB_BOOL, GrB_BOOL, NULL) ;
    if (info != GrB_SUCCESS) return (info) ;

    //--------------------------------------------------------------------------
    // call the jit kernel and return result
    //--------------------------------------------------------------------------

    #include "include/GB_pedantic_disable.h"
    GB_jit_dl_function GB_jit_kernel = (GB_jit_dl_function) dl_function ;
    return (GB_jit_kernel (Key_input, A, stream, gridsz)) ;
}

