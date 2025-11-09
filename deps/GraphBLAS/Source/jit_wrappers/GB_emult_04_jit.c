//------------------------------------------------------------------------------
// GB_emult_04_jit: C<M>=A.*B emult_04 method, via the JIT
//------------------------------------------------------------------------------

// SuiteSparse:GraphBLAS, Timothy A. Davis, (c) 2017-2025, All Rights Reserved.
// SPDX-License-Identifier: Apache-2.0

//------------------------------------------------------------------------------

#include "GB.h"
#include "jitifyer/GB_stringify.h"

typedef GB_JIT_KERNEL_EMULT_04_PROTO ((*GB_jit_dl_function)) ;

GrB_Info GB_emult_04_jit      // C<M>=A.*B, emult_04, via the JIT
(
    // input/output:
    GrB_Matrix C,
    // input:
    const int C_sparsity,
    const GrB_Matrix M,
    const bool Mask_struct,
    const GrB_BinaryOp binaryop,
    const bool flipij,
    const GrB_Matrix A,
    const GrB_Matrix B,
    const uint64_t *restrict Cp_kfirst,
    const int64_t *M_ek_slicing,
    const int M_ntasks,
    const int M_nthreads
)
{ 

    //--------------------------------------------------------------------------
    // encodify the problem
    //--------------------------------------------------------------------------

    GB_jit_encoding encoding ;
    char *suffix ;
    uint64_t hash = GB_encodify_ewise (&encoding, &suffix,
        GB_JIT_KERNEL_EMULT4, true,
        false, false, C_sparsity, C->type, C->p_is_32, C->j_is_32, C->i_is_32,
        M, Mask_struct, false, binaryop, flipij, false, A, B) ;

    //--------------------------------------------------------------------------
    // get the kernel function pointer, loading or compiling it if needed
    //--------------------------------------------------------------------------

    void *dl_function ;
    GrB_Info info = GB_jitifyer_load (&dl_function,
        GB_jit_ewise_family, "emult_04",
        hash, &encoding, suffix, NULL, NULL,
        (GB_Operator) binaryop, C->type, A->type, B->type) ;
    if (info != GrB_SUCCESS) return (info) ;

    //--------------------------------------------------------------------------
    // call the jit kernel and return result
    //--------------------------------------------------------------------------

    #include "include/GB_pedantic_disable.h"
    GB_jit_dl_function GB_jit_kernel = (GB_jit_dl_function) dl_function ;
    return (GB_jit_kernel (C, M, Mask_struct, A, B, Cp_kfirst, M_ek_slicing,
        M_ntasks, M_nthreads, binaryop->theta, &GB_callback)) ;
}

