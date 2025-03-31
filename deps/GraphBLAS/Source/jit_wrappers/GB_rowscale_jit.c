//------------------------------------------------------------------------------
// GB_rowscale_jit: C=D*B rowscale method, via the JIT
//------------------------------------------------------------------------------

// SuiteSparse:GraphBLAS, Timothy A. Davis, (c) 2017-2025, All Rights Reserved.
// SPDX-License-Identifier: Apache-2.0

//------------------------------------------------------------------------------

#include "GB.h"
#include "jitifyer/GB_stringify.h"

typedef GB_JIT_KERNEL_ROWSCALE_PROTO ((*GB_jit_dl_function)) ;

GrB_Info GB_rowscale_jit      // C=D*B, rowscale, via the JIT
(
    // input/output:
    GrB_Matrix C,
    // input:
    const GrB_Matrix D,
    const GrB_Matrix B,
    const GrB_BinaryOp binaryop,
    const bool flipxy,
    const int nthreads
)
{ 

    //--------------------------------------------------------------------------
    // encodify the problem
    //--------------------------------------------------------------------------

    GB_jit_encoding encoding ;
    char *suffix ;
    uint64_t hash = GB_encodify_ewise (&encoding, &suffix,
        GB_JIT_KERNEL_ROWSCALE, false,
        /* C_iso: */ false, /* C_in_iso: */ false, GB_sparsity (C), C->type,
        C->p_is_32, C->j_is_32, C->i_is_32,
        /* M: */ NULL, /* Mask_comp: */ false, /* Mask_struct: */ false,
        binaryop, false, flipxy, D, B) ;

    //--------------------------------------------------------------------------
    // get the kernel function pointer, loading or compiling it if needed
    //--------------------------------------------------------------------------

    void *dl_function ;
    GrB_Info info = GB_jitifyer_load (&dl_function,
        GB_jit_ewise_family, "rowscale",
        hash, &encoding, suffix, NULL, NULL,
        (GB_Operator) binaryop, C->type, D->type, B->type) ;
    if (info != GrB_SUCCESS) return (info) ;

    //--------------------------------------------------------------------------
    // call the jit kernel and return result
    //--------------------------------------------------------------------------

    #include "include/GB_pedantic_disable.h"
    GB_jit_dl_function GB_jit_kernel = (GB_jit_dl_function) dl_function ;
    return (GB_jit_kernel (C, D, B, nthreads, &GB_callback)) ;
}

