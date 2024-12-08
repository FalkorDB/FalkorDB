//------------------------------------------------------------------------------
// GB_kroner_jit: kronecker product
//------------------------------------------------------------------------------

// SuiteSparse:GraphBLAS, Timothy A. Davis, (c) 2017-2024, All Rights Reserved.
// SPDX-License-Identifier: Apache-2.0

//------------------------------------------------------------------------------

#include "GB.h"
#include "jitifyer/GB_stringify.h"

typedef GB_JIT_KERNEL_KRONER_PROTO ((*GB_jit_dl_function)) ;

GrB_Info GB_kroner_jit
(
    // output:
    GrB_Matrix C,
    // input:
    const GrB_BinaryOp binaryop,
    const bool flipij,
    const GrB_Matrix A,
    const GrB_Matrix B,
    const int nthreads
)
{ 

    //--------------------------------------------------------------------------
    // encodify the problem
    //--------------------------------------------------------------------------

    // C is sparse, hypersparse, or full, where sparse/hypersparse leads to
    // the same kernel, so C is treated as if sparse.  C->h is constructed
    // in the caller, not in the JIT kernel.
    int C_sparsity = (GB_IS_SPARSE (C) || GB_IS_HYPERSPARSE (C)) ?
        GxB_HYPERSPARSE : GxB_FULL ;

    GB_jit_encoding encoding ;
    char *suffix ;
    uint64_t hash = GB_encodify_ewise (&encoding, &suffix,
        GB_JIT_KERNEL_KRONER, /* is_ewisemult: */ false, /* C_iso: */ C->iso,
        /* C_in_iso: */ false, C_sparsity, C->type, /* M: */ NULL, true, false,
        binaryop, flipij, false, A, B) ;

    //--------------------------------------------------------------------------
    // get the kernel function pointer, loading or compiling it if needed
    //--------------------------------------------------------------------------

    void *dl_function ;
    GrB_Info info = GB_jitifyer_load (&dl_function,
        GB_jit_ewise_family, "kroner",
        hash, &encoding, suffix, NULL, NULL,
        (GB_Operator) binaryop, C->type, A->type, B->type) ;
    if (info != GrB_SUCCESS) return (info) ;

    //--------------------------------------------------------------------------
    // call the jit kernel and return result
    //--------------------------------------------------------------------------

    GB_jit_dl_function GB_jit_kernel = (GB_jit_dl_function) dl_function ;
    return (GB_jit_kernel (C, A, B, nthreads, binaryop->theta)) ;
}

