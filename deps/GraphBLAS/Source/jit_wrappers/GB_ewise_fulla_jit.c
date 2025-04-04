//------------------------------------------------------------------------------
// GB_ewise_fulla_jit: C+=A+B via the JIT
//------------------------------------------------------------------------------

// SuiteSparse:GraphBLAS, Timothy A. Davis, (c) 2017-2025, All Rights Reserved.
// SPDX-License-Identifier: Apache-2.0

//------------------------------------------------------------------------------

#include "GB.h"
#include "jitifyer/GB_stringify.h"

typedef GB_JIT_KERNEL_EWISE_FULLA_PROTO ((*GB_jit_dl_function)) ;

GrB_Info GB_ewise_fulla_jit    // C+=A+B via the JIT
(
    // input/output:
    GrB_Matrix C,
    // input:
    const GrB_BinaryOp binaryop,
    const GrB_Matrix A,
    const GrB_Matrix B,
    const int nthreads
)
{ 

    //--------------------------------------------------------------------------
    // encodify the problem
    //--------------------------------------------------------------------------

    GB_jit_encoding encoding ;
    char *suffix ;
    uint64_t hash = GB_encodify_ewise (&encoding, &suffix,
        GB_JIT_KERNEL_EWISEFA, /* is_eWiseMult: */ false,
        /* C_iso: */ false, false, GxB_FULL, C->type,
        /* is_32: */ false, false, false,
        /* M: */ NULL, false, false, binaryop,
        /* flipij: */ false, /* flipxy: */ false, A, B) ;

    //--------------------------------------------------------------------------
    // get the kernel function pointer, loading or compiling it if needed
    //--------------------------------------------------------------------------

    void *dl_function ;
    GrB_Info info = GB_jitifyer_load (&dl_function,
        GB_jit_ewise_family, "ewise_fulla",
        hash, &encoding, suffix, NULL, NULL,
        (GB_Operator) binaryop, C->type, A->type, B->type) ;
    if (info != GrB_SUCCESS) return (info) ;

    //--------------------------------------------------------------------------
    // call the jit kernel and return result
    //--------------------------------------------------------------------------

    bool A_is_B = GB_all_aliased (A, B) ;
    #include "include/GB_pedantic_disable.h"
    GB_jit_dl_function GB_jit_kernel = (GB_jit_dl_function) dl_function ;
    return (GB_jit_kernel (C, A, B, nthreads, A_is_B, &GB_callback)) ;
}

