//------------------------------------------------------------------------------
// GB_transpose_bind2nd_jit: C=op(x,A') via the JIT
//------------------------------------------------------------------------------

// SuiteSparse:GraphBLAS, Timothy A. Davis, (c) 2017-2023, All Rights Reserved.
// SPDX-License-Identifier: Apache-2.0

//------------------------------------------------------------------------------

#include "GB.h"
#include "GB_stringify.h"

typedef GB_JIT_KERNEL_TRANS_BIND2ND_PROTO ((*GB_jit_dl_function)) ;

GrB_Info GB_transpose_bind2nd_jit
(
    // output:
    GrB_Matrix C,
    // input:
    const GrB_BinaryOp binaryop,
    const GrB_Matrix A,
    const GB_void *yscalar,
    int64_t *restrict *Workspaces,
    const int64_t *restrict A_slice,
    int nworkspaces,
    int nthreads
)
{ 

    //--------------------------------------------------------------------------
    // encodify the problem
    //--------------------------------------------------------------------------

    GB_jit_encoding encoding ;
    char *suffix ;
    uint64_t hash = GB_encodify_ewise (&encoding, &suffix,
        GB_JIT_KERNEL_TRANSBIND2, false,
        false, false, GB_sparsity (C), C->type, NULL, false, false,
        binaryop, false, A, NULL) ;

    //--------------------------------------------------------------------------
    // get the kernel function pointer, loading or compiling it if needed
    //--------------------------------------------------------------------------

    void *dl_function ;
    GrB_Info info = GB_jitifyer_load (&dl_function,
        GB_jit_ewise_family, "trans_bind2nd",
        hash, &encoding, suffix, NULL, NULL,
        (GB_Operator) binaryop, C->type, A->type, NULL) ;
    if (info != GrB_SUCCESS) return (info) ;

    //--------------------------------------------------------------------------
    // call the jit kernel and return result
    //--------------------------------------------------------------------------

    GB_jit_dl_function GB_jit_kernel = (GB_jit_dl_function) dl_function ;
    return (GB_jit_kernel (C, A, yscalar, Workspaces, A_slice, nworkspaces,
        nthreads)) ;
}

