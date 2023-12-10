//------------------------------------------------------------------------------
// GB_AxB_dot4_jit: C+=A'*B dot4 method, via the JIT
//------------------------------------------------------------------------------

// SuiteSparse:GraphBLAS, Timothy A. Davis, (c) 2017-2023, All Rights Reserved.
// SPDX-License-Identifier: Apache-2.0

//------------------------------------------------------------------------------

#include "GB.h"
#include "GB_stringify.h"

typedef GB_JIT_KERNEL_AXB_DOT4_PROTO ((*GB_jit_dl_function)) ;

GrB_Info GB_AxB_dot4_jit            // C+=A'*B, dot4 method, via the JIT
(
    // input/output:
    GrB_Matrix C,
    // input:
    const bool C_in_iso,
    const GrB_Matrix A,
    const GrB_Matrix B,
    const GrB_Semiring semiring,
    const bool flipxy,
    const int64_t *restrict A_slice,
    const int64_t *restrict B_slice,
    const int naslice,
    const int nbslice,
    const int nthreads,
    GB_Werk Werk
)
{ 

    //--------------------------------------------------------------------------
    // encodify the problem
    //--------------------------------------------------------------------------

    GB_jit_encoding encoding ;
    char *suffix ;
    uint64_t hash = GB_encodify_mxm (&encoding, &suffix,
        GB_JIT_KERNEL_AXB_DOT4,
        false, C_in_iso, GxB_FULL, C->type,
        NULL, true, false, semiring, flipxy, A, B) ;

    //--------------------------------------------------------------------------
    // get the kernel function pointer, loading or compiling it if needed
    //--------------------------------------------------------------------------

    void *dl_function ;
    GrB_Info info = GB_jitifyer_load (&dl_function,
        GB_jit_mxm_family, "AxB_dot4",
        hash, &encoding, suffix, semiring, NULL,
        NULL, C->type, A->type, B->type) ;
    if (info != GrB_SUCCESS) return (info) ;

    //--------------------------------------------------------------------------
    // call the jit kernel and return result
    //--------------------------------------------------------------------------

    GB_jit_dl_function GB_jit_kernel = (GB_jit_dl_function) dl_function ;
    return (GB_jit_kernel (C, A, B, A_slice, B_slice, naslice, nbslice,
        nthreads, Werk, &GB_callback)) ;
}

