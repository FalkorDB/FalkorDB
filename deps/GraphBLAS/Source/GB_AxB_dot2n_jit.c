//------------------------------------------------------------------------------
// GB_AxB_dot2n_jit: C<M>=A'*B dot2n method, via the JIT
//------------------------------------------------------------------------------

// SuiteSparse:GraphBLAS, Timothy A. Davis, (c) 2017-2023, All Rights Reserved.
// SPDX-License-Identifier: Apache-2.0

//------------------------------------------------------------------------------

#include "GB.h"
#include "GB_stringify.h"

typedef GB_JIT_KERNEL_AXB_DOT2N_PROTO ((*GB_jit_dl_function)) ;

GrB_Info GB_AxB_dot2n_jit        // C<M>=A*B, dot2n method, via the JIT
(
    // input/output:
    GrB_Matrix C,
    // input:
    const GrB_Matrix M,
    const bool Mask_comp,
    const bool Mask_struct,
    const GrB_Matrix A,
    const int64_t *restrict A_slice,
    const GrB_Matrix B,
    const int64_t *restrict B_slice,
    const GrB_Semiring semiring,
    const bool flipxy,
    const int nthreads,
    const int naslice,
    const int nbslice
)
{ 

    //--------------------------------------------------------------------------
    // encodify the problem
    //--------------------------------------------------------------------------

    GB_jit_encoding encoding ;
    char *suffix ;
    uint64_t hash = GB_encodify_mxm (&encoding, &suffix,
        GB_JIT_KERNEL_AXB_DOT2N,
        C->iso, false, GB_sparsity (C), C->type,
        M, Mask_struct, Mask_comp, semiring, flipxy, A, B) ;

    //--------------------------------------------------------------------------
    // get the kernel function pointer, loading or compiling it if needed
    //--------------------------------------------------------------------------

    void *dl_function ;
    GrB_Info info = GB_jitifyer_load (&dl_function,
        GB_jit_mxm_family, "AxB_dot2n",
        hash, &encoding, suffix, semiring, NULL,
        NULL, C->type, A->type, B->type) ;
    if (info != GrB_SUCCESS) return (info) ;

    //--------------------------------------------------------------------------
    // call the jit kernel and return result
    //--------------------------------------------------------------------------

    GB_jit_dl_function GB_jit_kernel = (GB_jit_dl_function) dl_function ;
    return (GB_jit_kernel (C, M, A, A_slice, B, B_slice, nthreads, naslice,
        nbslice)) ;
}

