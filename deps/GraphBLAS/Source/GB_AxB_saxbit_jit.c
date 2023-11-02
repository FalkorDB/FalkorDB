//------------------------------------------------------------------------------
// GB_AxB_saxbit_jit: C<M>=A*B saxbit method, via the JIT
//------------------------------------------------------------------------------

// SuiteSparse:GraphBLAS, Timothy A. Davis, (c) 2017-2023, All Rights Reserved.
// SPDX-License-Identifier: Apache-2.0

//------------------------------------------------------------------------------

#include "GB.h"
#include "GB_stringify.h"

typedef GB_JIT_KERNEL_AXB_SAXBIT_PROTO ((*GB_jit_dl_function)) ;

GrB_Info GB_AxB_saxbit_jit      // C<M>=A*B, saxbit, via the JIT
(
    // input/output:
    GrB_Matrix C,
    // input:
    const GrB_Matrix M,
    const bool Mask_comp,
    const bool Mask_struct,
    const GrB_Matrix A,
    const GrB_Matrix B,
    const GrB_Semiring semiring,
    const bool flipxy,
    const int ntasks,
    const int nthreads,
    const int nfine_tasks_per_vector,
    const bool use_coarse_tasks,
    const bool use_atomics,
    const int64_t *restrict M_ek_slicing,
    const int M_nthreads,
    const int M_ntasks,
    const int64_t *restrict A_slice,
    const int64_t *restrict H_slice,
    GB_void *restrict Wcx,
    int8_t *restrict Wf
)
{

    //--------------------------------------------------------------------------
    // encodify the problem
    //--------------------------------------------------------------------------

    GB_jit_encoding encoding ;
    char *suffix ;
    uint64_t hash = GB_encodify_mxm (&encoding, &suffix,
        GB_JIT_KERNEL_AXB_SAXBIT,
        false, false, GxB_BITMAP, C->type,
        M, Mask_struct, Mask_comp, semiring, flipxy, A, B) ;

    //--------------------------------------------------------------------------
    // get the kernel function pointer, loading or compiling it if needed
    //--------------------------------------------------------------------------

    void *dl_function ;
    GrB_Info info = GB_jitifyer_load (&dl_function,
        GB_jit_mxm_family, "AxB_saxbit",
        hash, &encoding, suffix, semiring, NULL,
        NULL, C->type, A->type, B->type) ;
    if (info != GrB_SUCCESS) return (info) ;

    double chunk = GB_Context_chunk ( ) ;
    int nthreads_max = GB_Context_nthreads_max ( ) ;

    //--------------------------------------------------------------------------
    // call the jit kernel and return result
    //--------------------------------------------------------------------------

    GB_jit_dl_function GB_jit_kernel = (GB_jit_dl_function) dl_function ;
    return (GB_jit_kernel (C, M, A, B, ntasks, nthreads,
        nfine_tasks_per_vector, use_coarse_tasks, use_atomics,
        M_ek_slicing, M_nthreads, M_ntasks, A_slice, H_slice, Wcx, Wf,
        nthreads_max, chunk, &GB_callback)) ;
}

