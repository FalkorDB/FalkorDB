//------------------------------------------------------------------------------
// GB_AxB_saxpy5_jit: C+=A*B saxpy5 method, via the JIT
//------------------------------------------------------------------------------

// SuiteSparse:GraphBLAS, Timothy A. Davis, (c) 2017-2023, All Rights Reserved.
// SPDX-License-Identifier: Apache-2.0

//------------------------------------------------------------------------------

#include "GB.h"
#include "GB_stringify.h"

typedef GB_JIT_KERNEL_AXB_SAXPY5_PROTO ((*GB_jit_dl_function)) ;

GrB_Info GB_AxB_saxpy5_jit          // C+=A*B, saxpy5 method, via the JIT
(
    // input/output:
    GrB_Matrix C,
    // input:
    const GrB_Matrix A,
    const GrB_Matrix B,
    const GrB_Semiring semiring,
    const bool flipxy,
    const int ntasks,
    const int nthreads,
    const int64_t *B_slice
)
{ 

    //--------------------------------------------------------------------------
    // encodify the problem
    //--------------------------------------------------------------------------

    GB_jit_encoding encoding ;
    char *suffix ;
    ASSERT (!C->iso) ;
    ASSERT (GB_IS_FULL (C)) ;
    uint64_t hash = GB_encodify_mxm (&encoding, &suffix,
        GB_JIT_KERNEL_AXB_SAXPY5,
        false, false, GxB_FULL, C->type,
        NULL, true, false, semiring, flipxy, A, B) ;

    //--------------------------------------------------------------------------
    // get the kernel function pointer, loading or compiling it if needed
    //--------------------------------------------------------------------------

    void *dl_function ;
    GrB_Info info = GB_jitifyer_load (&dl_function,
        GB_jit_mxm_family, "AxB_saxpy5",
        hash, &encoding, suffix, semiring, NULL,
        NULL, C->type, A->type, B->type) ;
    if (info != GrB_SUCCESS) return (info) ;

    //--------------------------------------------------------------------------
    // call the jit kernel and return result
    //--------------------------------------------------------------------------

    bool cpu_has_avx2 = GB_Global_cpu_features_avx2 ( ) ;
    bool cpu_has_avx512f = GB_Global_cpu_features_avx512f ( ) ;
    GB_jit_dl_function GB_jit_kernel = (GB_jit_dl_function) dl_function ;
    return (GB_jit_kernel (C, A, B, ntasks, nthreads, B_slice,
        cpu_has_avx2, cpu_has_avx512f)) ;
}

