//------------------------------------------------------------------------------
// GB_unjumble_jit: JIT kernel to sort the vectors of a sparse/hyper matrix
//------------------------------------------------------------------------------

// SuiteSparse:GraphBLAS, Timothy A. Davis, (c) 2017-2025, All Rights Reserved.
// SPDX-License-Identifier: Apache-2.0

//------------------------------------------------------------------------------

#include "GB.h"
#include "jitifyer/GB_stringify.h"

typedef GB_JIT_KERNEL_UNJUMBLE_PROTO ((*GB_jit_dl_function)) ;

GrB_Info GB_unjumble_jit
(
    // input/output:
    const GrB_Matrix A,
    const GB_Operator op,           // identity op, unused
    const int64_t *A_slice,
    const int ntasks,
    const int nthreads
)
{ 

    //--------------------------------------------------------------------------
    // encodify the problem
    //--------------------------------------------------------------------------

    GB_jit_encoding encoding ;
    char *suffix ;
    uint64_t hash = GB_encodify_apply (&encoding, &suffix,
        GB_JIT_KERNEL_UNJUMBLE, GxB_FULL, false, A->type, false, false, false,
        op, false, GxB_SPARSE, true, A->type,
        A->p_is_32, A->j_is_32, A->i_is_32, false, 0) ;

    //--------------------------------------------------------------------------
    // get the kernel function pointer, loading or compiling it if needed
    //--------------------------------------------------------------------------

    void *dl_function ;
    GrB_Info info = GB_jitifyer_load (&dl_function,
        GB_jit_apply_family, "unjumble",
        hash, &encoding, suffix, NULL, NULL,
        op, A->type, A->type, NULL) ;
    if (info != GrB_SUCCESS) return (info) ;

    //--------------------------------------------------------------------------
    // call the jit kernel and return result
    //--------------------------------------------------------------------------

    #include "include/GB_pedantic_disable.h"
    GB_jit_dl_function GB_jit_kernel = (GB_jit_dl_function) dl_function ;
    return (GB_jit_kernel (A, A_slice, ntasks, nthreads, &GB_callback)) ;
}

