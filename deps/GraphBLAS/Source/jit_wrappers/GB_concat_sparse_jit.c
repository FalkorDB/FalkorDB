//------------------------------------------------------------------------------
// GB_concat_sparse_jit: concat A into Ci, Cx where C is sparse
//------------------------------------------------------------------------------

// SuiteSparse:GraphBLAS, Timothy A. Davis, (c) 2017-2025, All Rights Reserved.
// SPDX-License-Identifier: Apache-2.0

//------------------------------------------------------------------------------

#include "GB.h"
#include "jitifyer/GB_stringify.h"

typedef GB_JIT_KERNEL_CONCAT_SPARSE_PROTO ((*GB_jit_dl_function)) ;

GrB_Info GB_concat_sparse_jit      // concatenate A into a sparse matrix C
(
    // input/output
    GrB_Matrix C,
    // input:
    int64_t cistart,
    const GB_Operator op,
    const GrB_Matrix A,
    void *W,                        // integer type matches C->p
    const int64_t *restrict A_ek_slicing,
    const int A_ntasks,
    const int A_nthreads
)
{ 

    //--------------------------------------------------------------------------
    // encodify the problem
    //--------------------------------------------------------------------------

    GB_jit_encoding encoding ;
    char *suffix ;
    uint64_t hash = GB_encodify_apply (&encoding, &suffix,
        GB_JIT_KERNEL_CONCAT_SPARSE, GxB_SPARSE, true, C->type, C->p_is_32,
        C->i_is_32, false, op, false, GB_sparsity (A), true, A->type,
        A->p_is_32, A->j_is_32, A->i_is_32, A->iso, A->nzombies) ;

    //--------------------------------------------------------------------------
    // get the kernel function pointer, loading or compiling it if needed
    //--------------------------------------------------------------------------

    void *dl_function ;
    GrB_Info info = GB_jitifyer_load (&dl_function,
        GB_jit_apply_family, "concat_sparse",
        hash, &encoding, suffix, NULL, NULL,
        op, C->type, A->type, NULL) ;
    if (info != GrB_SUCCESS) return (info) ;

    //--------------------------------------------------------------------------
    // call the jit kernel and return result
    //--------------------------------------------------------------------------

    #include "include/GB_pedantic_disable.h"
    GB_jit_dl_function GB_jit_kernel = (GB_jit_dl_function) dl_function ;
    return (GB_jit_kernel (C, cistart, A, W, A_ek_slicing, A_ntasks,
        A_nthreads, &GB_callback)) ;
}

