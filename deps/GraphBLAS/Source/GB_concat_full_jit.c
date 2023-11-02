//------------------------------------------------------------------------------
// GB_concat_full_jit: concat A into a full matrix C
//------------------------------------------------------------------------------

// SuiteSparse:GraphBLAS, Timothy A. Davis, (c) 2017-2023, All Rights Reserved.
// SPDX-License-Identifier: Apache-2.0

//------------------------------------------------------------------------------

#include "GB.h"
#include "GB_stringify.h"

typedef GB_JIT_KERNEL_CONCAT_FULL_PROTO ((*GB_jit_dl_function)) ;

GrB_Info GB_concat_full_jit      // concatenate A into a full matrix C
(
    // input/output
    GrB_Matrix C,
    // input:
    int64_t cistart,
    int64_t cvstart,
    const GB_Operator op,
    const GrB_Matrix A,
    const int A_nthreads
)
{ 

    //--------------------------------------------------------------------------
    // encodify the problem
    //--------------------------------------------------------------------------

    GB_jit_encoding encoding ;
    char *suffix ;
    uint64_t hash = GB_encodify_apply (&encoding, &suffix,
        GB_JIT_KERNEL_CONCAT_FULL, GxB_FULL, true, C->type, op, false, A) ;

    //--------------------------------------------------------------------------
    // get the kernel function pointer, loading or compiling it if needed
    //--------------------------------------------------------------------------

    void *dl_function ;
    GrB_Info info = GB_jitifyer_load (&dl_function,
        GB_jit_apply_family, "concat_full",
        hash, &encoding, suffix, NULL, NULL,
        op, C->type, A->type, NULL) ;
    if (info != GrB_SUCCESS) return (info) ;

    //--------------------------------------------------------------------------
    // call the jit kernel and return result
    //--------------------------------------------------------------------------

    GB_jit_dl_function GB_jit_kernel = (GB_jit_dl_function) dl_function ;
    return (GB_jit_kernel (C, cistart, cvstart, A, A_nthreads)) ;
}

