//------------------------------------------------------------------------------
// GB_select_bitmap_jit: select phase 2 for the JIT
//------------------------------------------------------------------------------

// SuiteSparse:GraphBLAS, Timothy A. Davis, (c) 2017-2025, All Rights Reserved.
// SPDX-License-Identifier: Apache-2.0

//------------------------------------------------------------------------------

#include "GB.h"
#include "jitifyer/GB_stringify.h"

typedef GB_JIT_KERNEL_SELECT_BITMAP_PROTO ((*GB_jit_dl_function)) ;

GrB_Info GB_select_bitmap_jit      // select bitmap
(
    // input/output:
    GrB_Matrix C,                   // C->b and C->nvals are computed
    // input:
    const GrB_Matrix A,
    const bool flipij,
    const GB_void *restrict ythunk,
    const GrB_IndexUnaryOp op,
    const int nthreads
)
{ 

    //--------------------------------------------------------------------------
    // encodify the problem
    //--------------------------------------------------------------------------

    GB_jit_encoding encoding ;
    char *suffix ;
    uint64_t hash = GB_encodify_select (&encoding, &suffix,
        GB_JIT_KERNEL_SELECT_BITMAP, C, op, flipij, A) ;

    //--------------------------------------------------------------------------
    // get the kernel function pointer, loading or compiling it if needed
    //--------------------------------------------------------------------------

    void *dl_function ;
    GrB_Info info = GB_jitifyer_load (&dl_function,
        GB_jit_select_family, "select_bitmap",
        hash, &encoding, suffix, NULL, NULL,
        (GB_Operator) op, A->type, NULL, NULL) ;
    if (info != GrB_SUCCESS) return (info) ;

    //--------------------------------------------------------------------------
    // call the jit kernel and return result
    //--------------------------------------------------------------------------

    #include "include/GB_pedantic_disable.h"
    GB_jit_dl_function GB_jit_kernel = (GB_jit_dl_function) dl_function ;
    return (GB_jit_kernel (C, A, ythunk, nthreads, &GB_callback)) ;
}

