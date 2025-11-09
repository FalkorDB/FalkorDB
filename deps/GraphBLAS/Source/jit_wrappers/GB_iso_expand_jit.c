//------------------------------------------------------------------------------
// GB_iso_expand_jit: JIT kernel to expand an iso scalar into an array
//------------------------------------------------------------------------------

// SuiteSparse:GraphBLAS, Timothy A. Davis, (c) 2017-2025, All Rights Reserved.
// SPDX-License-Identifier: Apache-2.0

//------------------------------------------------------------------------------

#include "GB.h"
#include "jitifyer/GB_stringify.h"

typedef GB_JIT_KERNEL_ISO_EXPAND_PROTO ((*GB_jit_dl_function)) ;

GrB_Info GB_iso_expand_jit  // expand an iso scalar into an entire array
(
    void *restrict X,               // output array to expand into
    const int64_t n,                // # of entries in X
    const void *restrict scalar,    // scalar to expand into X
    const GrB_Type xtype,           // the type of the X and the scalar
    const GB_Operator op,           // identity operator
    const int nthreads              // # of threads to use
)
{ 

    //--------------------------------------------------------------------------
    // encodify the problem
    //--------------------------------------------------------------------------

    GB_jit_encoding encoding ;
    char *suffix ;
    uint64_t hash = GB_encodify_apply (&encoding, &suffix,
        GB_JIT_KERNEL_ISO_EXPAND, /* C sparsity: */ GxB_FULL, false, xtype,
        /* C is_32: */ false, false, false,
        op, /* flipij: */ false, /* A sparsity: */ GxB_FULL, false, xtype,
        /* A is_32: */ false, false, false, /* A_iso: */ true,
        /* nzombies: */ 0) ;

    //--------------------------------------------------------------------------
    // get the kernel function pointer, loading or compiling it if needed
    //--------------------------------------------------------------------------

    void *dl_function ;
    GrB_Info info = GB_jitifyer_load (&dl_function,
        GB_jit_apply_family, "iso_expand",
        hash, &encoding, suffix, NULL, NULL,
        op, xtype, xtype, NULL) ;
    if (info != GrB_SUCCESS) return (info) ;

    //--------------------------------------------------------------------------
    // call the jit kernel and return result
    //--------------------------------------------------------------------------

    #include "include/GB_pedantic_disable.h"
    GB_jit_dl_function GB_jit_kernel = (GB_jit_dl_function) dl_function ;
    return (GB_jit_kernel (X, n, scalar, nthreads, &GB_callback)) ;
}

