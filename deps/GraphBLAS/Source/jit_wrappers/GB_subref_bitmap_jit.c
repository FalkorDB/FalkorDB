//------------------------------------------------------------------------------
// GB_subref_bitmap_jit: JIT kernel for GB_bitmap_subref, C=A(I,J)
//------------------------------------------------------------------------------

// SuiteSparse:GraphBLAS, Timothy A. Davis, (c) 2017-2025, All Rights Reserved.
// SPDX-License-Identifier: Apache-2.0

//------------------------------------------------------------------------------

#include "GB.h"
#include "jitifyer/GB_stringify.h"

typedef GB_JIT_KERNEL_BITMAP_SUBREF_PROTO ((*GB_jit_dl_function)) ;

GrB_Info GB_subref_bitmap_jit
(
    // input/output:
    GrB_Matrix C,
    // input:
    GrB_Matrix A,
    // I:
    const void *I,
    const bool I_is_32,
    const int64_t nI,
    const int Ikind,
    const int64_t Icolon [3],
    // J:
    const void *J,
    const bool J_is_32,
    const int64_t nJ,
    const int Jkind,
    const int64_t Jcolon [3],
    GB_Werk Werk
)
{ 

    //--------------------------------------------------------------------------
    // encodify the problem
    //--------------------------------------------------------------------------

    GB_jit_encoding encoding ;
    char *suffix ;
    uint64_t hash = GB_encodify_subref (&encoding, &suffix,
        GB_JIT_KERNEL_BITMAP_SUBREF, C, I_is_32, J_is_32,
        Ikind, Jkind, false, NULL, A) ;

    //--------------------------------------------------------------------------
    // get the kernel function pointer, loading or compiling it if needed
    //--------------------------------------------------------------------------

    void *dl_function ;
    GrB_Info info = GB_jitifyer_load (&dl_function,
        GB_jit_subref_family, "subref_bitmap",
        hash, &encoding, suffix, NULL, NULL,
        NULL, C->type, NULL, NULL) ;
    if (info != GrB_SUCCESS) return (info) ;

    //--------------------------------------------------------------------------
    // call the jit kernel and return result
    //--------------------------------------------------------------------------

    double chunk = GB_Context_chunk ( ) ;
    int nthreads_max = GB_Context_nthreads_max ( ) ;

    #include "include/GB_pedantic_disable.h"
    GB_jit_dl_function GB_jit_kernel = (GB_jit_dl_function) dl_function ;
    return (GB_jit_kernel (C, A, I, nI, Icolon, J, nJ, Jcolon,
        Werk, nthreads_max, chunk, &GB_callback)) ;
}

