//------------------------------------------------------------------------------
// GB_subassign_jit: interface to JIT kernels for all assign/subassign methods
//------------------------------------------------------------------------------

// SuiteSparse:GraphBLAS, Timothy A. Davis, (c) 2017-2023, All Rights Reserved.
// SPDX-License-Identifier: Apache-2.0

//------------------------------------------------------------------------------

#include "GB.h"
#include "GB_stringify.h"

typedef GB_JIT_KERNEL_SUBASSIGN_PROTO ((*GB_jit_dl_function)) ;

GrB_Info GB_subassign_jit
(
    // input/output:
    GrB_Matrix C,
    // input:
    const bool C_replace,
    // I:
    const GrB_Index *I,
    const int64_t ni,
    const int64_t nI,
    const int Ikind,
    const int64_t Icolon [3],
    // J:
    const GrB_Index *J,
    const int64_t nj,
    const int64_t nJ,
    const int Jkind,
    const int64_t Jcolon [3],
    // mask M:
    const GrB_Matrix M,
    const bool Mask_comp,
    const bool Mask_struct,
    // accum, if present:
    const GrB_BinaryOp accum,   // may be NULL
    // A matrix or scalar:
    const GrB_Matrix A,         // NULL for scalar assignment
    const void *scalar,
    const GrB_Type scalar_type,
    // kind and kernel:
    const int assign_kind,      // row assign, col assign, assign, or subassign
    const int assign_kernel,    // GB_JIT_KERNEL_SUBASSIGN_01, ... etc
    const char *kname,          // kernel base name
    GB_Werk Werk
)
{ 

    //--------------------------------------------------------------------------
    // encodify the problem
    //--------------------------------------------------------------------------

    GB_jit_encoding encoding ;
    char *suffix ;
    uint64_t hash = GB_encodify_assign (&encoding, &suffix,
        assign_kernel, C, C_replace, Ikind, Jkind, M, Mask_struct,
        Mask_comp, accum, A, scalar_type, assign_kind) ;

    //--------------------------------------------------------------------------
    // get the kernel function pointer, loading or compiling it if needed
    //--------------------------------------------------------------------------

    GrB_Type atype = (A == NULL) ? scalar_type : A->type ;

    void *dl_function ;
    GrB_Info info = GB_jitifyer_load (&dl_function,
        GB_jit_assign_family, kname,
        hash, &encoding, suffix, NULL, NULL,
        (GB_Operator) accum, C->type, atype, NULL) ;
    if (info != GrB_SUCCESS) return (info) ;

    double chunk = GB_Context_chunk ( ) ;
    int nthreads_max = GB_Context_nthreads_max ( ) ;

    //--------------------------------------------------------------------------
    // call the jit kernel and return result
    //--------------------------------------------------------------------------

    GB_jit_dl_function GB_jit_kernel = (GB_jit_dl_function) dl_function ;
    return (GB_jit_kernel (C, I, ni, nI, Icolon, J, nj, nJ, Jcolon, M, A,
        scalar, nthreads_max, chunk, Werk, &GB_callback)) ;
}

