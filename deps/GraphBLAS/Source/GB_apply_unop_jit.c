//------------------------------------------------------------------------------
// GB_apply_unop_jit: Cx=op(A) apply unop method, via the JIT
//------------------------------------------------------------------------------

// SuiteSparse:GraphBLAS, Timothy A. Davis, (c) 2017-2023, All Rights Reserved.
// SPDX-License-Identifier: Apache-2.0

//------------------------------------------------------------------------------

#include "GB.h"
#include "GB_stringify.h"

typedef GB_JIT_KERNEL_APPLY_UNOP_PROTO ((*GB_jit_dl_function)) ;

GrB_Info GB_apply_unop_jit      // Cx = op (A), apply unop via the JIT
(
    // output:
    GB_void *Cx,
    // input:
    const GrB_Type ctype,
    const GB_Operator op,       // unary or index unary op
    const bool flipij,          // if true, use z = f(x,j,i,y)
    const GrB_Matrix A,
    const void *ythunk,         // for index unary ops (op->ytype scalar)
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
        GB_JIT_KERNEL_APPLYUNOP, GxB_FULL, false, ctype, op, flipij, A) ;

    //--------------------------------------------------------------------------
    // get the kernel function pointer, loading or compiling it if needed
    //--------------------------------------------------------------------------

    void *dl_function ;
    GrB_Info info = GB_jitifyer_load (&dl_function,
        GB_jit_apply_family, "apply_unop",
        hash, &encoding, suffix, NULL, NULL,
        op, ctype, A->type, NULL) ;
    if (info != GrB_SUCCESS) return (info) ;

    //--------------------------------------------------------------------------
    // call the jit kernel and return result
    //--------------------------------------------------------------------------

    GB_jit_dl_function GB_jit_kernel = (GB_jit_dl_function) dl_function ;
    return (GB_jit_kernel (Cx, A, ythunk, A_ek_slicing, A_ntasks, A_nthreads)) ;
}

