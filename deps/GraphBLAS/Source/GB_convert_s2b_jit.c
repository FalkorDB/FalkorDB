//------------------------------------------------------------------------------
// GB_convert_s2b_jit: JIT kernel to convert sparse to bitmap
//------------------------------------------------------------------------------

// SuiteSparse:GraphBLAS, Timothy A. Davis, (c) 2017-2023, All Rights Reserved.
// SPDX-License-Identifier: Apache-2.0

//------------------------------------------------------------------------------

// Currently, the kernel does not do any typecasting or apply an operator
// (except for identity), but this could be revised in the future.

#include "GB.h"
#include "GB_stringify.h"

typedef GB_JIT_KERNEL_CONVERT_S2B_PROTO ((*GB_jit_dl_function)) ;

GrB_Info GB_convert_s2b_jit    // convert sparse to bitmap
(
    // output:
    GB_void *Ax_new,
    int8_t *Ab,
    // input:
    GB_Operator op,
    const GrB_Matrix A,
    const int64_t *A_ek_slicing,
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
        GB_JIT_KERNEL_CONVERTS2B, GxB_FULL, false, A->type, op, false, A) ;

    //--------------------------------------------------------------------------
    // get the kernel function pointer, loading or compiling it if needed
    //--------------------------------------------------------------------------

    void *dl_function ;
    GrB_Info info = GB_jitifyer_load (&dl_function,
        GB_jit_apply_family, "convert_s2b",
        hash, &encoding, suffix, NULL, NULL,
        op, A->type, A->type, NULL) ;
    if (info != GrB_SUCCESS) return (info) ;

    //--------------------------------------------------------------------------
    // call the jit kernel and return result
    //--------------------------------------------------------------------------

    GB_jit_dl_function GB_jit_kernel = (GB_jit_dl_function) dl_function ;
    return (GB_jit_kernel (Ax_new, Ab, A, A_ek_slicing, A_ntasks, A_nthreads)) ;
}

