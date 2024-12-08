//------------------------------------------------------------------------------
// GB_sort_jit: sort a matrix
//------------------------------------------------------------------------------

// SuiteSparse:GraphBLAS, Timothy A. Davis, (c) 2017-2024, All Rights Reserved.
// SPDX-License-Identifier: Apache-2.0

//------------------------------------------------------------------------------

#include "GB.h"
#include "jitifyer/GB_stringify.h"

typedef GB_JIT_KERNEL_SORT_PROTO ((*GB_jit_dl_function)) ;

GrB_Info GB_sort_jit
(
    // input/output:
    GrB_Matrix C,
    // input:
    const GrB_BinaryOp binaryop,
    int nthreads,
    GB_Werk Werk
)
{ 

    //--------------------------------------------------------------------------
    // encodify the problem
    //--------------------------------------------------------------------------

    GB_jit_encoding encoding ;
    char *suffix ;
    uint64_t hash = GB_encodify_sort (&encoding, &suffix,
        GB_JIT_KERNEL_SORT, C, binaryop) ;

    //--------------------------------------------------------------------------
    // get the kernel function pointer, loading or compiling it if needed
    //--------------------------------------------------------------------------

    void *dl_function ;
    GrB_Info info = GB_jitifyer_load (&dl_function,
        GB_jit_sort_family, "sort",
        hash, &encoding, suffix, NULL, NULL,
        (GB_Operator) binaryop, C->type, NULL, NULL) ;
    if (info != GrB_SUCCESS) return (info) ;

    //--------------------------------------------------------------------------
    // call the jit kernel and return result
    //--------------------------------------------------------------------------

    GB_jit_dl_function GB_jit_kernel = (GB_jit_dl_function) dl_function ;
    return (GB_jit_kernel (C, nthreads, Werk, &GB_callback)) ;
}

