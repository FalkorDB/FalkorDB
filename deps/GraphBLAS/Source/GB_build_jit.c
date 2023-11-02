//------------------------------------------------------------------------------
// GB_build_jit: JIT for GB_builder
//------------------------------------------------------------------------------

// SuiteSparse:GraphBLAS, Timothy A. Davis, (c) 2017-2023, All Rights Reserved.
// SPDX-License-Identifier: Apache-2.0

//------------------------------------------------------------------------------

#include "GB.h"
#include "GB_stringify.h"

typedef GB_JIT_KERNEL_BUILD_PROTO ((*GB_jit_dl_function)) ;

GrB_Info GB_build_jit               // GB_builder JIT kernel
(
    // output:
    GB_void *restrict Tx,
    int64_t *restrict Ti,
    // input:
    const GB_void *restrict Sx,
    const GrB_Type ttype,           // type of Tx
    const GrB_Type stype,           // type of Sx
    const GrB_BinaryOp dup,         // operator for summing duplicates
    const int64_t nvals,            // number of tuples
    const int64_t ndupl,            // number of duplicates
    const int64_t *restrict I_work,
    const int64_t *restrict K_work,
    const int64_t *restrict tstart_slice,
    const int64_t *restrict tnz_slice,
    int nthreads
)
{ 

    //--------------------------------------------------------------------------
    // encodify the problem
    //--------------------------------------------------------------------------

    GB_jit_encoding encoding ;
    char *suffix ;
    uint64_t hash = GB_encodify_build (&encoding, &suffix,
        GB_JIT_KERNEL_BUILD, dup, ttype, stype) ;

    //--------------------------------------------------------------------------
    // get the kernel function pointer, loading or compiling it if needed
    //--------------------------------------------------------------------------

    void *dl_function ;
    GrB_Info info = GB_jitifyer_load (&dl_function,
        GB_jit_build_family, "build",
        hash, &encoding, suffix, NULL, NULL,
        (GB_Operator) dup, ttype, stype, NULL) ;
    if (info != GrB_SUCCESS) return (info) ;

    //--------------------------------------------------------------------------
    // call the jit kernel and return result
    //--------------------------------------------------------------------------

    GB_jit_dl_function GB_jit_kernel = (GB_jit_dl_function) dl_function ;
    return (GB_jit_kernel (Tx, Ti, Sx, nvals, ndupl, I_work, K_work,
        tstart_slice, tnz_slice, nthreads)) ;
}

