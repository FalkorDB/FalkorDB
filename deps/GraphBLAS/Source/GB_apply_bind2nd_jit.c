//------------------------------------------------------------------------------
// GB_apply_bind2nd_jit: Cx=op(A,y) apply bind2nd method, via the JIT
//------------------------------------------------------------------------------

// SuiteSparse:GraphBLAS, Timothy A. Davis, (c) 2017-2023, All Rights Reserved.
// SPDX-License-Identifier: Apache-2.0

//------------------------------------------------------------------------------

#include "GB.h"
#include "GB_stringify.h"

typedef GB_JIT_KERNEL_APPLY_BIND2ND_PROTO ((*GB_jit_dl_function)) ;

GrB_Info GB_apply_bind2nd_jit   // Cx = op (x,B), apply bind2nd via the JIT
(
    // output:
    GB_void *Cx,
    // input:
    const GrB_Type ctype,
    const GrB_BinaryOp binaryop,
    const GrB_Matrix A,
    const GB_void *yscalar,
    const int nthreads
)
{ 

    //--------------------------------------------------------------------------
    // encodify the problem
    //--------------------------------------------------------------------------

    GB_jit_encoding encoding ;
    char *suffix ;
    uint64_t hash = GB_encodify_ewise (&encoding, &suffix,
        GB_JIT_KERNEL_APPLYBIND2, false,
        false, false, GxB_FULL, ctype, NULL, false, false,
        binaryop, false, A, NULL) ;

    //--------------------------------------------------------------------------
    // get the kernel function pointer, loading or compiling it if needed
    //--------------------------------------------------------------------------

    void *dl_function ;
    GrB_Info info = GB_jitifyer_load (&dl_function,
        GB_jit_ewise_family, "apply_bind2nd",
        hash, &encoding, suffix, NULL, NULL,
        (GB_Operator) binaryop, ctype, A->type, NULL) ;
    if (info != GrB_SUCCESS) return (info) ;

    //--------------------------------------------------------------------------
    // call the jit kernel and return result
    //--------------------------------------------------------------------------

    GB_jit_dl_function GB_jit_kernel = (GB_jit_dl_function) dl_function ;
    return (GB_jit_kernel (Cx, A->x, yscalar, A->b, GB_nnz_held (A),
        nthreads)) ;
}

