//------------------------------------------------------------------------------
// GB_convert_b2s_jit: JIT kernel to convert bitmap to sparse
//------------------------------------------------------------------------------

// SuiteSparse:GraphBLAS, Timothy A. Davis, (c) 2017-2024, All Rights Reserved.
// SPDX-License-Identifier: Apache-2.0

//------------------------------------------------------------------------------

#include "GB.h"
#include "jitifyer/GB_stringify.h"

typedef GB_JIT_KERNEL_CONVERT_B2S_PROTO ((*GB_jit_dl_function)) ;

GrB_Info GB_convert_b2s_jit         // extract CSC/CSR or triplets from bitmap
(
    // input:
    const int64_t *restrict Cp,     // vector pointers for CSC/CSR form
    // outputs:
    int64_t *restrict Ci,           // indices for CSC/CSR or triplet form
    int64_t *restrict Cj,           // vector indices for triplet form
    GB_void *restrict Cx,           // values for CSC/CSR or triplet form
    // inputs: not modified
    const GrB_Type ctype,           // type of Cx
    GB_Operator op,
    const GrB_Matrix A,             // matrix to extract; not modified
    const int64_t *restrict W,      // workspace
    int nthreads                    // # of threads to use
)
{ 

    //--------------------------------------------------------------------------
    // encodify the problem
    //--------------------------------------------------------------------------

    GB_jit_encoding encoding ;
    char *suffix ;
    uint64_t hash = GB_encodify_apply (&encoding, &suffix,
        GB_JIT_KERNEL_CONVERT_B2S, GxB_SPARSE, false, ctype, op, false,
        GxB_BITMAP, true, A->type, A->iso, 0) ;

    //--------------------------------------------------------------------------
    // get the kernel function pointer, loading or compiling it if needed
    //--------------------------------------------------------------------------

    void *dl_function ;
    GrB_Info info = GB_jitifyer_load (&dl_function,
        GB_jit_apply_family, "convert_b2s",
        hash, &encoding, suffix, NULL, NULL,
        op, ctype, A->type, NULL) ;
    if (info != GrB_SUCCESS) return (info) ;

    //--------------------------------------------------------------------------
    // call the jit kernel and return result
    //--------------------------------------------------------------------------

    GB_jit_dl_function GB_jit_kernel = (GB_jit_dl_function) dl_function ;
    return (GB_jit_kernel (Cp, Ci, Cj, Cx, A, W, nthreads)) ;
}

