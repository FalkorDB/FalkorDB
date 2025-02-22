//------------------------------------------------------------------------------
// GB_convert_b2s_jit: JIT kernel to convert bitmap to sparse
//------------------------------------------------------------------------------

// SuiteSparse:GraphBLAS, Timothy A. Davis, (c) 2017-2025, All Rights Reserved.
// SPDX-License-Identifier: Apache-2.0

//------------------------------------------------------------------------------

#include "GB.h"
#include "jitifyer/GB_stringify.h"

typedef GB_JIT_KERNEL_CONVERT_B2S_PROTO ((*GB_jit_dl_function)) ;

GrB_Info GB_convert_b2s_jit         // extract CSC/CSR or triplets from bitmap
(
    // input:
    const void *Cp,                 // vector pointers for CSC/CSR form
    // outputs:
    void *Ci,                       // indices for CSC/CSR or triplet form
    void *Cj,                       // vector indices for triplet form
    GB_void *restrict Cx,           // values for CSC/CSR or triplet form
    // inputs: not modified
    const bool Cp_is_32,            // if true, Cp is uint32_t, else uint64_t
    const bool Ci_is_32,            // if true, Ci is uint32_t, else uint64_t
    const bool Cj_is_32,            // if true, Cj is uint32_t, else uint64_t
    const GrB_Type ctype,           // type of Cx
    GB_Operator op,
    const GrB_Matrix A,             // matrix to extract; not modified
    const void *W,                  // workspace
    int nthreads                    // # of threads to use
)
{ 

    //--------------------------------------------------------------------------
    // encodify the problem
    //--------------------------------------------------------------------------

    GB_jit_encoding encoding ;
    char *suffix ;
    uint64_t hash = GB_encodify_apply (&encoding, &suffix,
        GB_JIT_KERNEL_CONVERT_B2S, GxB_SPARSE, false, ctype, Cp_is_32,
        Ci_is_32, Cj_is_32, op, false, GxB_BITMAP, true, A->type,
        A->p_is_32, A->j_is_32, A->i_is_32, A->iso, 0) ;

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

    #include "include/GB_pedantic_disable.h"
    GB_jit_dl_function GB_jit_kernel = (GB_jit_dl_function) dl_function ;
    return (GB_jit_kernel (Cp, Ci, Cj, Cx, A, W, nthreads, &GB_callback)) ;
}

