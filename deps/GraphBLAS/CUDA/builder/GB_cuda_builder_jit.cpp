//------------------------------------------------------------------------------
// GraphBLAS/CUDA/builder/GB_cuda_builder_jit
//------------------------------------------------------------------------------

// SuiteSparse:GraphBLAS, Timothy A. Davis, (c) 2017-2025, All Rights Reserved.
// SPDX-License-Identifier: Apache-2.0

//------------------------------------------------------------------------------

#include "builder/GB_cuda_builder.hpp"

extern "C"
{
    typedef GB_JIT_CUDA_KERNEL_BUILDER_PROTO ((*GB_jit_dl_function)) ;
}

GrB_Info GB_cuda_builder_jit
(
    // output:
    GrB_Matrix *Thandle,
    // input:
    const GrB_Type ttype,   // type of output matrix T
    const int64_t vlen,     // length of each vector of T
    const int64_t vdim,     // number of vectors in T
    const bool is_csc,      // true if T is CSC, false if CSR
    const bool is_matrix,   // true if T a GrB_Matrix, false if vector
    const GB_void *Key_input,  // present if Key_in preloaded
    const GB_void *I,       // original indices, size nvals
    const GB_void *J,       // original indices, size nvals
    const GB_void *X,       // array of values of tuples, size nvals,
                            // or size 1 if X is iso
    const bool X_iso,       // true if X is iso
    const int64_t nvals,    // number of tuples
    GrB_BinaryOp dup,       // binary function to assemble duplicates,
                            // if NULL use the SECOND operator to
                            // keep the most recent duplicate.
    const GrB_Type xtype,   // the type of X
    bool I_is_32,       // true if I is 32 bit, false if 64
    bool J_is_32,       // true if J is 32 bit, false if 64
    bool Tp_is_32,      // true if T->p is built as 32 bit, false if 64
    bool Tj_is_32,      // true if T->h is built as 32 bit, false if 64
    bool Ti_is_32,      // true if T->i is built as 32 bit, false if 64
    bool known_no_duplicates,   // true if tuples known to have no duplicates
    bool known_sorted,          // true if tuples known to be sorted on input
    // CUDA stream and launch parameters:
    cudaStream_t stream,
    int32_t gridsz
)
{

    //--------------------------------------------------------------------------
    // encodify the problem
    //--------------------------------------------------------------------------

    bool Key_is_32 = GB_cuda_builder_key_is_32 (vlen, vdim) ;

    GB_jit_encoding encoding ;
    char *suffix ;
    uint64_t hash = GB_encodify_build (&encoding, &suffix,
        GB_JIT_CUDA_KERNEL_BUILD, dup, ttype, xtype, is_matrix, X_iso,
        Tp_is_32, Tj_is_32, Ti_is_32, I_is_32, J_is_32,
        /* K_is_32, not used: */ true,
        /* K_is_null, (K is not used in CUDA): */ true,
        Key_input, Key_is_32, known_no_duplicates, known_sorted) ;

    //--------------------------------------------------------------------------
    // get the kernel function pointer, loading or compiling it if needed
    //--------------------------------------------------------------------------

    void *dl_function ;
    GrB_Info info = GB_jitifyer_load (&dl_function,
        GB_jit_build_family, "cuda_builder",
        hash, &encoding, suffix, NULL, NULL,
        (GB_Operator) dup, ttype, xtype, NULL) ;
    if (info != GrB_SUCCESS) return (info) ;

    //--------------------------------------------------------------------------
    // call the jit kernel and return result
    //--------------------------------------------------------------------------

    GB_jit_dl_function GB_jit_kernel = (GB_jit_dl_function) dl_function ;
    return (GB_jit_kernel (Thandle, ttype, vlen, vdim, is_csc,
        Key_input, I, J, X, nvals, stream, gridsz, &GB_callback)) ;
}

