//------------------------------------------------------------------------------
// GB_cuda_reduce_to_scalar_jit: reduce a matrix to a scalar, via the CUDA JIT
//------------------------------------------------------------------------------

// SuiteSparse:GraphBLAS, Timothy A. Davis, (c) 2017-2025, All Rights Reserved.
// SPDX-License-Identifier: Apache-2.0

//------------------------------------------------------------------------------

#include "reduce/GB_cuda_reduce.hpp"

extern "C"
{
    typedef GB_JIT_CUDA_KERNEL_REDUCE_PROTO ((*GB_jit_dl_function)) ;
}

GrB_Info GB_cuda_reduce_to_scalar_jit   // z = reduce_to_scalar (A) via CUDA JIT
(
    // output:
    GB_void *z,                 // result if has_cheeseburger is true
    GrB_Matrix V,               // result if has_cheeseburger is false
    // input:
    const GrB_Monoid monoid,    // monoid to do the reduction
    const GrB_Matrix A,         // matrix to reduce
    // CUDA stream and launch parameters:
    cudaStream_t stream,
    int32_t gridsz,
    int32_t blocksz
)
{ 

    //--------------------------------------------------------------------------
    // encodify the problem
    //--------------------------------------------------------------------------

    GB_jit_encoding encoding ;
    char *suffix ;
    uint64_t hash = GB_encodify_reduce (&encoding, &suffix,
        GB_JIT_CUDA_KERNEL_REDUCE, monoid, A) ;

    // FIXME: could get has_cheesburger here, and allocate zscalar
    // and V accordingly.

    //--------------------------------------------------------------------------
    // get the kernel function pointer, loading or compiling it if needed
    //--------------------------------------------------------------------------

    void *dl_function ;
    GrB_Info info = GB_jitifyer_load (&dl_function,
        GB_jit_reduce_family, "cuda_reduce",
        hash, &encoding, suffix, NULL, monoid,
        NULL, A->type, NULL, NULL) ;
    if (info != GrB_SUCCESS) return (info) ;

    //--------------------------------------------------------------------------
    // call the jit kernel and return result
    //--------------------------------------------------------------------------

    GB_jit_dl_function GB_jit_kernel = (GB_jit_dl_function) dl_function ;
    return (GB_jit_kernel (z, V, A, stream, gridsz, blocksz, &GB_callback)) ;
}

