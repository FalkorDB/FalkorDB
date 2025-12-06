//------------------------------------------------------------------------------
// GB_cuda_reduce.hpp: CPU definitions for CUDA reductions
//------------------------------------------------------------------------------

// SuiteSparse:GraphBLAS, Timothy A. Davis, (c) 2017-2025, All Rights Reserved.
// SPDX-License-Identifier: Apache-2.0

//------------------------------------------------------------------------------

#ifndef GB_CUDA_REDUCE_H
#define GB_CUDA_REDUCE_H

#include "GB_cuda.hpp"

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
) ;

#endif

