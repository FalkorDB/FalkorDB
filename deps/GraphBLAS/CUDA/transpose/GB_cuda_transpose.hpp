//------------------------------------------------------------------------------
// GB_cuda_transpose.hpp: CPU definitions for CUDA transpose operations
//------------------------------------------------------------------------------

// SPDX-License-Identifier: Apache-2.0

//------------------------------------------------------------------------------

#ifndef GB_CUDA_APPLY_H
#define GB_CUDA_APPLY_H

#include "GB_cuda.hpp"

GrB_Info GB_cuda_transpose_prep_jit
(
    // output:
    GB_void *Key_input,
    // input:
    bool Key_is_32,
    const GrB_Matrix A,
    cudaStream_t stream,
    int32_t gridsz
) ;

#endif

