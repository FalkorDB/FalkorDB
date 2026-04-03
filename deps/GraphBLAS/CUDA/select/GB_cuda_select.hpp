//------------------------------------------------------------------------------
// GB_cuda_select.hpp: CPU definitions for CUDA select operations
//------------------------------------------------------------------------------

// SPDX-License-Identifier: Apache-2.0

//------------------------------------------------------------------------------

#ifndef GB_CUDA_SELECT_H
#define GB_CUDA_SELECT_H

#include "GB_cuda.hpp"
#include "select/GB_select_iso.h"

GrB_Info GB_cuda_select_bitmap_jit
(
    // input/output:
    GrB_Matrix C,
    // input:
    const GrB_Matrix A,
    const bool flipij,
    const GB_void *ythunk,
    const GrB_IndexUnaryOp op,
    // CUDA stream and launch parameters:
    cudaStream_t stream,
    int32_t gridsz
) ;

GrB_Info GB_cuda_select_sparse_jit
(
    // output:
    GrB_Matrix C,
    // input:
    const GrB_Matrix A,
    const bool flipij,
    const GB_void *ythunk,
    const GrB_IndexUnaryOp op,
    // CUDA stream and launch parameters:
    cudaStream_t stream,
    int32_t gridsz
) ;

#endif
