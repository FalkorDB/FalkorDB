//------------------------------------------------------------------------------
// GB_cuda_select.hpp: CPU definitions for CUDA select operations
//------------------------------------------------------------------------------

// SPDX-License-Identifier: Apache-2.0

//------------------------------------------------------------------------------

#ifndef GB_CUDA_SELECT_H
#define GB_CUDA_SELECT_H

#include "GB_cuda.hpp"

GrB_Info GB_cuda_select_bitmap_jit
(
    // output:
    int8_t *Cb,
    uint64_t *cnvals,
    // input:
    const bool C_iso,
    const GrB_Matrix A,
    const bool flipij,
    const GB_void *ythunk,
    const GrB_IndexUnaryOp op,
    // CUDA stream and launch parameters:
    cudaStream_t stream,
    int32_t gridsz,
    int32_t blocksz
) ;

GrB_Info GB_cuda_select_sparse_jit
(
    // output:
    GrB_Matrix C,
    // input:
    const bool C_iso,
    const GrB_IndexUnaryOp op,
    const bool flipij,
    const GrB_Matrix A,
    const GB_void *ythunk,
    // CUDA stream and launch parameters:
    cudaStream_t stream,
    int32_t gridsz,
    int32_t blocksz
) ;

#endif
