//------------------------------------------------------------------------------
// GraphBLAS/CUDA/GB_cuda_mask.hpp
//------------------------------------------------------------------------------

// SuiteSparse:GraphBLAS, Timothy A. Davis, (c) 2017-2026, All Rights Reserved.
// SPDX-License-Identifier: Apache-2.0

//------------------------------------------------------------------------------

#ifndef GB_CUDA_MASK__H
#define GB_CUDA_MASK_H

#include "GB_cuda.hpp"

GrB_Info GB_cuda_mask_assign_jit
(
    // output:
    GrB_Matrix C,
    // input:
    const GrB_Matrix A,
    const GrB_Matrix B,                     // B == M is the mask
    const bool Mask_struct,
    const bool Mask_comp,
    const GrB_BinaryOp binaryop,            // FIRST_atype
    // CUDA stream and launch parameters:
    int device,
    cudaStream_t stream,
    int32_t gridsz
) ;

#endif

