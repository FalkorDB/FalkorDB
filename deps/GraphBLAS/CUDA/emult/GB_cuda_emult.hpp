//------------------------------------------------------------------------------
// GraphBLAS/CUDA/emult/GB_cuda_emult.hpp
//------------------------------------------------------------------------------

// SuiteSparse:GraphBLAS, Timothy A. Davis, (c) 2017-2026, All Rights Reserved.
// SPDX-License-Identifier: Apache-2.0

//------------------------------------------------------------------------------

#ifndef GB_CUDA_EMULT_H
#define GB_CUDA_EMULT_H

#include "GB_cuda.hpp"
extern "C"
{
    #include "GB_emult_iso.h"
}

GrB_Info GB_cuda_emult_jit
(
    // output:
    GrB_Matrix C,
    // input:
    const bool C_iso,
    const GrB_Matrix A,
    const GrB_Matrix B,
    const void *theta,
    const GrB_BinaryOp binaryop,
    const bool flipij,
    // CUDA stream and launch parameters:
    int device,
    cudaStream_t stream,
    int32_t gridsz
) ;

#endif

