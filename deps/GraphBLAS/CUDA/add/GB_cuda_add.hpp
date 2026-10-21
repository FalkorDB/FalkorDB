//------------------------------------------------------------------------------
// GraphBLAS/CUDA/GB_cuda_add.hpp
//------------------------------------------------------------------------------

// SuiteSparse:GraphBLAS, Timothy A. Davis, (c) 2017-2025, All Rights Reserved.
// SPDX-License-Identifier: Apache-2.0

//------------------------------------------------------------------------------

#ifndef GB_CUDA_ADD_H
#define GB_CUDA_ADD_H

#include "GB_cuda.hpp"
extern "C"
{
    #include "GB_add_iso.h"
}

GrB_Info GB_cuda_add_jit
(
    // output:
    GrB_Matrix C,
    // input:
    const bool C_iso,
    const GrB_Matrix A,
    const GrB_Matrix B,
    const bool is_eWiseUnion,
    const GB_void *alpha_scalar,
    const GB_void *beta_scalar,
    const void *theta,
    const GrB_BinaryOp binaryop,
    const bool flipij,
    const bool A_and_B_are_disjoint,
    // CUDA stream and launch parameters:
    int device,
    cudaStream_t stream,
    int32_t gridsz
) ;

#endif

