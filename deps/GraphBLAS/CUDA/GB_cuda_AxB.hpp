//------------------------------------------------------------------------------
// GraphBLAS/CUDA/GB_cuda_AxB.hpp
//------------------------------------------------------------------------------

// SuiteSparse:GraphBLAS, Timothy A. Davis, (c) 2017-2025, All Rights Reserved.
// SPDX-License-Identifier: Apache-2.0

//------------------------------------------------------------------------------

#ifndef GB_CUDA_AXB_H
#define GB_CUDA_AXB_H

#include "GB_cuda.hpp"

GrB_Info GB_cuda_AxB_dot3_jit
(
    // input/output:
    GrB_Matrix C,               // FIXME: allow iso for this kernel
    // input:
    const GrB_Matrix M, const bool Mask_struct,
    const GrB_Matrix A,
    const GrB_Matrix B,
    const GrB_Semiring semiring,
    const bool flipxy,
    // CUDA stream, device, and # of ms
    cudaStream_t stream,
    int device,
    int number_of_sms
) ;

#endif

