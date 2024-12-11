//------------------------------------------------------------------------------
// GB_jit_kernel.h:  JIT kernel #include for all kernels (both CPU and CUDA)
//------------------------------------------------------------------------------

// SuiteSparse:GraphBLAS, Timothy A. Davis, (c) 2017-2023, All Rights Reserved.
// SPDX-License-Identifier: Apache-2.0

//------------------------------------------------------------------------------

// This file is #include'd into all JIT kernels on the CPU and the GPU.

#ifndef GB_JIT_KERNEL_H
#define GB_JIT_KERNEL_H

#define GB_JIT_KERNEL

#ifndef GB_CUDA_KERNEL
    // for CPU JIT kernels:
    #include "include/GB_include.h"
#else
    // for CUDA JIT kernels:
    #include "include/GB_cuda_kernel.cuh"
#endif

// for all JIT kernels
#include "include/GB_jit_kernel_proto.h"
#if defined (_MSC_VER) && !(defined (__INTEL_COMPILER) || defined(__INTEL_CLANG_COMPILER))
    #define GB_JIT_GLOBAL extern __declspec ( dllexport )
#else
    #define GB_JIT_GLOBAL
#endif

#ifndef GB_JIT_RUNTIME
    // for PreJIT kernels (CPU and CUDA)
    #include "callback/GB_callbacks.h"
#endif

// these macros are redefined by the JIT kernels to specialize them for each
// specific matrix format.

// accessing the C matrix
#undef GBP_C
#undef GBH_C
#undef GBI_C
#undef GBB_C
#undef GBX_C
#undef GB_C_NVALS
#undef GB_C_NHELD

// accessing the A matrix
#undef GBP_A
#undef GBH_A
#undef GBI_A
#undef GBB_A
#undef GBX_A
#undef GB_A_NVALS
#undef GB_A_NHELD

// accessing the B matrix
#undef GBP_B
#undef GBH_B
#undef GBI_B
#undef GBB_B
#undef GBX_B
#undef GB_B_NVALS
#undef GB_B_NHELD

// accessing the M matrix
#undef GBP_M
#undef GBH_M
#undef GBI_M
#undef GBB_M
#undef GBX_M
#undef GB_M_NVALS
#undef GB_M_NHELD

#undef GB_M_TYPE
#undef GB_MCAST

// accessing the S matrix for assign methods
#undef GBP_S
#undef GBH_S
#undef GBI_S
#undef GBB_S

// accessing the Z matrix for masker methods
#undef GBP_Z
#undef GBH_Z
#undef GBI_Z
#undef GBB_Z
#undef GBX_Z
#undef GB_Z_NVALS
#undef GB_Z_NHELD

// accessing the R matrix for masker methods
#undef GBP_R
#undef GBH_R
#undef GBI_R
#undef GBB_R
#undef GBX_R
#undef GB_R_NVALS
#undef GB_R_NHELD

#endif

