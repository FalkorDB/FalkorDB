//------------------------------------------------------------------------------
// GB_jit_kernel.h:  JIT kernel #include for all kernels (both CPU and CUDA)
//------------------------------------------------------------------------------

// SuiteSparse:GraphBLAS, Timothy A. Davis, (c) 2017-2025, All Rights Reserved.
// SPDX-License-Identifier: Apache-2.0

//------------------------------------------------------------------------------

// This file is #include'd into all JIT and PreJIT kernels on the CPU and the
// GPU.  It is not used outside of the JIT and PreJIT kernels.

#ifndef GB_JIT_KERNEL_H
#define GB_JIT_KERNEL_H

#define GB_JIT_KERNEL

#ifndef GB_CUDA_KERNEL
    // for CPU JIT and PreJIT kernels:
    #include "include/GB_include.h"
#else
    // for CUDA JIT and PreJIT kernels:
    #include "include/GB_cuda_kernel.cuh"
#endif

// for all JIT kernels:  the GB_jit_kernel and GB_jit_query functions must be
// exported so that GB_jitifyer can find the symbols when loading the kernels.
#include "include/GB_jit_kernel_proto.h"
#if defined (_MSC_VER) && !(defined (__INTEL_COMPILER) || defined(__INTEL_CLANG_COMPILER))
    #define GB_JIT_GLOBAL extern __declspec ( dllexport )
#else
    #define GB_JIT_GLOBAL
#endif

// Runtime JIT kernels are compiled with -DGB_JIT_RUNTIME, which PreJIT
// kernels do not have.  PreJIT kernels do not use callback function pointers,
// so they require the constant function declarations in GB_callbacks.h.
#ifndef GB_JIT_RUNTIME
    // for PreJIT kernels (CPU and CUDA)
    #include "callback/GB_callbacks.h"
#endif

#endif

