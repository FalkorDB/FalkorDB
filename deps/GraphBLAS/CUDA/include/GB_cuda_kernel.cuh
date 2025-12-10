//------------------------------------------------------------------------------
// GraphBLAS/CUDA/template/GB_cuda_kernel.cuh: definitions for CUDA kernels
//------------------------------------------------------------------------------

// SuiteSparse:GraphBLAS, Timothy A. Davis, (c) 2017-2025, All Rights Reserved.
// This file: Copyright (c) 2024-2025, NVIDIA CORPORATION. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//------------------------------------------------------------------------------

// This file is #include'd into all device functions for CUDA JIT kernels for
// GraphBLAS.  It provides a subset of GraphBLAS.h and GB.h, plus other
// definitions.  It is not used on the host.

#pragma once

//------------------------------------------------------------------------------
// C++ and CUDA #include files
//------------------------------------------------------------------------------

#include <limits>
#include <type_traits>
#include <cstdint>
#include <cmath>
#include <stdio.h>
// #include <cub/block/block_scan.cuh>
#include <cub/cub.cuh>
#include <cooperative_groups.h>
using namespace cooperative_groups ;

//------------------------------------------------------------------------------
// CUDA kernel definitions
//------------------------------------------------------------------------------

#define GB_CUDA_KERNEL

// for internal static inline functions
#undef  GB_STATIC_INLINE
#define GB_STATIC_INLINE static __device__ __inline__

//------------------------------------------------------------------------------
// subset of GraphBLAS.h
//------------------------------------------------------------------------------

#include "include/GraphBLAS_cuda.hpp"

//------------------------------------------------------------------------------
// internal #include files
//------------------------------------------------------------------------------

extern "C"
{
    #include "include/GB_opaque.h"
    #include "include/GB_index.h"
    #include "include/GB_math_macros.h"
    #include "include/GB_bytes.h"
    #include "include/GB_pun.h"
    #include "include/GB_partition.h"
    #include "include/GB_zombie.h"
    #include "include/GB_binary_search.h"
    #include "include/GB_int64_mult.h"
    #include "include/GB_math_macros.h"
    #include "include/GB_hash.h"
    #include "include/GB_complex.h"
    #include "include/GB_iceil.h"
    #include "include/GB_memory_macros.h"
    #include "include/GB_printf_kernels.h"
    #include "include/GB_clear_matrix_header.h"
    #include "include/GB_werk.h"
    #include "include/GB_task_struct.h"
    #include "include/GB_callback_proto.h"
    #include "include/GB_saxpy3task_struct.h"
    #include "include/GB_callback.h"
    #include "include/GB_hyper_hash_lookup.h"
    #include "include/GB_ok.h"
    #include "include/GB_omp_kernels.h"
}

#include "include/GB_cuda_error.hpp"
#include "include/GB_cuda_atomics.cuh"
#include "include/GB_cuda_timer.hpp"

