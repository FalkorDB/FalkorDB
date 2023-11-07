//------------------------------------------------------------------------------
// GB_cuda.h: definitions for using CUDA in GraphBLAS
//------------------------------------------------------------------------------

// SuiteSparse:GraphBLAS/CUDA, (c) NVIDIA Corp. 2017-2019, All Rights Reserved.
// SPDX-License-Identifier: Apache-2.0

//------------------------------------------------------------------------------

// This file is #include'd only in the GraphBLAS/CUDA/GB_cuda*.cu source files.

#ifndef GB_CUDA_H
#define GB_CUDA_H

extern "C"
{ 
    #include "GB_dev.h"
    #include "GB_compiler.h"
    #include "GB_cpu_features.h"
    #include "GB_warnings.h"
}

#define GB_LIBRARY
#include "GraphBLAS.h"

extern "C"
{
    #include <cassert>
    #include <cmath>
    #include "GB.h"
}

// Finally, include the CUDA definitions
#include "cuda_runtime.h"
#include "cuda.h"
// #include "cub.cuh"
#include "jitify.hpp"
#include "GB_cuda_mxm_factory.hpp"

#include <iostream>

#define CHECK_CUDA_SIMPLE(call)                                           \
  do {                                                                    \
    cudaError_t err = call;                                               \
    if (err != cudaSuccess) {                                             \
      const char* str = cudaGetErrorName( err);                           \
      std::cout << "(CUDA runtime) returned " << str;                     \
      std::cout << " (" << __FILE__ << ":" << __LINE__ << ":" << __func__ \
                << "())" << std::endl;                                    \
      return (GrB_PANIC) ;                                                \
    }                                                                     \
  } while (0)

#define CU_OK(call) CHECK_CUDA_SIMPLE(call)

//------------------------------------------------------------------------------
// GB_CUDA_CATCH: catch error from a try { ... } region
//------------------------------------------------------------------------------

//  #define GB_FREE_ALL { some macro to free all temporaries }
//  GrB_Info info ;
//  try { ... do stuff that can throw an exception }
//  GB_CUDA_CATCH (info) ;

#define GB_CUDA_CATCH(info)                                     \
    catch (std::exception& e)                                   \
    {                                                           \
        printf ("CUDA error: %s\n", e.what ( )) ;               \
        info = GrB_PANIC ;                                      \
        /* out_of_memory : info = GrB_OUT_OF_MEMORY ; */        \
        /* nulltpr:  info = ... ; */                            \
        /* no gpus here: info = GrB_PANIC ; */                  \
    }                                                           \
    if (info != GrB_SUCCESS)                                    \
    {                                                           \
        /* CUDA failed */                                       \
        GB_FREE_ALL ;                                           \
        return (info) ;                                         \
    }

// NBUCKETS buckets: computed by up to NBUCKETS-1 kernel launches (zombies need
// no work...), using different kernels (with different configurations
// depending on the bucket).

#include "GB_cuda_buckets.h"

extern "C"
{
    #include "GB_stringify.h"
}

//------------------------------------------------------------------------------
// prefetch and memadvise
//------------------------------------------------------------------------------

// for the "which" parameter of GB_cuda_matrix_prefetch:
// FIXME: rename this to GB_WHATEVER_P for GB_cuda_matrix_advise
#define GB_PREFETCH_P   1
#define GB_PREFETCH_H   2
#define GB_PREFETCH_Y   4
#define GB_PREFETCH_B   8
#define GB_PREFETCH_I  16
#define GB_PREFETCH_X  32
#define GB_PREFETCH_PIX   (GB_PREFETCH_P + GB_PREFETCH_I + GB_PREFETCH_X)
#define GB_PREFETCH_PYI   (GB_PREFETCH_P + GB_PREFETCH_Y + GB_PREFETCH_I)
#define GB_PREFETCH_PYBI  (GB_PREFETCH_PYI + GB_PREFETCH_B)
#define GB_PREFETCH_PYBIX (GB_PREFETCH_PYBI + GB_PREFETCH_X)
#define GB_PREFETCH_PHI   (GB_PREFETCH_P + GB_PREFETCH_H + GB_PREFETCH_I)
#define GB_PREFETCH_PHBI  (GB_PREFETCH_PHI + GB_PREFETCH_B)
#define GB_PREFETCH_PHBIX (GB_PREFETCH_PHBI + GB_PREFETCH_X)

GrB_Info GB_cuda_matrix_prefetch
(
    GrB_Matrix A,
    int which,              // which components to prefetch (phybix control)
    int device,             // GPU device or cudaCpuDeviceId
    cudaStream_t stream
) ;

#if 0
// do we need this function too?
GrB_Info GB_cuda_matrix_advise
(
    GrB_Matrix A,

    p, h, y, b, i, x?   6 bools

    what to do:  advise (prefer location? access by)?  prefetch? nothing?
        avdice: enum (1 to 6)

    int device,             // GPU device or cudaCpuDeviceId
) ;
#endif

void GB_cuda_upscale_identity
(
    GB_void *identity_upscaled,     // output: at least sizeof (uint16_t)
    GrB_Monoid monoid               // input: monoid to upscale
) ;

#endif

