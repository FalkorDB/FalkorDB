//------------------------------------------------------------------------------
// GB_include.h: internal definitions for GraphBLAS, including CPU JIT kernels
//------------------------------------------------------------------------------

// SuiteSparse:GraphBLAS, Timothy A. Davis, (c) 2017-2025, All Rights Reserved.
// SPDX-License-Identifier: Apache-2.0

//------------------------------------------------------------------------------

#ifndef GB_INCLUDE_H
#define GB_INCLUDE_H

//------------------------------------------------------------------------------
// definitions that modify GraphBLAS.h
//------------------------------------------------------------------------------

#undef GRAPHBLAS_VANILLA

#include "include/GB_dev.h"
#include "include/GB_compiler.h"
#include "include/GB_warnings.h"
#include "include/GB_coverage.h"

#if defined ( GB_JIT_KERNEL ) || defined ( GBCOMPACT )
// Because of the JIT code generation, the kernels often have unused variables,
// parameters, and functions.  These warnings are disabled here.  They may also
// generate warnings with -Wpedantic, so those are disabled as well.
#include "include/GB_unused.h"
#include "include/GB_pedantic_disable.h"
#endif

//------------------------------------------------------------------------------
// user-visible GraphBLAS.h
//------------------------------------------------------------------------------

#include "GraphBLAS.h"
#undef I

#ifdef GBMATLAB
#undef GRAPHBLAS_HAS_CUDA
#endif

//------------------------------------------------------------------------------
// handle the restrict and 'static inline' keywords
//------------------------------------------------------------------------------

// Intentionally shadow the built-in "restrict" keyword.  See GraphBLAS.h for
// the definition of GB_restrict.  It becomes empty for C++, and "__restrict"
// for MS Visual Studio.  Otherwise, GB_restrict is just "restrict" on C11
// compliant compilers.  I prefer to use the "restrict" keyword to make the
// code readable.  This #define is a patch for compilers that don't support it:

#define restrict GB_restrict

// for internal static inline functions (will be redefined for CUDA)
#undef  GB_STATIC_INLINE
#define GB_STATIC_INLINE static inline

//------------------------------------------------------------------------------
// internal #include files
//------------------------------------------------------------------------------

#include "include/GB_abort.h"
#include "include/GB_prefix.h"
#include "include/GB_defaults.h"
#include "include/GB_rand.h"

#ifdef GB_JIT_KERNEL

    //--------------------------------------------------------------------------
    // include files for JIT kernels
    //--------------------------------------------------------------------------

    // Placed in the SuiteSparse/GrB(version)/src/include folder by GrB_init,
    // via the JITPackage.  These files are used by the CPU JIT kernels (via
    // this file) and the CUDA JIT kernels (CUDA/include/GB_cuda_kernel.cuh):
    #include "include/GB_opaque.h"
    #include "include/GB_math_macros.h"
    #include "include/GB_bytes.h"
    #include "include/GB_pun.h"
    #include "include/GB_partition.h"
    #include "include/GB_zombie.h"
    #include "include/GB_binary_search.h"
    #include "include/GB_int64_mult.h"
    #include "include/GB_index.h"
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

    // not used by CUDA
    #include "include/GB_ijlist.h"
    #include "include/GB_atomics.h"
    #include "include/GB_assert_kernels.h"
    #include "include/GB_nthreads.h"
    #include "include/GB_log2.h"
    #include "include/GB_wait_macros.h"
    #include "include/GB_AxB_macros.h"
    #include "include/GB_ek_slice_kernels.h"
    #include "include/GB_bitmap_scatter.h"

#else

    //--------------------------------------------------------------------------
    // include files for the GraphBLAS libary
    //--------------------------------------------------------------------------

    // Original location in the GraphBLAS/Source folder, for compiling
    // the GraphBLAS library, including PreJIT kernels:
    #include "builtin/include/GB_opaque.h"
    #include "math/include/GB_math_macros.h"
    #include "type/include/GB_bytes.h"
    #include "type/include/GB_pun.h"
    #include "slice/include/GB_partition.h"
    #include "math/include/GB_zombie.h"
    #include "math/include/GB_binary_search.h"
    #include "math/include/GB_int64_mult.h"
    #include "matrix/include/GB_index.h"
    #include "math/include/GB_hash.h"
    #include "math/include/GB_complex.h"
    #include "math/include/GB_iceil.h"
    #include "memory/include/GB_memory_macros.h"
    #include "print/include/GB_printf_kernels.h"
    #include "matrix/include/GB_clear_matrix_header.h"
    #include "werk/include/GB_werk.h"
    #include "slice/include/GB_task_struct.h"
    #include "callback/include/GB_callback_proto.h"
    #include "mxm/include/GB_saxpy3task_struct.h"
    #include "callback/include/GB_callback.h"
    #include "hyper/include/GB_hyper_hash_lookup.h"
    #include "ok/include/GB_ok.h"
    #include "omp/include/GB_omp_kernels.h"

    // not used by CUDA
    #include "ij/include/GB_ijlist.h"
    #include "omp/include/GB_atomics.h"
    #include "ok/include/GB_assert_kernels.h"
    #include "omp/include/GB_nthreads.h"
    #include "math/include/GB_log2.h"
    #include "wait/include/GB_wait_macros.h"
    #include "mxm/include/GB_AxB_macros.h"
    #include "slice/include/GB_ek_slice_kernels.h"
    #include "assign/include/GB_bitmap_scatter.h"

#endif

#endif

