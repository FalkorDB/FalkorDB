//------------------------------------------------------------------------------
// GB_Template.h: definitions for Template methods
//------------------------------------------------------------------------------

// SuiteSparse:GraphBLAS, Timothy A. Davis, (c) 2017-2023, All Rights Reserved.
// SPDX-License-Identifier: Apache-2.0

//------------------------------------------------------------------------------

#ifndef GB_TEMPLATE_H
#define GB_TEMPLATE_H

//------------------------------------------------------------------------------
// definitions that modify GraphBLAS.h
//------------------------------------------------------------------------------

#include "GB_dev.h"
#include "GB_compiler.h"
#include "GB_warnings.h"
#include "GB_coverage.h"

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

#include "GB_prefix.h"
#include "GB_bytes.h"
#include "GB_defaults.h"
#include "GB_index.h"
#include "GB_complex.h"
#include "GB_pun.h"
#include "GB_atomics.h"
#include "GB_printf_kernels.h"
#include "GB_assert_kernels.h"
#include "GB_opaque.h"
#include "GB_math_macros.h"
#include "GB_binary_search.h"
#include "GB_zombie.h"
#include "GB_partition.h"
#include "GB_memory_macros.h"
#include "GB_werk.h"
#include "GB_nthreads.h"
#include "GB_log2.h"
#include "GB_task_struct.h"
#include "GB_hash.h"
#include "GB_wait_macros.h"
#include "GB_AxB_macros.h"
#include "GB_ek_slice_kernels.h"
#include "GB_bitmap_scatter.h"
#include "GB_int64_mult.h"
#include "GB_hyper_hash_lookup.h"
#include "GB_omp_kernels.h"
#include "GB_callback.h"

#endif

