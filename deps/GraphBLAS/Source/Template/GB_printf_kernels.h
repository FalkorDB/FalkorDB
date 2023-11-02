//------------------------------------------------------------------------------
// GB_printf_kernels.h: definitions for printing from GraphBLAS
//------------------------------------------------------------------------------

// SuiteSparse:GraphBLAS, Timothy A. Davis, (c) 2017-2023, All Rights Reserved.
// SPDX-License-Identifier: Apache-2.0

//------------------------------------------------------------------------------

#ifndef GB_PRINTF_KERNELS_H
#define GB_PRINTF_KERNELS_H

#define GB_STRING_MATCH(s,t) (strcmp (s,t) == 0)

// format strings, normally %llu and %lld, for GrB_Index values
#define GBu "%" PRIu64
#define GBd "%" PRId64

//------------------------------------------------------------------------------
// GBDUMP: print to stdout
//------------------------------------------------------------------------------

// print to the standard output, and flush the result.  No error check is done.
// This function is used for the BURBLE, and for debugging output. 

// the JIT run time kernels use printf and flush directly from libc:
#undef  GBDUMP
#define GBDUMP(...)                                                 \
{                                                                   \
    printf (__VA_ARGS__) ;                                          \
    fflush (stdout) ;                                               \
}

//------------------------------------------------------------------------------
// GBPR: print to a file, or stdout if the file is NULL
//------------------------------------------------------------------------------

// print to a file f, or to stdout if f is NULL, and check the result.  This
// macro is used by all user-callable GxB_*print and GB_*check functions.
// The method is not used in any JIT run time kernel.

// JIT runtime kernels do not use these methods
#undef  GBPR
#define GBPR(...)

#undef  GBPR0
#define GBPR0(...)

#undef  GB_CHECK_MAGIC
#define GB_CHECK_MAGIC(object)

// JIT kernels cannot burble
#undef  GBURBLE
#define GBURBLE(...)

#undef  GB_BURBLE_DENSE
#define GB_BURBLE_DENSE(A,format)

#undef  GB_BURBLE_START
#define GB_BURBLE_START(func)

#undef  GB_BURBLE_END
#define GB_BURBLE_END

#undef  GB_BURBLE_N
#define GB_BURBLE_N(n,...)

#undef  GB_BURBLE_MATRIX
#define GB_BURBLE_MATRIX(A, ...)

#endif

