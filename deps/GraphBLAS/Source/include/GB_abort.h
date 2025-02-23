//------------------------------------------------------------------------------
// GB_abort.h: assertions for all of GraphBLAS, including JIT kernels
//------------------------------------------------------------------------------

// SuiteSparse:GraphBLAS, Timothy A. Davis, (c) 2017-2025, All Rights Reserved.
// SPDX-License-Identifier: Apache-2.0

//------------------------------------------------------------------------------

#ifndef GB_ABORT_H
#define GB_ABORT_H

typedef void (*GB_abort_f) (const char *file, int line) ;

#ifdef GB_JIT_RUNTIME

    // JIT kernels call GB_abort via a callback function pointer
    static GB_abort_f GB_abort = (GB_abort_f) NULL ;

#else

    // primary functions in the GraphBLAS library, including PreJIT kernels,
    // call GB_abort as a compile-time function, not a function pointer.
    void GB_abort (const char *file, int line) ;

#endif

// this assertion is always enabled
#define GB_assert(X)                        \
{                                           \
    if (!(X))                               \
    {                                       \
        GB_abort (__FILE__, __LINE__) ;     \
    }                                       \
}

#ifdef GB_DEBUG
    // assert X is true
    #define ASSERT(X) GB_assert (X)
#else
    // debugging disabled
    #define ASSERT(X)
#endif

#define GB_IMPLIES(p,q) (!(p) || (q))

#endif

