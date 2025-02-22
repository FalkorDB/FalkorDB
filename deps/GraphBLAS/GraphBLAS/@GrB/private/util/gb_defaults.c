//------------------------------------------------------------------------------
// gb_defaults: set global GraphBLAS defaults for MATLAB
//------------------------------------------------------------------------------

// SuiteSparse:GraphBLAS, Timothy A. Davis, (c) 2017-2025, All Rights Reserved.
// SPDX-License-Identifier: Apache-2.0

//------------------------------------------------------------------------------

// This function accesses GB_methods inside GraphBLAS.

#include "gb_interface.h"

typedef void (*function_pointer) (void) ;

void gb_defaults (void)     // set global GraphBLAS defaults for MATLAB
{
    // for debug only
    GB_Global_abort_set (gb_abort) ;

    // mxMalloc, mxCalloc, mxRealloc, and mxFree are not thread safe
    GB_Global_malloc_is_thread_safe_set (false) ;

    // must use mexPrintf to print to Command Window
    OK (GrB_Global_set_VOID (GrB_GLOBAL, (void *) mexPrintf, GxB_PRINTF,
        sizeof (function_pointer))) ;
    OK (GrB_Global_set_VOID (GrB_GLOBAL, (void *) gb_flush, GxB_FLUSH,
        sizeof (function_pointer))) ;

    // enable the JIT
    OK (GrB_Global_set_INT32 (GrB_GLOBAL, GxB_JIT_ON, GxB_JIT_C_CONTROL)) ;

    // built-in matrices are stored by column
    OK (GrB_Global_set_INT32 (GrB_GLOBAL,
        GrB_COLMAJOR, GrB_STORAGE_ORIENTATION_HINT)) ;

    // print 1-based indices
    OK (GrB_Global_set_INT32 (GrB_GLOBAL, true, GxB_PRINT_1BASED)) ;

    // burble is off
    OK (GrB_Global_set_INT32 (GrB_GLOBAL, false, GxB_BURBLE)) ;

    // default # of threads from omp_get_max_threads
    int nthreads = GB_omp_get_max_threads ( ) ;
    OK (GrB_Global_set_INT32 (GrB_GLOBAL, nthreads, GxB_NTHREADS)) ;

    // default chunk
    GrB_Scalar chunk_default = NULL ;
    OK (GrB_Scalar_new (&chunk_default, GrB_FP64)) ;
    OK (GrB_Scalar_setElement_FP64 (chunk_default, (double) (64 * 1024))) ;
    OK (GrB_Global_set_Scalar (GrB_GLOBAL, chunk_default, GxB_CHUNK)) ;
    OK (GrB_Scalar_free (&chunk_default)) ;

    // for printing memory sizes of matrices
    OK (GrB_Global_set_INT32 (GrB_GLOBAL, true,
        GxB_INCLUDE_READONLY_STATISTICS)) ;
}

