//------------------------------------------------------------------------------
// gbmx_defaults: set global GraphBLAS defaults for MATLAB
//------------------------------------------------------------------------------

// SuiteSparse:GraphBLAS, Timothy A. Davis, (c) 2017-2026, All Rights Reserved.
// SPDX-License-Identifier: Apache-2.0

//------------------------------------------------------------------------------

// GraphBLAS methods are called, but none of them allocate any memory, so this
// will not fail.  Each call to GraphBLAS is checked just in case, however.

typedef void (*function_pointer) (void) ;

GrB_Info gbmx_defaults      // set global GraphBLAS defaults for MATLAB
(
    char err [ERRLEN]
)
{ 
    #ifdef MALLOC_TRACKING
    // debugging only; disabled in production
    GB_Global_malloc_tracking_set (true) ;
    #endif

    // for debug assertions only, for the ASSERT (...) macro
    GB_Global_abort_set (gbmx_abort) ;

    // must use mexPrintf to print to Command Window
    OK (GrB_Global_set_VOID (GrB_GLOBAL, (void *) mexPrintf, GxB_PRINTF,
        sizeof (function_pointer))) ;
    OK (GrB_Global_set_VOID (GrB_GLOBAL, (void *) gbmx_flush, GxB_FLUSH,
        sizeof (function_pointer))) ;

    // enable the JIT
    OK (GrB_Global_set_INT32 (GrB_GLOBAL, GxB_JIT_ON, GxB_JIT_C_CONTROL)) ;

    // built-in matrices are stored by column
    OK (GrB_Global_set_INT32 (GrB_GLOBAL, GrB_COLMAJOR,
        GrB_STORAGE_ORIENTATION_HINT)) ;

    // print 1-based indices
    OK (GrB_Global_set_INT32 (GrB_GLOBAL, true, GxB_PRINT_1BASED)) ;

    // burble is off
    OK (GrB_Global_set_INT32 (GrB_GLOBAL, false, GxB_BURBLE)) ;

    // default # of threads from omp_get_max_threads
    int nthreads = GB_omp_get_max_threads ( ) ;
    OK (GrB_Global_set_INT32 (GrB_GLOBAL, nthreads, GxB_NTHREADS)) ;

    // default chunk: use the historical method to avoid any memory allocation
    OK (GxB_Global_Option_set_FP64 (GxB_CHUNK, (double) (64 * 1024))) ;

    // for printing memory sizes of matrices
    OK (GrB_Global_set_INT32 (GrB_GLOBAL, true,
        GxB_INCLUDE_READONLY_STATISTICS)) ;

    return (GrB_SUCCESS) ;
}

