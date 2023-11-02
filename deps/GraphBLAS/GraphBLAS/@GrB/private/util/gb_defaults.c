//------------------------------------------------------------------------------
// gb_defaults: set global GraphBLAS defaults for MATLAB
//------------------------------------------------------------------------------

// SuiteSparse:GraphBLAS, Timothy A. Davis, (c) 2017-2023, All Rights Reserved.
// SPDX-License-Identifier: Apache-2.0

//------------------------------------------------------------------------------

#include "gb_interface.h"

void gb_defaults (void)     // set global GraphBLAS defaults for MATLAB
{
    // mxMalloc, mxCalloc, mxRealloc, and mxFree are not thread safe
    GB_Global_malloc_is_thread_safe_set (false) ;

    // must use mexPrintf to print to Command Window
    OK (GxB_Global_Option_set (GxB_PRINTF, mexPrintf)) ;
    OK (GxB_Global_Option_set (GxB_FLUSH, gb_flush)) ;

    // enable the JIT
    OK (GxB_Global_Option_set (GxB_JIT_C_CONTROL, GxB_JIT_ON)) ;

    // built-in matrices are stored by column
    OK (GxB_Global_Option_set (GxB_FORMAT, GxB_BY_COL)) ;

    // print 1-based indices
    OK (GxB_Global_Option_set (GxB_PRINT_1BASED, true)) ;

    // burble is off
    OK (GxB_Global_Option_set (GxB_BURBLE, false)) ;

    // default # of threads from omp_get_max_threads
    OK (GxB_Global_Option_set (GxB_NTHREADS, GB_omp_get_max_threads ( ))) ;

    // default chunk
    OK (GxB_Global_Option_set (GxB_CHUNK, GB_CHUNK_DEFAULT)) ;

    // for debug only
    GB_Global_abort_set (gb_abort) ;

    // for printing memory sizes of matrices
    GB_Global_print_mem_shallow_set (true) ;
}

