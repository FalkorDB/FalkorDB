//------------------------------------------------------------------------------
// GB_init.h: definitions for GB_init
//------------------------------------------------------------------------------

// SuiteSparse:GraphBLAS, Timothy A. Davis, (c) 2017-2025, All Rights Reserved.
// SPDX-License-Identifier: Apache-2.0

//------------------------------------------------------------------------------

#ifndef GB_INIT_H
#define GB_INIT_H

GrB_Info GB_init            // start up GraphBLAS
(
    int mode,               // blocking or non-blocking mode

    // pointers to memory management functions:
    GB_malloc_function_t malloc_function,           // required
    GB_calloc_function_t calloc_function,           // unused, can be NULL
    GB_realloc_function_t realloc_function,         // optional, can be NULL
    GB_free_function_t free_function,               // required

    GB_Werk Werk      // from GrB_init or GxB_init
) ;

#endif

