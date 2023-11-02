//------------------------------------------------------------------------------
// GB_init.h: definitions for GB_init
//------------------------------------------------------------------------------

// SuiteSparse:GraphBLAS, Timothy A. Davis, (c) 2017-2023, All Rights Reserved.
// SPDX-License-Identifier: Apache-2.0

//------------------------------------------------------------------------------

#ifndef GB_INIT_H
#define GB_INIT_H

GrB_Info GB_init            // start up GraphBLAS
(
    const GrB_Mode mode,    // blocking or non-blocking mode

    // pointers to memory management functions.
    void * (* malloc_function  ) (size_t),          // required
    void * (* calloc_function  ) (size_t, size_t),  // optional, can be NULL
    void * (* realloc_function ) (void *, size_t),  // optional, can be NULL
    void   (* free_function    ) (void *),          // required

    GB_Werk Werk      // from GrB_init or GxB_init
) ;

#endif

