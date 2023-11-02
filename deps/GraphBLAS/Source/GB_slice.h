//------------------------------------------------------------------------------
// GB_slice.h: definitions for GB_eslice and GB_pslice
//------------------------------------------------------------------------------

// SuiteSparse:GraphBLAS, Timothy A. Davis, (c) 2017-2023, All Rights Reserved.
// SPDX-License-Identifier: Apache-2.0

//------------------------------------------------------------------------------

#ifndef GB_SLICE_H
#define GB_SLICE_H

void GB_pslice                      // slice Ap
(
    int64_t *restrict Slice,     // size ntasks+1
    const int64_t *restrict Ap,  // array size n+1 (NULL if full or bitmap)
    const int64_t n,
    const int ntasks,               // # of tasks
    const bool perfectly_balanced
) ;

void GB_eslice
(
    // output:
    int64_t *Slice,         // array of size ntasks+1
    // input:
    int64_t e,              // number items to partition amongst the tasks
    const int ntasks        // # of tasks
) ;

#endif

