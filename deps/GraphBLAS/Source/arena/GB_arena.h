//------------------------------------------------------------------------------
// GB_arena.h: utilities for arenas
//------------------------------------------------------------------------------

// SuiteSparse:GraphBLAS, Timothy A. Davis, (c) 2017-2026, All Rights Reserved.
// SPDX-License-Identifier: Apache-2.0

//------------------------------------------------------------------------------

#ifndef GB_ARENA_H
#define GB_ARENA_H

GrB_Info GB_set_arena           // set arena of a block of memory
(
    // input/output:
    void **p_handle,            // block of memory to change
    uint64_t *p_mem_handle,     // memsize and arena of block of memory
    // input
    const int new_arena,        // arena to move to
    const uint64_t new_memsize, // new size of the block of memory
    const uint64_t n,           // # of bytes that must be copied
    const int nthreads          // max # of threads to use
) ;

GrB_Info GB_set_arenas          // modify all arenas of a matrix
(
    // input/output
    GrB_Matrix *Ahandle,        // handle of matrix to modify
    // input
    const int new_header_arena, // new arena for the header of A
    const int new_data_arena    // new arena for the data content of A
) ;

GrB_Info GB_get_arena_alias
(
    // output
    GrB_Matrix *Chandle,    // output matrix, (*Chandle) is NULL on input
    // inputs
    const int new_header_arena, // arena for C header
    const GrB_Matrix A      // input matrix
) ;

void GB_put_arena_alias
(
    // input/outputs
    GrB_Matrix *Chandle,    // alias of A to be freed; NULL on output
    const GrB_Matrix A      // updated with any revisions in the alias header C
) ;

bool GB_arenas_will_wait    // true if GrB_wait will change arenas of A
(
    // input/output:
    GrB_Matrix A
) ;

bool GB_shallow_arenas_ok
(
    // input/output:
    GrB_Matrix A
) ;

GrB_Info GB_wait_arenas         // align data with A->data_arena
(
    GrB_Matrix A                // input/output matrix
) ;

#endif

