//------------------------------------------------------------------------------
// GB_memory.h: memory allocation
//------------------------------------------------------------------------------

// SuiteSparse:GraphBLAS, Timothy A. Davis, (c) 2017-2025, All Rights Reserved.
// SPDX-License-Identifier: Apache-2.0

//------------------------------------------------------------------------------

#ifndef GB_MEMORY_H
#define GB_MEMORY_H

//------------------------------------------------------------------------------
// memory management
//------------------------------------------------------------------------------

void GB_memoryUsage         // count # allocated blocks and their sizes
(
    int64_t *nallocs,       // # of allocated memory blocks
    uint64_t *mem_deep,       // # of bytes in blocks owned by this matrix
    uint64_t *mem_shallow,    // # of bytes in blocks owned by another matrix
    const GrB_Matrix A,     // matrix to query
    bool count_hyper_hash   // if true, include A->Y
) ;

void *GB_realloc_memory     // pointer to reallocated block of memory, or
                            // to original block if the reallocation failed.
(
    uint64_t nitems_new,    // new number of items in the object
    uint64_t size_of_item,  // size of each item
    // input/output
    void *p,                // old object to reallocate
    uint64_t *p_mem,        // memsize and arena of object p to reallocate
    // output
    bool *ok                // true if successful, false otherwise
) ;

void *GB_xalloc_memory      // return the newly-allocated space
(
    // input
    bool use_calloc,        // if true, use calloc
    bool iso,               // if true, only allocate a single entry
    uint64_t nentries,      // # of entries to allocate if non iso
    uint64_t sizeof_entry,  // size of each entry
    // input/output
    uint64_t *mem           // resulting memsize and arena
) ;

//------------------------------------------------------------------------------
// parallel memcpy and memset
//------------------------------------------------------------------------------

void GB_memcpy                  // parallel memcpy
(
    void *dest,                 // destination
    const void *src,            // source
    size_t n,                   // # of bytes to copy
    int nthreads                // # of threads to use
) ;

//------------------------------------------------------------------------------
// GB_string_copy: a replacement for strncpy
//------------------------------------------------------------------------------

void GB_string_copy
(
    char *dest,
    const char *source,
    size_t dest_size
) ;

//------------------------------------------------------------------------------
// GB_rmm_malloc and GB_rmm_free: wrappers for Rapids, or empty stubs if no CUDA
//------------------------------------------------------------------------------

// If CUDA is enabled, these functions are defined in
// GraphBLAS/CUDA/rmm/GB_rmm_wrap.cpp, as C-callable wrappers for the C++
// Rapids Memory Manager allocate/deallocate methods.  If CUDA is not enabled,
// these functions are defined in GraphBLAS/Source/memory/GB_no_malloc_free.c,
// as functions that do nothing (the malloc method returns NULL and the free
// method does nothing).  These functions defined memory arenas 8 to 71, in
// both cases, so that when CUDA is not enabled, the arenas 8 to 71 are still
// reserved in case an application uses a CUDA-enabled copy of GraphBLAS.

#define GB_RMM_MALLOC_FREE_DECLARE(id)               \
    void *GB_rmm_malloc_ ## id (size_t size) ;       \
    void GB_rmm_free_ ## id (void *p) ;

GB_RMM_MALLOC_FREE_DECLARE (0) ;
GB_RMM_MALLOC_FREE_DECLARE (1) ;
GB_RMM_MALLOC_FREE_DECLARE (2) ;
GB_RMM_MALLOC_FREE_DECLARE (3) ;
GB_RMM_MALLOC_FREE_DECLARE (4) ;
GB_RMM_MALLOC_FREE_DECLARE (6) ;
GB_RMM_MALLOC_FREE_DECLARE (7) ;
GB_RMM_MALLOC_FREE_DECLARE (8) ;
GB_RMM_MALLOC_FREE_DECLARE (9) ;

GB_RMM_MALLOC_FREE_DECLARE (10) ;
GB_RMM_MALLOC_FREE_DECLARE (11) ;
GB_RMM_MALLOC_FREE_DECLARE (12) ;
GB_RMM_MALLOC_FREE_DECLARE (13) ;
GB_RMM_MALLOC_FREE_DECLARE (14) ;
GB_RMM_MALLOC_FREE_DECLARE (16) ;
GB_RMM_MALLOC_FREE_DECLARE (17) ;
GB_RMM_MALLOC_FREE_DECLARE (18) ;
GB_RMM_MALLOC_FREE_DECLARE (19) ;

GB_RMM_MALLOC_FREE_DECLARE (20) ;
GB_RMM_MALLOC_FREE_DECLARE (21) ;
GB_RMM_MALLOC_FREE_DECLARE (22) ;
GB_RMM_MALLOC_FREE_DECLARE (23) ;
GB_RMM_MALLOC_FREE_DECLARE (24) ;
GB_RMM_MALLOC_FREE_DECLARE (26) ;
GB_RMM_MALLOC_FREE_DECLARE (27) ;
GB_RMM_MALLOC_FREE_DECLARE (28) ;
GB_RMM_MALLOC_FREE_DECLARE (29) ;

GB_RMM_MALLOC_FREE_DECLARE (30) ;
GB_RMM_MALLOC_FREE_DECLARE (31) ;
GB_RMM_MALLOC_FREE_DECLARE (32) ;
GB_RMM_MALLOC_FREE_DECLARE (33) ;
GB_RMM_MALLOC_FREE_DECLARE (34) ;
GB_RMM_MALLOC_FREE_DECLARE (36) ;
GB_RMM_MALLOC_FREE_DECLARE (37) ;
GB_RMM_MALLOC_FREE_DECLARE (38) ;
GB_RMM_MALLOC_FREE_DECLARE (39) ;

GB_RMM_MALLOC_FREE_DECLARE (40) ;
GB_RMM_MALLOC_FREE_DECLARE (41) ;
GB_RMM_MALLOC_FREE_DECLARE (42) ;
GB_RMM_MALLOC_FREE_DECLARE (43) ;
GB_RMM_MALLOC_FREE_DECLARE (44) ;
GB_RMM_MALLOC_FREE_DECLARE (46) ;
GB_RMM_MALLOC_FREE_DECLARE (47) ;
GB_RMM_MALLOC_FREE_DECLARE (48) ;
GB_RMM_MALLOC_FREE_DECLARE (49) ;

GB_RMM_MALLOC_FREE_DECLARE (50) ;
GB_RMM_MALLOC_FREE_DECLARE (51) ;
GB_RMM_MALLOC_FREE_DECLARE (52) ;
GB_RMM_MALLOC_FREE_DECLARE (53) ;
GB_RMM_MALLOC_FREE_DECLARE (54) ;
GB_RMM_MALLOC_FREE_DECLARE (56) ;
GB_RMM_MALLOC_FREE_DECLARE (57) ;
GB_RMM_MALLOC_FREE_DECLARE (58) ;
GB_RMM_MALLOC_FREE_DECLARE (59) ;

GB_RMM_MALLOC_FREE_DECLARE (60) ;
GB_RMM_MALLOC_FREE_DECLARE (61) ;
GB_RMM_MALLOC_FREE_DECLARE (62) ;
GB_RMM_MALLOC_FREE_DECLARE (63) ;

#undef GB_RMM_MALLOC_FREE_DECLARE

#endif

