//------------------------------------------------------------------------------
// GB_subref_sparse_jit: C=A(I,J) when C and A are sparse/hypersparse
//------------------------------------------------------------------------------

// SuiteSparse:GraphBLAS, Timothy A. Davis, (c) 2017-2024, All Rights Reserved.
// SPDX-License-Identifier: Apache-2.0

//------------------------------------------------------------------------------

#include "GB.h"
#include "jitifyer/GB_stringify.h"

typedef GB_JIT_KERNEL_SUBREF_SPARSE_PROTO ((*GB_jit_dl_function)) ;

GrB_Info GB_subref_sparse_jit
(
    // output matrix
    GrB_Matrix C,                       // same type as A
    // from phase1:
    const GB_task_struct *restrict TaskList,  // list of tasks
    const int ntasks,                   // # of tasks
    const int nthreads,                 // # of threads to use
    const bool post_sort,               // true if post-sort needed
    const int64_t *Mark,                // for I inverse buckets, size A->vlen
    const int64_t *Inext,               // for I inverse buckets, size nI
    const bool I_has_duplicates,        // true if I has duplicates
    // from phase0:
    const int64_t *restrict Ap_start,
    const int64_t *restrict Ap_end,
    const bool need_qsort,
    const int Ikind,
    const int64_t nI,
    const int64_t Icolon [3],
    // original input:
    const GrB_Matrix A,
    const GrB_Index *I
)
{ 

    //--------------------------------------------------------------------------
    // encodify the problem
    //--------------------------------------------------------------------------

    GB_jit_encoding encoding ;
    char *suffix ;
    uint64_t hash = GB_encodify_subref (&encoding, &suffix,
        GB_JIT_KERNEL_SUBREF_SPARSE, C, Ikind, 0,
        need_qsort, I_has_duplicates, A) ;

    //--------------------------------------------------------------------------
    // get the kernel function pointer, loading or compiling it if needed
    //--------------------------------------------------------------------------

    void *dl_function ;
    GrB_Info info = GB_jitifyer_load (&dl_function,
        GB_jit_subref_family, "subref_sparse",
        hash, &encoding, suffix, NULL, NULL,
        NULL, C->type, NULL, NULL) ;
    if (info != GrB_SUCCESS) return (info) ;

    //--------------------------------------------------------------------------
    // call the jit kernel and return result
    //--------------------------------------------------------------------------

    GB_jit_dl_function GB_jit_kernel = (GB_jit_dl_function) dl_function ;
    return (GB_jit_kernel (C, TaskList, ntasks, nthreads, post_sort, Mark,
        Inext, Ap_start, Ap_end, nI, Icolon, A, I, &GB_callback)) ;
}

