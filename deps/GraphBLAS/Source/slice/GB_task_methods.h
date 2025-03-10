//------------------------------------------------------------------------------
// GB_task_methods.h: parallel task descriptor
//------------------------------------------------------------------------------

// SuiteSparse:GraphBLAS, Timothy A. Davis, (c) 2017-2025, All Rights Reserved.
// SPDX-License-Identifier: Apache-2.0

//------------------------------------------------------------------------------

#ifndef GB_TASK_METHODS_H
#define GB_TASK_METHODS_H

// The element-wise computations (GB_add, GB_emult, and GB_mask) compute
// C(:,j)<M(:,j)> = op (A (:,j), B(:,j)).  They are parallelized by slicing the
// work into tasks, described by the GB_task_struct.

// There are two kinds of tasks.  For a coarse task, kfirst <= klast, and the
// task computes all vectors in C(:,kfirst:klast), inclusive.  None of the
// vectors are sliced and computed by other tasks.  For a fine task, klast is
// -1.  The task computes part of the single vector C(:,kfirst).  It starts at
// pA in Ai,Ax, at pB in Bi,Bx, and (if M is present) at pM in Mi,Mx.  It
// computes C(:,kfirst), starting at pC in Ci,Cx.

// GB_subref also uses the TaskList.  It has 12 kinds of fine tasks,
// corresponding to each of the 12 methods used in GB_subref_template.  For
// those fine tasks, method = -TaskList [taskid].klast defines the method to
// use.

// The GB_subassign functions use the TaskList, in many different ways.

// typedef GB_task_struct:
#include "include/GB_task_struct.h"

// GB_REALLOC_TASK_WORK: Allocate or reallocate the TaskList so that it can
// hold at least ntasks.  Double the size if it's too small.

#define GB_REALLOC_TASK_WORK(TaskList,ntasks,max_ntasks)                    \
{                                                                           \
    if ((ntasks) >= max_ntasks)                                             \
    {                                                                       \
        bool ok ;                                                           \
        int nold = (max_ntasks == 0) ? 0 : (max_ntasks + 1) ;               \
        int nnew = 2 * (ntasks) + 1 ;                                       \
        GB_REALLOC_MEMORY (TaskList, nnew, sizeof (GB_task_struct),         \
            &TaskList_size, &ok) ;                                          \
        if (!ok)                                                            \
        {                                                                   \
            /* out of memory */                                             \
            GB_FREE_ALL ;                                                   \
            return (GrB_OUT_OF_MEMORY) ;                                    \
        }                                                                   \
        for (int t = nold ; t < nnew ; t++)                                 \
        {                                                                   \
            TaskList [t].kfirst = -1 ;                                      \
            TaskList [t].klast  = INT64_MIN ;                               \
            TaskList [t].pA     = INT64_MIN ;                               \
            TaskList [t].pA_end = INT64_MIN ;                               \
            TaskList [t].pB     = INT64_MIN ;                               \
            TaskList [t].pB_end = INT64_MIN ;                               \
            TaskList [t].pC     = INT64_MIN ;                               \
            TaskList [t].pC_end = INT64_MIN ;                               \
            TaskList [t].pM     = INT64_MIN ;                               \
            TaskList [t].pM_end = INT64_MIN ;                               \
            TaskList [t].len    = INT64_MIN ;                               \
        }                                                                   \
        max_ntasks = 2 * (ntasks) ;                                         \
    }                                                                       \
    ASSERT ((ntasks) < max_ntasks) ;                                        \
}

void GB_slice_vector
(
    // output: return i, pA, and pB
    int64_t *p_i,                   // work starts at A(i,kA) and B(i,kB)
    int64_t *p_pM,                  // M(i:end,kM) starts at pM
    int64_t *p_pA,                  // A(i:end,kA) starts at pA
    int64_t *p_pB,                  // B(i:end,kB) starts at pB
    // input:
    const int64_t pM_start,         // M(:,kM) starts at pM_start in Mi,Mx
    const int64_t pM_end,           // M(:,kM) ends at pM_end-1 in Mi,Mx
    const void *Mi,                 // indices of M (or NULL)
    const bool Mi_is_32,            // if true, Mi is 32-bit; else 64 bit
    const int64_t pA_start,         // A(:,kA) starts at pA_start in Ai,Ax
    const int64_t pA_end,           // A(:,kA) ends at pA_end-1 in Ai,Ax
    const void *Ai,                 // indices of A (or NULL)
    const bool Ai_is_32,            // if true, Ai is 32-bit; else 64 bit
    const int64_t pB_start,         // B(:,kB) starts at pB_start in Bi,Bx
    const int64_t pB_end,           // B(:,kB) ends at pB_end-1 in Bi,Bx
    const void *Bi,                 // indices of B (or NULL)
    const bool Bi_is_32,            // if true, Bi is 32-bit; else 64 bit
    const int64_t vlen,             // A->vlen and B->vlen
    const double target_work        // target work
) ;

void GB_task_cumsum
(
    void *Cp,                           // size Cnvec+1
    const bool Cp_is_32,                // if true, Cp is 32-bit, else 64-bit
    const int64_t Cnvec,
    int64_t *Cnvec_nonempty,            // # of non-empty vectors in C
    GB_task_struct *restrict TaskList,  // array of structs
    const int ntasks,                   // # of tasks
    const int nthreads,                 // # of threads
    GB_Werk Werk
) ;

void GB_p_slice_32       // slice Work, 32-bit
(
    // output:
    int64_t *restrict Slice,    // size ntasks+1
    // input:
    const uint32_t *Work,       // array size n+1
    const int64_t n,
    const int ntasks,           // # of tasks
    const bool perfectly_balanced
) ;

void GB_p_slice_64       // slice Work, 64-bit
(
    // output:
    int64_t *restrict Slice,    // size ntasks+1
    // input:
    const uint64_t *Work,       // array size n+1
    const int64_t n,
    const int ntasks,           // # of tasks
    const bool perfectly_balanced
) ;

void GB_p_slice_float       // slice Work, float
(
    // output:
    int64_t *restrict Slice,    // size ntasks+1
    // input:
    const float *Work,          // array size n+1
    const int64_t n,
    const int ntasks            // # of tasks
) ;

#endif

