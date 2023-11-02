//------------------------------------------------------------------------------
// GB_msort_3.h: definitions for GB_msort_3.c
//------------------------------------------------------------------------------

// SuiteSparse:GraphBLAS, Timothy A. Davis, (c) 2017-2023, All Rights Reserved.
// SPDX-License-Identifier: Apache-2.0

//------------------------------------------------------------------------------

// A parallel mergesort of an array of 3-by-n integers.  Each key consists
// of three integers.

#include "GB_sort.h"

//------------------------------------------------------------------------------
// prototypes only needed for GB_msort_3
//------------------------------------------------------------------------------

void GB_msort_3_create_merge_tasks
(
    // output:
    int64_t *restrict L_task,        // L_task [t0...t0+ntasks-1] computed
    int64_t *restrict L_len,         // L_len  [t0...t0+ntasks-1] computed
    int64_t *restrict R_task,        // R_task [t0...t0+ntasks-1] computed
    int64_t *restrict R_len,         // R_len  [t0...t0+ntasks-1] computed
    int64_t *restrict S_task,        // S_task [t0...t0+ntasks-1] computed
    // input:
    const int t0,                       // first task tid to create
    const int ntasks,                   // # of tasks to create
    const int64_t pS_start,             // merge into S [pS_start...]
    const int64_t *restrict L_0,     // Left = L [pL_start...pL_end-1]
    const int64_t *restrict L_1,
    const int64_t *restrict L_2,
    const int64_t pL_start,
    const int64_t pL_end,
    const int64_t *restrict R_0,     // Right = R [pR_start...pR_end-1]
    const int64_t *restrict R_1,
    const int64_t *restrict R_2,
    const int64_t pR_start,
    const int64_t pR_end
) ;

