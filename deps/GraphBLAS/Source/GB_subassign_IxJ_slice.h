//------------------------------------------------------------------------------
// GB_subassign_IxJ_slice.h: definitions for GB_subassign_IxJ_slice
//------------------------------------------------------------------------------

// SuiteSparse:GraphBLAS, Timothy A. Davis, (c) 2017-2023, All Rights Reserved.
// SPDX-License-Identifier: Apache-2.0

//------------------------------------------------------------------------------

#ifndef GB_SUBASSIGN_IXJ_SLICE_H
#define GB_SUBASSIGN_IXJ_SLICE_H

//------------------------------------------------------------------------------
// GB_subassign_IxJ_slice
//------------------------------------------------------------------------------

// Slice IxJ for a scalar assignment method (Methods 01, 03, 13, 15, 17, 19),
// and for bitmap assignments (in GB_bitmap_assign_IxJ_template).

GrB_Info GB_subassign_IxJ_slice
(
    // output:
    GB_task_struct **p_TaskList,    // array of structs
    size_t *p_TaskList_size,        // size of TaskList
    int *p_ntasks,                  // # of tasks constructed
    int *p_nthreads,                // # of threads to use
    // input:
//  const GrB_Index *I,
    const int64_t nI,
//  const int Ikind,
//  const int64_t Icolon [3],
//  const GrB_Index *J,
    const int64_t nJ,
//  const int Jkind,
//  const int64_t Jcolon [3],
    GB_Werk Werk
) ;

#endif

