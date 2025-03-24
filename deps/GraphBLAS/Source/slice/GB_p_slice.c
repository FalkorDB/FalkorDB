//------------------------------------------------------------------------------
// GB_p_slice: partition Work for a set of tasks
//------------------------------------------------------------------------------

// SuiteSparse:GraphBLAS, Timothy A. Davis, (c) 2017-2025, All Rights Reserved.
// SPDX-License-Identifier: Apache-2.0

//------------------------------------------------------------------------------

// Work [0..n] is an array with monotonically increasing entries, of type
// uint32_t, uint64_t, or float.  This function slices Work so that each chunk
// has the same number of total values of its entries.  Work can be A->p for a
// matrix and then n = A->nvec.  Or it can be the work needed for computing n
// tasks, where Work [p] is the work for task p.

// If Work is NULL then the matrix A (not provided here) is full or bitmap,
// which this function handles (Work is implicit).

#include "GB.h"

//------------------------------------------------------------------------------
// GB_p_slice_32 and GB_p_slice_64: for Work as uint32_t or uint64_t
//------------------------------------------------------------------------------

#define GB_ENABLE_PERFECT_BALANCE
#define GB_Work_TYPE           uint32_t
#define GB_p_slice_TYPE        GB_p_slice_32
#define GB_p_slice_worker_TYPE GB_p_slice_worker_32
#define GB_trim_binary_search_TYPE GB_trim_binary_search_32
#include "slice/factory/GB_p_slice_template.c"

#define GB_Work_TYPE           uint64_t
#define GB_p_slice_TYPE        GB_p_slice_64
#define GB_p_slice_worker_TYPE GB_p_slice_worker_64
#define GB_trim_binary_search_TYPE GB_trim_binary_search_64
#include "slice/factory/GB_p_slice_template.c"

//------------------------------------------------------------------------------
// GB_p_slice_float: for Work as float
//------------------------------------------------------------------------------

// no perfect balance when Work is float

#undef  GB_ENABLE_PERFECT_BALANCE
#define GB_Work_TYPE           float
#define GB_p_slice_TYPE        GB_p_slice_float
#define GB_p_slice_worker_TYPE GB_p_slice_worker_float
#include "slice/factory/GB_p_slice_template.c"

//------------------------------------------------------------------------------
// GB_p_slice: partition Work for a set of tasks (uint32_t or uint64_t only)
//------------------------------------------------------------------------------

//  void GB_p_slice                 // slice Work, 32-bit or 64-bit
//  (
//      // output:
//      int64_t *restrict Slice,    // size ntasks+1
//      // input:
//      const void *Work,           // array size n+1 (full/bitmap: NULL)
//      bool Work_is_32,            // if true, Work is uint32_t, else uint64_t
//      const int64_t n,
//      const int ntasks,           // # of tasks
//      const bool perfectly_balanced
//  )

GB_CALLBACK_P_SLICE_PROTO (GB_p_slice)
{

    //--------------------------------------------------------------------------
    // check inputs
    //--------------------------------------------------------------------------

    ASSERT (Slice != NULL) ;

    //--------------------------------------------------------------------------
    // slice the work
    //--------------------------------------------------------------------------

    if (Work == NULL)
    { 
        // A is full or bitmap: slice 0:n equally for all tasks
        GB_e_slice (Slice, n, ntasks) ;
    }
    else
    {
        // A is sparse or hypersparse
        if (Work_is_32)
        { 
            // Work is uint32_t
            GB_p_slice_32 (Slice, Work, n, ntasks, perfectly_balanced) ;
        }
        else
        { 
            // Work is uint64_t
            GB_p_slice_64 (Slice, Work, n, ntasks, perfectly_balanced) ;
        }
    }
}

