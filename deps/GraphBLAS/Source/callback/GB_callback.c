//------------------------------------------------------------------------------
// GB_callback.c: global callback struct for kernels
//------------------------------------------------------------------------------

// SuiteSparse:GraphBLAS, Timothy A. Davis, (c) 2017-2024, All Rights Reserved.
// SPDX-License-Identifier: Apache-2.0

//------------------------------------------------------------------------------

#include "GB.h"
#include "callback/include/GB_callback.h"

GB_callback_struct GB_callback =
{
    .GB_AxB_saxpy3_cumsum_func      = GB_AxB_saxpy3_cumsum,
    .GB_bitmap_M_scatter_whole_func = GB_bitmap_M_scatter_whole,
    .GB_bix_alloc_func              = GB_bix_alloc,
    .GB_ek_slice_func               = GB_ek_slice,
    .GB_ek_slice_merge1_func        = GB_ek_slice_merge1,
    .GB_free_memory_func            = GB_free_memory,
    .GB_malloc_memory_func          = GB_malloc_memory,
    .GB_memset_func                 = GB_memset,
    .GB_qsort_1_func                = GB_qsort_1,
    .GB_werk_pop_func               = GB_werk_pop,
    .GB_werk_push_func              = GB_werk_push,

    // added for assign:
    .GB_hyper_hash_build_func       = GB_hyper_hash_build,
    .GB_subassign_one_slice_func    = GB_subassign_one_slice,
    .GB_add_phase0_func             = GB_add_phase0,
    .GB_ewise_slice_func            = GB_ewise_slice,
    .GB_subassign_IxJ_slice_func    = GB_subassign_IxJ_slice,
    .GB_Pending_ensure_func         = GB_Pending_ensure,
    .GB_subassign_08n_slice_func    = GB_subassign_08n_slice,
    .GB_bitmap_assign_to_full_func  = GB_bitmap_assign_to_full,

    // added for sort:
    .GB_p_slice_func                = GB_p_slice,

} ;

