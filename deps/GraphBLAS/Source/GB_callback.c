//------------------------------------------------------------------------------
// GB_callback.c: global callback struct for kernels
//------------------------------------------------------------------------------

// SuiteSparse:GraphBLAS, Timothy A. Davis, (c) 2017-2023, All Rights Reserved.
// SPDX-License-Identifier: Apache-2.0

//------------------------------------------------------------------------------

#include "GB_Template.h"
#include "GB_callback.h"
#include "GB_AxB_saxpy3.h"
#include "GB_bitmap_assign_methods.h"
#include "GB_ek_slice.h"
#include "GB_sort.h"

GB_callback_struct GB_callback =
{
    .GB_AxB_saxpy3_cumsum_func      = GB_AxB_saxpy3_cumsum,
    .GB_bitmap_M_scatter_func       = GB_bitmap_M_scatter,
    .GB_bitmap_M_scatter_whole_func = GB_bitmap_M_scatter_whole,
    .GB_bix_alloc_func              = GB_bix_alloc,
    .GB_ek_slice_func               = GB_ek_slice,
    .GB_ek_slice_merge1_func        = GB_ek_slice_merge1,
    .GB_free_memory_func            = GB_free_memory,
    .GB_malloc_memory_func          = GB_malloc_memory,
    .GB_memset_func                 = GB_memset,
    .GB_qsort_1_func                = GB_qsort_1,
    .GB_werk_pop_func               = GB_werk_pop,
    .GB_werk_push_func              = GB_werk_push
} ;

