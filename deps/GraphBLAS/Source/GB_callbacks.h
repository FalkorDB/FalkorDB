//------------------------------------------------------------------------------
// GB_callbacks.h: prototypes for kernel callbacks for PreJIT
//------------------------------------------------------------------------------

// SuiteSparse:GraphBLAS, Timothy A. Davis, (c) 2017-2023, All Rights Reserved.
// SPDX-License-Identifier: Apache-2.0

//------------------------------------------------------------------------------

#ifndef GB_CALLBACKS_H
#define GB_CALLBACKS_H

GB_CALLBACK_SAXPY3_CUMSUM_PROTO (GB_AxB_saxpy3_cumsum) ;
GB_CALLBACK_EK_SLICE_PROTO (GB_ek_slice) ;
GB_CALLBACK_EK_SLICE_MERGE1_PROTO (GB_ek_slice_merge1) ;
GB_CALLBACK_MALLOC_MEMORY_PROTO (GB_malloc_memory) ;
GB_CALLBACK_FREE_MEMORY_PROTO (GB_free_memory) ;
GB_CALLBACK_MEMSET_PROTO (GB_memset) ;
GB_CALLBACK_BIX_ALLOC_PROTO (GB_bix_alloc) ;
GB_CALLBACK_QSORT_1_PROTO (GB_qsort_1) ; 
GB_CALLBACK_WERK_PUSH_PROTO (GB_werk_push) ;
GB_CALLBACK_WERK_POP_PROTO (GB_werk_pop) ;
GB_CALLBACK_BITMAP_M_SCATTER_PROTO (GB_bitmap_M_scatter) ;
GB_CALLBACK_BITMAP_M_SCATTER_WHOLE_PROTO (GB_bitmap_M_scatter_whole) ;

#endif

