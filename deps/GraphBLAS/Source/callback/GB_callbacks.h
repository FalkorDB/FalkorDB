//------------------------------------------------------------------------------
// GB_callbacks.h: prototypes for kernel callbacks
//------------------------------------------------------------------------------

// SuiteSparse:GraphBLAS, Timothy A. Davis, (c) 2017-2024, All Rights Reserved.
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
GB_CALLBACK_BITMAP_M_SCATTER_WHOLE_PROTO (GB_bitmap_M_scatter_whole) ;

// added for assign:
GB_CALLBACK_HYPER_HASH_BUILD_PROTO (GB_hyper_hash_build) ;
GB_CALLBACK_SUBASSIGN_ONE_SLICE_PROTO (GB_subassign_one_slice) ;
GB_CALLBACK_ADD_PHASE0_PROTO (GB_add_phase0) ;
GB_CALLBACK_EWISE_SLICE_PROTO (GB_ewise_slice) ;
GB_CALLBACK_SUBASSIGN_IXJ_SLICE_PROTO (GB_subassign_IxJ_slice) ;
GB_CALLBACK_PENDING_ENSURE_PROTO (GB_Pending_ensure) ;
GB_CALLBACK_SUBASSIGN_08N_SLICE_PROTO (GB_subassign_08n_slice) ;
GB_CALLBACK_BITMAP_ASSIGN_TO_FULL_PROTO (GB_bitmap_assign_to_full) ;

// added for sort:
GB_CALLBACK_P_SLICE_PROTO (GB_p_slice) ;

#endif

