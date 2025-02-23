//------------------------------------------------------------------------------
// GB_callback.h: typedefs for kernel callbacks
//------------------------------------------------------------------------------

// SuiteSparse:GraphBLAS, Timothy A. Davis, (c) 2017-2025, All Rights Reserved.
// SPDX-License-Identifier: Apache-2.0

//------------------------------------------------------------------------------

#ifndef GB_CALLBACK_H
#define GB_CALLBACK_H

//------------------------------------------------------------------------------
// function pointers to callback methods
//------------------------------------------------------------------------------

typedef GB_CALLBACK_SAXPY3_CUMSUM_PROTO ((*GB_AxB_saxpy3_cumsum_f)) ;
typedef GB_CALLBACK_BITMAP_M_SCATTER_WHOLE_PROTO ((*GB_bitmap_M_scatter_whole_f)) ;
typedef GB_CALLBACK_BIX_ALLOC_PROTO ((*GB_bix_alloc_f)) ;
typedef GB_CALLBACK_EK_SLICE_PROTO ((*GB_ek_slice_f)) ;
typedef GB_CALLBACK_FREE_MEMORY_PROTO ((*GB_free_memory_f)) ;
typedef GB_CALLBACK_MALLOC_MEMORY_PROTO ((*GB_malloc_memory_f)) ;
typedef GB_CALLBACK_MEMSET_PROTO ((*GB_memset_f)) ;
typedef GB_CALLBACK_WERK_POP_PROTO ((*GB_werk_pop_f)) ;
typedef GB_CALLBACK_WERK_PUSH_PROTO ((*GB_werk_push_f)) ;
typedef GB_CALLBACK_HYPER_HASH_BUILD_PROTO ((*GB_hyper_hash_build_f)) ;
typedef GB_CALLBACK_SUBASSIGN_ONE_SLICE_PROTO ((*GB_subassign_one_slice_f)) ;
typedef GB_CALLBACK_ADD_PHASE0_PROTO ((*GB_add_phase0_f)) ;
typedef GB_CALLBACK_EWISE_SLICE_PROTO ((*GB_ewise_slice_f)) ;
typedef GB_CALLBACK_SUBASSIGN_IXJ_SLICE_PROTO ((*GB_subassign_IxJ_slice_f)) ;
typedef GB_CALLBACK_PENDING_ENSURE_PROTO ((*GB_Pending_ensure_f)) ;
typedef GB_CALLBACK_SUBASSIGN_08N_SLICE_PROTO ((*GB_subassign_08n_slice_f)) ;
typedef GB_CALLBACK_BITMAP_ASSIGN_TO_FULL_PROTO ((*GB_bitmap_assign_to_full_f));
typedef GB_CALLBACK_P_SLICE_PROTO ((*GB_p_slice_f)) ;

//------------------------------------------------------------------------------
// GB_callback: a struct to pass to kernels to give them their callback methods
//------------------------------------------------------------------------------

typedef struct
{
    GB_AxB_saxpy3_cumsum_f      GB_AxB_saxpy3_cumsum_func ;
    GB_bitmap_M_scatter_whole_f GB_bitmap_M_scatter_whole_func ;
    GB_bix_alloc_f              GB_bix_alloc_func ;
    GB_ek_slice_f               GB_ek_slice_func ;
    GB_free_memory_f            GB_free_memory_func ;
    GB_malloc_memory_f          GB_malloc_memory_func ;
    GB_memset_f                 GB_memset_func ;
    GB_werk_pop_f               GB_werk_pop_func ;
    GB_werk_push_f              GB_werk_push_func ;
    GB_hyper_hash_build_f       GB_hyper_hash_build_func ;
    GB_subassign_one_slice_f    GB_subassign_one_slice_func ;
    GB_add_phase0_f             GB_add_phase0_func ;
    GB_ewise_slice_f            GB_ewise_slice_func ;
    GB_subassign_IxJ_slice_f    GB_subassign_IxJ_slice_func ;
    GB_Pending_ensure_f         GB_Pending_ensure_func ;
    GB_subassign_08n_slice_f    GB_subassign_08n_slice_func ;
    GB_bitmap_assign_to_full_f  GB_bitmap_assign_to_full_func ;
    GB_p_slice_f                GB_p_slice_func ;
    GB_abort_f                  GB_abort_func ;
}
GB_callback_struct ;

GB_GLOBAL GB_callback_struct GB_callback ;

#endif

