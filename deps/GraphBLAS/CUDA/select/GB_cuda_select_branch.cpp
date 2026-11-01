//------------------------------------------------------------------------------
// GraphBLAS/CUDA/select/GB_cuda_select_branch
//------------------------------------------------------------------------------

// SuiteSparse:GraphBLAS, Timothy A. Davis, (c) 2017-2026, All Rights Reserved.
// SPDX-License-Identifier: Apache-2.0

//------------------------------------------------------------------------------

#include "GB_cuda.hpp"

bool GB_cuda_select_branch
(
    const GrB_Matrix A,
    const GrB_IndexUnaryOp op
)
{
//  printf ("HACK: turn off cuda select\n") ; return (false) ;

    if (GB_ngpus_to_use (1) == 0) return (false) ;

    int data_arena = A->data_arena ;
    int header_arena = GB_arena (A->header_mem) ;
    int dev = data_arena - GxB_NARENAS ;

    bool use_cuda =
           (dev >= 0 && dev <= GB_Global_gpu_count_get ( )) // data on GPU
        && (data_arena == header_arena)                 // header on same GPU
        && (GB_jitifyer_get_control ( ) >= GxB_JIT_RUN) // JIT is running
        && (op->hash != UINT64_MAX)                     // op is jitable
        && GB_cuda_type_branch (A->type)                // types OK for CUDA
        && GB_cuda_type_branch (op->xtype)
        && GB_cuda_type_branch (op->ytype)
        && GB_cuda_type_branch (op->ztype)
        && GB_shallow_arenas_ok (A) ;       // shallow data on same GPU

    return (use_cuda) ;
}

