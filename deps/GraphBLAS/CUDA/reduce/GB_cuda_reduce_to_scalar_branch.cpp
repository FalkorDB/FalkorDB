//------------------------------------------------------------------------------
// CUDA/reduce/GB_cuda_reduce_to_scalar_branch: decide to use GPU for reduce
//------------------------------------------------------------------------------

// SuiteSparse:GraphBLAS, Timothy A. Davis, (c) 2017-2026, All Rights Reserved.
// SPDX-License-Identifier: Apache-2.0

//------------------------------------------------------------------------------

// Decide branch direction for GPU use for the reduction to scalar

#include "reduce/GB_cuda_reduce.hpp"

bool GB_cuda_reduce_to_scalar_branch    // return true to use the GPU
(
    const GrB_Monoid monoid,        // monoid to do the reduction
    const GrB_Matrix A              // input matrix
)
{
//  printf ("HACK: turn off cuda reduce\n") ; return (false) ;

    if (GB_ngpus_to_use (1) == 0) return (false) ;

    int data_arena = A->data_arena ;
    int header_arena = GB_arena (A->header_mem) ;
    int dev = data_arena - GxB_NARENAS ;

    bool use_cuda =
           (!A->iso)                                    // A is not iso
        && (monoid->op->opcode != GB_ANY_binop_code)    // monoid is not ANY
        && (dev >= 0 && dev <= GB_Global_gpu_count_get ( )) // data on GPU
        && (data_arena == header_arena)                 // header on same GPU
        && (GB_jitifyer_get_control ( ) >= GxB_JIT_RUN) // JIT is running
        && (monoid->hash != UINT64_MAX)                 // monoid is jitable
        && GB_cuda_type_branch (A->type)                // types OK for CUDA
        && GB_cuda_type_branch (monoid->op->ztype)
        && GB_shallow_arenas_ok (A) ;       // shallow data on same GPU

    return (use_cuda) ;
}

