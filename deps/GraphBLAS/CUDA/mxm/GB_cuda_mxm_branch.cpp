//------------------------------------------------------------------------------
// GraphBLAS/CUDA/GB_cuda_mxm_branch: decide to use GPU for GrB_mxm
//------------------------------------------------------------------------------

// SuiteSparse:GraphBLAS, Timothy A. Davis, (c) 2017-2026, All Rights Reserved.
// SPDX-License-Identifier: Apache-2.0

//------------------------------------------------------------------------------

#include "GB_cuda.hpp"

bool GB_cuda_mxm_branch 
(
    const GrB_Matrix C,             // output matrix
    const GrB_Matrix M,             // mask matrix
    const GrB_Matrix A,             // input matrix
    const GrB_Matrix B,             // input matrix
    const GrB_Semiring semiring     // semiring that defines C=A*B
)
{
//  printf ("HACK: turn off cuda mxm\n") ; return (false) ;

    if (GB_ngpus_to_use (1) == 0) return (false) ;

    int data_arena = C->data_arena ;
    int header_arena = GB_arena (C->header_mem) ;
    int dev = data_arena - GxB_NARENAS ;

    bool use_cuda =
           (dev >= 0 && dev <= GB_Global_gpu_count_get ( )) // data on GPU
        && (data_arena == header_arena)                 // header on same GPU
        && (data_arena == A->data_arena)                // A on same GPU
        && (data_arena == GB_arena (A->header_mem))
        && (data_arena == B->data_arena)                // B on same GPU
        && (data_arena == GB_arena (B->header_mem))
        && (GB_jitifyer_get_control ( ) >= GxB_JIT_RUN) // JIT is running
        && (semiring->hash != UINT64_MAX)               // semiring is jitable
        && GB_cuda_type_branch (A->type)                // types OK for CUDA
        && GB_cuda_type_branch (B->type)
        && GB_cuda_type_branch (semiring->multiply->xtype)
        && GB_cuda_type_branch (semiring->multiply->ytype)
        && GB_cuda_type_branch (semiring->multiply->ztype)
        && GB_shallow_arenas_ok (A)         // shallow data on same GPU
        && GB_shallow_arenas_ok (B) ;

    if (M != NULL)
    {
        use_cuda = use_cuda
            && (data_arena == M->data_arena)            // M on same GPU
            && (data_arena == GB_arena (M->header_mem))
            && GB_shallow_arenas_ok (M) ;   // shallow data on same GPU
    }

    return (use_cuda) ;
}

