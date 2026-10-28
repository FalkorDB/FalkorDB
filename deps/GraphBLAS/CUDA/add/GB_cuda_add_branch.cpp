//------------------------------------------------------------------------------
// GraphBLAS/CUDA/add/GB_cuda_add_branch: decide to use GPU for GB_add
//------------------------------------------------------------------------------

// SuiteSparse:GraphBLAS, Timothy A. Davis, (c) 2017-2026, All Rights Reserved.
// SPDX-License-Identifier: Apache-2.0

//------------------------------------------------------------------------------

#include "GB_cuda.hpp"

bool GB_cuda_add_branch 
(
    const GrB_Matrix C,     // output matrix, empty header with just data arena
    const GrB_Type ctype,   // type of C
    const GrB_Matrix M,     // input mask matrix
    const bool Mask_struct, // if true, mask is structural
    const bool Mask_comp,   // if true, mask is complemented
    const GrB_Matrix A,     // input matrix
    const GrB_Matrix B,     // input matrix
    const GrB_BinaryOp op   // op that defines C=A+B
)
{

    if (GB_ngpus_to_use (1) == 0) return (false) ;

    int data_arena = C->data_arena ;
    int header_arena = GB_arena (C->header_mem) ;
    int dev = data_arena - GxB_NARENAS ;

    // The CUDA kernel cannot yet apply a valued mask, and currently only
    // handles the case when all matrices are sparse/hypersparse.
    // FIXME: CUDA needs to handle all of these cases below

    int A_sparsity = GB_sparsity (A) ;
    int B_sparsity = GB_sparsity (B) ;

    bool use_cuda =
           (A_sparsity == GxB_SPARSE || A_sparsity == GxB_HYPERSPARSE) // FIXME
        && (B_sparsity == GxB_SPARSE || B_sparsity == GxB_HYPERSPARSE) // FIXME
        && (dev >= 0 && dev <= GB_Global_gpu_count_get ( )) // data on GPU
        && (data_arena == header_arena)                 // header on same GPU
        && (data_arena == A->data_arena)                // A on same GPU
        && (data_arena == GB_arena (A->header_mem))
        && (data_arena == B->data_arena)                // B on same GPU
        && (data_arena == GB_arena (B->header_mem))
        && (GB_jitifyer_get_control ( ) >= GxB_JIT_RUN) // JIT is running
        && (op->hash != UINT64_MAX)                     // op is jitable
        && GB_cuda_type_branch (A->type)                // types OK for CUDA
        && GB_cuda_type_branch (B->type)
        && GB_cuda_type_branch (ctype)
        && GB_cuda_type_branch (op->xtype)
        && GB_cuda_type_branch (op->ytype)
        && GB_cuda_type_branch (op->ztype)
        && GB_shallow_arenas_ok (A)         // shallow data on same GPU
        && GB_shallow_arenas_ok (B) ;

    if (M != NULL)
    {
        int M_sparsity = GB_sparsity (M) ;
        use_cuda = use_cuda
        && (Mask_struct)    // FIXME, also handle a valued mask
        && (M_sparsity == GxB_SPARSE || M_sparsity == GxB_HYPERSPARSE) // FIXME
        && (data_arena == M->data_arena)                // M on same GPU
        && (data_arena == GB_arena (M->header_mem))
        && (Mask_struct || GB_cuda_type_branch (M->type))
        && GB_shallow_arenas_ok (M) ;
    }

    return (use_cuda) ;
}

