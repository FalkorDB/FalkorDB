//------------------------------------------------------------------------------
// GB_cuda_transpose_branch: determine if the GPU can transpose the matrix
//------------------------------------------------------------------------------

// SuiteSparse:GraphBLAS, Timothy A. Davis, (c) 2017-2026, All Rights Reserved.
// SPDX-License-Identifier: Apache-2.0

//------------------------------------------------------------------------------

#include "GB_cuda.hpp"

bool GB_cuda_transpose_branch
(
    const int C_arena,              // arena of header and data of C
    const GrB_Type ctype,
    const GrB_Matrix A,
    const GB_Operator op,           // any type of operator
    const GrB_Scalar scalar
)
{
//  printf ("HACK: turn off cuda transpose\n") ; return (false) ;

    if (GB_ngpus_to_use (1) == 0) return (false) ;

    int dev = C_arena - GxB_NARENAS ;

    bool use_cuda =
           (dev >= 0 && dev <= GB_Global_gpu_count_get ( )) // data on GPU
        && (C_arena == A->data_arena)                   // A on same GPU
        && (C_arena == GB_arena (A->header_mem))
        && (GB_jitifyer_get_control ( ) >= GxB_JIT_RUN) // JIT is running
        && GB_cuda_type_branch (ctype)                  // types OK for CUDA
        && GB_cuda_type_branch (A->type)
        && GB_shallow_arenas_ok (A) ;       // shallow data on same GPU

    if (op != NULL)
    {
        use_cuda = use_cuda
            && (op->hash != UINT64_MAX)         // op is jitable
            && GB_cuda_type_branch (op->xtype)
            && GB_cuda_type_branch (op->ytype)
            && GB_cuda_type_branch (op->ztype)  ;
    }

    if (scalar != NULL)
    {
        use_cuda = use_cuda && GB_cuda_type_branch (scalar->type) ;
    }

    return (use_cuda) ;
}

