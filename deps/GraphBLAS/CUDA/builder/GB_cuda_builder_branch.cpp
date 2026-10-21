//------------------------------------------------------------------------------
// GraphBLAS/CUDA/builder/GB_cuda_builder_branch
//------------------------------------------------------------------------------

// SuiteSparse:GraphBLAS, Timothy A. Davis, (c) 2017-2026, All Rights Reserved.
// SPDX-License-Identifier: Apache-2.0

//------------------------------------------------------------------------------

// Determine if the GPU should be used for GB_build

#include "GB_cuda.hpp"

bool GB_cuda_builder_branch
(
    const GrB_Matrix C,
    const GrB_BinaryOp dup,
    const GrB_Type xtype,
    const void *I,
    const void *J,
    const void *X,
    const uint64_t nvals
)
{
//  printf ("HACK: turn off cuda builder\n") ; return (false) ;

    if (GB_ngpus_to_use (1) == 0) return (false) ;

    int data_arena = C->data_arena ;
    int header_arena = GB_arena (C->header_mem) ;
    int dev = data_arena - GxB_NARENAS ;

    bool use_cuda =
           (dev >= 0 && dev <= GB_Global_gpu_count_get ( )) // data on GPU
        && (data_arena == header_arena)                 // header on same GPU
        && (GB_jitifyer_get_control ( ) >= GxB_JIT_RUN) // JIT is running
        && GB_cuda_type_branch (xtype)                  // types OK for CUDA
        && GB_cuda_type_branch (C->type)
        && GB_cuda_pointer_ok (I, "I")                  // I,J,X on the GPU
        && GB_cuda_pointer_ok (J, "J")
        && GB_cuda_pointer_ok (X, "X") ;

    if (dup != NULL && dup != GxB_IGNORE_DUP)
    {
        // check the dup operator
        use_cuda = use_cuda
            && dup->hash != UINT64_MAX
            && GB_cuda_type_branch (dup->xtype)
            && GB_cuda_type_branch (dup->ytype)
            && GB_cuda_type_branch (dup->ztype) ;
    }

    return (use_cuda) ;
}

