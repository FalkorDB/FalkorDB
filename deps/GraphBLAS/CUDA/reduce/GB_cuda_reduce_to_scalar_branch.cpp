//------------------------------------------------------------------------------
// GraphBLAS/CUDA/GB_cuda_reduce_to_scalar_branch: decide to use GPU for reduce
//------------------------------------------------------------------------------

// SuiteSparse:GraphBLAS, Timothy A. Davis, (c) 2017-2025, All Rights Reserved.
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

    int jit_control = GB_jitifyer_get_control ( ) ;
    if (jit_control <= GxB_JIT_PAUSE)
    { 
        // JIT is off or paused
        return (false) ;
    }

    if (monoid->hash == UINT64_MAX)
    {
        return false ;
    }

    if (!GB_cuda_type_branch (A->type) ||
        !GB_cuda_type_branch (monoid->op->ztype))
    {
        // one or more types are not yet supported on the GPU
        return (false) ;
    }

    if (monoid->op->opcode == GB_ANY_binop_code)
    {
        // the ANY monoid takes O(1) time; do it on the CPU:
        return (false) ;
    }

    if (A->iso)
    {
        // A iso takes O(log(nvals(A))) time; do it on the CPU:
        return (false) ;
    }

    // see if there is enough work to do on the GPU
    double work = GB_nnz_held (A) ;
    int ngpus_to_use = GB_ngpus_to_use (work) ;
    GBURBLE (" work:%g gpus:%d ", work, ngpus_to_use) ;
    if (ngpus_to_use > 0)
    {
        return (true) ;
    }
    else
    {
        return (false) ;
    }
}

