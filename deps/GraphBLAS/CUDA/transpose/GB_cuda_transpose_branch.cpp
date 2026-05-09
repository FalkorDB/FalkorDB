//------------------------------------------------------------------------------
// GB_cuda_transpose_branch: determine if the GPU can transpose the matrix
//------------------------------------------------------------------------------

// SuiteSparse:GraphBLAS, Timothy A. Davis, (c) 2017-2026, All Rights Reserved.
// SPDX-License-Identifier: Apache-2.0

//------------------------------------------------------------------------------

#include "GB_cuda.hpp"

bool GB_cuda_transpose_branch
(
    const GrB_Type ctype,
    const GrB_Matrix A,
    const GB_Operator op,           // any type of operator
    const GrB_Scalar scalar
)
{

    int jit_control = GB_jitifyer_get_control ( ) ;
    if (jit_control <= GxB_JIT_PAUSE)
    {
        // JIT is off or paused
        return (false) ;
    }

    if (A->header_mem == 0)
    {
        // FIXME memlane: check all of A
        return (false) ;
    }

    bool ok = GB_cuda_type_branch (ctype) && GB_cuda_type_branch (A->type) ;

    if (op != NULL)
    {
        if (op->hash == UINT64_MAX)
        {
            // op cannot be JIT'd
            return false ;
        }
        ok = ok && GB_cuda_type_branch (op->xtype)
                && GB_cuda_type_branch (op->ytype)
                && GB_cuda_type_branch (op->ztype)
                && GB_cuda_type_branch (op->theta_type) ;
    }

    if (scalar != NULL)
    {
        ok = ok && GB_cuda_type_branch (scalar->type) ;
    }

    double work = GB_nnz_held (A) ;
    int gpu_count = GB_ngpus_to_use (work) ;
    int ngpus_max = GB_Context_gpu_ids (NULL) ;     // FIXME: get gpu_ids
    gpu_count = std::min (gpu_count, ngpus_max) ;
    ok = ok && (gpu_count > 0);
    return (ok) ;
}

