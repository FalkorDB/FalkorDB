//------------------------------------------------------------------------------
// GraphBLAS/CUDA/select/GB_cuda_select_branch
//------------------------------------------------------------------------------

// SuiteSparse:GraphBLAS, Timothy A. Davis, (c) 2017-2025, All Rights Reserved.
// SPDX-License-Identifier: Apache-2.0

//------------------------------------------------------------------------------

#include "GB_cuda.hpp"

bool GB_cuda_select_branch
(
    const GrB_Matrix A,
    const GrB_IndexUnaryOp op
)
{

    int jit_control = GB_jitifyer_get_control ( ) ;
    if (jit_control <= GxB_JIT_PAUSE)
    { 
        // JIT is off or paused
        return (false) ;
    }

    ASSERT (A != NULL && op != NULL) ;

    if (op->hash == UINT64_MAX)
    {
        return false ;
    }

    if (A->header_mem == 0)
    {
        // FIXME memlane: check all of A
        return (false) ;
    }

    bool ok = (GB_cuda_type_branch (A->type)) ;

    if (op->xtype != NULL)
    {
        ok = ok && (GB_cuda_type_branch (op->xtype)) ;
    }
    if (op->ytype != NULL)
    {
        ok = ok && (GB_cuda_type_branch (op->ytype)) ;
    }
    if (op->ztype != NULL)
    {
        ok = ok && (GB_cuda_type_branch (op->ztype)) ;
    }

    double work = GB_nnz_held (A) ;
    int gpu_count = GB_ngpus_to_use (work) ;
    int ngpus_max = GB_Context_gpu_ids (NULL) ;     // FIXME: get gpu_ids
    gpu_count = std::min (gpu_count, ngpus_max) ;
    ok = ok && (gpu_count > 0);
    return ok ;
}

