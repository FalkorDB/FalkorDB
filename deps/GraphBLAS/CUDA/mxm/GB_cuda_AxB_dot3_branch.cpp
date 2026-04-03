//------------------------------------------------------------------------------
// GraphBLAS/CUDA/GB_cuda_AxB_dot3_branch: decide to use GPU for dot3
//------------------------------------------------------------------------------

// SuiteSparse:GraphBLAS, Timothy A. Davis, (c) 2017-2025, All Rights Reserved.
// SPDX-License-Identifier: Apache-2.0

//------------------------------------------------------------------------------

// Decide branch direction for GPU use for the dot-product C<M>=A'*B

#include "GB_cuda.hpp"

bool GB_cuda_AxB_dot3_branch 
(
    const GrB_Matrix M,             // mask matrix
    const bool Mask_struct,         // if true, use the only structure of M
    const GrB_Matrix A,             // input matrix
    const GrB_Matrix B,             // input matrix
    const GrB_Semiring semiring,    // semiring that defines C=A*B
    const bool flipxy               // if true, do z=fmult(b,a) vs fmult(a,b)
)
{

    int jit_control = GB_jitifyer_get_control ( ) ;
    if (jit_control <= GxB_JIT_PAUSE)
    { 
        // JIT is off or paused
        return (false) ;
    }

    if (semiring->hash == UINT64_MAX)
    {
        return false ;
    }

    if (!GB_cuda_type_branch (A->type) ||
        !GB_cuda_type_branch (B->type) ||
        !GB_cuda_type_branch (semiring->multiply->xtype) ||
        !GB_cuda_type_branch (semiring->multiply->ytype) ||
        !GB_cuda_type_branch (semiring->multiply->ztype))
    {
        // one or more types are not yet supported on the GPU
        return (false) ;
    }

    if (A->vlen == 0)
    {
        // C has no entries: no need to compute it on the GPU
        return (false) ;
    }

    // very rough estimate of the work to do
    double adeg = ((double) GB_nnz (A)) / ((double) GB_IMAX (1, A->nvec)) ;
    double bdeg = ((double) GB_nnz (B)) / ((double) GB_IMAX (1, B->nvec)) ;
    double work = GB_nnz (M) * GB_IMIN (adeg, bdeg) ;

    int ngpus_to_use = GB_ngpus_to_use (work) ;
    GBURBLE (" work:%g GPUs:%d ", work, ngpus_to_use) ;
    if (ngpus_to_use > 0)
    {
        // FIXME: determine which GPU from the context object
        return true ;
    }
    else
    {
        return false ;
    }
}

