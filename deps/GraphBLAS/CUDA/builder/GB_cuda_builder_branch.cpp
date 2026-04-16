//------------------------------------------------------------------------------
// GraphBLAS/CUDA/builder/GB_cuda_builder_branch
//------------------------------------------------------------------------------

// SuiteSparse:GraphBLAS, Timothy A. Davis, (c) 2017-2025, All Rights Reserved.

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

    int jit_control = GB_jitifyer_get_control ( ) ;
    if (jit_control <= GxB_JIT_PAUSE)
    { 
        // JIT is off or paused
        return (false) ;
    }

    ASSERT (C != NULL && xtype != NULL) ;

    if (dup != NULL && dup != GxB_IGNORE_DUP && dup->hash == UINT64_MAX)
    {
        return (false) ;
    }

    if (!GB_cuda_pointer_ok (I, "I") ||
        !GB_cuda_pointer_ok (J, "J") ||
        !GB_cuda_pointer_ok (X, "X"))
    {
        // FIXME: better yet, memcpy to GPU-accessible memory
        printf ("%s: (I,J,X) pointers not accessible by the GPU\n", __FILE__) ;
        return (false) ;
    }

    if (C->header_mem == 0)
    {
        // FIXME memlane: check all of C
        return (false) ;
    }

    bool ok = (GB_cuda_type_branch (C->type) && GB_cuda_type_branch (xtype)) ;

    if (dup != NULL && dup != GxB_IGNORE_DUP)
    {
        if (dup->xtype != NULL)
        {
            ok = ok && (GB_cuda_type_branch (dup->xtype)) ;
        }
        if (dup->ytype != NULL)
        {
            ok = ok && (GB_cuda_type_branch (dup->ytype)) ;
        }
        if (dup->ztype != NULL)
        {
            ok = ok && (GB_cuda_type_branch (dup->ztype)) ;
        }
    }

    double work = nvals ;
    int gpu_count = GB_ngpus_to_use (work) ;
    int ngpus_max = GB_Context_gpu_ids (NULL) ;     // FIXME: get gpu_ids
    gpu_count = std::min (gpu_count, ngpus_max) ;
    ok = ok && (gpu_count > 0);
    return (ok) ;
}

