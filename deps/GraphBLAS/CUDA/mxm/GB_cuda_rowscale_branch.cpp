#include "GB_cuda.hpp"

bool GB_cuda_rowscale_branch
(
    const GrB_Matrix D,
    const GrB_Matrix B,
    const GrB_Semiring semiring,
    const bool flipxy
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

    if (D->header_mem == 0 || B->header_mem == 0)
    {
        // FIXME memlane: check all of D, B
        return (false) ;
    }

    if (!GB_cuda_type_branch (D->type) ||
        !GB_cuda_type_branch (B->type) ||
        !GB_cuda_type_branch (semiring->multiply->ztype))
    {
        return false;
    }

    double work = GB_nnz_held (B) ;
    int gpu_count = GB_ngpus_to_use (work) ;
    int ngpus_max = GB_Context_gpu_ids (NULL) ;     // FIXME: get gpu_ids
    gpu_count = std::min (gpu_count, ngpus_max) ;
    return (gpu_count > 0) ;
}

