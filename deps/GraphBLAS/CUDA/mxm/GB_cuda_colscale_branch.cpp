#include "GB_cuda.hpp"

bool GB_cuda_colscale_branch
(
    const GrB_Matrix A,
    const GrB_Matrix D,
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

    if (A->header_size == 0)
    {
        return false ;
    }
    if (D->header_size == 0)
    {
        return false ;
    }
    
    if (!GB_cuda_type_branch (A->type) || 
        !GB_cuda_type_branch (D->type) ||
        !GB_cuda_type_branch (semiring->multiply->ztype))
    {
        return false;
    }

    double work = GB_nnz_held (A) ;
    int gpu_count = GB_ngpus_to_use (work) ;

    return (gpu_count > 0);
}
