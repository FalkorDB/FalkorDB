#include "GB_cuda.hpp"

bool GB_cuda_colscale_branch
(
    const GrB_Matrix A,
    const GrB_Matrix D,
    const GrB_Semiring semiring,
    const bool flipxy
)
{
    if (A->static_header)
    {
        return false ;
    }
    if (D->static_header)
    {
        return false ;
    }
    
    if (!GB_cuda_type_branch (A->type) || 
        !GB_cuda_type_branch (D->type) ||
        !GB_cuda_type_branch (semiring->multiply->ztype))
    {
        return false;
    }
    return true;
}
