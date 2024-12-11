#include "GB_cuda.hpp"

bool GB_cuda_rowscale_branch
(
    const GrB_Matrix D,
    const GrB_Matrix B,
    const GrB_Semiring semiring,
    const bool flipxy
)
{
    if (D->static_header)
    {
        return false ;
    }
    if (B->static_header)
    {
        return false ;
    }

    if (!GB_cuda_type_branch (D->type) || 
        !GB_cuda_type_branch (B->type) ||
        !GB_cuda_type_branch (semiring->multiply->ztype))
    {
        return false;
    }
    return true;
}
