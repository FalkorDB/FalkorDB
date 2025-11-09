#include "GB_cuda.hpp"

bool GB_cuda_rowscale_branch
(
    const GrB_Matrix D,
    const GrB_Matrix B,
    const GrB_Semiring semiring,
    const bool flipxy
)
{
    if (D->header_size == 0)
    {
        return false ;
    }
    if (B->header_size == 0)
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
