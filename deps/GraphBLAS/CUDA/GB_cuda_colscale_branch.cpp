#include "GB_cuda.hpp"

bool GB_cuda_colscale_branch
(
    const GrB_Matrix A,
    const GrB_Matrix D,
    const GrB_Semiring semiring,
    const bool flipxy
)
{
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
    return true;
}
