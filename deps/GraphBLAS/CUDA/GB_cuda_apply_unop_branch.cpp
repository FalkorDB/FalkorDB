#include "GB_cuda.hpp"

bool GB_cuda_apply_unop_branch
(
    const GrB_Type ctype,
    const GrB_Matrix A,
    const GB_Operator op
)
{
    if (op == NULL)
    {
        return false ;
    }

    if (A->header_size == 0)
    {
        return false ;
    }
    
    bool ok = (GB_cuda_type_branch (ctype) && GB_cuda_type_branch (A->type)) ;

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
    
    ok = ok && (op->hash != UINT64_MAX) ;

    return ok ;
}
