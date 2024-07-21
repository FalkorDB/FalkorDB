#include "GraphBLAS_cuda.hpp"
#include "GB_cuda.hpp"

bool GB_cuda_apply_binop_branch
(
    const GrB_Type ctype,
    const GrB_BinaryOp op,
    const GrB_Matrix A
)
{
    bool ok = GB_cuda_type_branch (ctype) && GB_cuda_type_branch (A->type) ;

    if (op->xtype != NULL)
    {
        ok = ok && GB_cuda_type_branch (op->xtype) ;
    }
    if (op->ytype != NULL)
    {
        ok = ok && GB_cuda_type_branch (op->ytype) ;
    }
    if (op->ztype != NULL)
    {
        ok = ok && GB_cuda_type_branch (op->ztype) ;
    }

    ok = ok && (op != NULL && op->hash != UINT64_MAX) ; 

    return (ok) ;
}

