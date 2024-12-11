#include "GB_cuda.hpp"

bool GB_cuda_select_branch
(
    const GrB_Matrix A,
    const GrB_IndexUnaryOp op
)
{
    
    ASSERT (A != NULL && op != NULL) ;

    if (A->static_header)
    {
        // see Source/matrix/GB_static_header.h for details.  If A has a
        // static header, it cannot be done on the GPU.  However, if GraphBLAS
        // is compiled to use CUDA, there should be no static headers anyway,
        // so this is likely dead code.  Just a sanity check.
        return false ;
    }

    bool ok = (GB_cuda_type_branch (A->type)) ;

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

