#include "GB_cuda.hpp"

bool GB_cuda_apply_unop_branch
(
    const GrB_Type ctype,
    const GrB_Matrix A,
    const GB_Operator op
)
{

    int jit_control = GB_jitifyer_get_control ( ) ;
    if (jit_control <= GxB_JIT_PAUSE)
    { 
        // JIT is off or paused
        return (false) ;
    }

    if (op == NULL || op->hash == UINT64_MAX)
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
    
    double work = GB_nnz_held (A) ;
    int gpu_count = GB_ngpus_to_use (work) ;
    ok = ok && (gpu_count > 0);
    return ok ;
}
