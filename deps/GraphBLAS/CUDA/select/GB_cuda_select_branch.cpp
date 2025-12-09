#include "GB_cuda.hpp"

bool GB_cuda_select_branch
(
    const GrB_Matrix A,
    const GrB_IndexUnaryOp op
)
{

    int jit_control = GB_jitifyer_get_control ( ) ;
    if (jit_control <= GxB_JIT_PAUSE)
    { 
        // JIT is off or paused
        return (false) ;
    }

    ASSERT (A != NULL && op != NULL) ;

    if (op->hash == UINT64_MAX)
    {
        return false ;
    }

    if (A->header_size == 0)
    {
        // see Source/matrix/GB_clear_matrix_header.h for details.  If A has a
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

    double work = GB_nnz_held (A) ;
    int gpu_count = GB_ngpus_to_use (work) ;
    ok = ok && (gpu_count > 0);
    return ok ;
}

