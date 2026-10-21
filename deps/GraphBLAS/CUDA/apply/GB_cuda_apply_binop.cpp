//------------------------------------------------------------------------------
// CUDA/apply/GB_cuda_apply_binop: apply a binary op
//------------------------------------------------------------------------------

// SuiteSparse:GraphBLAS, Timothy A. Davis, (c) 2017-2026, All Rights Reserved.
// SPDX-License-Identifier: Apache-2.0

//------------------------------------------------------------------------------

#include "apply/GB_cuda_apply.hpp"

#undef  GB_FREE_WORKSPACE
#define GB_FREE_WORKSPACE                                   \
{                                                           \
    GB_FREE_MEMORY (&scalarx_cuda, scalarx_cuda_mem) ;      \
}

#undef  GB_FREE_ALL
#define GB_FREE_ALL                                         \
{                                                           \
    GB_FREE_WORKSPACE ;                                     \
    GB_cuda_stream_pool_release (&stream) ;                 \
}

GrB_Info GB_cuda_apply_binop
(
    GB_void *Cx,
    const int Cx_arena,
    const GrB_Type ctype,
    const GrB_BinaryOp op,
    const GrB_Matrix A,
    const GB_void *scalarx,
    const bool bind1st
)
{

    GrB_Info info ;
    GB_void *scalarx_cuda = NULL ;
    cudaStream_t stream = nullptr ;
    int device = Cx_arena - GxB_NARENAS ;
    uint64_t scalarx_cuda_mem = GB_mem (Cx_arena, 0) ;

    GB_OK (GB_cuda_stream_pool_acquire (device, &stream)) ;
    GB_OK (GB_wait_arenas (A)) ;

    ASSERT (scalarx != NULL) ;

    // make a copy of scalarx to ensure it's not on the CPU stack
    if (bind1st)
    {
        ASSERT (op->xtype != NULL) ;
        scalarx_cuda = (GB_void *) GB_MALLOC_MEMORY (1, op->xtype->size,
            &scalarx_cuda_mem) ;
    }
    else
    {
        ASSERT (op->ytype != NULL) ;
        scalarx_cuda = (GB_void *) GB_MALLOC_MEMORY (1, op->ytype->size,
            &scalarx_cuda_mem) ;
    }
    if (scalarx_cuda == NULL)
    {
        GB_FREE_ALL ;
        return (GrB_OUT_OF_MEMORY) ;
    }
    memcpy (scalarx_cuda, scalarx, GB_memsize (scalarx_cuda_mem)) ;

    GrB_Index anz = GB_nnz_held (A) ;

    int32_t number_of_sms = GB_Global_gpu_sm_get (device) ;
    int64_t raw_gridsz = GB_ICEIL (anz, GB_CUDA_APPLY_CHUNKSIZE) ;
    // cap #of blocks to 256 * #of sms
    int32_t gridsz = std::min (raw_gridsz, (int64_t) (number_of_sms * 256)) ;

    GBURBLE ("(cuda apply binop, device %d, sms: %d, gridsz: %d) ",
        device, number_of_sms, gridsz) ;

    if (bind1st)
    {
        GB_OK (GB_cuda_apply_bind1st_jit (Cx, ctype, op, A, 
            scalarx_cuda, device, stream, gridsz)) ;
    }
    else
    {
        GB_OK (GB_cuda_apply_bind2nd_jit (Cx, ctype, op, A,
            scalarx_cuda, device, stream, gridsz)) ;
    }

    GB_FREE_WORKSPACE ;
    GB_OK (GB_cuda_stream_pool_release (&stream)) ;
    return GrB_SUCCESS ; 
}

