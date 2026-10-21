//------------------------------------------------------------------------------
// GB_cuda_apply_unop: apply a unary op
//------------------------------------------------------------------------------

// SuiteSparse:GraphBLAS, Timothy A. Davis, (c) 2017-2026, All Rights Reserved.
// SPDX-License-Identifier: Apache-2.0

//------------------------------------------------------------------------------

#include "apply/GB_cuda_apply.hpp"

#undef  GB_FREE_WORKSPACE
#define GB_FREE_WORKSPACE                                   \
{                                                           \
    GB_FREE_MEMORY (&ythunk_cuda, ythunk_cuda_mem) ;        \
}

#undef  GB_FREE_ALL
#define GB_FREE_ALL                                         \
{                                                           \
    GB_FREE_WORKSPACE                                       \
    GB_cuda_stream_pool_release (&stream) ;                 \
}

GrB_Info GB_cuda_apply_unop
(
    GB_void *Cx,
    const int Cx_arena,
    const GrB_Type ctype,
    const GB_Operator op,
    const bool flipij,
    const GrB_Matrix A,
    const GB_void *ythunk
)
{

    GrB_Info info ;
    GB_void *ythunk_cuda = nullptr ;
    int device = Cx_arena - GxB_NARENAS ;
    uint64_t ythunk_cuda_mem = GB_mem (Cx_arena, 0) ;
    cudaStream_t stream = nullptr ;

    GB_OK (GB_cuda_stream_pool_acquire (device, &stream)) ;
    GB_OK (GB_wait_arenas (A)) ;

    GrB_Index anz = GB_nnz_held (A) ;
    if (anz == 0) return (GrB_SUCCESS) ;

    // fixme: make this a CUDA helper function
    if (ythunk != NULL && op != NULL && op->ytype != NULL)
    {
        // make a copy of ythunk, since ythunk might be allocated on
        // the CPU stack and thus not accessible to the CUDA kernel.
        ythunk_cuda = (GB_void *) GB_MALLOC_MEMORY (1, op->ytype->size,
            &ythunk_cuda_mem) ;
        if (ythunk_cuda == NULL)
        {
            GB_FREE_ALL ;
            return (GrB_OUT_OF_MEMORY) ;
        }
        memcpy (ythunk_cuda, ythunk, op->ytype->size) ;
    }

    int32_t number_of_sms = GB_Global_gpu_sm_get (device) ;
    int64_t raw_gridsz = GB_ICEIL (anz, GB_CUDA_APPLY_CHUNKSIZE) ;
    // cap #of blocks to 256 * #of sms
    int32_t gridsz = std::min (raw_gridsz, (int64_t) (number_of_sms * 256)) ;

    GBURBLE ("(cuda apply unop, device %d, sms: %d, gridsz: %d) ",
        device, number_of_sms, gridsz) ;

    GB_OK (GB_cuda_apply_unop_jit (Cx, ctype, op, flipij, A,
        ythunk_cuda, device, stream, gridsz)) ;

    GB_FREE_WORKSPACE ;
    GB_OK (GB_cuda_stream_pool_release (&stream)) ;
    return GrB_SUCCESS ;
}

