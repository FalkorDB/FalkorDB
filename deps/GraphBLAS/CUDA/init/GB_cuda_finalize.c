//------------------------------------------------------------------------------
// GraphBLAS/CUDA/GB_cuda_finalize: finalize GPUs used by GraphBLAS
//------------------------------------------------------------------------------

// SuiteSparse:GraphBLAS, Timothy A. Davis, (c) 2017-2026, All Rights Reserved.
// SPDX-License-Identifier: Apache-2.0

//------------------------------------------------------------------------------

#include "GB.h"

GrB_Info GB_cuda_finalize (void)
{
    GB_cuda_stream_pool_finalize ( ) ;
    rmm_wrap_finalize ( ) ;
    return GrB_SUCCESS ;
}

