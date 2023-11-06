//------------------------------------------------------------------------------
// GB_reduce_to_scalar_cuda.cpp: reduce on the GPU with semiring 
//------------------------------------------------------------------------------

// SuiteSparse:GraphBLAS, Timothy A. Davis, (c) 2017-2022, All Rights Reserved.
// SPDX-License-Identifier: Apache-2.0

//------------------------------------------------------------------------------

// Reduce a matrix A to a scalar s, or to a smaller matrix V if the GPU was
// only able to do a partial reduction.  This case occurs if the GPU does not
// cannot do an atomic update for the monoid.  To handle this case, the GPU
// returns a full GrB_Matrix V, of size gridsize-by-1, with one entry per
// threadblock.  Then GB_reduce_to_scalar on the CPU sees this V as the result,
// and calls itself recursively to continue the reduction.

#include "GraphBLAS.h"
extern "C"
{
    #include "GB_reduce.h"
}

#include "GB_cuda.h"
#include "GB_jit_cache.h"
#include "GB_cuda_common_jitFactory.hpp"
#include "GB_cuda_reduce_jitFactory.hpp"
#include "GB_cuda_type_wrap.hpp"

GrB_Info GB_reduce_to_scalar_cuda
(
    // output:
    GB_void *s,                 // note: statically allocated on CPU stack; if
                                // the result is in s then V is NULL.
    GrB_Matrix *V_handle,       // partial result if unable to reduce to scalar;
                                // NULL if result is in s.
    // input:
    const GrB_Monoid monoid,
    const GrB_Matrix A
)
{

    // FIXME: use the stream pool
    cudaStream_t stream ;
    CHECK_CUDA (cudaStreamCreate (&stream)) ;

    //--------------------------------------------------------------------------
    // reduce C to a scalar
    //--------------------------------------------------------------------------

    // FIXME: check error conditions (out of memory, etc)
    GB_cuda_reduce_factory myreducefactory ;
    myreducefactory.reduce_factory (monoid, A) ;

    // FIXME: get GrB_Info result from GB_cuda_reduce
    GB_cuda_reduce (myreducefactory, A, s, V_handle, monoid, stream) ;

    CHECK_CUDA (cudaStreamSynchronize (stream)) ;
    CHECK_CUDA (cudaStreamDestroy (stream)) ;

    return (GrB_SUCCESS) ;
}

