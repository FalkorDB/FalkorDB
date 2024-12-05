//------------------------------------------------------------------------------
// GraphBLAS/CUDA/GB_cuda_reduce_to_scalar: reduce on the GPU with semiring 
//------------------------------------------------------------------------------

// SuiteSparse:GraphBLAS, Timothy A. Davis, (c) 2017-2024, All Rights Reserved.
// This file: Copyright (c) 2024, NVIDIA CORPORATION. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//------------------------------------------------------------------------------

// Reduce a matrix A to a scalar s, or to a smaller matrix V if the GPU was
// only able to do a partial reduction.  This case occurs if the GPU does not
// cannot do an atomic update for the monoid.  To handle this case, the GPU
// returns a full GrB_Matrix V, of size gridsize-by-1, with one entry per
// threadblock.  Then GB_reduce_to_scalar on the CPU sees this V as the result,
// and calls itself recursively to continue the reduction.

#define GB_FREE_ALL                                         \
{                                                           \
    GB_FREE_WORK (&zscalar, zscalar_size) ;                 \
    GB_Matrix_free (&V) ;                                   \
    if (stream != nullptr) cudaStreamDestroy (stream) ;     \
    stream = nullptr ;                                      \
}

#include "GB_cuda_reduce.hpp"

GrB_Info GB_cuda_reduce_to_scalar
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

    //--------------------------------------------------------------------------
    // check inputs
    //--------------------------------------------------------------------------

    GB_void *zscalar = NULL ;
    size_t zscalar_size = 0 ;
    GrB_Matrix V = NULL ;
    (*V_handle) = NULL ;
    GrB_Info info = GrB_SUCCESS ;
    cudaStream_t stream = nullptr ;

    //--------------------------------------------------------------------------
    // create the stream
    //--------------------------------------------------------------------------

    // FIXME: use the stream pool
    CUDA_OK (cudaStreamCreate (&stream)) ;

    //--------------------------------------------------------------------------
    // determine problem characteristics and allocate worksbace
    //--------------------------------------------------------------------------

    int blocksz = 320 ;             // # threads in each block
    int work_per_thread = 256 ;     // work each thread does in a single block
    int number_of_sms = GB_Global_gpu_sm_get (0) ;

    GrB_Type ztype = monoid->op->ztype ;
    size_t zsize = ztype->size ;

    // determine kernel launch geometry
    int64_t anvals = GB_nnz_held (A) ;
    int64_t work_per_block = work_per_thread*blocksz ;
    // gridsz = ceil (anvals / work_per_block)
    int64_t raw_gridsz = GB_ICEIL (anvals, work_per_block) ;
    raw_gridsz = std::min (raw_gridsz, (int64_t) (number_of_sms * 256)) ;
    int gridsz = (int) raw_gridsz ;

    // FIXME: GB_enumify_reduce is called twice: here (to get has_cheeseburger)
    // and in GB_cuda_reduce_to_scalar_jit.  Can we just call it once?

    uint64_t rcode ;
    GB_enumify_reduce (&rcode, monoid, A) ;
    bool has_cheeseburger = GB_RSHIFT (rcode, 16, 1) ;
    GBURBLE ("has_cheeseburger %d\n", has_cheeseburger) ;

    // determine the kind of reduction: partial (to &V), or complete
    // (to the scalar output)
    if (has_cheeseburger)
    {
        // the kernel launch can reduce A to zscalar all by itself
        // allocate and initialize zscalar (upscaling it to at least 32 bits)
        size_t zscalar_space = GB_IMAX (zsize, sizeof (uint32_t)) ;
        zscalar = GB_MALLOC (zscalar_space, GB_void, &zscalar_size) ;
        if (zscalar == NULL)
        {
            // out of memory
            GB_FREE_ALL ;
            return (GrB_OUT_OF_MEMORY) ;
        }
        GB_cuda_upscale_identity (zscalar, monoid) ;
    }
    else
    {
        // allocate a full GrB_Matrix V for the partial result, of size
        // gridsz-by-1, and of type ztype.  V is allocated but not
        // initialized.
        GB_OK (GB_new_bix (&V, ztype, gridsz, 1, GB_Ap_null,
            true, GxB_FULL, false, 0, -1, gridsz, true, false)) ;
    }

    GBURBLE ("(cuda reduce launch %d threads in %d blocks)",
        blocksz, gridsz ) ;

    //--------------------------------------------------------------------------
    // reduce C to a scalar via the CUDA JIT
    //--------------------------------------------------------------------------

//  final call looks like this:
//  GB_OK (GB_cuda_reduce_to_scalar_jit (zscalar, V, monoid, A,
//      stream, gridsz, blocksz)) ;

//  debugging for now, to die early if the CUDA fails to compile, load, or run:
    info = (GB_cuda_reduce_to_scalar_jit (zscalar, V, monoid, A,
        stream, gridsz, blocksz)) ;
    if (info == GrB_NO_VALUE) info = GrB_PANIC ;
    GB_OK (info) ;

    //--------------------------------------------------------------------------
    // return result and destroy the stream
    //--------------------------------------------------------------------------

    CUDA_OK (cudaStreamSynchronize (stream)) ;

    if (has_cheeseburger)
    {
        // return the scalar result
        // s = zscalar (but only the first zsize bytes of it)
        memcpy (s, zscalar, zsize) ;
        GB_FREE_WORK (&zscalar, zscalar_size) ;
    }
    else
    {
        // return the partial reduction
        (*V_handle) = V ;
    }

    CUDA_OK (cudaStreamDestroy (stream)) ;
    return (GrB_SUCCESS) ;
}

