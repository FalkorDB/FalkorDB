//------------------------------------------------------------------------------
// CUDA/device/GB_cuda_stream_pool.cu:  stream pools for each CUDA device
//------------------------------------------------------------------------------

// SuiteSparse:GraphBLAS, Timothy A. Davis, (c) 2017-2026, All Rights Reserved.
// SPDX-License-Identifier: Apache-2.0

//------------------------------------------------------------------------------

#include "GB_cuda.hpp"

#define STREAMS_PER_DEVICE 32

// fixme: avoid std::, no need for it here; there are at most GxB_NARENAS_GPU
// gpus.
struct GB_cuda_stream_pool
{
    std::vector<std::array<cudaStream_t, STREAMS_PER_DEVICE>> streams ;
    std::vector<int> nstreams_avail;
} ;

static GB_cuda_stream_pool pool ;   // a global variable limited to this file

#undef  GB_FREE_ALL
#define GB_FREE_ALL ;

//------------------------------------------------------------------------------
// GB_cuda_stream_pool_release
//------------------------------------------------------------------------------

GrB_Info GB_cuda_stream_pool_release (cudaStream_t *stream)
{

    //--------------------------------------------------------------------------
    // check inputs, get current device, and sync the stream before releasing it
    //--------------------------------------------------------------------------

    if (stream == nullptr || (*stream) == nullptr)
    {
        // nothing to do
        return (GrB_SUCCESS) ;
    }

    int device = 0 ;
    CUDA_OK (cudaStreamGetDevice (*stream, &device)) ;
    CUDA_OK (cudaSetDevice (device)) ;

    // Fixme: do not sync here: Instead, assert stream already sync'd
    // with cudaStreamQuery (only when debugging)
    CUDA_OK (cudaStreamSynchronize (*stream)) ;

    cudaError_t cuda_error1 = cudaSuccess ;

    //--------------------------------------------------------------------------
    // release the stream inside a process-wide critical section
    //--------------------------------------------------------------------------

    // fixme: use one OpenMP lock per GPU

    GB_OPENMP_LOCK_SET (4)      // CUDA stream pool
    {
        if (pool.nstreams_avail[device] == STREAMS_PER_DEVICE)
        {
            // Pool is full; destroy the stream
            cuda_error1 = cudaStreamDestroy (*stream) ;
        }
        else
        {
            // Check the stream back in; it's OK if the stream wasn't
            // created at init time. Whatever stream was will be destroyed
            // when it is checked back in.
            size_t stream_idx = pool.nstreams_avail[device];
            pool.streams[device][stream_idx] = (*stream) ;
            pool.nstreams_avail[device]++ ;
        }
    }
    GB_OPENMP_LOCK_UNSET (4)    // CUDA stream pool

    //--------------------------------------------------------------------------
    // handle any error and return results
    //--------------------------------------------------------------------------

    CUDA_OK (cuda_error1) ;
    (*stream) = nullptr ;
    return (GrB_SUCCESS) ;
}

//------------------------------------------------------------------------------
// GB_cuda_stream_pool_acquire
//------------------------------------------------------------------------------

GrB_Info GB_cuda_stream_pool_acquire (int device, cudaStream_t *stream)
{

    //--------------------------------------------------------------------------
    // check inputs and set current device
    //--------------------------------------------------------------------------

    if (stream == nullptr)
    {
        return (GrB_NULL_POINTER) ;
    }

    (*stream) = nullptr ;

    if (device < 0 || device > GxB_NARENAS_GPU)
    { 
        return (GrB_INVALID_VALUE) ;
    }

    CUDA_OK (cudaSetDevice (device)) ;
    cudaError_t cuda_error1 = cudaSuccess ;

    //--------------------------------------------------------------------------
    // acquire the stream inside a process-wide critical section
    //--------------------------------------------------------------------------

    // fixme: use one OpenMP lock per GPU

    GB_OPENMP_LOCK_SET (4)      // CUDA stream pool
    {
        if (!pool.nstreams_avail[device])
        {
            // Pool is empty; create a stream
            cuda_error1 = cudaStreamCreate (stream) ;
        }
        else
        {
            // Checkout a stream
            size_t stream_idx = pool.nstreams_avail[device] - 1;
            (*stream) = pool.streams[device][stream_idx];
            pool.nstreams_avail[device]-- ;
        }
    }
    GB_OPENMP_LOCK_UNSET (4)    // CUDA stream pool

    //--------------------------------------------------------------------------
    // handle any error and return results
    //--------------------------------------------------------------------------

    CUDA_OK (cuda_error1) ;
    return (GrB_SUCCESS) ;
}

//------------------------------------------------------------------------------
// GB_cuda_stream_pool_init: initialize all streams on all devices
//------------------------------------------------------------------------------

GrB_Info GB_cuda_stream_pool_init (void)
{
    int ngpus = GB_Global_gpu_count_get ( ) ;
    for (int device = 0 ; device < ngpus ; device++)
    {
        printf ("CUDA stream pool, device: %d, # streams %d\n",
            device, STREAMS_PER_DEVICE) ;
        pool.nstreams_avail.push_back (0) ;
        pool.streams.push_back (std::array<cudaStream_t, STREAMS_PER_DEVICE>()) ;
        CUDA_OK (cudaSetDevice (device)) ;
        for (int k = 0 ; k < STREAMS_PER_DEVICE ; k++)
        {
            cudaStream_t tmp ;
            CUDA_OK (cudaStreamCreate (&tmp)) ;
            pool.streams[device][k] = tmp ;
            pool.nstreams_avail[device]++ ;
        }
    }

    return GrB_SUCCESS ;
}

//------------------------------------------------------------------------------
// GB_cuda_stream_pool_finalize: destroy all streams on all devices
//------------------------------------------------------------------------------

GrB_Info GB_cuda_stream_pool_finalize (void)
{
    // destroy all streams
    while (pool.streams.size())
    {
        auto curr = pool.streams.back() ;
        int end = pool.nstreams_avail.back() - 1 ;
        for (int k = end; k >= 0 ; k--)
        {
            // CUDA_OK (cudaSetDevice (k)) ;
            CUDA_OK (cudaStreamDestroy (curr[k])) ;
        }
        pool.streams.pop_back() ;
        pool.nstreams_avail.pop_back() ;
    }
    return GrB_SUCCESS ;
}

