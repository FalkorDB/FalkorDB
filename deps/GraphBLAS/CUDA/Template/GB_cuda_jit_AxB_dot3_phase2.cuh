//------------------------------------------------------------------------------
// GraphBLAS/CUDA/template/GB_cuda_jit_GB_AxB_dot3_phase2.cuh
//------------------------------------------------------------------------------

// SuiteSparse:GraphBLAS, Timothy A. Davis, (c) 2017-2025, All Rights Reserved.
// This file: Copyright (c) 2024-2025, NVIDIA CORPORATION. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//------------------------------------------------------------------------------

// AxB_dot3_phase2: fill the global buckets

//------------------------------------------------------------------------------
// BlockPrefixCallbackOp
//------------------------------------------------------------------------------

// A stateful callback functor that maintains a running prefix to be applied
// during consecutive scan operations.
struct BlockPrefixCallbackOp
{
    // Running prefix
    int64_t running_total ;

    // Constructor
    __device__ BlockPrefixCallbackOp (int64_t running_total) :
        running_total(running_total) {}

    // Callback operator to be entered by the first warp of threads in the
    // block.  Thread-0 is responsible for returning a value for seeding the
    // block-wide scan.
    __device__ int64_t operator()(int64_t block_aggregate)
    {
        int64_t old_prefix = running_total ;
        running_total += block_aggregate ;
        return old_prefix ;
    }
} ;

//------------------------------------------------------------------------------
// blockBucketExclusiveSum
//------------------------------------------------------------------------------

__inline__ __device__ void blockBucketExclusiveSum
(
    int bucketId,
    int64_t *d_data,
    int nblocks
)
{

    // Specialize BlockScan for a 1D block of 32 threads
    typedef cub::BlockScan<int64_t, 32, cub::BLOCK_SCAN_WARP_SCANS> BlockScan ;

    // Allocate shared memory for BlockScan
    __shared__ typename BlockScan::TempStorage temp_storage ;

    // Initialize running total
    BlockPrefixCallbackOp prefix_op (0) ;

    // Have the block iterate over segments of items
    int64_t data = 0 ;

    int64_t *blockbucket = d_data ;

    for (int block_id = 0 ; block_id < nblocks ; block_id += blocksize)
    {
        // Load a segment of consecutive items that are blocked across threads

        int loc = block_id + threadIdx.x;
        if (loc < nblocks)
        {
            data = blockbucket [bucketId*nblocks + loc] ;
        }
        this_thread_block().sync() ;

        // Collectively compute the block-wide exclusive prefix sum
        BlockScan(temp_storage).ExclusiveSum (data, data, prefix_op) ;
        this_thread_block().sync() ;

        if (loc < nblocks)
        {
            blockbucket [bucketId*nblocks + loc] = data ;
        }

        // this_thread_block().sync();

        data = 0 ;
    }
}

//------------------------------------------------------------------------------
// GB_cuda_AxB_dot3_phase2_kernel
//------------------------------------------------------------------------------

// GB_cuda_AxB__dot3_phase2 is a CUDA kernel that takes as input the
// nanobuckets and blockbucket arrays computed by the first phase kernel,
// GB_cuda_AxB__dot3_phase1.  The launch geometry of this kernel must match
// the GB_cuda_AxB_dot3_phase1 kernel, with the same # of threads and
// threadblocks.

__global__ void GB_cuda_AxB_dot3_phase2_kernel
(
    // input, not modified:
    int64_t *__restrict__ blockbucket,  // global bucket count,
                                        // of size NBUCKETS*nblocks
    // output:
    int64_t *__restrict__ offset,       // global offsets, for each bucket
    // inputs, not modified:
    const int nblocks               // input number of blocks to reduce
                                    // across, ie size of vector for 1 bucket
)
{

    //--------------------------------------------------------------------------
    // sum up the bucket counts of prior threadblocks
    //--------------------------------------------------------------------------

    // blockbucket is an array of size NBUCKETS-by-nblocks, held by row.  The
    // entry blockbucket [bucket * nblocks + t] holds the # of entries
    // in the bucket (in range 0 to NBUCKETS-1) found by threadblock t.

    uint64_t s [NBUCKETS] ;

    #pragma unroll
    for (int b = 0 ; b < NBUCKETS ; b++)
    {
        s [b] = 0 ;
    }

    thread_block_tile<32> tile = tiled_partition<32>(this_thread_block() );

     #pragma unroll
     for (int b = 0 ; b < NBUCKETS ; b++)
     {
        for (int64_t tid = threadIdx.x + blockIdx.x * blockDim.x ;
              tid < nblocks ;
              tid += blockDim.x*gridDim.x)
        {
            s [b] += blockbucket [b * nblocks + tid] ;
        }
        this_thread_block().sync(); 

        s [b] = GB_cuda_tile_sum_uint64 (tile, s [b]) ;
     }

    if (threadIdx.x == 0)
    {
        #pragma unroll
        for (int b = 0 ; b < NBUCKETS ; b++)
        {
            atomicAdd ((unsigned long long int*) &(offset [b]), s [b]) ;
        }
    }
    this_thread_block().sync(); 

    if (gridDim.x >= NBUCKETS)
    {
        // Cumulative sum across blocks for each bucket
        if (blockIdx.x <NBUCKETS)
        {
            blockBucketExclusiveSum (blockIdx.x, blockbucket, nblocks) ;
        }
    }
    else
    {
        if (blockIdx.x == 0)
        {
            #pragma unroll
            for (int b = 0 ; b < NBUCKETS ; b++)
            {
                blockBucketExclusiveSum (b, blockbucket, nblocks) ;
            }
        }
    }
}

