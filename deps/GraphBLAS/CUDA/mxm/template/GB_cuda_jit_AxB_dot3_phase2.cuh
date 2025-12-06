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
    int64_t *Blockbucket,
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

    for (int block_id = 0 ; block_id <= nblocks ; block_id += threads_per_block)
    {
        int64_t data = 0 ;

        // Load a segment of consecutive items that are blocked across threads

        int loc = block_id + threadIdx.x;
        if (loc <= nblocks)
        {
            data = Blockbucket [bucketId*(nblocks+1) + loc] ;
        }
        this_thread_block().sync() ;

        // Collectively compute the block-wide exclusive prefix sum
        BlockScan(temp_storage).ExclusiveSum (data, data, prefix_op) ;
        this_thread_block().sync() ;

        if (loc <= nblocks)
        {
            Blockbucket [bucketId*(nblocks+1) + loc] = data ;
        }
    }
}

//------------------------------------------------------------------------------
// GB_cuda_AxB_dot3_phase2_kernel
//------------------------------------------------------------------------------

// GB_cuda_AxB__dot3_phase2 is a CUDA kernel that takes as input the
// nanobuckets and Blockbucket arrays computed by the first phase kernel,
// GB_cuda_AxB__dot3_phase1.

__global__ void GB_cuda_AxB_dot3_phase2_kernel
(
    // input, not modified:
    int64_t *__restrict__ Blockbucket,  // global bucket count,
                                        // of size NBUCKETS*(nblocks+1)
    // inputs, not modified:
    const int nblocks               // input number of blocks to reduce
                                    // across, ie size of vector for 1 bucket
)
{
    blockBucketExclusiveSum (blockIdx.x, Blockbucket, nblocks) ;
}

