//------------------------------------------------------------------------------
// GraphBLAS/CUDA/template/GB_cuda_threadblock_reduce_ztype.cuh
//------------------------------------------------------------------------------

// SuiteSparse:GraphBLAS, Timothy A. Davis, (c) 2017-2025, All Rights Reserved.
// This file: Copyright (c) 2024-2025, NVIDIA CORPORATION. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//------------------------------------------------------------------------------

// Reduce across an entire threadblock a single scalar of type ztype,
// using the given monoid (used in GB_cuda_tile_reduce_ztype).

// Compare with template/GB_cuda_threadblock_sum_uint64.

__inline__ __device__ GB_Z_TYPE GB_cuda_threadblock_reduce_ztype
(
    GB_Z_TYPE val
)
{
    // The thread_block g that calls this method has a number of threads
    // defined by the kernel launch geometry (dim3 block (blocksz)).
    thread_block g = this_thread_block ( ) ;
    // here, g.sync() is not needed

    // The threads in this thread block are partitioned into tiles, each with
    // GB_CUDA_TILE_SIZE threads.
    thread_block_tile<GB_CUDA_TILE_SIZE> tile =
        tiled_partition<GB_CUDA_TILE_SIZE> (g) ;
    // here, tile.sync() is implicit

    // threadId_in_tile: a local thread id, for all threads in a single tile,
    // ranging from 0 to the size of the tile minus one.  Normally the tile has
    // size 32, but it could be a power of 2 less than or equal to 32.
    int threadId_in_tile = threadIdx.x & (GB_CUDA_TILE_SIZE-1) ;
    // tile_id: is the id for a single tile, each with GB_CUDA_TILE_SIZE
    // threads in it.
    int tile_id = threadIdx.x >> GB_CUDA_LOG2_TILE_SIZE ;

    // Each tile performs partial reduction
    val = GB_cuda_tile_reduce_ztype (tile, val) ;

    // shared result for partial sums of all threads in a tile:
    static __shared__ GB_Z_TYPE shared [GB_CUDA_TILE_SIZE] ;

    if (threadId_in_tile == 0)
    {
        shared [tile_id] = val ;    // Write reduced value to shared memory
    }

    // This g.sync() is required:
    g.sync() ;                      // Wait for all partial reductions

    // This method requires blockDim.x <= GB_CUDA_TILE_SIZE^2 = 1024, but this
    // is always enforced in the CUDA standard since the our geometry is 1D.

    // Final reduce within first tile
    if (tile_id == 0)
    {
        GB_DECLARE_IDENTITY_CONST (zid) ;   // const GB_Z_TYPE zid = identity ;
        // read from shared memory only if that tile existed
        val = (threadIdx.x < (blockDim.x >> GB_CUDA_LOG2_TILE_SIZE)) ?
            shared [threadId_in_tile] : zid ;
        val = GB_cuda_tile_reduce_ztype (tile, val) ;
    }

    // Not needed:
    // g.sync() ;
    return (val) ;
}

