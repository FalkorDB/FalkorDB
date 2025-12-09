//------------------------------------------------------------------------------
// GraphBLAS/CUDA/template/GB_cuda_threadblock_sum_uint64.cuh
//------------------------------------------------------------------------------

// SuiteSparse:GraphBLAS, Timothy A. Davis, (c) 2017-2025, All Rights Reserved.
// This file: Copyright (c) 2024-2025, NVIDIA CORPORATION. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//------------------------------------------------------------------------------

// Sum across an entire threadblock a single uint64_t scalar.

// Compare with template/GB_cuda_threadblock_reduce_ztype.
// The #include'ing file must define tile_sz and log2_tile_sz.

// On input, there is no need for this_thread_block().sync(), because the
// first reduction is across a single tile.  The creation of the tile with
// tiled_partition<tile_sz>(g) ensures each tile is synchronized, which is
// sufficient for the following call to GB_cuda_tile_sum_uint64.

__inline__ __device__ uint64_t GB_cuda_threadblock_sum_uint64
(
    uint64_t val
)
{
    // The thread_block g that calls this method has a number of threads
    // defined by the kernel launch geometry (dim3 block (blocksz)).
    thread_block g = this_thread_block ( ) ;
    // here, g.sync() is not needed (see comments above).

    // The threads in this thread block are partitioned into tiles, each with
    // tile_sz threads.
    thread_block_tile<tile_sz> tile = tiled_partition<tile_sz> (g) ;
    // here, tile.sync() is implicit (see comments above)

    // lane: a local thread id, for all threads in a single tile, ranging from
    // 0 to the size of the tile minus one.  Normally the tile has size 32, but
    // it could be a power of 2 less than or equal to 32.
    int lane = threadIdx.x & (tile_sz-1) ;
    // tile_id: is the id for a single tile, each with tile_sz threads in it.
    int tile_id = threadIdx.x >> log2_tile_sz ;

    // Each tile performs partial reduction
    val = GB_cuda_tile_sum_uint64 (tile, val) ;    

    // shared result for partial sums of all threads in a tile:
    static __shared__ uint64_t shared [tile_sz] ;

    if (lane == 0)
    {
        shared [tile_id] = val ;    // Write reduced value to shared memory
    }

    // This g.sync() is essential:  All tiles must finish their work so that
    // the first tile can reduce the shared array down to the scalar val.
    g.sync() ;                      // Wait for all partial reductions

    // This method requires blockDim.x <= tile_sz^2 = 1024, but this is always
    // enforced in the CUDA standard since the our geometry is 1D.

    // Final reduce within first tile
    if (tile_id == 0)
    {
        // read from shared memory only if that tile existed
        val = (threadIdx.x < (blockDim.x >> log2_tile_sz)) ? shared [lane] : 0 ;
        val = GB_cuda_tile_sum_uint64 (tile, val) ;
    }

    // The following sync is not necessary because only tile zero will have the
    // final result in val anyway.  Other tiles (aka warps) will have garbage
    // in val, even with the g.sync().
    // g.sync() ;
    return (val) ;
}

