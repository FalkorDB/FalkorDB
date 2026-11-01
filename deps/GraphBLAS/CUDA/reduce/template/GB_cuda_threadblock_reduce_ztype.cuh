//------------------------------------------------------------------------------
// CUDA/reduce/template/GB_cuda_threadblock_reduce_ztype.cuh
//------------------------------------------------------------------------------

// SuiteSparse:GraphBLAS, Timothy A. Davis, (c) 2017-2026, All Rights Reserved.
// SPDX-License-Identifier: Apache-2.0

//------------------------------------------------------------------------------

// Reduce across an entire threadblock a single scalar of type ztype,
// using the given monoid (used in GB_cuda_tile_reduce_ztype).

// On input, there is no need for this_thread_block().sync(), because the first
// reduction is across a single tile.  The creation of the tile with
// tiled_partition<GB_CUDA_TILE_SIZE>(g) ensures each tile is synchronized,
// which is sufficient for the following call to GB_cuda_tile_reduce_ztype.

// Compare with CUDA/cumsum/template/GB_cuda_threadblock_sum_uint64.

__inline__ __device__ GB_Z_TYPE GB_cuda_threadblock_reduce_ztype
(
    GB_Z_TYPE val
)
{
    // The thread_block g that calls this method has a number of threads
    // defined by the kernel launch geometry (dim3 block (...)).
    thread_block g = this_thread_block ( ) ;
    // here, g.sync() is not needed (see comments above).

    // The threads in this thread block are partitioned into tiles, each with
    // GB_CUDA_TILE_SIZE threads.
    thread_block_tile<GB_CUDA_TILE_SIZE> tile =
        tiled_partition<GB_CUDA_TILE_SIZE> (g) ;
    // here, tile.sync() is implicit (see comments above)

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
    __shared__ GB_Z_TYPE shared [GB_CUDA_TILE_SIZE] ;

    if (threadId_in_tile == 0)
    {
        // the first thread in each tile writes the result of its entire tile
        // to shared memory
        shared [tile_id] = val ;
    }

    // This g.sync() is essential:  All tiles must finish their work so that
    // the first tile can reduce the shared array down to the scalar val.
    g.sync() ;                      // Wait for all partial reductions

    // This method requires blockDim.x <= GB_CUDA_TILE_SIZE^2 = 1024, but this
    // is always enforced in the CUDA standard since our geometry is 1D.

    // Final reduce within first tile
    if (tile_id == 0)
    {
        GB_DECLARE_IDENTITY_CONST (zid) ;   // const GB_Z_TYPE zid = identity ;
        // read from shared memory only if that tile existed
        val = (threadIdx.x < (blockDim.x >> GB_CUDA_LOG2_TILE_SIZE)) ?
            shared [threadId_in_tile] : zid ;
        val = GB_cuda_tile_reduce_ztype (tile, val) ;
    }

    // The following g.sync() is not necessary because only tile zero will have
    // the final result in val anyway.  Other tiles (aka warps) will have
    // garbage in val, even with the g.sync().

    // g.sync() ;
    return (val) ;
}

