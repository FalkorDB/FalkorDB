//------------------------------------------------------------------------------
// GraphBLAS/CUDA/template/GB_cuda_tile_sum_uint64.cuh:  warp-level reductions
//------------------------------------------------------------------------------

// SuiteSparse:GraphBLAS, Timothy A. Davis, (c) 2017-2025, All Rights Reserved.
// This file: Copyright (c) 2024-2025, NVIDIA CORPORATION. All rights reserved.
// SPDX-License-Identifier: BSD-3-Clause

//------------------------------------------------------------------------------

// See template/GB_cuda_tile_reduce_ztype.cuh for a description of
// tile.shfl_down.

//------------------------------------------------------------------------------
// GB_cuda_tile_sum_uint64: reduce a uint64_t value across a single warp
//------------------------------------------------------------------------------

// On input, each thread in the tile holds a single uint64_t value.  On output,
// thread zero holds the sum of values from all the warps.

__device__ __inline__ uint64_t GB_cuda_tile_sum_uint64
(
    thread_block_tile<tile_sz> tile,
    uint64_t value
)
{

    //--------------------------------------------------------------------------
    // ensure the tile is ready
    //--------------------------------------------------------------------------

    tile.sync ( ) ;

    //--------------------------------------------------------------------------
    // sum value on all threads to a single value
    //--------------------------------------------------------------------------

    #if (tile_sz == 32)
    {
        // this is the typical case
        value += tile.shfl_down (value, 16) ;
        value += tile.shfl_down (value,  8) ;
        value += tile.shfl_down (value,  4) ;
        value += tile.shfl_down (value,  2) ;
        value += tile.shfl_down (value,  1) ;
    }
    #else
    {
        // tile_sz is less than 32 (either 1, 2, 4, 8, or 16)
        #pragma unroll
        for (int offset = tile_sz >> 1 ; offset > 0 ; offset >>= 1)
        {
            value += tile.shfl_down (value, offset) ;
        }
    }
    #endif

    //--------------------------------------------------------------------------
    // return result
    //--------------------------------------------------------------------------

    // Note that only thread 0 will have the full summation of all values in
    // the tile.  To broadcast it to all threads, use the following:

//  value = tile.shfl (value, 0) ;

    return (value) ;
}

