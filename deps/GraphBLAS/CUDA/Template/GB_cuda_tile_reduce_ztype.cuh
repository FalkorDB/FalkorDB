//------------------------------------------------------------------------------
// GraphBLAS/CUDA/template/GB_cuda_tile_reduce_ztype.cuh:  warp-level reductions
//------------------------------------------------------------------------------

// SuiteSparse:GraphBLAS, Timothy A. Davis, (c) 2017-2025, All Rights Reserved.
// This file: Copyright (c) 2024-2025, NVIDIA CORPORATION. All rights reserved.
// SPDX-License-Identifier: BSD-3-Clause

//------------------------------------------------------------------------------

// shfl_down is a method in the cooperative_groups namespace.  It allows all
// threads in a warp (or other thread partition) to work together in a
// cooperative fashion.
//
// Suppose we have a tile that defines a single warp of 32 threads:
//
//      #define tile_sz 32
//      thread_block_tile<tile_sz> tile =
//          tiled_partition<tile_sz> (this_thread_block()) ;
//
// Suppose each thread has two scalars dest and src of type T.  Then:
//
//      T dest, src ;
//      dest = tile.shfl_down (src, offset) ;
//
// performs the following computation for each thread tid:
//
//      if (tid+offset < tile_sz)
//      {
//          dest = (the value of src on thread tid+offset)
//      }
//
// Where tid ranges from 0 to the tile_sz-1, which is the warp size of 32
// (the size of the tile, given by tile.num_threads() and also the #define'd
// value tile_sz), minus one.  If tid+offset >= tile_sz for the ith thread,
// then nothing happens for that thread, and the thread is inactive.
//
// Restrictions:  tile_sz must be a power of 2, and it must be 32 or less for
// tile.shfl_down().  The type T must be trivially-copyable (that is
// is_trivially_copyable<T>::value must be true), and sizeof (T) <= 32 must
// hold (that is, the size of T must be 32 bytes or less).  The 32-byte limit
// is handled by GB_cuda_shfl_down_large_ztype, which uses repeated calls to
// tile.shfl_down on 32-byte chunks.

//------------------------------------------------------------------------------
// GB_cuda_shfl_down_large_ztype: shfl_down a type larger than 32 bytes
//------------------------------------------------------------------------------

// This returns result = tile.shfl_down (value, offset), where value has type
// GB_Z_TYPE, and sizeof (GB_Z_TYPE) > 32.

#if ( GB_Z_SIZE > 32 )

    // # of 32-byte chunks to hold a single GB_Z_TYPE, excluding leftover
    // chunk; GB_Z_SIZE is sizeof (GB_Z_TYPE) as a hard-coded constant.
    #define GB_Z_NCHUNKS ( GB_Z_SIZE / 32 )

    // ztype_chunk is always 32 bytes in size
    typedef struct { uint8_t bytes [32] ; } ztype_chunk ;

    // size of the single leftover chunk of size 0 to < 32 bytes
    #define GB_Z_LEFTOVER ( GB_Z_SIZE - ( GB_Z_NCHUNKS * 32 ) )

    #if ( GB_Z_LEFTOVER > 0 )
    // leftover chunk is not defined if GB_Z_SIZE is a multiple of 32
    typedef struct { uint8_t bytes [GB_Z_LEFTOVER] ; } ztype_leftover ;
    #endif

    __device__ __inline__ void GB_cuda_shfl_down_large_ztype
    (
        GB_Z_TYPE *result,
        thread_block_tile<tile_sz> tile,
        GB_Z_TYPE *value,
        int offset
    )
    {

        // get pointers to value and result, as chunks of size 32 bytes
        ztype_chunk *v = (ztype_chunk *) value ;
        ztype_chunk *r = (ztype_chunk *) result ;

        // shfl_down value into result, one chunk at a time
        #pragma unroll
        for (int chunk = 0 ; chunk < GB_Z_NCHUNKS ; chunk++, r++, v++)
        {
            (*r) = tile.shfl_down (*v, offset) ;
        }

        #if ( GB_Z_LEFTOVER > 0 )
        // handle the leftover chunk, if it has nonzero size
        ztype_leftover *v_leftover = (ztype_leftover *) v ;
        ztype_leftover *r_leftover = (ztype_leftover *) r ;
        (*r_leftover) = tile.shfl_down (*v_leftover, offset) ;
        #endif
    }

#endif

//------------------------------------------------------------------------------
// GB_cuda_tile_reduce_ztype: reduce a ztype to a scalar, on a single warp
//------------------------------------------------------------------------------

__device__ __inline__ GB_Z_TYPE GB_cuda_tile_reduce_ztype
(
    thread_block_tile<tile_sz> tile,
    GB_Z_TYPE value
)
{

    #if ( GB_Z_SIZE <= 32 )
    {

        //----------------------------------------------------------------------
        // GB_Z_TYPE can done with a single shfl_down
        //----------------------------------------------------------------------

        #if ( tile_sz == 32 )
        {
            // this is the typical case
            GB_Z_TYPE next ;
            next = tile.shfl_down (value, 16) ; GB_ADD (value, value, next) ;
            next = tile.shfl_down (value,  8) ; GB_ADD (value, value, next) ;
            next = tile.shfl_down (value,  4) ; GB_ADD (value, value, next) ;
            next = tile.shfl_down (value,  2) ; GB_ADD (value, value, next) ;
            next = tile.shfl_down (value,  1) ; GB_ADD (value, value, next) ;
        }
        #else
        {

            #pragma unroll
            for (int offset = tile_sz >> 1 ; offset > 0 ; offset >>= 1)
            {
                GB_Z_TYPE next = tile.shfl_down (value, offset) ;
                GB_ADD (value, value, next) ;
            }

        }
        #endif
    }
    #else
    {

        //----------------------------------------------------------------------
        // sizeof (GB_Z_TYPE) is too large for a single shfl_down
        //----------------------------------------------------------------------

        #pragma unroll
        for (int offset = tile_sz >> 1 ; offset > 0 ; offset >>= 1)
        {
            GB_Z_TYPE next ;
            GB_cuda_shfl_down_large_ztype (&next, tile, &value, offset) ;
            GB_ADD (value, value, next) ;
        }
    }
    #endif

    //--------------------------------------------------------------------------
    // return result
    //--------------------------------------------------------------------------

    // Note that only thread 0 will have the full summation of all values in
    // the tile.  To broadcast it to all threads, use the following:

    // value = tile.shfl (value, 0) ;

    // or if the ztype is large:
    // GB_cuda_shfl_down_large_ztype (&value, tile, &value, 0) ;

    return (value) ;
}

