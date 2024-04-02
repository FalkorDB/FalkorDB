//------------------------------------------------------------------------------
// GraphBLAS/CUDA/Template/GB_cuda_shfl_down.cuh:  warp-level reductions
//------------------------------------------------------------------------------

// SuiteSparse:GraphBLAS, Timothy A. Davis, (c) 2017-2024, All Rights Reserved.
// This file: Copyright (c) 2024, NVIDIA CORPORATION. All rights reserved.
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
//      dest = tile.shfl_down (src, delta) ;
//
// performs the following computation for each thread i:
//
//      if (i+delta < tile_sz)
//      {
//          dest = (the value of src on thread i+delta)
//      }
//
// Where i ranges from 0 to the tile_size-1, which is the warp size of 32 (the
// size of the tile, given by tile.num_threads() and also the #define'd value
// tile_sz), minus one.  If i+delta >= tile_sz for the ith thread, then nothing
// happens for that thread, and the thread is inactive.
//
// Restrictions:  tile_sz must be a power of 2, and it must be 32 or less for
// tile.shfl_down().  The type T must be trivially-copyable (that is
// is_trivially_copyable<T>::value must be true), and sizeof (T) <= 32 must
// hold (that is, the size of T must be 32 bytes or less).  The 32-byte limit
// is handled by GB_cuda_shfl_down_large_ztype, which uses repeated calls to
// tile.shfl_down on 32-byte chunks.

// FIXME for tile.shfl_down(...), delta is an int, so can it be negative?
// For the __shfl_down warp shuffle function, delta is an unsigned int.

//------------------------------------------------------------------------------
// GB_cuda_warp_sum_uint64: reduce a uint64_t value across a single warp
//------------------------------------------------------------------------------

// On input, each thread in the tile holds a single uint64_t value.  On output,
// thread zero holds the sum of values from all the warps.

__device__ __inline__ uint64_t GB_cuda_warp_sum_uint64
(
    thread_block_tile<tile_sz> tile,
    uint64_t value
)
{

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
        #pragma unroll
        for (int i = tile_sz >> 1 ; i > 0 ; i >>= 1)
        {
            value += tile.shfl_down (value, i) ;
        }
    }
    #endif

    //--------------------------------------------------------------------------
    // return result
    //--------------------------------------------------------------------------

    // Note that only thread 0 will have the full summation of all values in
    // the tile.  To broadcast it to all threads, use the following:

    // value = tile.shfl (value, 0) ;

    return (value) ;
}

#if 0

//------------------------------------------------------------------------------
// warp_ReduceSumPlus_uint64: for dot3_phase2
//------------------------------------------------------------------------------

__inline__ __device__ uint64_t warp_ReduceSumPlus_uint64
(
    thread_block_tile<tile_sz> tile,
    uint64_t val
)
{
    // Each iteration halves the number of active threads
    // Each thread adds its partial sum[i] to sum[lane+i]
    for (int i = tile.num_threads() / 2; i > 0; i /= 2)
    {
        val += tile.shfl_down (val, i) ;
    }
    return val; // note: only thread 0 will return full sum
}

//------------------------------------------------------------------------------
// GB_warp_ReduceSumPlus_uint64_vsvs: for vsvs kernel
//------------------------------------------------------------------------------

__inline__ __device__ uint64_t GB_warp_ReduceSumPlus_uint64_vsvs
(
    thread_block_tile<tile_sz> g,
    uint64_t val
)
{
    // Each iteration halves the number of active threads
    // Each thread adds its partial sum[i] to sum[lane+i]
    /*
    #pragma unroll
    for (int i = tile_sz >> 1; i > 0; i >>= 1) {
        val +=  g.shfl_down( val, i);
    }
    */
    // assuming tile_sz is 32:
    val +=  g.shfl_down( val, 16);
    val +=  g.shfl_down( val, 8);
    val +=  g.shfl_down( val, 4);
    val +=  g.shfl_down( val, 2);
    val +=  g.shfl_down( val, 1);
    return val; // note: only thread 0 will return full sum
}

//------------------------------------------------------------------------------
// reduce_sum_int64: for vsdn
//------------------------------------------------------------------------------

// for counting zombies only (always int64_t)
__device__ int64_t reduce_sum_int64
(
    thread_block_tile<tile_sz> g,
    int64_t val
)
{
    // Each iteration halves the number of active threads
    // Each thread adds its partial sum[i] to sum[lane+i]
    for (int64_t i = g.num_threads() / 2; i > 0; i /= 2)
    {
        val += g.shfl_down(val,i) ;
    }
    return val; // note: only thread 0 will return full sum
}

#endif

//------------------------------------------------------------------------------
// GB_cuda_shfl_down_large_ztype: shfl_down a type larger than 32 bytes
//------------------------------------------------------------------------------

// This returns result = tile.shfl_down (value, delta), where value has type
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
        int delta
    )
    {

        // get pointers to value and result, as chunks of size 32 bytes
        struct ztype_chunk *v = (struct ztype_chunk *) value ;
        struct ztype_chunk *r = (struct ztype_chunk *) result ;

        // shfl_down value into result, one chunk at a time
        #pragma unroll
        for (int chunk = 0 ; chunk < GB_Z_NCHUNKS ; chunk++, r++, v++)
        {
            (*r) = tile.shfl_down (*v, delta) ;
        }

        #if ( GB_Z_LEFTOVER > 0 )
        // handle the leftover chunk, if it has nonzero size
        struct ztype_leftover *v_leftover = (struct ztype_leftover *) v ;
        struct ztype_leftover *r_leftover = (struct ztype_leftover *) r ;
        (*r_leftover) = tile.shfl_down (*v_leftover, delta) ;
        #endif
    }

#endif

//------------------------------------------------------------------------------
// GB_cuda_warp_reduce_ztype: reduce a ztype to a scalar, on a single warp
//------------------------------------------------------------------------------

// FIXME: make value parameter *value, and return type void?

__device__ __inline__ GB_Z_TYPE GB_cuda_warp_reduce_ztype
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
            next = tile.shfl_down (value, 16) ;
            GB_ADD (value, value, next) ;
            next = tile.shfl_down (value,  8) ;
            GB_ADD (value, value, next) ;
            next = tile.shfl_down (value,  4) ;
            GB_ADD (value, value, next) ;
            next = tile.shfl_down (value,  2) ;
            GB_ADD (value, value, next) ;
            next = tile.shfl_down (value,  1) ;
            GB_ADD (value, value, next) ;
        }
        #else
        {

            #pragma unroll
            for (int i = tile_sz >> 1 ; i > 0 ; i >>= 1)
            {
                GB_Z_TYPE next = tile.shfl_down (value, i) ;
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
        for (int i = tile_sz >> 1 ; i > 0 ; i >>= 1)
        {
            GB_Z_TYPE next ;
            GB_cuda_shfl_down_large_ztype (&next, tile, &value, i) ;
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

#if 0

//------------------------------------------------------------------------------
// warp_ReduceSum_dndn: for dndn kernel
//------------------------------------------------------------------------------

__inline__ __device__ GB_Z_TYPE warp_ReduceSum_dndn
(
    thread_block_tile<32> g,
    GB_Z_TYPE val
)
{
    // Each iteration halves the number of active threads
    // Each thread adds its partial sum[i] to sum[lane+i]
    // FIXME: only works if sizeof(GB_Z_TYPE) <= 32 bytes
    // FIXME: the ANY monoid needs the cij_exists for each thread
    for (int i = g.num_threads() / 2; i > 0; i /= 2)
    {
        GB_Z_TYPE next = g.shfl_down( val, i) ;
        GB_ADD( val, val, next ); 
    }
    return val; // note: only thread 0 will return full sum
}

//------------------------------------------------------------------------------
// GB_reduce_sum: for dot3 mp and spdn
//------------------------------------------------------------------------------

__device__ __inline__ GB_Z_TYPE GB_reduce_sum
(
    thread_block_tile<tile_sz> g,
    GB_Z_TYPE val
)
{
    // Each iteration halves the number of active threads
    // Each thread adds its partial sum[i] to sum[lane+i]
    // Temporary GB_Z_TYPE is necessary to handle arbirary ops
    // FIXME: only works if sizeof(GB_Z_TYPE) <= 32 bytes
    // FIXME: the ANY monoid needs the cij_exists for each thread
    #pragma unroll
    for (int i = tile_sz >> 1 ; i > 0 ; i >>= 1)
    {
        GB_Z_TYPE next = g.shfl_down (val, i) ;
        GB_ADD (val, val, next) ; 
    }
    return val;
}

//------------------------------------------------------------------------------
// GB_warp_Reduce: for cuda_reduce
//------------------------------------------------------------------------------

__device__ __inline__ GB_Z_TYPE GB_warp_Reduce
(
    thread_block_tile<tile_sz> g,
    GB_Z_TYPE val
)
{
    // Each iteration halves the number of active threads
    // Each thread adds its partial val[k] to val[lane+k]

    // FIXME: doesn't work unless sizeof(GB_Z_TYPE) <= 32 bytes

#if ( GB_Z_SIZE <= 32 )
    // assumes tile_sz is 32:
    GB_Z_TYPE fold = g.shfl_down ( val, 16) ;
    GB_ADD ( val, val, fold ) ;
    fold = g.shfl_down ( val, 8) ;
    GB_ADD ( val, val, fold ) ;
    fold = g.shfl_down ( val, 4) ;
    GB_ADD ( val, val, fold ) ;
    fold = g.shfl_down ( val, 2) ;
    GB_ADD ( val, val, fold ) ;
    fold = g.shfl_down ( val, 1) ;
    GB_ADD ( val, val, fold ) ;
#else
    // use shared memory and do not use shfl_down?
    // or use repeated calls to shfl_down, on chunks of 32 bytes each?
    #error "not implemented yet"
#endif

    return (val) ; // note: only thread 0 will return full val
}
#endif
