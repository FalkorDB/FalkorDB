//------------------------------------------------------------------------------
// templates/GB_AxB_cuda_dot3_phase2: fill the global buckets
//------------------------------------------------------------------------------

// TODO describe me
#pragma once
#include "GB_cuda_kernel.h"
#include "GB_mxm_shared_definitions.h"
#include "GB_cuda_buckets.h"
#include <stdint.h>
#include <cooperative_groups.h>
#include <cub/block/block_scan.cuh>

using namespace cooperative_groups;

//------------------------------------------------------------------------------
// BlockPrefixCallbackOp
//------------------------------------------------------------------------------

// A stateful callback functor that maintains a running prefix to be applied
// during consecutive scan operations.
struct BlockPrefixCallbackOp
{
   // Running prefix
   int64_t running_total;
   // Constructor
   __device__ BlockPrefixCallbackOp(int64_t running_total) : running_total(running_total) {}

   // Callback operator to be entered by the first warp of threads in the block.
   // Thread-0 is responsible for returning a value for seeding the block-wide scan.
   __device__ int64_t operator()(int64_t block_aggregate)
   {
     int64_t old_prefix = running_total;
     running_total += block_aggregate;
     return old_prefix;
   }
};

//------------------------------------------------------------------------------
// blockBucketExclusiveSum
//------------------------------------------------------------------------------

__inline__
__device__ void blockBucketExclusiveSum(int bucketId, int64_t *d_data, int nblocks)
{
   #define blocksize  32

   // Specialize BlockScan for a 1D block of 32 threads
   typedef cub::BlockScan<int64_t, 32, cub::BLOCK_SCAN_WARP_SCANS> BlockScan;

   // Allocate shared memory for BlockScan
   __shared__ typename BlockScan::TempStorage temp_storage;

   // Initialize running total
   BlockPrefixCallbackOp prefix_op(0);

   // Have the block iterate over segments of items
   int64_t data=0;

   int64_t *blockbucket= d_data;

   for (int block_id = 0; block_id < nblocks; block_id += blocksize)
   {
    // Load a segment of consecutive items that are blocked across threads

    //printf("block %d entering sum\n",blockIdx.x);
      int loc = block_id + threadIdx.x;
      if ( loc < nblocks)
      {
        //printf("block %di loading tid=%d\n",block_id,tid);
        data  = blockbucket[bucketId*nblocks +loc ] ;
      }
      this_thread_block().sync();

      //printf("bb%d_%d s0 before prefix= %ld \n", block_id,bucketId,
      //                     blockbucket[bucketId*nblocks +loc] )  ;
      // Collectively compute the block-wide exclusive prefix sum
      BlockScan(temp_storage).ExclusiveSum( data, data, prefix_op);
      this_thread_block().sync();

      if ( loc < nblocks)
      {
        blockbucket[bucketId*nblocks   +loc ]  = data  ;
      }
      //this_thread_block().sync();

      //printf("bb%d_%d = %ld \n", block_id, bucketId, blockbucket[bucketId*nblocks +loc] )  ;

      data = 0;
   }
}

//------------------------------------------------------------------------------
// warp_ReduceSumPlus_uint64
//------------------------------------------------------------------------------

template< int tile_sz>
__inline__ __device__ uint64_t warp_ReduceSumPlus_uint64( thread_block_tile<tile_sz> tile, uint64_t val)
{
    // Each iteration halves the number of active threads
    // Each thread adds its partial sum[i] to sum[lane+i]
    for (int i = tile.size() / 2; i > 0; i /= 2) {
        val +=  tile.shfl_down( val, i);
    }
    return val; // note: only thread 0 will return full sum
}

//------------------------------------------------------------------------------
// AxB_phase2
//------------------------------------------------------------------------------

// GB_AxB_cuda_dot3_phase2 is a CUDA kernel that takes as input the
// nanobuckets and blockbucket arrays computed by the first phase kernel,
// GB_AxB_cuda_dot3_phase1.  The launch geometry of this kernel must match the
// GB_AxB_cuda_dot3_phase1 kernel, with the same # of threads and threadblocks.

__global__ void AxB_phase2
(
    // input, not modified:
    int64_t *__restrict__ blockbucket,    // global bucket count, of size NBUCKETS*nblocks
    // output:
    int64_t *__restrict__ offset,         // global offsets, for each bucket
    // inputs, not modified:
    const int nblocks        // input number of blocks to reduce across, ie size of vector for 1 bucket
)
{

    //--------------------------------------------------------------------------
    // sum up the bucket counts of prior threadblocks
    //--------------------------------------------------------------------------

    // blockbucket is an array of size NBUCKETS-by-nblocks, held by row.  The
    // entry blockbucket [bucket * nblocks + t] holds the # of entries
    // in the bucket (in range 0 to NBUCKETS-1) found by threadblock t.

    //__shared__ uint64_t offset [NBUCKETS] ;
    uint64_t s[NBUCKETS];

    #pragma unroll
    for(int b = 0; b < NBUCKETS; ++b){
        s[b] = 0;
    }

    thread_block_tile<32> tile = tiled_partition<32>(this_thread_block() );

    //printf("block %d,dim %d entering sum %d nblocks\n",blockIdx.x, blockDim.x, nblocks);
    int64_t tid = threadIdx.x  + blockIdx.x * blockDim.x;


     #pragma unroll
     for(int b = 0; b < NBUCKETS; ++b) {
         for( tid = threadIdx.x + blockIdx.x * blockDim.x;
              tid < nblocks;
              tid += blockDim.x*gridDim.x) {
            s[b]  += blockbucket[  b * nblocks +tid] ;
         }
         this_thread_block().sync(); 

         s[b]  = warp_ReduceSumPlus_uint64<32>( tile, s[b]);
     }

    if (threadIdx.x ==0 )
    {
        #pragma unroll
        for(int b = 0; b < NBUCKETS; ++b) {
            atomicAdd( (unsigned long long int*)&(offset[b]), s[b]);
        }
    }
    this_thread_block().sync(); 

    if( gridDim.x >= NBUCKETS)
    {
        // Cumulative sum across blocks for each bucket
        if (blockIdx.x <NBUCKETS) {
            blockBucketExclusiveSum( blockIdx.x, blockbucket, nblocks ) ;
        }
    }
    else
    {
        if (blockIdx.x == 0)
        {
            #pragma unroll
            for(int b = 0; b < NBUCKETS; ++b) {
                blockBucketExclusiveSum( b, blockbucket, nblocks ) ;
            }
        }
    }
} // phase2
