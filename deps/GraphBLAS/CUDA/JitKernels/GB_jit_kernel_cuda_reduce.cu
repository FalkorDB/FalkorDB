//------------------------------------------------------------------------------
// GraphBLAS/CUDA/JitKernels/GB_jit_cuda_reduce.cu
//------------------------------------------------------------------------------

// SuiteSparse:GraphBLAS, Timothy A. Davis, (c) 2017-2024, All Rights Reserved.
// This file: Copyright (c) 2024, NVIDIA CORPORATION. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//------------------------------------------------------------------------------

// The GB_cuda_jit_reduce CUDA kernel reduces a GrB_Matrix A of any type
// GB_A_TYPE, to a scalar of type GB_Z_TYPE.  Each threadblock (blockIdx.x)
// reduces its portion of Ax to a single scalar, and then atomics are used
// across the threadblocks.

// Both the grid and block are 1D, so blockDim.x is the # threads in a
// threadblock, and the # of threadblocks is grid.x

// Let b = blockIdx.x, and let s be blockDim.x.
// Each threadblock owns s*8 contiguous items in the input data.

// Thus, threadblock b owns Ax [b*s*8 ... min(n,(b+1)*s*8-1)].  Its job
// is to reduce this data to a scalar, and write it to its scalar.

// If the reduction is done on the GPU, A will never be iso-valued.

#if GB_C_ISO || GB_A_ISO
#error "kernel undefined for C or A iso"
#endif

// FIXME: put these definitions in GB_cuda_kernel.h:
#define tile_sz 32
#define log2_tile_sz 5

#include "GB_cuda_shfl_down.cuh"

//------------------------------------------------------------------------------
// GB_block_Reduce: reduce across all warps into a single scalar
//------------------------------------------------------------------------------

__inline__ __device__ GB_Z_TYPE GB_block_Reduce
(
    thread_block g,
    GB_Z_TYPE val
)
{
    static __shared__ GB_Z_TYPE shared [tile_sz] ;
    int lane = threadIdx.x & (tile_sz-1) ;
    int wid  = threadIdx.x >> log2_tile_sz ;
    thread_block_tile<tile_sz> tile = tiled_partition<tile_sz>( g ) ;

    // Each warp performs partial reduction
    val = GB_cuda_warp_reduce_ztype (tile, val) ;

    // Wait for all partial reductions
    if (lane == 0)
    {
        shared [wid] = val ; // Write reduced value to shared memory
    }
    this_thread_block().sync() ;        // Wait for all partial reductions

    GB_DECLARE_IDENTITY_CONST (zid) ;   // const GB_Z_TYPE zid = identity ;

    val = (threadIdx.x < (blockDim.x >> LOG2_WARPSIZE)) ?  shared [lane] : zid ;

    // Final reduce within first warp
    val = GB_cuda_warp_reduce_ztype (tile, val) ;
    return (val) ;
}

//------------------------------------------------------------------------------
// GB_cuda_reduce_kernel: reduce all entries in a matrix to a single scalar
//------------------------------------------------------------------------------

__global__ void GB_cuda_reduce_kernel
(
    // output:
    void *zscalar,  // scalar result, at least sizeof (uint32_t)
    GrB_Matrix V,   // matrix result, for partial reduction (or NULL)
    // input:
    GrB_Matrix A,   // matrix to reduce
    int64_t anz     // # of entries in A
)
{

    //--------------------------------------------------------------------------
    // initializations
    //--------------------------------------------------------------------------

    const GB_A_TYPE *__restrict__ Ax = (GB_A_TYPE *) A->x ;

    // each thread reduces its result into zmine, of type GB_Z_TYPE
    GB_DECLARE_IDENTITY (zmine) ; // GB_Z_TYPE zmine = identity ;

    // On input, zscalar is already initialized to the monoid identity value.
    // If GB_Z_TYPE has size less than 4 bytes, zscalar has been upscaled to 4
    // bytes.

    //--------------------------------------------------------------------------
    // phase 1: each thread reduces a part of the matrix to its own scalar
    //--------------------------------------------------------------------------

    #if GB_A_IS_SPARSE || GB_A_IS_HYPER
    {

        //----------------------------------------------------------------------
        // A is sparse or hypersparse
        //----------------------------------------------------------------------

        #if GB_A_HAS_ZOMBIES
        {
            // check for zombies during the reduction
            const int64_t *__restrict__ Ai = A->i ;
            // grid-stride loop:
            for (int64_t p = blockIdx.x * blockDim.x + threadIdx.x ;
                p < anz ;
                p += blockDim.x * gridDim.x)
            {
                if (Ai [p] < 0) continue ;          // skip zombies
                GB_GETA_AND_UPDATE (zmine, Ax, p) ; // zmine += (ztype) Ax [p]
            }
        }
        #else
        {
            // no zombies present
            for (int64_t p = blockIdx.x * blockDim.x + threadIdx.x ;
                p < anz ;
                p += blockDim.x * gridDim.x)
            {
                GB_GETA_AND_UPDATE (zmine, Ax, p) ; // zmine += (ztype) Ax [p]
            }
        }
        #endif

    }
    #elif GB_A_IS_FULL
    {

        //----------------------------------------------------------------------
        // A is full
        //----------------------------------------------------------------------

        for (int64_t p = blockIdx.x * blockDim.x + threadIdx.x ;
            p < anz ;
            p += blockDim.x * gridDim.x)
        {
            GB_GETA_AND_UPDATE (zmine, Ax, p) ; // zmine += (ztype) Ax [p]
        }

    }
    #else
    {

        //----------------------------------------------------------------------
        // A is bitmap
        //----------------------------------------------------------------------

        const int8_t *__restrict__ Ab = A->b ;
        for (int64_t p = blockIdx.x * blockDim.x + threadIdx.x ;
            p < anz ;
            p += blockDim.x * gridDim.x)
        {
            if (Ab [p] == 0) continue ;         // skip if entry not in bitmap
            GB_GETA_AND_UPDATE (zmine, Ax, p) ; // zmine += (ztype) Ax [p]
        }
    }
    #endif

    this_thread_block().sync() ;

    //--------------------------------------------------------------------------
    // phase 2: each threadblock reduces all threads into a scalar
    //--------------------------------------------------------------------------

    zmine = GB_block_Reduce( this_thread_block(), zmine) ;
    this_thread_block().sync() ;

    //--------------------------------------------------------------------------
    // phase 3: reduce across threadblocks
    //--------------------------------------------------------------------------

    if (threadIdx.x == 0)
    {
        #if GB_Z_HAS_CUDA_ATOMIC_USER

            // user-defined monoid can be done automically:
            // zscalar "+=" zmine using a CUDA atomic directly
            GB_cuda_atomic_user (zscalar, zmine) ;

        #elif GB_Z_HAS_CUDA_ATOMIC_BUILTIN

            // cast the zmine result to the CUDA atomic type, and reduce
            // atomically to the global zscalar
            // zscalar "+=" zmine using a CUDA atomic pun
            GB_Z_CUDA_ATOMIC_TYPE *z = (GB_Z_CUDA_ATOMIC_TYPE *) zscalar ;
            GB_Z_CUDA_ATOMIC_TYPE zsum = (GB_Z_CUDA_ATOMIC_TYPE) zmine ;
            GB_Z_CUDA_ATOMIC <GB_Z_CUDA_ATOMIC_TYPE> (z, zsum) ;

        #else

            // save my zmine result in V
            GB_Z_TYPE *Vx = (GB_Z_TYPE *) V->x ;
            Vx [blockIdx.x] = zmine ;

        #endif
    }
}

//------------------------------------------------------------------------------
// host function to launch the CUDA kernel
//------------------------------------------------------------------------------

extern "C"
{
    GB_JIT_CUDA_KERNEL_REDUCE_PROTO (GB_jit_kernel) ;
}

GB_JIT_CUDA_KERNEL_REDUCE_PROTO (GB_jit_kernel)
{
    dim3 grid (gridsz) ;
    dim3 block (blocksz) ;
    GB_A_NHELD (anz) ;      // anz = # of entries held in A
    GB_cuda_reduce_kernel <<<grid, block, 0, stream>>> (zscalar, V, A, anz) ;
    return (GrB_SUCCESS) ;
}

