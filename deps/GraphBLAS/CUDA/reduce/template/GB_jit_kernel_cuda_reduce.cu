//------------------------------------------------------------------------------
// GraphBLAS/CUDA/jit_kernels/GB_jit_cuda_reduce.cu
//------------------------------------------------------------------------------

// SuiteSparse:GraphBLAS, Timothy A. Davis, (c) 2017-2025, All Rights Reserved.
// This file: Copyright (c) 2024-2025, NVIDIA CORPORATION. All rights reserved.
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

#define GB_FREE_ALL ;

#if GB_C_ISO || GB_A_ISO
#error "kernel undefined for C or A iso"
#endif

// tile_sz can vary per algorithm.  It must a power of 2, and because we
// use tile.shfl_down(), it must also be <= 32.
#define tile_sz 32
#define log2_tile_sz 5

#include "template/GB_cuda_tile_sum_uint64.cuh"
#include "template/GB_cuda_tile_reduce_ztype.cuh"
#include "template/GB_cuda_threadblock_reduce_ztype.cuh"

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
            const GB_Ai_SIGNED_TYPE *__restrict__ Ai = (GB_Ai_SIGNED_TYPE *) A->i ;
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

    zmine = GB_cuda_threadblock_reduce_ztype (zmine) ;

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
    GB_GET_CALLBACKS ;
    dim3 grid (gridsz) ;    // gridsz: # of threadblocks
    dim3 block (blocksz) ;  // blocksz: # of threads in each threadblock
    GB_A_NHELD (anz) ;      // anz = # of entries held in A

    CUDA_OK (cudaGetLastError ( )) ;
    CUDA_OK (cudaStreamSynchronize (stream)) ;
    GB_cuda_reduce_kernel <<<grid, block, 0, stream>>> (zscalar, V, A, anz) ;
    CUDA_OK (cudaGetLastError ( )) ;
    CUDA_OK (cudaStreamSynchronize (stream)) ;
    return (GrB_SUCCESS) ;
}

