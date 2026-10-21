//------------------------------------------------------------------------------
// CUDA/reduce/template/GB_jit_kernel_cuda_reduce.cu
//------------------------------------------------------------------------------

// SuiteSparse:GraphBLAS, Timothy A. Davis, (c) 2017-2026, All Rights Reserved.
// SPDX-License-Identifier: Apache-2.0

//------------------------------------------------------------------------------

// The GB_cuda_reduce_kernel CUDA kernel reduces a GrB_Matrix A of any type
// GB_A_TYPE to a scalar of type GB_Z_TYPE, using a monoid.  The work is done
// in three phases:
//
//      phase 1: each thread reduces a portion of A's entries into a per-thread
//               scalar (zmine) using a grid-stride loop.
//      phase 2: the threads of a threadblock reduce their zmine values into one
//               per-block scalar.
//      phase 3: the per-block scalars are combined into the final scalar: with
//               a CUDA atomic into zscalar if the monoid has one, else each
//               block writes its scalar to V [blockIdx.x] and the host reduces
//               V.

// Both the grid and block are 1D, so blockDim.x is the # of threads in a
// threadblock, and gridDim.x is the # of threadblocks.  In phase 1 each thread
// uses a grid-stride loop over A's entries: thread threadIdx.x of block
// blockIdx.x starts at p0 = blockIdx.x*blockDim.x + threadIdx.x, advancing by
// stride = blockDim.x*gridDim.x, reading p0, p0+stride, p0+2*stride, ...

// Terminal (absorbing) monoids (GB_MONOID_IS_TERMINAL): the reduction may stop
// early as soon as any thread reads the terminal value, since the result is
// then settled (true for OR, -inf for MIN, etc.).  A single global
// terminal_flag coordinates the blocks (it is NULL for non-terminal monoids):
//      * at entry, a block reads terminal_flag once; if an earlier block has
//        already hit the terminal, the whole block returns immediately.
//      * during phase 1, a thread that hits the terminal sets a block-shared
//        flag (blk_terminal), and the block stops at the end of the current
//        chunk (of GB_CUDA_REDUCE_CHECK entries).  This in-loop check is a
//        cheap shared-memory read, with no global atomic in the hot loop.
//      * after phase 1, a block that hit the terminal writes terminal_flag
//        once (one atomic per block), so later blocks short-circuit at the
//        entry check above.

// If the reduction is done on the GPU, A will never be iso-valued.

#define GB_FREE_ALL ;

#if GB_C_ISO || GB_A_ISO
#error "kernel undefined for C or A iso"
#endif

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
    GrB_Matrix A,           // matrix to reduce
    int32_t *terminal_flag, // early-exit flag; NULL for non-terminal
    int64_t anz             // # of entries in A
)
{

#if GB_MONOID_IS_TERMINAL
    // a prior block may have already hit the terminal value - if so, skip this
    // whole block. read once, decide for the entire block, return together.

    // blk_terminal: block-level shared flag, set during phase 1 when any
    // thread in this block hits the terminal - a cheap shared-memory check
    // replaces the per-chunk global-atomic poll.
    __shared__ int blk_stop ;
    __shared__ volatile int blk_terminal ;
    if (threadIdx.x == 0)
    {
        blk_stop = atomicAdd (terminal_flag, 0) ;
        blk_terminal = 0 ;
    }

    this_thread_block ().sync () ;

    if (blk_stop)
    {
        return ;
    }
#endif

    //--------------------------------------------------------------------------
    // initializations
    //--------------------------------------------------------------------------

    const GB_A_TYPE *__restrict__ Ax = (GB_A_TYPE *) A->x ;

    // each thread reduces its result into zmine, of type GB_Z_TYPE
    GB_DECLARE_IDENTITY (zmine) ; // GB_Z_TYPE zmine = identity ;

#if GB_MONOID_IS_TERMINAL
    GB_DECLARE_TERMINAL_CONST (zterminal) ;  // const GB_Z_TYPE zterminal = T
#endif

    // On input, zscalar is already initialized to the monoid identity value.
    // If GB_Z_TYPE has size less than 4 bytes, zscalar has been upscaled to 4
    // bytes.

    //--------------------------------------------------------------------------
    // phase 1: each thread reduces a part of the matrix to its own scalar
    //--------------------------------------------------------------------------

    // select the per-entry body for the (compile-time) storage format
    #if GB_A_IS_SPARSE || GB_A_IS_HYPER

        //----------------------------------------------------------------------
        // A is sparse or hypersparse
        //----------------------------------------------------------------------

        #if GB_A_HAS_ZOMBIES
            // check for zombies during the reduction
            const GB_Ai_SIGNED_TYPE *__restrict__ Ai =
                (GB_Ai_SIGNED_TYPE *) A->i ;
            #define GB_REDUCE_ONE(p) \
                { if (Ai [p] >= 0) { GB_GETA_AND_UPDATE (zmine, Ax, p) ; } }
        #else
            // no zombies present
            #define GB_REDUCE_ONE(p) \
                { GB_GETA_AND_UPDATE (zmine, Ax, p) ; }
        #endif

    #elif GB_A_IS_FULL

        //----------------------------------------------------------------------
        // A is full
        //----------------------------------------------------------------------

        #define GB_REDUCE_ONE(p) \
            { GB_GETA_AND_UPDATE (zmine, Ax, p) ; }

    #else

        //----------------------------------------------------------------------
        // A is bitmap
        //----------------------------------------------------------------------

        const int8_t *__restrict__ Ab = A->b ;
        #define GB_REDUCE_ONE(p) \
            { if (Ab [p] != 0) { GB_GETA_AND_UPDATE (zmine, Ax, p) ; } }

    #endif

    const int64_t stride = (int64_t) blockDim.x * gridDim.x ;
    const int64_t p0     = (int64_t) blockIdx.x * blockDim.x + threadIdx.x ;

#if GB_MONOID_IS_TERMINAL
    // chunked: the inner loop is pristine.  The checkpoint is a cheap
    // block-level test in shared memory, with no global atomics in the hot
    // loop.  A thread that hits the terminal flips the block-shared flag, and
    // the whole block then bails out.  The global terminal_flag is written
    // once, after phase 1 (see below), only to short-circuit future blocks via
    // the entry early-out.  (break, not return.)
    for (int64_t p = p0 ; p < anz ; )
    {
        const int64_t span = (int64_t) GB_CUDA_REDUCE_CHECK * stride ;
        const int64_t pend = (anz < p + span) ? anz : p + span ;
        for ( ; p < pend ; p += stride)
        {
            GB_REDUCE_ONE (p) ;
        }

        if (GB_TERMINAL_CONDITION (zmine, zterminal))
        {
            blk_terminal = 1 ;      // tell my block (shared, monotonic 0 -> 1)
        }
        if (blk_terminal) break ;   // my block hit the terminal -> stop
    }
#else
    // non-terminal: the original pristine grid-stride loop, unchanged
    for (int64_t p = p0 ; p < anz ; p += stride)
    {
        GB_REDUCE_ONE (p) ;
    }
#endif
    #undef GB_REDUCE_ONE

    this_thread_block().sync() ;

#if GB_MONOID_IS_TERMINAL
    // propagate this block's finding to global memory once (one atomic per
    // block, not per chunk), so later blocks short-circuit at their entry
    // early-out.
    if (threadIdx.x == 0 && blk_terminal)
    {
        atomicExch (terminal_flag, 1) ;
    }
#endif

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
    CUDA_OK (cudaSetDevice (device)) ;
    dim3 grid (gridsz) ;    // gridsz: # of threadblocks
    dim3 block (GB_CUDA_REDUCE_BLOCKDIM) ;  // # of threads in each threadblock
    GB_A_NHELD (anz) ;      // anz = # of entries held in A

    CUDA_OK (cudaGetLastError ( )) ;
    CUDA_OK (cudaStreamSynchronize (stream)) ;
    GB_cuda_reduce_kernel <<<grid, block, 0, stream>>>
        (zscalar, V, A, terminal_flag, anz) ;
    CUDA_OK (cudaGetLastError ( )) ;
    CUDA_OK (cudaStreamSynchronize (stream)) ;
    return (GrB_SUCCESS) ;
}

