//------------------------------------------------------------------------------
// GraphBLAS/CUDA/template/GB_jit_kernel_cuda_transpose_prep.cu
//------------------------------------------------------------------------------

// SuiteSparse:GraphBLAS, Timothy A. Davis, (c) 2017-2026, All Rights Reserved.
// SPDX-License-Identifier: Apache-2.0

//------------------------------------------------------------------------------

// Constructs the Key_in data structure to pass to GB_cuda_builder, as part of
// the CUDA tranpose process in GB_cuda_transpose.

//------------------------------------------------------------------------------
// declarations
//------------------------------------------------------------------------------

// FIXME: place this "using ..." in GB_cuda_kernel.cuh
using namespace cooperative_groups ;

#include "template/GB_cuda_ek_slice.cuh"

#undef  GB_FREE_ALL
#define GB_FREE_ALL ;

#define CHUNKSIZE       GB_CUDA_TRANSPOSE_PREP_CHUNKSIZE
#define LOG2_CHUNKSIZE  GB_CUDA_TRANSPOSE_PREP_CHUNKSIZE_LOG2
#define BLOCKDIM        GB_CUDA_TRANSPOSE_PREP_BLOCKDIM

// There is no matrix C, but the Ci_is_32 is used in place of the Key type;
// it is uint32_t or uint64_t, and must match the expected type of
// the CUDA builder kernel
#define GB_KEY_TYPE GB_Ci_TYPE

struct GB_key_t
{
    GB_KEY_TYPE j ;
    GB_KEY_TYPE i ;
    GB_key_t ( ) = default ;
} ;

//------------------------------------------------------------------------------
// GB_cuda_transpose_prep_kernel: load indices from A into Key_in
//------------------------------------------------------------------------------

__global__ void GB_cuda_transpose_prep_kernel
(
    // outputs:
    GB_key_t *Key_in,       // Key_in [-1..anz-1]
    // inputs, not modified:
    GrB_Matrix A
)
{

    //--------------------------------------------------------------------------
    // get A
    //--------------------------------------------------------------------------

    const int64_t anvec = A->nvec ;
    const GB_Ap_TYPE *__restrict__ Ap = (GB_Ap_TYPE *) A->p ;
    const GB_Ai_SIGNED_TYPE *__restrict__ Ai = (GB_Ai_SIGNED_TYPE *) A->i ;
    #if ( GB_A_IS_HYPER )
    const GB_Aj_TYPE *__restrict__ Ah = (GB_Aj_TYPE *) A->h ;
    #endif
    GB_A_NHELD (anz) ;          // # of entries in A
    // # of chunks in A:
    int64_t nchunks = (anz + CHUNKSIZE - 1) >> LOG2_CHUNKSIZE ;

    if (threadIdx.x == 0 && blockIdx.x == 0)
    {
        // first thread loads the sentinel value of Key_in
        memset (&(Key_in [-1]), 0xFF, sizeof (GB_key_t)) ;
    }

    //--------------------------------------------------------------------------
    // construct Key_in with the (i,j) indices from A
    //--------------------------------------------------------------------------

    for (int64_t chunk = blockIdx.x ;
                 chunk < nchunks ;
                 chunk += gridDim.x)        // grid-stride loop
    {

        //----------------------------------------------------------------------
        // determine the chunk
        //----------------------------------------------------------------------

        int64_t pfirst = chunk << LOG2_CHUNKSIZE ;
        int64_t my_chunk_size ;
        // detemine the slope, for computing Ak and j
        int64_t anvec1, kfirst, klast ;
        float slope ;
        GB_cuda_ek_slice_setup<GB_Ap_TYPE> (Ap, anvec, anz, pfirst,
            CHUNKSIZE, &kfirst, &klast, &my_chunk_size, &anvec1, &slope) ;

        //----------------------------------------------------------------------
        // find the kA-th vector that contains each entry p = pfirst:plast-1
        //----------------------------------------------------------------------

        int64_t pdelta = threadIdx.x ;
        for ( ; pdelta < my_chunk_size ;
                pdelta += blockDim.x)       // block-stride loop
        {

            //------------------------------------------------------------------
            // this thread works on the p-th entry
            //------------------------------------------------------------------

            int64_t p = pfirst + pdelta ;

            //------------------------------------------------------------------
            // determine the indices of the p-th entry and load them into Key_in
            //------------------------------------------------------------------

            int64_t kA = GB_cuda_ek_slice_entry<GB_Ap_TYPE> (p, pdelta, Ap,
                anvec1, kfirst, slope) ;
            Key_in [p].i = GBh_A (Ah, kA) ;
            Key_in [p].j = Ai [p] ;
        }
    }
}

//------------------------------------------------------------------------------
// cuda transpose prep, host method
//------------------------------------------------------------------------------

extern "C"
{
    GB_JIT_CUDA_KERNEL_TRANSPOSE_PREP_PROTO (GB_jit_kernel) ;
}

GB_JIT_CUDA_KERNEL_TRANSPOSE_PREP_PROTO (GB_jit_kernel)
{

    //--------------------------------------------------------------------------
    // get inputs
    //--------------------------------------------------------------------------

    ASSERT (GB_A_IS_HYPER || GB_A_IS_SPARSE) ;
    GB_key_t *Key_in = ((GB_key_t *) Key_input) + 1 ;
    dim3 grid (gridsz) ;        // = min (ceil (anz/CHUNKSIZE), 256*(#sms))
    dim3 block (BLOCKDIM) ;

    //--------------------------------------------------------------------------
    // launch the kernel
    //--------------------------------------------------------------------------

    GB_cuda_transpose_prep_kernel <<<grid, block, 0, stream>>>
        (/* outputs: */ Key_in,
         /* inputs: */  A) ;
    CUDA_OK (cudaGetLastError ( )) ;
    CUDA_OK (cudaStreamSynchronize (stream)) ;

    //--------------------------------------------------------------------------
    // return result
    //--------------------------------------------------------------------------

    return (GrB_SUCCESS) ;
}

