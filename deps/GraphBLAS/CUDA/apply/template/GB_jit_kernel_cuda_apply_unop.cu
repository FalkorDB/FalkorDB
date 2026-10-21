//------------------------------------------------------------------------------
// CUDA/apply/template/GB_jit_kernel_cuda_apply_unop
//------------------------------------------------------------------------------

// SuiteSparse:GraphBLAS, Timothy A. Davis, (c) 2017-2026, All Rights Reserved.
// SPDX-License-Identifier: Apache-2.0

//------------------------------------------------------------------------------

#define GB_FREE_ALL ;

#include "template/GB_cuda_ek_slice.cuh"

//------------------------------------------------------------------------------
// GB_cuda_apply_unop_kernel: device kernel for unary apply
//------------------------------------------------------------------------------

__global__ void GB_cuda_apply_unop_kernel
(
    GB_void *Cx_out,
    const GB_void *thunk,
    GrB_Matrix A
)
{

    //--------------------------------------------------------------------------
    // get A, Cx, and thunk
    //--------------------------------------------------------------------------

    GB_A_NHELD (anz) ;

    #if ( GB_DEPENDS_ON_X )
    const GB_A_TYPE *__restrict__ Ax = (GB_A_TYPE *) A->x ;
    #endif

    #if ( GB_A_IS_SPARSE || GB_A_IS_HYPER )
        #if ( GB_DEPENDS_ON_I )
        const GB_Ai_TYPE *__restrict__ Ai = (GB_Ai_TYPE *) A->i ;
        #endif

        #if ( GB_DEPENDS_ON_J )
            #if ( GB_A_IS_HYPER )
            const GB_Aj_TYPE *__restrict__ Ah = (GB_Aj_TYPE *) A->h ;
            #endif
        const GB_Ap_TYPE *__restrict__ Ap = (GB_Ap_TYPE *) A->p ;
        #endif
    #endif

    #if ( GB_A_IS_BITMAP )
    const int8_t *__restrict__ Ab = (int8_t *) A->b ;
    #endif

    GB_C_TYPE *__restrict__ Cx = (GB_C_TYPE *) Cx_out;

    #define A_iso GB_A_ISO

    #if ( GB_DEPENDS_ON_Y )
        // get thunk value (of type GB_Y_TYPE)
        GB_Y_TYPE thunk_value = * ((GB_Y_TYPE *) thunk) ;
    #endif

    //--------------------------------------------------------------------------
    // apply the unary operator
    //--------------------------------------------------------------------------

    #if ( GB_A_IS_BITMAP || GB_A_IS_FULL )

        //----------------------------------------------------------------------
        // bitmap/full case
        //----------------------------------------------------------------------

        int tid = blockDim.x * blockIdx.x + threadIdx.x ;
        int nthreads = blockDim.x * gridDim.x ;
        #if ( GB_DEPENDS_ON_I ) || ( GB_DEPENDS_ON_J )
        const int64_t avlen = A->vlen ;
        #endif
        for (int64_t p = tid ; p < anz ; p += nthreads)
        {
            if (!GBb_A (Ab, p)) { continue ; }
            #if ( GB_DEPENDS_ON_I )
            int64_t i = p % avlen ;
            #endif
            #if ( GB_DEPENDS_ON_J )
            int64_t j = p / avlen ;
            #endif
            GB_UNOP (Cx, p, Ax, p, A_iso, i, j, thunk_value) ;
        }

    #else

        //----------------------------------------------------------------------
        // sparse/hypersparse case
        //----------------------------------------------------------------------

        #if ( GB_DEPENDS_ON_J )
            // operator depends on j; need to do ek_slice method
            const int64_t anvec = A->nvec ;
            const int64_t anvec1 = anvec - 1 ;
            for (int64_t pfirst = blockIdx.x << GB_CUDA_APPLY_CHUNKSIZE_LOG2 ;
                         pfirst < anz ;
                         pfirst += gridDim.x << GB_CUDA_APPLY_CHUNKSIZE_LOG2 )
            {
                int64_t my_chunk_size, kfirst ;
                float slope ;
                GB_cuda_ek_slice_setup<GB_Ap_TYPE> (Ap, anvec, anz, pfirst,
                    GB_CUDA_APPLY_CHUNKSIZE, &kfirst, &my_chunk_size, &slope) ;
                for (int64_t pdelta = threadIdx.x ;
                             pdelta < my_chunk_size ;
                             pdelta += blockDim.x)
                {
                    int64_t p = pfirst + pdelta ;
                    int64_t k = GB_cuda_ek_slice_entry<GB_Ap_TYPE> (p, pdelta,
                        Ap, anvec1, kfirst, slope) ;
                    int64_t j = GBh_A (Ah, k) ;
                    #if ( GB_DEPENDS_ON_I )
                    int64_t i = Ai [p] ;
                    #endif
                    GB_UNOP (Cx, p, Ax, p, A_iso, i, j, thunk_value) ;
                }
            }

        #else

            // operator does not require j
            int tid = blockDim.x * blockIdx.x + threadIdx.x ;
            int nthreads = blockDim.x * gridDim.x ;
            for (int64_t p = tid ; p < anz ; p += nthreads)
            {
                #if ( GB_DEPENDS_ON_I )
                int64_t i = Ai [p] ;
                #endif
                GB_UNOP (Cx, p, Ax, p, A_iso, i, /* j unused */, thunk_value) ;
            }

        #endif
    #endif
}

//------------------------------------------------------------------------------
// host CUDA JIT kernel for unary apply
//------------------------------------------------------------------------------

extern "C" {
    GB_JIT_CUDA_KERNEL_APPLY_UNOP_PROTO (GB_jit_kernel) ;
}

GB_JIT_CUDA_KERNEL_APPLY_UNOP_PROTO (GB_jit_kernel)
{
    GB_GET_CALLBACKS ;

    CUDA_OK (cudaSetDevice (device)) ;
    dim3 grid (gridsz) ;
    dim3 block (GB_CUDA_APPLY_BLOCKDIM) ;

    GB_A_NHELD (anz) ;
    if (anz == 0) return (GrB_SUCCESS) ;

    CUDA_OK (cudaGetLastError ( )) ;
    CUDA_OK (cudaStreamSynchronize (stream)) ;
    GB_cuda_apply_unop_kernel <<<grid, block, 0, stream>>> (Cx, ythunk, A) ;
    CUDA_OK (cudaGetLastError ( )) ;
    CUDA_OK (cudaStreamSynchronize (stream)) ;

    return (GrB_SUCCESS) ;
}

