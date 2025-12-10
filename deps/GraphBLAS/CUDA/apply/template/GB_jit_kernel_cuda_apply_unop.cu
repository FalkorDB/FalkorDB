#define GB_FREE_ALL ;

using namespace cooperative_groups ;

#include "template/GB_cuda_ek_slice.cuh"

#define log2_chunk_size 10
#define chunk_size 1024

__global__ void GB_cuda_apply_unop_kernel
(
    GB_void *Cx_out,
    const GB_void *thunk,
    GrB_Matrix A
)
{
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

    #if ( GB_A_IS_BITMAP || GB_A_IS_FULL )

        // bitmap/full case
        int tid = blockDim.x * blockIdx.x + threadIdx.x ;
        int nthreads = blockDim.x * gridDim.x ;
        #if ( GB_DEPENDS_ON_I ) || ( GB_DEPENDS_ON_J )
        const int64_t avlen = A->vlen ;
        #endif
        for (int64_t p = tid ; p < anz ; p += nthreads)
        {
            if (!GBb_A (Ab, p)) { continue ; }
            #if ( GB_DEPENDS_ON_I )
            int64_t row_idx = p % avlen ;
            #endif
            #if ( GB_DEPENDS_ON_J )
            int64_t col_idx = p / avlen ;
            #endif
            GB_UNOP (Cx, p, Ax, p, A_iso, row_idx, col_idx, thunk_value) ;
        }

    #else

        // sparse/hypersparse case
        #if ( GB_DEPENDS_ON_J )
            const int64_t anvec = A->nvec ;
            // need to do ek_slice method
            for (int64_t pfirst = blockIdx.x << log2_chunk_size ;
                        pfirst < anz ;
                        pfirst += gridDim.x << log2_chunk_size )
                {
                    int64_t my_chunk_size, anvec_sub1, kfirst, klast ;
                    float slope ;
                    GB_cuda_ek_slice_setup<GB_Ap_TYPE> (Ap, anvec, anz, pfirst, chunk_size,
                        &kfirst, &klast, &my_chunk_size, &anvec_sub1, &slope) ;
                    for (int64_t pdelta = threadIdx.x ; pdelta < my_chunk_size ; pdelta += blockDim.x)
                    {
                        int64_t p = pfirst + pdelta ;
                        int64_t k = GB_cuda_ek_slice_entry<GB_Ap_TYPE> (p, pdelta, Ap, anvec_sub1, kfirst, slope) ;
                        int64_t col_idx = GBh_A (Ah, k) ;
                        #if ( GB_DEPENDS_ON_I )
                        int64_t row_idx = Ai [p] ;
                        #endif
                        GB_UNOP (Cx, p, Ax, p, A_iso, row_idx, col_idx, thunk_value) ;
                    }
                }
        #else
            // can do normal method
            int tid = blockDim.x * blockIdx.x + threadIdx.x ;
            int nthreads = blockDim.x * gridDim.x ;
            for (int64_t p = tid ; p < anz ; p += nthreads)
            {
                #if ( GB_DEPENDS_ON_I )
                int64_t row_idx = Ai [p] ;
                #endif
                GB_UNOP (Cx, p, Ax, p, A_iso, row_idx, /* col_idx unused */, thunk_value) ;
            }
        #endif
    #endif
}

extern "C" {
    GB_JIT_CUDA_KERNEL_APPLY_UNOP_PROTO (GB_jit_kernel) ;
}

GB_JIT_CUDA_KERNEL_APPLY_UNOP_PROTO (GB_jit_kernel)
{
    GB_GET_CALLBACKS ;
    dim3 grid (gridsz) ;
    dim3 block (blocksz) ;

    GB_A_NHELD (anz) ;
    if (anz == 0) return (GrB_SUCCESS) ;

    CUDA_OK (cudaGetLastError ( )) ;
    CUDA_OK (cudaStreamSynchronize (stream)) ;
    GB_cuda_apply_unop_kernel <<<grid, block, 0, stream>>> (Cx, ythunk, A) ;
    CUDA_OK (cudaGetLastError ( )) ;
    CUDA_OK (cudaStreamSynchronize (stream)) ;

    return (GrB_SUCCESS) ;
}
