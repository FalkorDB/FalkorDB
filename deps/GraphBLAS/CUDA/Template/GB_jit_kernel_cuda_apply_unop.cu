using namespace cooperative_groups ;

#include "GB_cuda_ek_slice.cuh"

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
        const int64_t *__restrict__ Ai = (int64_t *) A->i ;
        #endif

        #if ( GB_DEPENDS_ON_J )
            #if ( GB_A_IS_HYPER )
            const int64_t *__restrict__ Ah = (int64_t *) A->h ;
            #endif
        const int64_t *__restrict__ Ap = (int64_t *) A->p ;
        #endif
    #endif

    #if ( GB_A_IS_BITMAP )
    const int8_t *__restrict__ Ab = (int8_t *) A->b ;
    #endif
    
    GB_C_TYPE *__restrict__ Cx = (GB_C_TYPE *) Cx_out;

    #define A_iso GB_A_ISO

    int tid = blockDim.x * blockIdx.x + threadIdx.x ;
    int nthreads = blockDim.x * gridDim.x ;

    #if ( GB_DEPENDS_ON_Y )
        // get thunk value (of type GB_Y_TYPE)
        GB_Y_TYPE thunk_value = * ((GB_Y_TYPE *) thunk) ;
    #endif

    #if ( GB_A_IS_BITMAP || GB_A_IS_FULL )
        // bitmap/full case
        for (int64_t p = tid ; p < anz ; p += nthreads)
        {
            if (!GBB_A (Ab, p)) { continue ; }

            #if ( GB_DEPENDS_ON_I )
            int64_t row_idx = p % A->vlen ;
            #endif

            #if ( GB_DEPENDS_ON_J )
            int64_t col_idx = p / A->vlen ;
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
                    GB_cuda_ek_slice_setup (Ap, anvec, anz, pfirst, chunk_size,
                        &kfirst, &klast, &my_chunk_size, &anvec_sub1, &slope) ;

                    for (int64_t pdelta = threadIdx.x ; pdelta < my_chunk_size ; pdelta += blockDim.x)
                    {
                        int64_t p_final ;
                        int64_t k = GB_cuda_ek_slice_entry (&p_final, pdelta, pfirst, Ap, anvec_sub1, kfirst, slope) ;
                        int64_t col_idx = GBH_A (Ah, k) ;
                        
                        #if ( GB_DEPENDS_ON_I )
                        int64_t row_idx = GBI_A (Ai, p_final, A->vlen) ;
                        #endif

                        GB_UNOP (Cx, p_final, Ax, p_final, 
                            A_iso, row_idx, col_idx, thunk_value) ;
                    }
                }
        #else
            const int64_t avlen = A->vlen ;
            // can do normal method
            for (int64_t p = tid ; p < anz ; p += nthreads)
            {
                #if ( GB_DEPENDS_ON_I )
                int64_t row_idx = GBI_A (Ai, p, avlen) ;
                #endif

                GB_UNOP (Cx, p, Ax, p, A_iso, row_idx, /* col_idx */, thunk_value) ;  
            }
        #endif
    #endif
}

extern "C" {
    GB_JIT_CUDA_KERNEL_APPLY_UNOP_PROTO (GB_jit_kernel) ;
}

GB_JIT_CUDA_KERNEL_APPLY_UNOP_PROTO (GB_jit_kernel)
{
    dim3 grid (gridsz) ;
    dim3 block (blocksz) ;

    GB_cuda_apply_unop_kernel <<<grid, block, 0, stream>>> (Cx, ythunk, A) ;

    return (GrB_SUCCESS) ;
}
