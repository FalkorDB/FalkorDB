using namespace cooperative_groups ;

#define GB_FREE_ALL             \
{                               \
    GB_FREE_WORK (W) ;          \
    GB_FREE_WORK (W_2) ;        \
    GB_FREE_WORK (W_3) ;        \
}

#include "GB_cuda_ek_slice.cuh"
#include "GB_cuda_cumsum.cuh"

//------------------------------------------------------------------------------
// GB_cuda_select_sparse_phase1: construct Keep array
//------------------------------------------------------------------------------

// Compute Keep array
__global__ void GB_cuda_select_sparse_phase1
(
    int64_t *Keep,
    GrB_Matrix A,
    void *ythunk
)
{
    const int64_t *__restrict__ Ap = A->p ;

    #if ( GB_A_IS_HYPER )
    const int64_t *__restrict__ Ah = A->h ;
    #endif

    #if ( GB_DEPENDS_ON_X )
    const GB_A_TYPE *__restrict__ Ax = (GB_A_TYPE *) A->x ;
    #endif

    #if ( GB_DEPENDS_ON_Y )
    const GB_Y_TYPE y = * ((GB_Y_TYPE *) thunk) ;
    #endif

    GB_A_NHELD (anz) ;

    #if ( GB_DEPENDS_ON_J )

        for (int64_t pfirst = blockIdx.x << log2_chunk_size ;
                     pfirst < anz ;
                     pfirst += gridDim.x << log2_chunk_size )
        {
            int64_t my_chunk_size, anvec1, kfirst, klast ;
            float slope ;
            GB_cuda_ek_slice_setup (Ap, anvec, anz, pfirst, chunk_size,
                &kfirst, &klast, &my_chunk_size, &anvec1, &slope) ;

            for (int64_t pdelta = threadIdx.x ;
                         pdelta < my_chunk_size ;
                         pdelta += blockDim.x)
            {
                int64_t pA ;
                int64_t k = GB_cuda_ek_slice_entry (&pA, pdelta, pfirst, Ap,
                    anvec1, kfirst, slope) ;
                int64_t j = GBH_A (Ah, k) ;

                #if ( GB_DEPENDS_ON_I )
                int64_t i = Ai [pA] ;
                #endif

                // keep = fselect (A (i,j)), true if A(i,j) is kept, else false
                GB_TEST_VALUE_OF_ENTRY (keep, pA) ; // FIXME: add Ax,i,j,y
                Keep[pA] = keep;
            }
        }

    #else

        int tid = blockIdx.x * blockDim.x + threadIdx.x ;
        int nthreads = blockDim.x * gridDim.x ;

        for (int64_t pA = tid; pA < anz; pA += nthreads)
        {
            #if ( GB_DEPENDS_ON_I )
            int64_t i = Ai [pA] ;
            #endif

            GB_TEST_VALUE_OF_ENTRY (keep, pA) ;
            Keep[pA] = keep;
        }

    #endif
}

//------------------------------------------------------------------------------
// GB_cuda_select_sparse_phase2:
//------------------------------------------------------------------------------

__global__ void GB_cuda_select_sparse_phase2
(
    int64_t *Map
    GrB_Matrix A,
    int64_t *Ak_keep,
    int64_t *Ck_delta,
    int64_t *Ci,
    GB_X_TYPE *Cx
)
{
    const int64_t *__restrict__ Ap = A->p ;
    const int64_t *__restrict__ Ai = A->i ;
    const GB_A_TYPE *__restrict__ Ax = (GB_A_TYPE *) A->x ;
    
    GB_A_NHELD (anz) ;
    int64_t cnz = Map [anz - 1];

    int tid = blockIdx.x * blockDim.x + threadIdx.x ;
    int nthreads = blockDim.x * gridDim.x ;

    for (int64_t pfirst = blockIdx.x << log2_chunk_size ;
                 pfirst < anz ;
                 pfirst += gridDim.x << log2_chunk_size )
    {
        int64_t my_chunk_size, anvec1, kfirst, klast ;
        float slope ;
        GB_cuda_ek_slice_setup (Ap, anvec, anz, pfirst, chunk_size,
            &kfirst, &klast, &my_chunk_size, &anvec1, &slope) ;

        for (int64_t pdelta = threadIdx.x ;
                     pdelta < my_chunk_size ;
                     pdelta += blockDim.x)
        {
            int64_t pA = pfirst + pdelta ;
            int64_t pC = Map [pA] ;     // Note: pC is off-by-1 (see below).
            if (Map [pA-1] < pC)
            {
                // This entry is kept; Keep [pA] was 1 but the contents of the
                // Keep has been overwritten by the Map array using an
                // inclusive cumsum.  Keep [pA] (before being overwritten) is
                // identical to the expression (Map [pA-1] < pC).

                // the A(i,j) is in the kA-th vector of A:
                int64_t kA = GB_cuda_ek_slice_entry (&pA, pdelta, pfirst, Ap,
                    anvec1, kfirst, slope) ;

                // Map is offset by 1 since it was computed as an inclusive
                // cumsum, so decrement pC here to get the actual position in
                // Ci,Cx.
                pC-- ;
                Ci [pC] = Ai [pA] ;

                // Cx [pC] = Ax [pA] ;
                // Q: In iso case, this just becomes
                // #define GB_ISO_SELECT 1? I would expect
                // Cx [0] = Ax [0]
                GB_SELECT_ENTRY (pC, pA) ;

                // save the name of the vector kA that holds this entry in A,
                // for the new position of this entry in C at pC.
                Ak_keep [pC] = kA ;
            }
        }

        // Build the Delta over Ck_delta
        this_thread_block().sync();

        for (int64_t pdelta = threadIdx.x ;
                     pdelta < my_chunk_size ;
                     pdelta += blockDim.x)
        {
            int64_t pA = pfirst + pdelta ;
            int64_t pC = Map[pA] ;
            if (Map [pA-1] < pC)
            {
                pC-- ;
                Ck_delta [pC] = (Ak_keep [pC] != Ak_keep [pC - 1]) ;
            }
        }
    }
}

// HERE

__global__ void GB_cuda_select_sparse_phase3
(
    GrB_Matrix A,
    int64_t cnz,
    int64_t *Ak_keep,
    int64_t *Ck_map,
    int64_t *Cp,
    int64_t *Ch
)
{
    #if ( GB_A_IS_HYPER ) 
    const int64_t *__restrict__ Ah = A->h;
    #endif

    int tid = blockIdx.x * blockDim.x + threadIdx.x ;
    int nthreads = blockDim.x * gridDim.x ;

    for (int64_t pC = tid; pC < cnz; pC += nthreads)
    {
        if (Ck_map[pC] != Ck_map[pC - 1])
        {
            int64_t kA = Ak_keep[pC] - 1 ;
            Cp[Ck_map[pC] - 1] = pC;
            Ch[Ck_map[pC] - 1] = GBH_A (Ah, kA);
        }
    }
}

//------------------------------------------------------------------------------
// select sparse, host method
//------------------------------------------------------------------------------

extern "C"
{
    GB_JIT_CUDA_KERNEL_SELECT_SPARSE_PROTO (GB_jit_kernel) ;
}

GB_JIT_CUDA_KERNEL_SELECT_SPARSE_PROTO (GB_jit_kernel)
{

    //--------------------------------------------------------------------------
    // get callback functions
    //--------------------------------------------------------------------------

    #ifdef GB_JIT_RUNTIME
    // get callback functions
    GB_free_memory_f GB_free_memory = my_callback->GB_free_memory_func ;
    GB_malloc_memory_f GB_malloc_memory = my_callback->GB_malloc_memory_func ;
    #endif

    //--------------------------------------------------------------------------
    // check inputs and declare workspace
    //--------------------------------------------------------------------------

    int64_t *W = NULL, *W_2 = NULL, *W_3 = NULL,
        *Ak_keep = NULL, *Ck_delta = NULL,
        *Keep = NULL ;
    size_t W_size = 0, W_2_size = 0, W_3_size = 0 ;
    int64_t cnz = 0 ;

    ASSERT (GB_A_IS_HYPER || GB_A_IS_SPARSE) ;

    dim3 grid (gridsz) ;
    dim3 block (blocksz) ;

    //--------------------------------------------------------------------------
    // phase 1: determine which entries of A to keep
    //--------------------------------------------------------------------------

    // Phase 1: Keep [p] = 1 if Ai,Ax [p] is kept, 0 otherwise; then cumsum

    W = GB_MALLOC_WORK (A->nvals + 1, int64_t, &W_size) ;
    if (W == NULL)
    {
        // out of memory
        GB_FREE_ALL ;
        return (GrB_OUT_OF_MEMORY) ;
    }

    // shift by one, to define Keep [-1] as 0
    W [0] = 0;      // placeholder for easier end-condition
    Keep = W + 1 ;  // Keep has size A->nvals and starts at W [1]

    GB_cuda_select_sparse_phase1 <<<grid, block, 0, stream>>>
        (Keep, A, ythunk) ;

    CUDA_OK (cudaStreamSynchronize (stream)) ;

    //--------------------------------------------------------------------------
    // phase1b: Map = cumsum (Keep)
    //--------------------------------------------------------------------------

    // in-place cumsum, overwriting Keep with its cumsum, then becomes Map
    CUDA_OK (GB_cuda_cumsum (Keep, Keep, A->nvals, stream,
        GB_CUDA_CUMSUM_INCLUSIVE)) ;
    CUDA_OK (cudaStreamSynchronize (stream)) ;

    int64_t *Map = Keep ;             // Keep has been replaced with Map
    cnz = Map [A->nvals - 1] ;        // total # of entries kept, for C

    // Q: need to allocate space for Cx, Ci?
    // If cnz = 0, just need to do Cp [0] = 0, then done?

    // allocate workspace
    W_2 = GB_MALLOC_WORK (cnz + 1, int64_t, &W_2_size) ;
    W_3 = GB_MALLOC_WORK (cnz + 1, int64_t, &W_3_size) ;
    if (W_2 == NULL || W_3 == NULL)
    {
        // out of memory
        GB_FREE_ALL ;
        return (GrB_OUT_OF_MEMORY) ;
    }

    // shift by one: to define Ck_delta [-1] as 0
    W_2 [0] = 0 ;
    Ck_delta = W_2 + 1 ;

    // shift by one: to define Ak_keep [-1] as -1
    W_3 [0] = -1 ;
    Ak_keep = W_3 + 1 ;


    //--------------------------------------------------------------------------
    // phase 2:
    //--------------------------------------------------------------------------

    // Phase 2: Build Ci, Cx, Ak_keep, and Ck_delta
    GB_cuda_select_sparse_phase2 <<<grid, block, 0, stream>>>
        (Map, A, Ak_keep, Ck_delta, Ci, Cx) ;
    
    CUDA_OK (cudaStreamSynchronize (stream)) ;

    // Cumsum over Ck_delta array to get Ck_map
    // Can reuse `Keep` to avoid a malloc
    //--------------------------------------------------------------------------
    // phase2b: Ck_map = cumsum (Ck_delta)
    //--------------------------------------------------------------------------

    CUDA_OK (GB_cuda_cumsum (Keep, Ck_delta, cnz, stream,
        GB_CUDA_CUMSUM_INCLUSIVE)) ;
    CUDA_OK (cudaStreamSynchronize (stream)) ;

    int64_t *Ck_map = Keep;
    int64_t cnk = Ck_map [cnz - 1] ;

    // Q: Need to allocate space for Cp, Ch?

    //--------------------------------------------------------------------------
    // Phase 3: Build Cp and Ch
    //--------------------------------------------------------------------------
    GB_cuda_select_sparse_phase3 <<<grid, block, 0, stream>>>
        (A, cnz, Ak_keep, Ck_map, Cp, Ch) ;
    CUDA_OK (cudaStreamSynchronize (stream)) ;
    
    // log the end of the last vector of C
    Cp [Ck_map [cnz - 1]] = cnz;

    GB_FREE_ALL ;
    return (GrB_SUCCESS) ;
}

