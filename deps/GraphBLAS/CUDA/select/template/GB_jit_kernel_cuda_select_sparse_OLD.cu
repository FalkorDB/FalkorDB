using namespace cooperative_groups ;

#include "template/GB_cuda_ek_slice.cuh"
#include "template/GB_cuda_cumsum.cuh"

#define GB_FREE_WORKSPACE                   \
{                                           \
    GB_FREE_MEMORY (&W, W_size) ;           \
    GB_FREE_MEMORY (&W_2, W_2_size) ;       \
    GB_FREE_MEMORY (&W_3, W_3_size) ;       \
}

#undef GB_FREE_ALL
#define GB_FREE_ALL GB_FREE_WORKSPACE ;

#define chunk_size      GB_CUDA_SELECT_SPARSE_CHUNKSIZE1
#define log2_chunk_size GB_CUDA_SELECT_SPARSE_CHUNKSIZE1_LOG2

//------------------------------------------------------------------------------
// GB_cuda_select_sparse_phase1: construct Keep array
//------------------------------------------------------------------------------

// Compute Keep array
__global__ void GB_cuda_select_sparse_phase1
(
    int64_t *Keep,
    GrB_Matrix A,
    const void *ythunk
)
{
    #if ( GB_DEPENDS_ON_I )
    const GB_Ai_SIGNED_TYPE *__restrict__ Ai = (GB_Ai_SIGNED_TYPE *) A->i ;
    #endif

    #if ( GB_DEPENDS_ON_J )
        #if ( GB_A_IS_HYPER )
        const GB_Aj_TYPE *__restrict__ Ah = (GB_Aj_TYPE *) A->h ;
        #endif
    const GB_Ap_TYPE *__restrict__ Ap = (GB_Ap_TYPE *) A->p ;
    #endif

    #if ( GB_DEPENDS_ON_X )
    const GB_A_TYPE *__restrict__ Ax = (GB_A_TYPE *) A->x ;
    #endif

    #if ( GB_DEPENDS_ON_Y )
    const GB_Y_TYPE y = * ((GB_Y_TYPE *) ythunk) ;
    #endif
    GB_A_NHELD (anz) ;

    #if ( GB_DEPENDS_ON_J )
        const int64_t anvec = A->nvec ;
        for (int64_t pfirst = blockIdx.x << log2_chunk_size ;
                     pfirst < anz ;
                     pfirst += gridDim.x << log2_chunk_size )
        {
            int64_t my_chunk_size, anvec1, kfirst, klast ;
            float slope ;
            GB_cuda_ek_slice_setup<GB_Ap_TYPE> (Ap, anvec, anz, pfirst, chunk_size,
                &kfirst, &klast, &my_chunk_size, &anvec1, &slope) ;

            for (int64_t pdelta = threadIdx.x ;
                         pdelta < my_chunk_size ;
                         pdelta += blockDim.x)
            {
                int64_t pA = pfirst + pdelta ;
                int64_t k = GB_cuda_ek_slice_entry<GB_Ap_TYPE> (pA, pdelta, Ap, anvec1, kfirst, slope) ;
                int64_t j = GBh_A (Ah, k) ;
                // Ak [pA] = k ;        // save the kth vector containing the pA-th entry

                #if ( GB_DEPENDS_ON_I )
                int64_t i = Ai [pA] ;
                #endif

                // keep = fselect (A (i,j)), true if A(i,j) is kept, else false
                GB_TEST_VALUE_OF_ENTRY (keep, pA) ;
                // FIXME: do not save in Keep array: make it threadlocal,
                // of size my_chunk_size
                Keep [pA] = keep ;
            }

            // HERE: do a cub::BlockScan on this threadblock's threadlocal Keep
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
            Keep [pA] = keep ;
        }

    #endif
}

//------------------------------------------------------------------------------
// GB_cuda_select_sparse_phase2:
//------------------------------------------------------------------------------

__global__ void GB_cuda_select_sparse_phase2
(
    int64_t *Map,
    GrB_Matrix A,
    int64_t *Ak_keep,
    GB_Ci_TYPE *Ci,
    GB_C_TYPE *Cx
)
{
    const GB_Ap_TYPE *__restrict__ Ap = (GB_Ap_TYPE *) A->p ;
    const GB_Ai_TYPE *__restrict__ Ai = (GB_Ai_TYPE *) A->i ;
    #if (!GB_ISO_SELECT)
    const GB_A_TYPE *__restrict__ Ax = (GB_A_TYPE *) A->x ;
    #endif
    GB_A_NHELD (anz) ;

    const int64_t anvec = A->nvec ;

    for (int64_t pfirst = blockIdx.x << log2_chunk_size ;
                 pfirst < anz ;
                 pfirst += gridDim.x << log2_chunk_size )
    {
        int64_t my_chunk_size, anvec1, kfirst, klast ;
        float slope ;
        GB_cuda_ek_slice_setup<GB_Ap_TYPE> (Ap, anvec, anz, pfirst, chunk_size,
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
                int64_t kA = GB_cuda_ek_slice_entry<GB_Ap_TYPE> (pA, pdelta, Ap, anvec1, kfirst, slope) ;

                // Map is offset by 1 since it was computed as an inclusive
                // cumsum, so decrement pC here to get the actual position in
                // Ci,Cx.
                pC-- ;
                Ci [pC] = Ai [pA] ;

                // Cx [pC] = Ax [pA] ;
                GB_SELECT_ENTRY (Cx, pC, Ax, pA) ;

                // save the name of the vector kA that holds this entry in A,
                // for the new position of this entry in C at pC.
                Ak_keep [pC] = kA ;
            }
        }
    }
}

__global__ void GB_cuda_select_sparse_phase3
(
    int64_t anz,
    int64_t *Map,
    int64_t *Ak_keep,
    int64_t *Ck_delta
)
{
    int tid = blockIdx.x * blockDim.x + threadIdx.x ;
    int nthreads = blockDim.x * gridDim.x ;

    for (int64_t pA = tid; pA < anz; pA += nthreads)
    {
        int64_t pC = Map [pA] ;
        if (Map [pA-1] < pC)
        {
            pC-- ;
            Ck_delta [pC] = (Ak_keep [pC] != Ak_keep [pC - 1]) ;
        }
    }
}

__global__ void GB_cuda_select_sparse_phase4
(
    GrB_Matrix A,
    int64_t cnz,
    int64_t *Ak_keep,
    int64_t *Ck_map,
    GB_Cp_TYPE *Cp,
    GB_Cj_TYPE *Ch
)
{
    #if ( GB_A_IS_HYPER ) 
    const GB_Aj_TYPE *__restrict__ Ah = (GB_Aj_TYPE *) A->h;
    #endif

    int tid = blockIdx.x * blockDim.x + threadIdx.x ;
    int nthreads = blockDim.x * gridDim.x ;

    for (int64_t pC = tid; pC < cnz; pC += nthreads)
    {
        if (Ck_map [pC] != Ck_map [pC - 1])
        {
            int64_t kA = Ak_keep [pC] ;
            Cp [Ck_map[pC] - 1] = pC ;
            Ch [Ck_map[pC] - 1] = GBh_A (Ah, kA) ;
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
    GB_GET_CALLBACKS ;
    GB_free_memory_f GB_free_memory = my_callback->GB_free_memory_func ;
    GB_malloc_memory_f GB_malloc_memory = my_callback->GB_malloc_memory_func ;
    GB_bix_alloc_f GB_bix_alloc = my_callback->GB_bix_alloc_func ;
    #endif

    //--------------------------------------------------------------------------
    // check inputs and declare workspace
    //--------------------------------------------------------------------------
    GrB_Info info ;
    int64_t *W = NULL, *W_2 = NULL, *W_3 = NULL,
        *Ak_keep = NULL, *Ck_delta = NULL,
        *Keep = NULL ;
    size_t W_size = 0, W_2_size = 0, W_3_size = 0 ;
    int64_t cnz = 0 ;

    ASSERT (GB_A_IS_HYPER || GB_A_IS_SPARSE) ;

    dim3 grid (gridsz) ;        // = min (ceil (nnz(A)/chunk_size), 256*(#sms))
    dim3 block (GB_CUDA_SELECT_SPARSE_BLOCKDIM1) ;

//  std::cout << std::endl << "--------start select sparse----" << std::endl ;
    CUDA_OK (cudaGetLastError ( )) ;    //FIXME: remove
    CUDA_OK (cudaStreamSynchronize (stream)) ;  //FIXME: remove
    CUDA_OK (cudaGetLastError ( )) ;    //FIXME: remove

    //--------------------------------------------------------------------------
    // phase 1: determine which entries of A to keep
    //--------------------------------------------------------------------------

    // Phase 1: Keep [p] = 1 if Ai,Ax [p] is kept, 0 otherwise; then cumsum

    W = (int64_t *) GB_MALLOC_MEMORY (A->nvals + 1, sizeof (int64_t), &W_size) ;
    if (W == NULL)
    {
        // out of memory
        GB_FREE_ALL ;
//      std::cout << std::endl << "------malloc dies----" << std::endl ;
        return (GrB_OUT_OF_MEMORY) ;
    }

    // shift by one, to define Keep [-1] as 0
    W [0] = 0;      // placeholder for easier end-condition
    Keep = W + 1 ;  // Keep has size A->nvals and starts at W [1]

//  std::cout << std::endl << "----------------------------------" << std::endl ;
//  std::cout << "Grid size: " << gridsz ; //FIXME: remove
    CUDA_OK (cudaGetLastError ( )) ;    //FIXME: remove
    CUDA_OK (cudaStreamSynchronize (stream)) ;  //FIXME: remove
    CUDA_OK (cudaGetLastError ( )) ;    //FIXME: remove

    // KERNEL LAUNCH 1
    GB_cuda_select_sparse_phase1 <<<grid, block, 0, stream>>>
        (Keep, A, ythunk) ;
    CUDA_OK (cudaGetLastError ( )) ;
    CUDA_OK (cudaStreamSynchronize (stream)) ;

    //--------------------------------------------------------------------------
    // phase1b: Map = cumsum (Keep)
    //--------------------------------------------------------------------------

    // in-place cumsum, overwriting Keep with its cumsum, then becomes Map
    // KERNEL LAUNCH 2,3
    GB_OK (GB_cuda_cumsum (Keep, Keep, A->nvals, stream,
        GB_CUDA_CUMSUM_INCLUSIVE, my_callback)) ;
    CUDA_OK (cudaStreamSynchronize (stream)) ;

    int64_t *Map = Keep ;             // Keep has been replaced with Map
    cnz = Map [A->nvals - 1] ;        // total # of entries kept, for C

    GB_OK (GB_bix_alloc (C, cnz, GxB_HYPERSPARSE, false, true, GB_ISO_SELECT)) ;
    C->nvals = cnz ;

    if (cnz == 0) {
        // C is empty; nothing more to do
//      printf ("C is empty, iso %d\n", C->iso) ;
        GB_FREE_WORKSPACE ;
        return (GrB_SUCCESS) ;
    }

    // allocate workspace
    W_2 = (int64_t *) GB_MALLOC_MEMORY (cnz + 1, sizeof (int64_t), &W_2_size) ;
    W_3 = (int64_t *) GB_MALLOC_MEMORY (cnz + 1, sizeof (int64_t), &W_3_size) ;
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
    // Phase 2: Build Ci, Cx, and Ak_keep
    //--------------------------------------------------------------------------
    
    // KERNEL LAUNCH 4
    GB_cuda_select_sparse_phase2 <<<grid, block, 0, stream>>>
        (Map, A, Ak_keep, (GB_Ci_TYPE *) C->i, (GB_C_TYPE *) C->x) ;
    CUDA_OK (cudaGetLastError ( )) ;
    CUDA_OK (cudaStreamSynchronize (stream)) ;

    //--------------------------------------------------------------------------
    // phase 3:
    //--------------------------------------------------------------------------

    // Phase 3a: Build Ck_delta
    // KERNEL LAUNCH 5
    GB_cuda_select_sparse_phase3 <<<grid, block, 0, stream>>>
        (A->nvals, Map, Ak_keep, Ck_delta) ;
    CUDA_OK (cudaGetLastError ( )) ;
    CUDA_OK (cudaStreamSynchronize (stream)) ;

    // Cumsum over Ck_delta array to get Ck_map
    // Can reuse `Keep` to avoid a malloc

    // Phase 3b: Ck_map = cumsum (Ck_delta)
    // KERNEL LAUNCH 6,7
    GB_OK (GB_cuda_cumsum (Keep, Ck_delta, cnz, stream,
        GB_CUDA_CUMSUM_INCLUSIVE, my_callback)) ;

    CUDA_OK (cudaStreamSynchronize (stream)) ;

    int64_t *Ck_map = Keep;
    int64_t cnvec = Ck_map [cnz - 1] ;

    // The caller has already allocated C->p, C->h for
    // a user-returnable empty hypersparse matrix.
    // Free them here before updating.
    GB_FREE_MEMORY (&(C->p), C->p_size) ;
    GB_FREE_MEMORY (&(C->h), C->h_size) ;

    // Allocate Cp, Ch, finalize matrix 
    C->plen = cnvec ;
    C->nvec = cnvec ;
    C->nvec_nonempty = cnvec ;  // FIXME
    C->nvals = cnz ;
    C->p = (GB_Cp_TYPE *) GB_MALLOC_MEMORY (C->plen + 1, sizeof (GB_Cp_TYPE),
        &(C->p_size)) ;
    C->h = (GB_Cj_TYPE *) GB_MALLOC_MEMORY (C->plen, sizeof (GB_Cj_TYPE),
        &(C->h_size)) ;
    if (C->p == NULL || C->h == NULL)
    {
        // The contents of C will be freed with GB_phybix_free()
        // in the caller (GB_cuda_select_sparse()) upon returning
        // an error.
        GB_FREE_ALL ;
        return (GrB_OUT_OF_MEMORY) ;
    }
    GB_Cp_TYPE *Cp = (GB_Cp_TYPE *) C->p ;

    //--------------------------------------------------------------------------
    // Phase 3: Build Cp and Ch
    //--------------------------------------------------------------------------
    // KERNEL LAUNCH 8
    GB_cuda_select_sparse_phase4 <<<grid, block, 0, stream>>>
        (A, cnz, Ak_keep, Ck_map, Cp, (GB_Cj_TYPE *) C->h) ;
    CUDA_OK (cudaGetLastError ( )) ;
    CUDA_OK (cudaStreamSynchronize (stream)) ;

    // log the end of the last vector of C
    // FIXME: isn't this just Cp [cnvec] = cnz?
    Cp [Ck_map [cnz - 1]] = cnz;

    GB_FREE_ALL ;
    return (GrB_SUCCESS) ;
}

