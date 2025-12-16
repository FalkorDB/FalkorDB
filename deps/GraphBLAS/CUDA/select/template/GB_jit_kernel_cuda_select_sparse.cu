//------------------------------------------------------------------------------
// GraphBLAS/CUDA/template/GB_jit_kernel_cuda_select_sparse
//------------------------------------------------------------------------------

// SuiteSparse:GraphBLAS, Timothy A. Davis, (c) 2017-2025, All Rights Reserved.
// SPDX-License-Identifier: Apache-2.0

//------------------------------------------------------------------------------

#define TIMING /* FIXME:remove */
#define Ak_SAVE 0

// C = select (A) kernel on the GPU.  The input matrix A may be jumbled; if
// so, and if C is not empty, then C->jumbled is set below.  The algorithm
// breaks down into 6 phases, 2 on the CPU and 4 on the GPU:
//
// phase1 (GPU): all entries in A are scanned, to determine if they are kept.
//      This phase constructs Ak, so that (Ai,Ak,Ax) is a triplet form of the
//      input matrix.  It also constructs a Map array, which tells each entry
//      in A where it needs to appear in C (after a cumulative sum).  Time
//      taken is O(nnz(A) + (nnz(A)/chunksize)*log2(nvec(A))), where the latter
//      is due to GB_cuda_ek_slice_setup.
//
// phase2 (CPU): a cumulative sum of the # of entries kept in each chunk of A.
//      Time taken is O(nnz(A)/chunksize).  This work could also be done on
//      the GPU, if there are many chunks.
//
// phase3 (GPU): entries to keep are copied from A into C, and Ck1 is
//      constructed, where Ck1 [pC] = kA if the pC-th entry of C comes from the
//      kA-th vector of A.  This is the same as Ck0.  Time taken is O(nnz(A)).
//
// phase4 (GPU): determine where each vector of C starts in Ci,Cx, by computing
//      Ck_Delta and its cumulative sum (per chunk of C); also computes the
//      ChunkSum, which tells how many vectors start in any given chunk of C.
//      Time taken is O(nnz(C)), where nnz(C) <= nnz(A).
//
// phase5 (CPU): a cumulative sum of the # of vectors that start in each chunk
//      of C.  Time taken is O(nnz(C)/chunksize).  This work could also be done
//      on the GPU, if there are many chunks.
//
// phase6 (GPU): construct Cp and Ch, from Ck_Delta, Ck, and Ck0.
//      Time taken is O(nnz(C)).
//
// Each phase is described with a small example below.  Currently, phase2 and
// phase5 are single-threaded, but can easily be done in parallel on either the
// CPU (using GB_cumsum) or on the GPU with a device-wide cumulative sum.

using namespace cooperative_groups ;

#include "template/GB_cuda_ek_slice.cuh"

// FIXME: put the following elsewhere, say GB_cuda_kernel.cuh:
#include <cub/cub.cuh>
#ifdef TIMING
#include "omp.h"
#endif

#define GB_FREE_WORKSPACE               \
{                                       \
    GB_FREE_MEMORY (&W_0, W_0_size) ;   \
    GB_FREE_MEMORY (&W_1, W_1_size) ;   \
    GB_FREE_MEMORY (&W_2, W_2_size) ;   \
    GB_FREE_MEMORY (&W_3, W_3_size) ;   \
}

#undef  GB_FREE_ALL
#define GB_FREE_ALL GB_FREE_WORKSPACE ;

// The chunk size is assumed to be < UINT16_MAX (65535), so that the cumsums
// can be done in an Int workspace array.  If the chunk size exceeds this
// value, then the Int type below must be replaced with a larger type.
#define CHUNKSIZE1         GB_CUDA_SELECT_SPARSE_CHUNKSIZE1
#define LOG2_CHUNKSIZE1    GB_CUDA_SELECT_SPARSE_CHUNKSIZE1_LOG2
#define BLOCKDIM1          GB_CUDA_SELECT_SPARSE_BLOCKDIM1

#define CHUNKSIZE2         GB_CUDA_SELECT_SPARSE_CHUNKSIZE2
#define LOG2_CHUNKSIZE2    GB_CUDA_SELECT_SPARSE_CHUNKSIZE2_LOG2
#define BLOCKDIM2          GB_CUDA_SELECT_SPARSE_BLOCKDIM2

// the # of items per thread must be an integer, so the chunk size must be a
// multiple of blockdim (the # of threads in each threadblock):
#define ITEMS_PER_THREAD1    (CHUNKSIZE1 / BLOCKDIM1)
#define ITEMS_PER_THREAD2    (CHUNKSIZE2 / BLOCKDIM2)

// Int can be uint16_t if CHUNKSIZE1 and CHUNKSIZE2 are both < 65,535:
#define Int uint16_t

//------------------------------------------------------------------------------
// GB_cuda_select_sparse_phase1: determine which entries in A to keep
//------------------------------------------------------------------------------

__global__ void GB_cuda_select_sparse_phase1
(
    // outputs:
    #if Ak_SAVE
    GB_Aj_SIGNED_TYPE *Ak,  // size anz, in Ak [0..anz-1], and values in range
                            // 0 to the # of vectors in A
    #endif
    Int *Map,               // size anz+1, in Map [-1..anz-1]
    GB_Ap_TYPE *ChunkSum,   // size nchunks_in_A+1, ChunkSum [-1..nchunks_in_A]
    // inputs, not modified:
    GrB_Matrix A,
    const void *ythunk,
    int64_t anz,            // # of entries in A
    int64_t nchunks_in_A    // # of chunks in A
)
{

    //--------------------------------------------------------------------------
    // get A and ythunk
    //--------------------------------------------------------------------------

    #if ( Ak_SAVE ) || ( GB_DEPENDS_ON_J )
    const int64_t anvec = A->nvec ;
    const GB_Ap_TYPE *__restrict__ Ap = (GB_Ap_TYPE *) A->p ;
    #endif
    #if ( GB_DEPENDS_ON_I )
    const GB_Ai_SIGNED_TYPE *__restrict__ Ai = (GB_Ai_SIGNED_TYPE *) A->i ;
    #endif
    #if ( GB_DEPENDS_ON_J) && ( GB_A_IS_HYPER )
    const GB_Aj_TYPE *__restrict__ Ah = (GB_Aj_TYPE *) A->h ;
    #endif
    #if ( GB_DEPENDS_ON_X )
    const GB_A_TYPE *__restrict__ Ax = (GB_A_TYPE *) A->x ;
    #endif
    #if ( GB_DEPENDS_ON_Y )
    const GB_Y_TYPE y = * ((GB_Y_TYPE *) ythunk) ;
    #endif

    //--------------------------------------------------------------------------
    // workspace for each threadblock
    //--------------------------------------------------------------------------

    __shared__ Int Local_Map [CHUNKSIZE1] ;

    // cub::Block* workspace:
    using BlockLoad  = cub::BlockLoad  <Int, BLOCKDIM1, ITEMS_PER_THREAD1> ;
    using BlockScan  = cub::BlockScan  <Int, BLOCKDIM1,
                                             cub::BLOCK_SCAN_WARP_SCANS> ;
    using BlockStore = cub::BlockStore <Int, BLOCKDIM1, ITEMS_PER_THREAD1> ;
    __shared__ union
    {
        typename BlockLoad::TempStorage load ;
        typename BlockScan::TempStorage scan ;
        typename BlockStore::TempStorage store ;
    } W ;

    //--------------------------------------------------------------------------
    // compute Ak, and each local chunk of Map
    //--------------------------------------------------------------------------

    for (int64_t chunk = blockIdx.x ;
                 chunk < nchunks_in_A ;
                 chunk += gridDim.x)        // "grid-stride" loop
    {

        //----------------------------------------------------------------------
        // determine the chunk and its slope
        //----------------------------------------------------------------------

        int64_t pfirst = chunk << LOG2_CHUNKSIZE1 ;
        int64_t my_chunk_size ;
        #if ( Ak_SAVE ) || ( GB_DEPENDS_ON_J )
        int64_t anvec1, kfirst, klast ;
        float slope ;
        GB_cuda_ek_slice_setup<GB_Ap_TYPE> (Ap, anvec, anz, pfirst,
            CHUNKSIZE1, &kfirst, &klast, &my_chunk_size, &anvec1, &slope) ;
        #else
        int64_t plast = pfirst + CHUNKSIZE1 ;
        plast = GB_IMIN (plast, anz) ;
        my_chunk_size = plast - pfirst ;
        #endif

        //----------------------------------------------------------------------
        // find the kA-th vector that contains each entry pA = pfirst:plast-1
        //----------------------------------------------------------------------

        int64_t pdelta = threadIdx.x ;
        for ( ; pdelta < my_chunk_size ;
                pdelta += blockDim.x)       // "block-stride" loop
        {

            //------------------------------------------------------------------
            // determine the kA-th vector that contains the pA-th entry
            //------------------------------------------------------------------

            int64_t pA = pfirst + pdelta ;
            #if ( Ak_SAVE ) || ( GB_DEPENDS_ON_J )
            int64_t kA = GB_cuda_ek_slice_entry<GB_Ap_TYPE> (pA, pdelta, Ap,
                anvec1, kfirst, slope) ;
            #endif

            //------------------------------------------------------------------
            // save the vector index kA, and determine if this entry is kept
            //------------------------------------------------------------------

            #if Ak_SAVE
            // FIXME: try recomputing Ak, not saving it
            Ak [pA] = kA ;
            #endif
            #if ( GB_DEPENDS_ON_J )
            int64_t j = GBh_A (Ah, kA) ;
            #endif
            #if ( GB_DEPENDS_ON_I )
            int64_t i = Ai [pA] ;
            #endif
            // keep = fselect (A (i,j)), 1 if A(i,j) is kept, else 0
            GB_TEST_VALUE_OF_ENTRY (keep, pA) ;
            Local_Map [pdelta] = keep ;
        }

        // clear the unused part of the Local_Map
        for ( ; pdelta < CHUNKSIZE1 ;
                pdelta += blockDim.x)
        {
            Local_Map [pdelta] = 0 ;
        }

        //----------------------------------------------------------------------
        // inclusive cumulative sum of Local_Map
        //----------------------------------------------------------------------

        // Map [pfirst..pfirst+CHUNKSIZE1-1] = inclusive cumsum of Local_Map,
        // where Local_Map [i] = sum (Local_Map [0:i]).  This entire phase
        // computes the following::
        /*
            for (int i = 1 ; i < CHUNKSIZE1 ; i++)
            {
                Local_Map [i] += Local_Map [i-1] ;
            }
            Map [pfirst + 0:CHUNKSIZE1-1] = Local_Map [0:CHUNKSIZE1-1]
            block_aggregate = Local_Map [CHUNKSIZE1-1]
            ChunkSum [chunk] = block_aggregate ;
        */

        this_thread_block ( ).sync ( ) ;
        Int t [ITEMS_PER_THREAD1] ;

        // each thread loads its data from Local_Map (in shared memory):
        /*
            for (int k = 0 ; k < ITEMS_PER_THREAD1 ; k++)
            {
                t [k] = Local_Map [ITEMS_PER_THREAD1 * threadIdx.x + k] ;
            }
        */
        BlockLoad (W.load).Load (Local_Map, t) ;
        this_thread_block ( ).sync ( ) ;

        // inclusive sum of data from Local_Map,
        // where Local_Map [i] = sum (Local_Map [0:i])
        Int block_aggregate ;
        BlockScan (W.scan).InclusiveSum (t, t, block_aggregate) ;
        this_thread_block ( ).sync ( ) ;

        // each thread saves its data into Map (in global memory):
        /*
            for (int k = 0 ; k < ITEMS_PER_THREAD1 ; k++)
            {
                Map [pfirst + ITEMS_PER_THREAD1 * threadIdx.x + k] = t [k] ;
            }
        }
        */
        BlockStore (W.store).Store (Map + pfirst, t) ;

        if (threadIdx.x == blockDim.x - 1)
        {
            // or try this:
//          ChunkSum [chunk] = tt [ITEMS_PER_THREAD1-1] ;   // in last thread
            ChunkSum [chunk] = block_aggregate ;
        }
    }

    //--------------------------------------------------------------------------
    // assign Map sentinal value
    //--------------------------------------------------------------------------

    if (threadIdx.x == 0 && blockIdx.x == 0)
    {
        Map [-1] = 0 ;  // sentinel value required for phase3; this also sets
        // Ck_Delta [-1] = 0 as a sentinal value required for phase6.
    }
}

//------------------------------------------------------------------------------
// GB_cuda_select_sparse_phase3: construct Ci, Cx, Ck1
//------------------------------------------------------------------------------

__global__ void GB_cuda_select_sparse_phase3
(
    // outputs:
    GrB_Matrix C,           // construct C->i and C->x
    GB_Aj_SIGNED_TYPE *Ck1, // size cnz+1, in Ck1 [0..cnz]
    // inputs, not modified:
    GrB_Matrix A,
    GB_Ap_TYPE *ChunkSum,   // size nchunks_in_A+1, ChunkSum [-1..nchunks_in_A]
    Int *Map,               // size anz+1, in Map [-1..anz-1]
    #if Ak_SAVE
    GB_Aj_SIGNED_TYPE *Ak,  // size anz, in Ak [0..anz-1]
    #endif
    int64_t anz,            // # of entries in A
    int64_t nchunks_in_A    // # of chunks in A
)
{

    //--------------------------------------------------------------------------
    // get C->i and C->x, shifting down by one to account for 1-based Map
    //--------------------------------------------------------------------------

    // in this method, the index pC = Map [pA] + ChunkSum [chunk] is 1-based,
    // so decrement Ci, Cx, to account for this.  Ck1 is already 1-based, and
    // appears in Ck1 [0..cnz] with Ck1 [0] = -1 set below.

    GB_Ci_TYPE *__restrict__ Ci = (GB_Ci_TYPE *) C->i ; Ci-- ;
    #if !GB_ISO_SELECT
    GB_C_TYPE  *__restrict__ Cx = (GB_C_TYPE  *) C->x ; Cx-- ;
    #endif

    //--------------------------------------------------------------------------
    // get A
    //--------------------------------------------------------------------------

    #if ( !Ak_SAVE )
    const int64_t anvec = A->nvec ;
    const GB_Ap_TYPE *__restrict__ Ap = (GB_Ap_TYPE *) A->p ;
    #endif
    const GB_Ai_SIGNED_TYPE *__restrict__ Ai = (GB_Ai_SIGNED_TYPE *) A->i ;
    #if !GB_ISO_SELECT
    const GB_A_TYPE *__restrict__ Ax = (GB_A_TYPE *) A->x ;
    #endif

    //--------------------------------------------------------------------------
    // select the entries and copy them into Ci, Cx; construct Ck1
    //--------------------------------------------------------------------------

    for (int64_t chunk = blockIdx.x ;
                 chunk < nchunks_in_A ;
                 chunk += gridDim.x)
    {

        //----------------------------------------------------------------------
        // get the chunk of A; entries in pA = pfirst:plast-1
        //----------------------------------------------------------------------

        int64_t pfirst = chunk << LOG2_CHUNKSIZE1 ;
        int64_t my_chunk_size ;
        #if !Ak_SAVE
        int64_t anvec1, kfirst, klast ;
        float slope ;
        GB_cuda_ek_slice_setup<GB_Ap_TYPE> (Ap, anvec, anz, pfirst,
            CHUNKSIZE1, &kfirst, &klast, &my_chunk_size, &anvec1, &slope) ;
        #else
        int64_t plast = pfirst + CHUNKSIZE1 ;
        plast = GB_IMIN (plast, anz) ;
        my_chunk_size = plast - pfirst ;
        #endif

        //----------------------------------------------------------------------
        // move the entries in this chunk of A into Ci and Cx
        //----------------------------------------------------------------------

        for (int64_t pdelta = threadIdx.x ;
                     pdelta < my_chunk_size ;
                     pdelta += blockDim.x)
        {
            int64_t pA = pfirst + pdelta ;
            // get the position pC in C of the pA-th entry in A
            GB_Ap_TYPE pC = Map [pA  ] + ChunkSum [chunk] ;
            // get the position p0 in C of the (pA-1)-st entry in A
            GB_Ap_TYPE p0 = Map [pA-1] + ChunkSum [chunk - (pdelta == 0)] ;
            if (p0 < pC)
            {
                // This entry is kept; it appears in a new position pC as
                // compared to the entry Map [pA-1] immediately to its left.
                // Map contains 1-based indices since it was computed as an
                // inclusive cumsum, so Ci and Cx have been shifted by one
                // above.  Ck1 is already set up to access as 1-based.  pC is a
                // 1-based index, and pA is 0-based.
                Ci [pC] = Ai [pA] ;
                // Cx [pC] = Ax [pA] ;
                GB_SELECT_ENTRY (Cx, pC, Ax, pA) ;
                // save the index of the kA-th vector kA that holds this entry
                #if Ak_SAVE
                Ck1 [pC] = Ak [pA] ;
                #else
                int64_t kA = GB_cuda_ek_slice_entry<GB_Ap_TYPE> (pA, pdelta, Ap,
                    anvec1, kfirst, slope) ;
                Ck1 [pC] = kA ;
                #endif
            }
        }
    }

    //--------------------------------------------------------------------------
    // assign Ck1/Ck0 sentinal value
    //--------------------------------------------------------------------------

    if (threadIdx.x == 0 && blockIdx.x == 0)
    {
        Ck1 [0] = -1 ;  // sentinel value required for phase4 (Ck0 [-1] = -1)
    }
}

//------------------------------------------------------------------------------
// GB_cuda_select_sparse_phase4: construct Ck_Delta
//------------------------------------------------------------------------------

__global__ void GB_cuda_select_sparse_phase4
(
    // outputs:
    Int *Ck_Delta,          // size cnz+1, in Ck_Delta [-1..cnz-1]
    GB_Ap_TYPE *ChunkSum,   // in ChunkSum [-1..nchunks_in_C]
    // inputs, not modified:
    GB_Aj_SIGNED_TYPE *Ck0, // size cnz+1, in Ck0 [-1..cnz-1]
    int64_t cnz,
    int64_t nchunks_in_C
)
{

    //--------------------------------------------------------------------------
    // workspace for each threadblock
    //--------------------------------------------------------------------------

    __shared__ Int Local_Ck_Delta [CHUNKSIZE2] ;

    // cub::Block* workspace:
    using BlockLoad  = cub::BlockLoad  <Int, BLOCKDIM2, ITEMS_PER_THREAD2> ;
    using BlockScan  = cub::BlockScan  <Int, BLOCKDIM2,
                                             cub::BLOCK_SCAN_WARP_SCANS> ;
    using BlockStore = cub::BlockStore <Int, BLOCKDIM2, ITEMS_PER_THREAD2> ;
    __shared__ union
    {
        typename BlockLoad::TempStorage load ;
        typename BlockScan::TempStorage scan ;
        typename BlockStore::TempStorage store ;
    } W ;

    //--------------------------------------------------------------------------
    // construct Ck_Delta and then cumsum each block
    //--------------------------------------------------------------------------

    for (int64_t chunk = blockIdx.x ;
                 chunk < nchunks_in_C ;
                 chunk += gridDim.x)
    {

        //----------------------------------------------------------------------
        // get the chunk of Ck0 and Ck_Delta this threadblock works on
        //----------------------------------------------------------------------

        // this threadblock works on Ck0 [pfirst:plast-1] and
        // Ck_Delta [pfirst:plast-1], where Ck0 uses 0-based indices

        int64_t pfirst = chunk << LOG2_CHUNKSIZE2 ;
        int64_t plast = pfirst + CHUNKSIZE2 ;
        plast = GB_IMIN (plast, cnz) ;
        int64_t my_chunk_size = plast - pfirst ;

        //----------------------------------------------------------------------
        // determine which entries of Ck0 start new vectors in C
        //----------------------------------------------------------------------

        int64_t pdelta = threadIdx.x ;
        for ( ; pdelta < my_chunk_size ;
                pdelta += blockDim.x)
        {
            // pC is a 0-based index, in range 0 to cnz-1
            int64_t pC = pfirst + pdelta ;
            Local_Ck_Delta [pdelta] = (Ck0 [pC-1] < Ck0 [pC]) ;
        }

        // clear the unused part of the Local_Ck_Delta
        for ( ; pdelta < CHUNKSIZE2 ;
                pdelta += blockDim.x)
        {
            Local_Ck_Delta [pdelta] = 0 ;
        }

        //----------------------------------------------------------------------
        // inclusive sum of Local_Ck_Delta
        //----------------------------------------------------------------------

        this_thread_block ( ).sync ( ) ;
        Int t [ITEMS_PER_THREAD2] ;

        // each thread loads its data from Local_Ck_Delta (in shared memory):
        /*
            for (int k = 0 ; k < ITEMS_PER_THREAD2 ; k++)
            {
                t [k] = Local_Ck_Delta [ITEMS_PER_THREAD2 * threadIdx.x + k] ;
            }
        */
        BlockLoad (W.load).Load (Local_Ck_Delta, t) ;
        this_thread_block ( ).sync ( ) ;

        // inclusive sum of data from Local_Ck_Delta,
        // where Local_Ck_Delta [i] = sum (Local_Ck_Delta [0:i])
        Int block_aggregate ;
        BlockScan (W.scan).InclusiveSum (t, t, block_aggregate) ;
        this_thread_block ( ).sync ( ) ;

        // each thread saves its data into Ck_Delta (in global memory):
        /*
            for (int k = 0 ; k < ITEMS_PER_THREAD2 ; k++)
            {
                Ck_Delta [pfirst + ITEMS_PER_THREAD2 * threadIdx.x + k] = t [k];
            }
        */
        BlockStore (W.store).Store (Ck_Delta + pfirst, t) ;

        // last thread writes the sum of the whole threadblock to global
        if (threadIdx.x == blockDim.x - 1)
        {
            ChunkSum [chunk] = block_aggregate ;
        }
    }

    // no need to set Ck_Delta [-1] = 0; already done in phase1
}

//------------------------------------------------------------------------------
// GB_cuda_select_sparse_phase6: construct Cp and Ch
//------------------------------------------------------------------------------

__global__ void GB_cuda_select_sparse_phase6
(
    // outputs:
    GrB_Matrix C,           // Cp and Ch are constructed
    // inputs, not modified
    Int *Ck_Delta,          // size cnz+1, in Ck_Delta [-1..cnz-1]
    GB_Ap_TYPE *ChunkSum,   // in ChunkSum [-1..nchunks_in_C]
    GB_Aj_SIGNED_TYPE *Ck0, // size cnz+1, in Ck0 [-1..cnz-1]
    #if ( GB_A_IS_HYPER )
    GrB_Matrix A,           // A->h is required if A is hypersparse
    #endif
    int64_t cnz,
    int64_t nchunks_in_C
)
{

    //--------------------------------------------------------------------------
    // get A and C
    //--------------------------------------------------------------------------

    // Cp and Ch use 1-based indexing below, so decrement them by 1
    GB_Cp_TYPE *__restrict__ Cp = (GB_Cp_TYPE *) C->p ; Cp-- ;
    GB_Cj_TYPE *__restrict__ Ch = (GB_Cj_TYPE *) C->h ; Ch-- ;
    #if ( GB_A_IS_HYPER )
    const GB_Aj_TYPE *__restrict__ Ah = (GB_Aj_TYPE *) A->h ;
    #endif

    //--------------------------------------------------------------------------
    // determine the start of each vector in C
    //--------------------------------------------------------------------------

    for (int64_t chunk = blockIdx.x ;
                 chunk < nchunks_in_C ;
                 chunk += gridDim.x)
    {

        //----------------------------------------------------------------------
        // get this chunk of C
        //----------------------------------------------------------------------

        int64_t pfirst = chunk << LOG2_CHUNKSIZE2 ;
        int64_t plast = pfirst + CHUNKSIZE2 ;
        plast = GB_IMIN (plast, cnz) ;
        int64_t my_chunk_size = plast - pfirst ;

        //----------------------------------------------------------------------
        // compute Cp and Ch for this chunk
        //----------------------------------------------------------------------

        for (int64_t pdelta = threadIdx.x ;
                     pdelta < my_chunk_size ;
                     pdelta += blockDim.x)
        {
            int64_t pC = pfirst + pdelta ;
            // get the vector kC that contains the pC-th entry of C
            GB_Cj_SIGNED_TYPE kC = Ck_Delta [pC  ] + ChunkSum [chunk] ;
            // get the vector k0 that contains the (pC-1)-st entry of C
            GB_Cj_SIGNED_TYPE k0 = Ck_Delta [pC-1] + ChunkSum [chunk -
                (pdelta == 0)] ;
            if (k0 < kC)
            {
                // The pC-th entry is the start of a vector kC in C;
                // note that kC is 1-based, so Cp and Ch are decremented above.
                int64_t kA = Ck0 [pC] ;
                Cp [kC] = pC ;
                Ch [kC] = GBh_A (Ah, kA) ;
            }
        }
    }

    //--------------------------------------------------------------------------
    // finalize the last vector of C
    //--------------------------------------------------------------------------

    if (threadIdx.x == 0 && blockIdx.x == 0)
    {
        // C->nvec is 0-based, so increment Cp to undo the Cp-- done above
        Cp++ ;
        Cp [C->nvec] = cnz ;
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
    #ifdef TIMING
    double t = omp_get_wtime ( ) ;
    #endif

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
    // declare workspace
    //--------------------------------------------------------------------------

    GrB_Info info ;

    // workspaces of size anz+2
    void *W_0 = NULL ; size_t W_0_size = 0 ;
    void *W_1 = NULL ; size_t W_1_size = 0 ;
    // workspace of size max (nchunks_in_A, nchunks_in_C)+1
    void *W_2 = NULL ; size_t W_2_size = 0 ;
    // workspace of size cnz+2, where cnz <= anz
    void *W_3 = NULL ; size_t W_3_size = 0 ;

    GB_A_NHELD (anz) ;          // # of entries in A
    int64_t cnz = 0 ;           // # of entries in C (which is <= anz)
    // # of chunks in A:
    int64_t nchunks_in_A = (anz + CHUNKSIZE1 - 1) >> LOG2_CHUNKSIZE1 ;
    // max # of chunks in C (revised below):
    int64_t nchunks_in_C = (anz + CHUNKSIZE2 - 1) >> LOG2_CHUNKSIZE2 ;
    int64_t nchunks_max = GB_IMAX (nchunks_in_A, nchunks_in_C) ;

    ASSERT (GB_A_IS_HYPER || GB_A_IS_SPARSE) ;

    dim3 grid (gridsz) ;        // = min (ceil (anz/CHUNKSIZE1), 256*(#sms))
    dim3 block1 (BLOCKDIM1) ;
    dim3 block2 (BLOCKDIM2) ;

    //--------------------------------------------------------------------------
    // phase 1: allocate workspace and determine which entries of A to keep
    //--------------------------------------------------------------------------

    // This phase constructs Ak [0..anz-1], where Ak [pA] = kA if the pA-th
    // entry is in the kA-th vector of A.  It also is the first phase in
    // constructing Map [-1..anz-1], where Map [pA] = pC + ChunkSum [chunk]
    // if the pA-th entry of A is the pC-th entry of C, and chunk is the
    // chunk of A that contains the pA-th entry.  The Map array is extended so
    // that it contains an integral number of chunks.

    // Example:  suppose the select operator fkeep(aij) keeps nonzero entries
    // in A, with the following input.  Suppose the CHUNKSIZE1 is 4, for this
    // diagram (it is 1024 as #define'd above), as delineated by "|" in the
    // diagram.  The matrix A is 10 by 7 with 18 entries, held by column in
    // sparse form (Ah is NULL, and the kth vector is simply column k of A).

    // col:      0 0 0 1 1 1 2 3 3 4 4 5 5 5 5 6 6 6 (computed and put in Ak)
    // Ap:     [ 0     3     6 7   9  11      15     18 ]  size: ncol+1 = 8
    // Ai:     [ 0 1 2 2|5 6 0 1|8 1 7 2|7 8 9 3|7 9|- -]  size: anz = 18
    // Ax:     [ 1 1 1 0|1 0 0 0|0 1 0 1|1 0 1 1|1 1|- -]  size: anz = 18

    // phase1 computes Ak [0..anz-1] and Map [-1..anz-1], where Ak [pA] = kA if
    // the pA-th entry is in the kA-th vector of A, and where Map [pA] is an
    // inclusive cumulative sum of fkeep(aij) for each chunk.  In this example,
    // the first entry is kept, and this can be checked later by comparing Map
    // [-1] with Map [0].  Since A has entries equal to 0 or 1, fkeep(aij) is
    // simply aij, held in the Ax array:

    // inputs:
    // Ax:     [ 1 1 1 0|1 0 0 0|0 1 0 1|1 0 1 1|1 1|- -] (- denotes empty)
    // output:
    // Map:  0 [ 1 2 3 3|1 1 1 1|0 1 1 2|1 1 2 3|1 2 2 2] (note the padding)
    // Ak:     [ 0 0 0 1|1 1 2 3|3 4 4 5|5 5 5 6|6 6 - -] (see col list above

    // output:
    // ChunkSum [-1..nchunks_in_A] is the # of entries kept in each chunk, with
    // ChunkSum [-1] = 0 and ChunkSum [nchunks_in_A] = 0 for now:
    //       0 [       3       1       2       3       2] 0

    // The (Ai,Ak,Ax) sets of arrays are a coordinate-form representation of
    // the entries in A, in sorted order, with Ai being the row indices, Ak the
    // column indices (if the matrix is in CSC format) and Ax being the values.
    // If A is hypersparse by column, then Ak [pA] = kA holds the value kA if
    // the entry is in the kA-th nonempty column of A, which is column Ah [kA].

    // The W_2 workspace is used only for ChunkSum, and must be accessible on
    // both the GPU and CPU, so the RMM memory manager must be used.  All other
    // workspaces (W_0, W_1, and W_3) exist only on the GPU, so cudaMalloc
    // could be used for them.  However, the RMM memory manager gives better
    // performance than cudaMalloc.
    #if Ak_SAVE
    W_0 = GB_MALLOC_MEMORY (anz+2, sizeof (GB_Aj_SIGNED_TYPE), &W_0_size) ;
    #endif
    W_1 = GB_MALLOC_MEMORY (anz+2 + CHUNKSIZE1, sizeof (Int), &W_1_size) ;
    W_2 = GB_MALLOC_MEMORY (nchunks_max+2, sizeof (GB_Ap_TYPE), &W_2_size) ;
    if ((Ak_SAVE && W_0 == NULL) || W_1 == NULL || W_2 == NULL)
    {
        // out of memory
        GB_FREE_ALL ;
        return (GrB_OUT_OF_MEMORY) ;
    }

    #if Ak_SAVE
    // use W_0 [0..anz-1] as workspace for Ak [0..anz-1]
    GB_Aj_SIGNED_TYPE *Ak = (GB_Aj_SIGNED_TYPE *) W_0 ;
    #endif

    // use W_1 workspace for Map, and shift by one to define Map [-1] as 0
    // (which is set in the phase1 kernel launch below).
    Int *Map = ((Int *) W_1) + 1 ;

    // ChunkSum [-1 .. nchunks_in_*] of size nchunks_max+2, in workspace W_2
    GB_Ap_TYPE *ChunkSum = (GB_Ap_TYPE *) W_2 + 1 ;
    ChunkSum [-1] = 0 ;     // sentinel value required for phase3 and phase6

    // KERNEL LAUNCH 1: phase1
    GB_cuda_select_sparse_phase1 <<<grid, block1, 0, stream>>>
        (/* outputs: */
            #if Ak_SAVE
            Ak,
            #endif
            Map, ChunkSum,
         /* inputs: */  A, ythunk, anz, nchunks_in_A) ;
    CUDA_OK (cudaGetLastError ( )) ;
    CUDA_OK (cudaStreamSynchronize (stream)) ;

    #ifdef TIMING
    t = omp_get_wtime ( ) - t ;
    printf ("\nselect sparse phase1: %g sec (gpu: Map, with cumsum)\n", t) ;
    t = omp_get_wtime ( ) ;
    #endif

    //--------------------------------------------------------------------------
    // phase 2: sum up the entries in each chunk (on the CPU)
    //--------------------------------------------------------------------------

    // At the start of phase2, ChunkSum [-1..nchunks_in_A] holds the # of
    // entries kept in C from each chunk of A, with cnz = 11 entries kept in C.
    //       0 [       3       1       2       3       2]
    // This phase computes an exclusive cumulative sum:
    //       0 [       0       3       4       6       9]    11

    // FIXME: do this on the GPU?  Or in parallel on the CPU?
    // FIXME: if on the CPU, use GB_cumsum.c (need both int32_t and int64_t)

    // overwrite ChunkSum [0..gridsdz] with its cumulative sum
    for (int64_t chunk = 0 ; chunk < nchunks_in_A ; chunk++)
    {
        // get the # of entries found by this threadblock
        int64_t s = ChunkSum [chunk] ;
        // overwrite the entry with the cumulative sum, so that the new
        // ChunkSum [chunk] = original ChunkSum [0..chunk-1]
        ChunkSum [chunk] = cnz ;
        cnz += s ;
    }
    ChunkSum [nchunks_in_A] = cnz ;

    #ifdef TIMING
    t = omp_get_wtime ( ) - t ;
    printf ("select sparse phase2: %g sec (cpu: ChunkSum of Map)\n", t) ;
    t = omp_get_wtime ( ) ;
    #endif

    //--------------------------------------------------------------------------
    // phase 3: allocate C and construct Ci, Cx, and Ck1
    //--------------------------------------------------------------------------

    // allocate the C matrix as hypersparse, with cnz entries
    GB_OK (GB_bix_alloc (C, cnz, GxB_HYPERSPARSE, false, true, GB_ISO_SELECT)) ;
    C->nvals = cnz ;
    if (cnz == 0)
    {
        // C is empty; nothing more to do
        GB_FREE_WORKSPACE ;
        return (GrB_SUCCESS) ;
    }

    // This kernel tolerates A in a jumbled state, for all operators.
    // TODO: GB_select does a wait on A, but this can be delayed.
    C->jumbled = A->jumbled ;

    // allocate workspace of size cnz+2
    W_3 = GB_MALLOC_MEMORY (cnz+2, sizeof (GB_Aj_SIGNED_TYPE), &W_3_size) ;
    if (W_3 == NULL)
    {
        // out of memory
        GB_FREE_ALL ;
        return (GrB_OUT_OF_MEMORY) ;
    }

    // use W_3 as workspace for Ck0 and Ck1, which are the same array, but Ck0
    // is accessed with 0-based indices (in range 0:cnz-1) and Ck1 uses 1-based
    // indices (in range 1:cnz).  That is, Ck0 [0] is the same as Ck1 [1].  The
    // first entry (Ck1 [0] and Ck0 [-1]) is a sentinel value (-1), set in the
    // phase3 kernel launch below.
    GB_Aj_SIGNED_TYPE *Ck0 = ((GB_Aj_SIGNED_TYPE *) W_3) + 1 ;
    GB_Aj_SIGNED_TYPE *Ck1 = ((GB_Aj_SIGNED_TYPE *) W_3) ;

    // The position where the pA-th entry in A appears in C is given by pC =
    // Map [pA] + ChunkSum [chunk], where pC is a 1-based index.  If this
    // position differs from the position of the (pA-1)st entry in A, then the
    // entry is kept, and is copied from A into C in phase3.

    // Ck1 [pC] = kA = Ak [pA] if the entry in C is in the vector of C that
    // corresponds to the kA-th vector of A.

    // On output, where "|" reflects the chunks of A, not C, and spaces are
    // added in C just for illustration (denoting entries in A not kept):

    // input:
    // Ai:     [ 0 1 2 2|5 6 0 1|8 1 7 2|7 8 9 3|7 9|- -]  size: anz = 18
    // Ax:     [ 1 1 1 0|1 0 0 0|0 1 0 1|1 0 1 1|1 1|- -] (- denotes empty)
    // Map:  0 [ 1 2 3 3|1 1 1 1|0 1 1 2|1 1 2 3|1 2 2 2] (note the padding)
    // Ak:     [ 0 0 0 1|1 1 2 3|3 4 4 5|5 5 5 6|6 6 - -]
    // ChunkSum:
    //       0 [       0       3       4       6       9]    11

    // output, with gaps denoting entries not in C:
    // Ci:     [ 0 1 2  |5      |  1   2|7   9 3|7 9 ]
    // Cx:     [ 1 1 1  |1      |  1   1|1   1 1|1 1 ]
    // Ck1: -1 [ 0 0 0  |1      |  4   5|5   5 6|6 6 ]
    //           ^       ^         ^   ^       ^---------start of vectors in C

    // Note that k=2 and k=3 in C are empty vectors since no entries in the 2nd
    // and 3rd columns of A were kept.

    // KERNEL LAUNCH 2: phase3
    GB_cuda_select_sparse_phase3 <<<grid, block1, 0, stream>>>
        (/* outputs: */ C, Ck1,
         /* inputs: */  A, ChunkSum, Map,
            #if Ak_SAVE
            Ak,
            #endif
            anz, nchunks_in_A) ;
    CUDA_OK (cudaGetLastError ( )) ;
    CUDA_OK (cudaStreamSynchronize (stream)) ;

    // Ak (in W_0) no longer needed, and W_0 is no longer needed, so free it
    GB_FREE_MEMORY (&W_0, W_0_size) ;

    // Map (in W_1) no longer needed; reused below for Ck_Delta

    #ifdef TIMING
    t = omp_get_wtime ( ) - t ;
    printf ("select sparse phase3: %g sec (gpu: create Ci,Cx,Ck)\n", t) ;
    t = omp_get_wtime ( ) ;
    #endif

    //--------------------------------------------------------------------------
    // phase 4: construct Ck_Delta and its local cumulative sum
    //--------------------------------------------------------------------------

    // Ck_Delta [pC] = 1 if the pC-th entry is the first in its vector of C, or
    // 0 otherwise.  Then each threadblock computes the inclusive cumulative
    // sum of its chunk of Ck_Delta, overwriting Ck_Delta with its cumulative
    // sum.  Note the spaces (for illustration above) are removed here.
    // The index pC is 0-based in phase 4.

    // input:
    // Ck0: -1 [ 0 0 0 1|4 5 5 5|6 6 6 ]
    //           ^     ^ ^ ^     ^---------start of vectors in C
    // output:
    // Ck_Delta as 0/1:
    //       0 [ 1 0 0 1|1 1 0 0|1 0 0 ]
    // Ck_Delta as inclusive cumsum, per chunk of C:
    //       0 [ 1 1 1 2|1 2 2 2|1 1 1 ]
    // ChunkSum of C:
    //       0 [       2|      2|    1 ]

    // # of chunks in C:
    nchunks_in_C = (cnz + CHUNKSIZE2 - 1) >> LOG2_CHUNKSIZE2 ;

    // using W_1 [-1..cnz-1] as workspace for Ck_Delta, which is accessed
    // with 0-based indices, using Ck_Delta [-1..cnz-1] where cnz <= anz.
    // Note that W_1 [-1] is already set to zero in phase1 (Map [-1] = 0),
    // and thus Ck_Delta [-1] is already equal to 0, as required.
    Int *Ck_Delta = ((Int *) W_1) + 1 ;

    // KERNEL LAUNCH 3: phase4
    GB_cuda_select_sparse_phase4 <<<grid, block2, 0, stream>>>
        (/* outputs: */ Ck_Delta, ChunkSum,
         /* inputs: */  Ck0, cnz, nchunks_in_C) ;
    CUDA_OK (cudaGetLastError ( )) ;
    CUDA_OK (cudaStreamSynchronize (stream)) ;

    #ifdef TIMING
    t = omp_get_wtime ( ) - t ;
    printf ("select sparse phase4: %g sec (gpu: Ck_Delta, with cumsum)\n", t) ;
    t = omp_get_wtime ( ) ;
    #endif

    //--------------------------------------------------------------------------
    // phase 5: construct global cumsum of Ck_Delta on the CPU
    //--------------------------------------------------------------------------

    // FIXME: do this on the GPU?  Or in parallel on the CPU?
    // FIXME: if on the CPU, use GB_cumsum.c (need both int32_t and int64_t)

    // overwrite ChunkSum [0..nchunks_in_C] with its cumulative sum
    int64_t cnvec = 0 ;
    for (int64_t chunk = 0 ; chunk < nchunks_in_C ; chunk++)
    {
        // get the # of entries found by this threadblock
        int64_t s = ChunkSum [chunk] ;
        // overwrite the entry with the cumulative sum, so that the new
        // ChunkSum [chunk] = original ChunkSum [0..chunk-1]
        ChunkSum [chunk] = cnvec ;
        cnvec += s ;
    }
    ChunkSum [nchunks_in_C] = cnvec ;

    // ChunkSum of C, before the cumsum:
    //       0 [       2|      2|    1 ]
    // ChunkSum of C, after the cumsum, where ChunkSum [nchunks_in_C] = 5
    // are the final number of nonempty vectors of C:
    //       0 [       0|      2|    4 ]  5

    #ifdef TIMING
    t = omp_get_wtime ( ) - t ;
    printf ("select sparse phase5: %g sec (cpu: ChunkSum for Ck_Delta\n", t) ;
    t = omp_get_wtime ( ) ;
    #endif

    //--------------------------------------------------------------------------
    // phase 6: construct Cp and Ch
    //--------------------------------------------------------------------------

    // The caller has already allocated C->p, C->h for a user-returnable empty
    // hypersparse matrix.  Free them here before reallocating them.
    GB_FREE_MEMORY (&(C->p), C->p_size) ;
    GB_FREE_MEMORY (&(C->h), C->h_size) ;

    // Allocate Cp and Ch
    C->plen = cnvec ;
    C->nvec = cnvec ;
    C->nvec_nonempty = cnvec ;
    C->p = (GB_Cp_TYPE *) GB_MALLOC_MEMORY (C->plen+1, sizeof (GB_Cp_TYPE),
        &(C->p_size)) ;
    C->h = (GB_Cj_TYPE *) GB_MALLOC_MEMORY (C->plen, sizeof (GB_Cj_TYPE),
        &(C->h_size)) ;
    if (C->p == NULL || C->h == NULL)
    {
        // out of memory
        GB_FREE_ALL ;
        return (GrB_OUT_OF_MEMORY) ;
    }

    // input:
    // Ck0: -1 [ 0 0 0 1|4 5 5 5|6 6 6 ]
    //           ^-----^-^-^-----^------^----- start of vectors in C
    //           |     | | |     |      |
    // pC is:    0 1 2 3 4 5 6 7 8 9 10 11  (pC is 0-based in phase6)

    // Ck_Delta as inclusive cumsum, per chunk of C:
    //       0 [ 1 1 1 2|1 2 2 2|1 1 1 ]
    // ChunkSum of C:
    //       0 [       2|      2|    1 ]

    // output:
    // Cp:   [ 0 3 4 5 8 11 ] of size cnvec+1
    // Ch:   [ 0 1 4 5 6    ] of size cnvec

    // KERNEL LAUNCH 4: phase6
    GB_cuda_select_sparse_phase6 <<<grid, block2, 0, stream>>>
        (/* outputs are C->p and C->h: */ C,
         /* inputs: */ Ck_Delta, ChunkSum, Ck0,
         #if ( GB_A_IS_HYPER )
         A,               // A->h is required if A is hypersparse
         #endif
         cnz, nchunks_in_C) ;
    CUDA_OK (cudaGetLastError ( )) ;
    CUDA_OK (cudaStreamSynchronize (stream)) ;

    #ifdef TIMING
    t = omp_get_wtime ( ) - t ;
    printf ("select sparse phase6: %g sec (gpu: Cp,Ch)\n", t) ;
    #endif

    //--------------------------------------------------------------------------
    // free workspace and return result
    //--------------------------------------------------------------------------

    GB_FREE_ALL ;
    return (GrB_SUCCESS) ;
}

