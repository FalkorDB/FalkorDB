//------------------------------------------------------------------------------
// GraphBLAS/CUDA/template/GB_jit_kernel_cuda_builder
//------------------------------------------------------------------------------

// SuiteSparse:GraphBLAS, Timothy A. Davis, (c) 2017-2025, All Rights Reserved.
// SPDX-License-Identifier: Apache-2.0

//------------------------------------------------------------------------------

#define GB_FREE_WORKSPACE                   \
{                                           \
    GB_FREE_MEMORY (&W_0, W_0_mem) ;        \
    GB_FREE_MEMORY (&W_1, W_1_mem) ;        \
    GB_FREE_MEMORY (&W_2, W_2_mem) ;        \
    GB_FREE_MEMORY (&W_3, W_3_mem) ;        \
    GB_FREE_MEMORY (&W_4, W_4_mem) ;        \
    GB_FREE_MEMORY (&W_5, W_5_mem) ;        \
    GB_FREE_MEMORY (&W_6, W_6_mem) ;        \
    GB_FREE_MEMORY (&W_7, W_7_mem) ;        \
    GB_FREE_MEMORY (&W_8, W_8_mem) ;        \
}

#define GB_FREE_ALL                         \
{                                           \
    GB_Matrix_free (&T) ;                   \
    GB_FREE_WORKSPACE ;                     \
}

#include "template/GB_cuda_tile_sum_uint64.cuh"
#include "template/GB_cuda_threadblock_sum_uint64.cuh"

//------------------------------------------------------------------------------
// typedefs
//------------------------------------------------------------------------------

// GB_key_t: sorting key type for CUB radix sort.
// GB_KEY_TYPE is uint32_t or uint64_t

#if GB_MTX_BUILD

    struct GB_key_t
    {
        GB_KEY_TYPE j ;
        GB_KEY_TYPE i ;
        GB_key_t ( ) = default ;
    } ;

    #define GB_KEY_LOAD(Key_in,p,i1,j1)                 \
        Key_in [p].i = (GB_KEY_TYPE) i1 ;               \
        Key_in [p].j = (GB_KEY_TYPE) j1 ;
    #define GB_KEY_UNLOAD_I(Key_out,p,i1)               \
        GB_KEY_TYPE i1 = (GB_KEY_TYPE) Key_out [p].i ;
    #define GB_KEY_UNLOAD_J(Key_out,p,j1)               \
        GB_KEY_TYPE j1 = (GB_KEY_TYPE) Key_out [p].j ;

    struct GB_key_decomposer_t
    {
        __host__ __device__
        cuda::std::tuple <GB_KEY_TYPE&, GB_KEY_TYPE&> operator()(GB_key_t& key)
        const
        {
            return {key.j, key.i} ;
        }
    } ;

#else

    // Buidling a GrB_Vector, so j is not used.
    // GB_key_t is not a struct; just a plain uint32_t or uint64_t
    typedef GB_KEY_TYPE GB_key_t ;
    #define GB_KEY_LOAD(Key_in,p,i1,j1)                 \
        Key_in [p] = (GB_KEY_TYPE) i1 ;
    #define GB_KEY_UNLOAD_I(Key_out,p,i1)               \
        GB_KEY_TYPE i1 = Key_out [p] ;
    #define GB_KEY_UNLOAD_J(Key_out,p,j1) ;

#endif

#define GB_KEY_UNLOAD(Key_out,p,i1,j1)                  \
    GB_KEY_UNLOAD_I (Key_out, p, i1) ;                  \
    GB_KEY_UNLOAD_J (Key_out, p, j1) ;

//------------------------------------------------------------------------------
// geometry
//------------------------------------------------------------------------------

#define CHUNKSIZE           GB_CUDA_BUILDER_CHUNKSIZE
#define LOG2_CHUNKSIZE      GB_CUDA_BUILDER_CHUNKSIZE_LOG2
#define BLOCKDIM            GB_CUDA_BUILDER_BLOCKDIM
#define ITEMS_PER_THREAD    (CHUNKSIZE / BLOCKDIM)

// Int can be uint16_t if CHUNKSIZE is < 65,535
#define Int uint16_t

#if CHUNKSIZE > 65535
#error "Int cannot be uint16_t"
#endif

//------------------------------------------------------------------------------
// GB_cuda_builder_phase1:
//------------------------------------------------------------------------------

// phase1 loads the (I,J) tuples into the Key_in workspace, and checks the
// indices (I,J) to ensure they are in range.  It sets the global (*bad) scalar
// to the # of tuples that are out of range.  Note that the CPU builder also
// returns the first invalid indices for the error message returned to the user
// application; this kernel does not do that.

// phase1 checks if the I,J tuples are already in order, and returns
// (*unsorted) as the # of tuples that are out of order.  If this count is
// zero, the sort (via CUB RadixSort) can be skipped.

// phase1 counts the # of adjacent duplicate entries (as (*dupls); if tuples
// are in order and this count is zero, then no duplicates exist in the I,J
// tuples.

// This phase is skipped if the caller passes in Key_input as the pre-loaded
// keys.

#if !GB_KEY_PRELOADED
__global__ void GB_cuda_builder_phase1
(
    // output
    GB_key_t *Key_in,   // size nvals+1: Key_in [-1...nvals-1]
    uint64_t *bad,      // count of # of invalid tuples (zero on input)
    #if !GB_KNOWN_SORTED
    uint64_t *unsorted, // count of # of tuples out of order (zero on input)
    #endif
    #if !GB_KNOWN_NO_DUPLICATES
    uint64_t *dupls,    // count of # duplicates if all in order (zero on input)
    #endif
    // input
    const GB_I_TYPE *__restrict__ I, // size nvals
    int64_t vlen,       // vector-length dimension of C (for I indices)
    #if GB_MTX_BUILD
    const GB_J_TYPE *__restrict__ J, // size nvals, NULL if C is a vector
    int64_t vdim,       // vector-dim dimension of C (for J indices)
    #endif
    int64_t nvals       // # of tuples in (I,J,X)
)
{

    //--------------------------------------------------------------------------
    // load the (I,J) tuples into the Key_in workspace
    //--------------------------------------------------------------------------

    if (threadIdx.x == 0 && blockIdx.x == 0)
    {
        // first thread loads the sentinel value of Key_in
        memset (&(Key_in [-1]), 0xFF, sizeof (GB_key_t)) ;
    }

    uint64_t my_bad = 0 ;
    #if !GB_KNOWN_SORTED
    uint64_t my_unsorted = 0 ;
    #endif
    #if !GB_KNOWN_NO_DUPLICATES
    uint64_t my_dupls = 0 ;
    #endif

    for (int64_t p = blockIdx.x * blockDim.x + threadIdx.x ;
                 p < nvals ;
                 p += blockDim.x * gridDim.x)       // grid-block-stride loop
    {
        // get the indices
        GB_I_TYPE i = I [p] ;
        #if GB_MTX_BUILD
        GB_J_TYPE j = J [p] ;
        #endif

        if (p > 0)
        {
            GB_I_TYPE i0 = I [p-1] ;
            #if GB_MTX_BUILD
            GB_J_TYPE j0 = J [p-1] ;
            #endif
            #if !GB_KNOWN_SORTED
            // count the number of indices that are out of order
            my_unsorted += (uint64_t)
                #if GB_MTX_BUILD
                ((j0 > j) || (j0 == j && i0 > i)) ;
                #else
                (i0 > i) ;
                #endif
            #endif
            #if !GB_KNOWN_NO_DUPLICATES
            // count the # of duplicates (only valid indices are all in order)
            my_dupls += (uint64_t)
                #if GB_MTX_BUILD
                ((j0 == j) && (i0 == i)) ;
                #else
                (i0 == i) ;
                #endif
            #endif
        }

        // count the # of tuples out of range
        #if GB_MTX_BUILD
        my_bad += (uint64_t) ((i >= vlen) || (j >= vdim)) ;
        #else
        my_bad += (uint64_t) ((i >= vlen)) ;
        #endif

        // load the indices into Key_in [p]
        GB_KEY_LOAD (Key_in, p, i, j) ;
    }

    //--------------------------------------------------------------------------
    // compute the global count of bad, unsorted, and duplicate tuples
    //--------------------------------------------------------------------------

    this_thread_block ( ).sync ( ) ;
    my_bad      = GB_cuda_threadblock_sum_uint64 (my_bad) ;
    this_thread_block ( ).sync ( ) ;
    #if !GB_KNOWN_SORTED
    my_unsorted = GB_cuda_threadblock_sum_uint64 (my_unsorted) ;
    this_thread_block ( ).sync ( ) ;
    #endif
    #if !GB_KNOWN_NO_DUPLICATES
    my_dupls    = GB_cuda_threadblock_sum_uint64 (my_dupls) ;
    this_thread_block ( ).sync ( ) ;
    #endif

    if (threadIdx.x == 0)
    {
        GB_cuda_atomic_add <uint64_t> (bad     , my_bad) ;
        #if !GB_KNOWN_SORTED
        GB_cuda_atomic_add <uint64_t> (unsorted, my_unsorted) ;
        #endif
        #if !GB_KNOWN_NO_DUPLICATES
        GB_cuda_atomic_add <uint64_t> (dupls   , my_dupls) ;
        #endif
    }
}
#endif

//------------------------------------------------------------------------------
// GB_cuda_builder_phase3_with_dupl
//------------------------------------------------------------------------------

// phase3 looks for duplicates in the sorted Key_out array.  It constructs a
// Map array which tells each entry in (Key_out,Sx) where it appears in C,
// after a cumulative sum.  This part of the phase is nearly identical to
// CUDA/select phase1.  builder/phase3 must also find the leading entry in each
// vector of the output matrix C.  It does this with JDelta, which is a
// cumulative sum of the number of leading entries in each chunk.

// Compare with select/phase1

#if !GB_KNOWN_NO_DUPLICATES
__global__ void GB_cuda_builder_phase3_with_dupl
(
    // outputs
    Int *Map,               // size nvals+1, in Map [-1...nvals-1]
    GB_Tp_TYPE *ChunkSum,   // size nchunks+1, in ChunkSum [-1..nchunks]
    #if GB_MTX_BUILD
    Int *JDelta,            // size nvals+1, in JDelta [-1..nvals-1]
    GB_Tp_TYPE *JDeltaSum,  // size nchunks+1
    #endif
    // inputs, not modified, except for the Key_out [-1] sentinel value:
    GB_key_t *Key_out,      // size nvals+1: Key_out [-1 ... nvals-1]
    int64_t nvals,          // # of tuples in (I,J,X)
    int64_t nchunks
)
{

    //--------------------------------------------------------------------------
    // workspace for each threadblock
    //--------------------------------------------------------------------------

    __shared__ Int Local_Map [CHUNKSIZE] ;
    #if GB_MTX_BUILD
    __shared__ Int Local_JDelta [CHUNKSIZE] ;
    #endif

    // cub::Block* workspace:
    GB_CUB_BLOCK_WORKSPACE (W, Int, BLOCKDIM, ITEMS_PER_THREAD) ;

    //--------------------------------------------------------------------------
    // the first thread of the threadblock fills in the sentinal values
    //--------------------------------------------------------------------------

    if (threadIdx.x == 0 && blockIdx.x == 0)
    {
        memset (&(Key_out [-1]), 0xFF, sizeof (GB_key_t)) ;
        Map [-1] = 0 ;
        ChunkSum [-1] = 0 ;
        #if GB_MTX_BUILD
        JDelta [-1] = 0 ;
        JDeltaSum [-1] = 0 ;
        #endif
    }

    // this_thread_block ( ).sync ( ) ; not needed since the thread that wrote
    // the Key_out [-1] entry is the only thread that reads it.

    //--------------------------------------------------------------------------
    // compute each local chunk of Map
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
        // this computation is just the 2nd #if case of select/phase1:
        int64_t plast = pfirst + CHUNKSIZE ;
        plast = GB_IMIN (plast, nvals) ;
        my_chunk_size = plast - pfirst ;

        //----------------------------------------------------------------------
        // determine the first unique tuple in each sequence of duplicates
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
            // determine if the p-th entry is 1st of duplicates, or leading
            //------------------------------------------------------------------

            // get the indices
            GB_KEY_UNLOAD (Key_out, p-1, iprev, jprev) ;
            GB_KEY_UNLOAD (Key_out, p,   i,     j    ) ;
            #if GB_MTX_BUILD
            // leading is true if this is the first entry in vector j
            bool leading = (j != jprev) ;
            #endif
            // keep = 1 if (i,j) is unique, 0 if duplicate
            bool keep = (i != iprev)
                #if GB_MTX_BUILD
                || leading
                #endif
                ;
            Local_Map [pdelta] = keep ;         // 1 if 1st in seq of dupls
            #if GB_MTX_BUILD
            Local_JDelta [pdelta] = leading ;   // 1 if leading entry of vector
            #endif
        }

        //----------------------------------------------------------------------
        // the remainder is similar to select/phase1:
        //----------------------------------------------------------------------

        // clear the unused part of the Local_Map and Local_JDelta
        for ( ; pdelta < CHUNKSIZE ;
                pdelta += blockDim.x)
        {
            Local_Map [pdelta] = 0 ;
            #if GB_MTX_BUILD
            Local_JDelta [pdelta] = 0 ;
            #endif
        }

        //----------------------------------------------------------------------
        // inclusive cumulative sum of Local_Map and Local_JDelta
        //----------------------------------------------------------------------

        // Map [pfirst..pfirst+CHUNKSIZE-1] = inclusive cumsum of Local_Map,
        // where Local_Map [i] = sum (Local_Map [0:i]) is computed.

        // Similarly, JDelta [pfirst..pfirst+CHUNKSIZE-1] = inclusive cumsum
        // of Local_JDelta, where Local_JDelta [i] = sum (Local_JDelta [0:i])
        // is computed.

        this_thread_block ( ).sync ( ) ;
        Int t_block_aggregate ;
        #if GB_MTX_BUILD
        Int s_block_aggregate ;
        #endif

#if 0
        // This entire phase computes the following:
        if (threadIdx.x == blockDim.x - 1)
        {
            // construct Map and ChunkSum
            for (int i = 1 ; i < CHUNKSIZE ; i++)
            {
                Local_Map [i] += Local_Map [i-1] ;
            }
            for (int i = 0 ; i < CHUNKSIZE ; i++)
            {
                Map [pfirst + i] = Local_Map [i] ;
            }
            t_block_aggregate = Local_Map [CHUNKSIZE-1] ;
            ChunkSum [chunk] = t_block_aggregate ;

            // construct JDelta and JDeltaSum
            #if GB_MTX_BUILD
            for (int i = 1 ; i < CHUNKSIZE ; i++)
            {
                Local_JDelta [i] += Local_JDelta [i-1] ;
            }
            for (int i = 0 ; i < CHUNKSIZE ; i++)
            {
                JDelta [pfirst + i] = Local_JDelta [i] ;
            }
            s_block_aggregate = Local_JDelta [CHUNKSIZE-1] ;
            JDeltaSum [chunk] = s_block_aggregate ;
            #endif
        }

#else
        Int t [ITEMS_PER_THREAD] ;

        BlockLoad (W.load).Load (Local_Map, t) ;
        this_thread_block ( ).sync ( ) ;
        BlockScan (W.scan).InclusiveSum (t, t, t_block_aggregate) ;
        this_thread_block ( ).sync ( ) ;
        BlockStore (W.store).Store (Map + pfirst, t) ;
        this_thread_block ( ).sync ( ) ;

        #if GB_MTX_BUILD
        BlockLoad (W.load).Load (Local_JDelta, t) ;
        this_thread_block ( ).sync ( ) ;
        BlockScan (W.scan).InclusiveSum (t, t, s_block_aggregate) ;
        this_thread_block ( ).sync ( ) ;
        BlockStore (W.store).Store (JDelta + pfirst, t) ;
        this_thread_block ( ).sync ( ) ;
        #endif

        // finally, the aggregate sums are written to ChunkSum and JDeltaSum
        if (threadIdx.x == blockDim.x - 1)
        {
            ChunkSum  [chunk] = t_block_aggregate ;
            #if GB_MTX_BUILD
            JDeltaSum [chunk] = s_block_aggregate ;
            #endif
        }

#endif

        this_thread_block ( ).sync ( ) ;
    }
}

#endif

//------------------------------------------------------------------------------
// GB_cuda_builder_phase3_no_dupl
//------------------------------------------------------------------------------

// builder/phase3 must also find the leading entry in each
// vector of the output matrix C.  It does this with JDelta, which is a
// cumulative sum of the number of leading entries in each chunk.

// This kernel is not needed if a vector is being built.

// Compare with select/phase1

#if GB_MTX_BUILD
__global__ void GB_cuda_builder_phase3_no_dupl
(
    // outputs
    Int *JDelta,            // size nvals+1, in JDelta [-1..nvals-1]
    GB_Tp_TYPE *JDeltaSum,  // size nchunks+1
    // inputs, not modified, except for the Key_out [-1] sentinel value:
    GB_key_t *Key_out,      // size nvals+1: Key_out [-1 ... nvals-1]
    int64_t nvals,          // # of tuples in (I,J,X)
    int64_t nchunks
)
{

    //--------------------------------------------------------------------------
    // workspace for each threadblock
    //--------------------------------------------------------------------------

    __shared__ Int Local_JDelta [CHUNKSIZE] ;

    // cub::Block* workspace:
    GB_CUB_BLOCK_WORKSPACE (W, Int, BLOCKDIM, ITEMS_PER_THREAD) ;

    //--------------------------------------------------------------------------
    // the first thread of the threadblock fills in the sentinal values
    //--------------------------------------------------------------------------

    if (threadIdx.x == 0 && blockIdx.x == 0)
    {
        memset (&(Key_out [-1]), 0xFF, sizeof (GB_key_t)) ;
        JDelta [-1] = 0 ;
        JDeltaSum [-1] = 0 ;
    }

    // this_thread_block ( ).sync ( ) ; not needed since the thread that wrote
    // the Key_out [-1] entry is the only thread that reads it.

    //--------------------------------------------------------------------------
    // compute each local chunk of Map
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
        // this computation is just the 2nd #if case of select/phase1:
        int64_t plast = pfirst + CHUNKSIZE ;
        plast = GB_IMIN (plast, nvals) ;
        my_chunk_size = plast - pfirst ;

        //----------------------------------------------------------------------
        // determine the first unique tuple in each sequence of duplicates
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
            // determine if the p-th entry is 1st of duplicates, or leading
            //------------------------------------------------------------------

            // get the indices
            GB_KEY_UNLOAD_J (Key_out, p-1, jprev) ;
            GB_KEY_UNLOAD_J (Key_out, p,   j) ;

            // leading is true if this is the first entry in vector j
            bool leading = (j != jprev) ;
            Local_JDelta [pdelta] = leading ;   // 1 if leading entry of vector
        }

        //----------------------------------------------------------------------
        // the remainder is similar to select/phase1:
        //----------------------------------------------------------------------

        // clear the unused part of the Local_Map and Local_JDelta
        for ( ; pdelta < CHUNKSIZE ;
                pdelta += blockDim.x)
        {
            Local_JDelta [pdelta] = 0 ;
        }

        //----------------------------------------------------------------------
        // inclusive cumulative sum of Local_Map and Local_JDelta
        //----------------------------------------------------------------------

        // Map [pfirst..pfirst+CHUNKSIZE-1] = inclusive cumsum of Local_Map,
        // where Local_Map [i] = sum (Local_Map [0:i]) is computed.

        // Similarly, JDelta [pfirst..pfirst+CHUNKSIZE-1] = inclusive cumsum
        // of Local_JDelta, where Local_JDelta [i] = sum (Local_JDelta [0:i])
        // is computed.

        this_thread_block ( ).sync ( ) ;
        Int s_block_aggregate ;

#if 0
        // This entire phase computes the following:
        if (threadIdx.x == blockDim.x - 1)
        {

            // construct JDelta and JDeltaSum
            for (int i = 1 ; i < CHUNKSIZE ; i++)
            {
                Local_JDelta [i] += Local_JDelta [i-1] ;
            }
            for (int i = 0 ; i < CHUNKSIZE ; i++)
            {
                JDelta [pfirst + i] = Local_JDelta [i] ;
            }
            s_block_aggregate = Local_JDelta [CHUNKSIZE-1] ;
            JDeltaSum [chunk] = s_block_aggregate ;
        }

#else
        Int t [ITEMS_PER_THREAD] ;

        BlockLoad (W.load).Load (Local_JDelta, t) ;
        this_thread_block ( ).sync ( ) ;
        BlockScan (W.scan).InclusiveSum (t, t, s_block_aggregate) ;
        this_thread_block ( ).sync ( ) ;
        BlockStore (W.store).Store (JDelta + pfirst, t) ;
        this_thread_block ( ).sync ( ) ;

        // finally, the aggregate sums are written to ChunkSum and JDeltaSum
        if (threadIdx.x == blockDim.x - 1)
        {
            JDeltaSum [chunk] = s_block_aggregate ;
        }

#endif

        this_thread_block ( ).sync ( ) ;
    }
}

#endif

//------------------------------------------------------------------------------
// GB_cuda_builder_phase5_with_dupl
//------------------------------------------------------------------------------

// phase5 constructs the output matrix T (Tp, Th, Ti, and Tx) from the
// (Key_out,Sx) tuples, summing up duplicates (at least one duplicate is
// present).

// compare with select/phase3 and select/phase6

#if !GB_KNOWN_NO_DUPLICATES
__global__ void GB_cuda_builder_phase5_with_dupl
(
    // outputs
    GrB_Matrix T,
    // inputs, not modified:
    Int *Map,               // size nvals+1, in Map [-1...nvals-1]
    GB_Tp_TYPE *ChunkSum,   // size nchunks+1, in ChunkSum [-1..nchunks]
    #if GB_MTX_BUILD
    Int *JDelta,            // size nvals+1, in JDelta [-1..nvals-1]
    GB_Tp_TYPE *JDeltaSum,  // size nchunks+1
    #endif
    GB_key_t *Key_out,      // size nvals+1: Key_out [-1 ... nvals-1]
    GB_Sx_TYPE *Sx,         // size nvals: Sx [0 ... nvals-1]
    int64_t nvals,          // # of tuples in (I,J,X)
    int64_t nchunks
)
{

    //--------------------------------------------------------------------------
    // get T->p, T->h, T->i, and T->x, shifting down by 1 since Map is 1-based
    //--------------------------------------------------------------------------

    GB_Tp_TYPE *__restrict__ Tp = (GB_Tp_TYPE *) T->p ; Tp-- ;
    #if GB_MTX_BUILD
    GB_Tj_TYPE *__restrict__ Th = (GB_Tj_TYPE *) T->h ; Th-- ;
    #endif
    GB_Ti_TYPE *__restrict__ Ti = (GB_Ti_TYPE *) T->i ; Ti-- ;
    #if !GB_ISO_BUILD
    GB_Tx_TYPE *__restrict__ Tx = (GB_Tx_TYPE *) T->x ; Tx-- ;
    #endif

    //--------------------------------------------------------------------------
    // copy the entries from (Key_out,Sx) into Tp, Th, Ti, and Tx, summing dupls
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
        // this computation is just the 2nd #if case of select/phase3:
        int64_t plast = pfirst + CHUNKSIZE ;
        plast = GB_IMIN (plast, nvals) ;
        my_chunk_size = plast - pfirst ;

        //----------------------------------------------------------------------
        // copy the entries, sum duplicates, and construct Tp and Th
        //----------------------------------------------------------------------

        for (int64_t pdelta = threadIdx.x ;
                     pdelta < my_chunk_size ;
                     pdelta += blockDim.x)       // block-stride loop
        {

            int64_t p = pfirst + pdelta ;

            //------------------------------------------------------------------
            // copy the entries, summing the duplicates
            //------------------------------------------------------------------

            // get the position pT in C of the p-th tuple in (Key_out,Sx)
            GB_Tp_TYPE pT = Map [p  ] + ChunkSum [chunk] ;
            // get the position p0 in C of the (p-1)-st tuple in (Key_out,Sx)
            GB_Tp_TYPE p0 = Map [p-1] + ChunkSum [chunk - (pdelta == 0)] ;
            if (p0 < pT)
            {
                // This entry is the first in a sequence of duplicates (perhaps
                // just a single entry with no duplicates)
                // Ti [pT] = Key_out [p].i ;
                GB_KEY_UNLOAD_I (Key_out, p, i1) ;
                Ti [pT] = (GB_Ti_TYPE) i1 ;

                #if !GB_ISO_BUILD
                GB_BLD_COPY (Tx, pT, Sx, p) ; // Tx [pT] = Sx [p]

                #if !defined (GB_DUP_IS_FIRST)
                // Sum up all duplicate entries, in order.  This can cross over
                // into subsequent chunks of (Key_out,Sx).  Warp divergence is
                // expected, but it should be OK since only a modest O(1)
                // number of duplicate are expected for each unique T(i,j)
                // entry.
                int64_t chunk2 = chunk ;
                bool is_duplicate = true ;
                for (uint64_t p2 = p+1 ; (p2 < nvals) && is_duplicate ; p2++)
                {
                    // get the next entry: increment the chunk2 of p2
                    chunk2 += ((p2 & (CHUNKSIZE-1)) == 0) ;
                    GB_Tp_TYPE pdupl = Map [p2] + ChunkSum [chunk2] ;
                    is_duplicate = (pT == pdupl) ;
                    if (is_duplicate)
                    {
                        // Tx [pT] += Sx [p2]
                        GB_BLD_DUP (Tx, pT, Sx, p2) ;
                    }
                }
                #endif
                #endif
            }

            //------------------------------------------------------------------
            // construct Tp and Th, if T is a matrix (skip if T is a vector)
            //------------------------------------------------------------------

            #if GB_MTX_BUILD
            GB_Tp_TYPE kT = JDelta [p  ] + JDeltaSum [chunk] ;
            GB_Tp_TYPE k0 = JDelta [p-1] + JDeltaSum [chunk - (pdelta == 0)] ;
            if (k0 < kT)
            {
                // The p-th entry is the leading entry of the kT-th vector of T
                Tp [kT] = pT - 1 ;      // shift by 1 since pT is 1-based
                // Th [kT] = Key_out [p].j ;
                GB_KEY_UNLOAD_J (Key_out, p, j1) ;
                Th [kT] = j1 ;
            }
            #endif
        }
    }

    //--------------------------------------------------------------------------
    // finalize the last vector of C
    //--------------------------------------------------------------------------

    if (threadIdx.x == 0 && blockIdx.x == 0)
    {
        // T->nvec is 0-based, so increment Tp to undo the Tp-- done above
        Tp++ ;
        #if GB_MTX_BUILD
        Tp [T->nvec] = T->nvals ;
        #else
        Tp [0] = 0 ;
        Tp [1] = T->nvals ;
        #endif
    }
}
#endif

//------------------------------------------------------------------------------
// GB_cuda_builder_phase5_transplant
//------------------------------------------------------------------------------

// phase5 constructs the output matrix T (Tp, Th, Ti, and Tx) from the
// (Key_out,Sx) tuples, where no duplicates appear.

// compare with select/phase3 and select/phase6

// Sx has already been transplanted into T->x

#define GB_TRANSPLANT_IS_POSSIBLE (GB_BLD_SXTYPE_IS_TXTYPE && !GB_ISO_BUILD)

#if GB_TRANSPLANT_IS_POSSIBLE
__global__ void GB_cuda_builder_phase5_transplant
(
    // outputs
    GrB_Matrix T,
    // inputs, not modified:
    #if GB_MTX_BUILD
    Int *JDelta,            // size nvals+1, in JDelta [-1..nvals-1]
    GB_Tp_TYPE *JDeltaSum,  // size nchunks+1
    #endif
    GB_key_t *Key_out,      // size nvals+1: Key_out [-1 ... nvals-1]
    int64_t nvals,          // # of tuples in (I,J,X)
    int64_t nchunks
)
{

    //--------------------------------------------------------------------------
    // get T->p, T->h, and T->i. kT is 1-based but p is 0-based
    //--------------------------------------------------------------------------

    GB_Tp_TYPE *__restrict__ Tp = (GB_Tp_TYPE *) T->p ; Tp-- ; // index with kT
    #if GB_MTX_BUILD
    GB_Tj_TYPE *__restrict__ Th = (GB_Tj_TYPE *) T->h ; Th-- ; // index with kT
    #endif
    GB_Ti_TYPE *__restrict__ Ti = (GB_Ti_TYPE *) T->i ;        // index with p

    //--------------------------------------------------------------------------
    // copy the entries from (Key_out,Sx) into Tp, Th, Ti, and Tx, no duplicates
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
        // this computation is just the 2nd #if case of select/phase3:
        int64_t plast = pfirst + CHUNKSIZE ;
        plast = GB_IMIN (plast, nvals) ;
        my_chunk_size = plast - pfirst ;

        //----------------------------------------------------------------------
        // copy the entries, sum duplicates, and construct Tp and Th
        //----------------------------------------------------------------------

        for (int64_t pdelta = threadIdx.x ;
                     pdelta < my_chunk_size ;
                     pdelta += blockDim.x)       // block-stride loop
        {

            int64_t p = pfirst + pdelta ;

            //------------------------------------------------------------------
            // copy the entries
            //------------------------------------------------------------------

            // FIXME: break this into 2 loops?  For Ti, and (Tp,Th)

            // Ti [p] = Key_out [p].i ;
            GB_KEY_UNLOAD_I (Key_out, p, i1) ;
            Ti [p] = (GB_Ti_TYPE) i1 ;

            //------------------------------------------------------------------
            // construct Tp and Th, if T is a matrix (skip if T is a vector)
            //------------------------------------------------------------------

            #if GB_MTX_BUILD
            GB_Tp_TYPE kT = JDelta [p  ] + JDeltaSum [chunk] ;
            GB_Tp_TYPE k0 = JDelta [p-1] + JDeltaSum [chunk - (pdelta == 0)] ;
            if (k0 < kT)
            {
                // The p-th entry is the leading entry of the kT-th vector of T
                Tp [kT] = p ;       // p is already 0-based
                // Th [kT] = Key_out [p].j ;
                GB_KEY_UNLOAD_J (Key_out, p, j1) ;
                Th [kT] = j1 ;
            }
            #endif
        }
    }

    //--------------------------------------------------------------------------
    // finalize the last vector of C
    //--------------------------------------------------------------------------

    if (threadIdx.x == 0 && blockIdx.x == 0)
    {
        // T->nvec is 0-based, so increment Tp to undo the Tp-- done above
        Tp++ ;
        #if GB_MTX_BUILD
        Tp [T->nvec] = T->nvals ;
        #else
        Tp [0] = 0 ;
        Tp [1] = T->nvals ;
        #endif
    }
}
#endif

//------------------------------------------------------------------------------
// GB_cuda_builder_phase5_no_dupl
//------------------------------------------------------------------------------

// phase5 constructs the output matrix T (Tp, Th, Ti, and Tx) from the
// (Key_out,Sx) tuples, where no duplicates appear.

// compare with select/phase3 and select/phase6

__global__ void GB_cuda_builder_phase5_no_dupl
(
    // outputs
    GrB_Matrix T,
    // inputs, not modified:
    #if GB_MTX_BUILD
    Int *JDelta,            // size nvals+1, in JDelta [-1..nvals-1]
    GB_Tp_TYPE *JDeltaSum,  // size nchunks+1
    #endif
    GB_key_t *Key_out,      // size nvals+1: Key_out [-1 ... nvals-1]
    GB_Sx_TYPE *Sx,         // size nvals+1: Sx  [-1 ... nvals-1]
    int64_t nvals,          // # of tuples in (I,J,X)
    int64_t nchunks
)
{

    //--------------------------------------------------------------------------
    // get T->p, T->h, T->i, and T->x. kT is 1-based but p is 0-based
    //--------------------------------------------------------------------------

    GB_Tp_TYPE *__restrict__ Tp = (GB_Tp_TYPE *) T->p ; Tp-- ; // index with kT
    #if GB_MTX_BUILD
    GB_Tj_TYPE *__restrict__ Th = (GB_Tj_TYPE *) T->h ; Th-- ; // index with kT
    #endif
    GB_Ti_TYPE *__restrict__ Ti = (GB_Ti_TYPE *) T->i ;        // index with p
    #if !GB_ISO_BUILD
    GB_Tx_TYPE *__restrict__ Tx = (GB_Tx_TYPE *) T->x ;        // index with p
    #endif

    //--------------------------------------------------------------------------
    // copy the entries from (Key_out,Sx) into Tp, Th, Ti, and Tx, no duplicates
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
        // this computation is just the 2nd #if case of select/phase3:
        int64_t plast = pfirst + CHUNKSIZE ;
        plast = GB_IMIN (plast, nvals) ;
        my_chunk_size = plast - pfirst ;

        //----------------------------------------------------------------------
        // copy the entries, sum duplicates, and construct Tp and Th
        //----------------------------------------------------------------------

        for (int64_t pdelta = threadIdx.x ;
                     pdelta < my_chunk_size ;
                     pdelta += blockDim.x)       // block-stride loop
        {

            int64_t p = pfirst + pdelta ;

            //------------------------------------------------------------------
            // copy the entries
            //------------------------------------------------------------------

            // FIXME: break this into 3 loops?  For Ti, Tx, and (Tp,Th)

            // Ti [p] = Key_out [p].i ;
            GB_KEY_UNLOAD_I (Key_out, p, i1) ;
            Ti [p] = (GB_Ti_TYPE) i1 ;

            #if !GB_ISO_BUILD
            GB_BLD_COPY (Tx, p, Sx, p) ;       // Tx [p] = Sx [p]
            #endif

            //------------------------------------------------------------------
            // construct Tp and Th, if T is a matrix (skip if T is a vector)
            //------------------------------------------------------------------

            #if GB_MTX_BUILD
            GB_Tp_TYPE kT = JDelta [p  ] + JDeltaSum [chunk] ;
            GB_Tp_TYPE k0 = JDelta [p-1] + JDeltaSum [chunk - (pdelta == 0)] ;
            if (k0 < kT)
            {
                // The p-th entry is the leading entry of the kT-th vector of T
                Tp [kT] = p ;       // p is already 0-based
                // Th [kT] = Key_out [p].j ;
                GB_KEY_UNLOAD_J (Key_out, p, j1) ;
                Th [kT] = j1 ;
            }
            #endif
        }
    }

    //--------------------------------------------------------------------------
    // finalize the last vector of C
    //--------------------------------------------------------------------------

    if (threadIdx.x == 0 && blockIdx.x == 0)
    {
        // T->nvec is 0-based, so increment Tp to undo the Tp-- done above
        Tp++ ;
        #if GB_MTX_BUILD
        Tp [T->nvec] = T->nvals ;
        #else
        Tp [0] = 0 ;
        Tp [1] = T->nvals ;
        #endif
    }
}

//------------------------------------------------------------------------------
// builder, host method
//------------------------------------------------------------------------------

extern "C"
{
    GB_JIT_CUDA_KERNEL_BUILDER_PROTO (GB_jit_kernel) ;
}

// #undef GB_TIMING
#define GB_TIMING

GB_JIT_CUDA_KERNEL_BUILDER_PROTO (GB_jit_kernel)
{

    //--------------------------------------------------------------------------
    // get callback functions
    //--------------------------------------------------------------------------

    #ifdef GB_TIMING
    double t1 = GB_OPENMP_GET_WTIME ;
    #endif
    #ifdef GB_JIT_RUNTIME
    // get callback functions
    GB_GET_CALLBACKS ;
    GB_GET_CALLBACK (GB_free_memory) ;
    GB_GET_CALLBACK (GB_malloc_memory) ;
    GB_GET_CALLBACK (GB_new_bix) ;
    GB_GET_CALLBACK (GB_Matrix_free) ;
    #endif

    //--------------------------------------------------------------------------
    // get inputs
    //--------------------------------------------------------------------------

    // GB_I_TYPE and GB_J_TYPE are uint32_t or uint64_t
    #if !GB_KEY_PRELOADED
    GB_I_TYPE  *I = (GB_I_TYPE  *) I_input ;
    GB_J_TYPE  *J = (GB_J_TYPE  *) J_input ;
    #endif
    GB_Sx_TYPE *X = (GB_Sx_TYPE *) X_input ;

    //--------------------------------------------------------------------------
    // declare workspace
    //--------------------------------------------------------------------------

    (*Thandle) = NULL ;
    GrB_Matrix T = NULL ;
    GrB_Info info = GrB_SUCCESS ;
    int memlane = GB_MEMLANE_RMM ;
    uint64_t mem = GB_mem (memlane, 0) ;

    // workspace needed for CUB radix sort of (Key_in,X):
    void *W_0 = NULL ; uint64_t W_0_mem = mem ;    // size nvals+1: Key_in
    void *W_1 = NULL ; uint64_t W_1_mem = mem ;    // size nvals+1: Key_out
    void *W_2 = NULL ; uint64_t W_2_mem = mem ;    // size nvals+1: Sx (or NULL)
    void *W_3 = NULL ; uint64_t W_3_mem = mem ;    // size nvals+1: CUB work

    // when the CUB radix sort is done, Key_in and the CUB workspace can
    // be freed.

    // workspace needed after CUB radix sort:
    void *W_4 = NULL ; uint64_t W_4_mem = mem ;    // size nvals+1: Map
    void *W_5 = NULL ; uint64_t W_5_mem = mem ;    // size nchunks+2: ChunkSum
    void *W_6 = NULL ; uint64_t W_6_mem = mem ;    // size nvals+1: JDelta
    void *W_7 = NULL ; uint64_t W_7_mem = mem ;    // size nchunks+2: JDeltaSum
    void *W_8 = NULL ; uint64_t W_8_mem = mem ;    // size 2: scalar workspace

    // # of entries, chunks, and vectors of T
    int64_t tnz = 0 ;   // # of unique tuples, and # of entries in T
    int64_t tnvec = 0 ; // # of vectors of T
    int64_t nchunks = (nvals + CHUNKSIZE - 1) >> LOG2_CHUNKSIZE ;

    dim3 grid (gridsz) ;        // = min (ceil (nvals/CHUNKSIZE), 256*(#sms))
    dim3 block1 (BLOCKDIM) ;

    //--------------------------------------------------------------------------
    // phase1: load the Key_in workspace and check if indices are in range
    //--------------------------------------------------------------------------

    // Example, assuming a chunksize of 4, with 14 tuples and 3 duplicates
    // marked with (*).  The first entry of each set of duplicates is marked
    // with (^), but this is not detected in phase1.

    // J:          [ 0 1 0 0 | 0 1 1 1 | 2 2 4 1 | 4 4 ]
    // I:          [ 0 3 1 2 | 1 4 5 5 | 3 4 3 5 | 0 1 ]
    // dupl:             ^     *   ^ *         *

    // builder/phase1 loads the (I,J) tuples into Key_in and ensures the
    // indices are in range.  If the input keys are preloaded in Key_input,
    // they are assumed to be valid, and not checked.

    #if GB_KEY_PRELOADED

        //----------------------------------------------------------------------
        // the caller has pre-loaded Key_in
        //----------------------------------------------------------------------

        GB_key_t *Key_in = ((GB_key_t *) Key_input) + 1 ;

        // Key_in is not checked for duplicates or sorted status at run time;
        // it only uses the cases known at JIT compile time
        #define known_no_duplicates GB_KNOWN_NO_DUPLICATES
        #define known_sorted        GB_KNOWN_SORTED

    #else

        //----------------------------------------------------------------------
        // load Key_in from I and J
        //----------------------------------------------------------------------

        // allocate Key_in and W_8 workspace
        W_0 = GB_MALLOC_MEMORY (nvals+1, sizeof (GB_key_t), &W_0_mem) ;
        W_8 = GB_MALLOC_MEMORY (3, sizeof (uint64_t), &W_8_mem) ;
        if (W_0 == NULL || W_8 == NULL)
        {
            // out of memory
            GB_FREE_ALL ;
            return (GrB_OUT_OF_MEMORY) ;
        }
        // shift by one so Key_in [-1...nvals-1] can be used
        GB_key_t *Key_in = ((GB_key_t *) W_0) + 1 ;

        // get 2 uint64_t scalars from the W_8 workspace
        uint64_t *bad = ((uint64_t *) W_8) ;
        uint64_t *unsorted = ((uint64_t *) W_8) + 1 ;
        uint64_t *dupls = ((uint64_t *) W_8) + 2 ;
        (*bad) = 0 ;
        (*unsorted) = 0 ;
        (*dupls) = 0 ;

        // FIXME: if I or J cannot be read by the GPU, do phase1 on the CPU
        // with OpenMP

        GB_cuda_builder_phase1 <<<grid, block1, 0, stream>>>
            (/* outputs: */ Key_in, bad,
                #if !GB_KNOWN_SORTED
                unsorted,
                #endif
                #if !GB_KNOWN_NO_DUPLICATES
                dupls,
                #endif
             /* inputs: */ I, vlen,
                #if GB_MTX_BUILD
                J, vdim,
                #endif
                nvals) ;

        CUDA_OK (cudaGetLastError ( )) ;
        CUDA_OK (cudaStreamSynchronize (stream)) ;

        // after the CUDA kernel launch is done, check if (I,J) indices are OK
        GB_OK (((*bad) == 0) ? GrB_SUCCESS : GrB_INVALID_INDEX) ;

        #if GB_KNOWN_SORTED
        // tuples are asserted to be sorted on input; this is not checked
        #define known_sorted true
        #else
        // check if tuples are sorted on input
        bool known_sorted = (*unsorted == 0) ;
        #endif

        #if GB_KNOWN_NO_DUPLICATES
        // tuples are asserted to have no duplicates on input
        #define known_no_duplicates true
        #else
        // check if tuples have no duplicates; if the tuples are sorted then
        // dupls is a valid count of the # of duplicates
        bool known_no_duplicates = known_sorted && (*dupls == 0) ;
        #endif

    #endif

    #ifdef GB_TIMING
    t1 = GB_OPENMP_GET_WTIME - t1 ;
    printf ("builder phase 1: %g sec,", t1) ;
    printf (" known_sorted: %d, known_no_duplicates: %d\n",
        known_sorted, known_no_duplicates) ;
    double t2 = GB_OPENMP_GET_WTIME ;
    #endif

    // Inputs I,J are no longer needed.

    //--------------------------------------------------------------------------
    // phase2: CUB radix sort of (Key_in,X) to obtain (Key_out,Sx)
    //--------------------------------------------------------------------------

    // builder/phase2 sorts the tuples in (Key_in,X) to obtain (Key_out,Sx).
    // The dupl state (of first entry in sequence of duplicates), or leading
    // state (if the entry is the first in its vector of C) are not yet
    // computed, but shown below for reference.  The (@) denotes the leading
    // entry of each vector in C.

    // output of builder/phase2 (Sx values not shown):
    // Key_out.j:inf [ 0 0 0 0 | 1 1 1 1 | 1 2 2 4 | 4 4 ]
    // Key_out.i:inf [ 0 1 1 2 | 3 4 5 5 | 5 3 4 0 | 1 3 ]
    // dupl:             ^ *         ^ *   *             <--1st entry in dupls
    // leading:        @         @           @   @       <--1st of vectors in C

    // FIXME: if X cannot be read by the GPU, then copy it from X into
    // another workspace allocated on the GPU using OpenMP, before doing
    // phase2.

    GB_key_t *Key_out ;
    GB_Sx_TYPE *Sx ;
    bool Sx_is_workspace = false ;  // true if Sx is allocated

    #if GB_KNOWN_SORTED
    {

        //----------------------------------------------------------------------
        // tuples are known to be sorted on input
        //----------------------------------------------------------------------

        Key_out = Key_in ;          // either Key_input, or allocated workspace
        Sx = X ;                    // Sx is X, and is not allocated

    }
    #else
    {

        //----------------------------------------------------------------------
        // tuples are not known to be sorted on input; but may be found to be so
        //----------------------------------------------------------------------

        if (known_sorted)
        {

            //------------------------------------------------------------------
            // tuples have been checked and found to be already sorted
            //------------------------------------------------------------------

            Key_out = Key_in ;      // either Key_input, or allocated workspace
            Sx = X ;                // Sx is X, and is not allocated

        }
        else
        {

            //------------------------------------------------------------------
            // tuples must be sorted
            //------------------------------------------------------------------

            // allocate Key_out, Sx, and CUB temporary workspace
            W_1 = GB_MALLOC_MEMORY (nvals+1, sizeof (GB_key_t), &W_1_mem) ;
            #if !GB_ISO_BUILD
            W_2 = GB_MALLOC_MEMORY (nvals+1, sizeof (GB_Sx_TYPE), &W_2_mem) ;
            #endif
            if (W_1 == NULL || (!GB_ISO_BUILD && W_2 == NULL))
            {
                // out of memory
                GB_FREE_ALL ;
                return (GrB_OUT_OF_MEMORY) ;
            }

            // shift by one so Key_out [-1...nvals-1], etc can be used
            Key_out = ((GB_key_t *) W_1) + 1 ;

            // no need to shift Sx
            Sx = ((GB_Sx_TYPE *) W_2) ;
            #if !GB_ISO_BUILD
            Sx_is_workspace = true ;    // Sx is allocated workspace
            #endif

            size_t W_3_memsize = 0 ;

            // determine the amount of workspace needed by CUB radix sort
            #if GB_ISO_BUILD
            CUDA_OK (cub::DeviceRadixSort::SortKeys (
                /* temp storage: */ W_3, W_3_memsize,
                Key_in, Key_out, nvals,
                #if GB_MTX_BUILD
                GB_key_decomposer_t { },
                #endif
                /* begin/end bits: */ 0, sizeof (GB_key_t) * 8,
                stream)) ;
            #else
            CUDA_OK (cub::DeviceRadixSort::SortPairs (
                /* temp storage: */ W_3, W_3_memsize,
                Key_in, Key_out,
                /* values in: */ X, /* values out: */ Sx, nvals,
                #if GB_MTX_BUILD
                GB_key_decomposer_t { },
                #endif
                /* begin/end bits: */ 0, sizeof (GB_key_t) * 8,
                stream)) ;
            #endif

            CUDA_OK (cudaGetLastError ( )) ;
            CUDA_OK (cudaStreamSynchronize (stream)) ;

            // allocate workspace for CUB radix sort
            W_3 = GB_MALLOC_MEMORY (W_3_memsize+1, 1, &W_3_mem) ;
            if (W_3 == NULL)
            {
                // out of memory
                GB_FREE_ALL ;
                return (GrB_OUT_OF_MEMORY) ;
            }

            // sort (Key_in,X) to get (Key_out,Sx)
            #if GB_ISO_BUILD
            CUDA_OK (cub::DeviceRadixSort::SortKeys (
                /* temp storage: */ W_3, W_3_memsize,
                Key_in, Key_out, nvals,
                #if GB_MTX_BUILD
                GB_key_decomposer_t { },
                #endif
                /* begin/end bits: */ 0, sizeof (GB_key_t) * 8,
                stream)) ;
            #else
            CUDA_OK (cub::DeviceRadixSort::SortPairs (
                /* temp storage: */ W_3, W_3_memsize,
                Key_in, Key_out,
                /* values in: */ X, /* values out: */ Sx, nvals,
                #if GB_MTX_BUILD
                GB_key_decomposer_t { },
                #endif
                /* begin/end bits: */ 0, sizeof (GB_key_t) * 8,
                stream)) ;
            #endif

            CUDA_OK (cudaGetLastError ( )) ;
            CUDA_OK (cudaStreamSynchronize (stream)) ;

            // Key_in and CUB workspace no longer needed
            GB_FREE_MEMORY (&W_0, W_0_mem) ;
            GB_FREE_MEMORY (&W_3, W_3_mem) ;
        }
    }
    #endif

    // sorted tuples are now in (Key_out,Sx).  Inputs I,J,X and Key_input are
    // no longer needed (unless Key_out is aliased to Key_input, and unless Sx
    // is aliased to X).  Workspace Key_in is no longer needed (either
    // Key_input or allocated W_2 workspace).

    Key_in = NULL ;

    //--------------------------------------------------------------------------
    // sanity check
    //--------------------------------------------------------------------------

    #if 0
    {
        printf ("\nafter sort:\n") ;
        bool ok4 = true ;
        for (int64_t p = 0 ; p < nvals ; p++)
        {
            int64_t j, jprev, i, iprev ;
            if (p == 0)
            {
                iprev = -1 ;
                jprev = -1 ;
            }
            else
            {
                GB_KEY_UNLOAD_I (Key_out, p-1, i0) ;
                GB_KEY_UNLOAD_J (Key_out, p-1, j0) ;
                iprev = i0 ;
                jprev = j0 ;
            }
            GB_KEY_UNLOAD_I (Key_out, p, i1) ;
            GB_KEY_UNLOAD_J (Key_out, p, j1) ;
            i = i1 ;
            j = j1 ;
            bool outoforder = (jprev > j) || (jprev == j && iprev > i) ;
            if (outoforder)
            {
                printf ("p: %ld, iprev,jprev: (%ld,%ld), i,j (%ld,%ld) "
                    "out of order!\n", p,iprev,jprev, i,j) ;
                ok4 = false ;
            }
            bool bad_index = (j > vdim) || (i > vlen) ;
            if (bad_index)
            {
                printf ("out of range!\n") ;
                return (GrB_PANIC) ;
            }
        }
        if (!ok4) { printf ("out of order!\n") ; return (GrB_PANIC) ; }
    }
    #endif

    #ifdef GB_TIMING
    t2 = GB_OPENMP_GET_WTIME - t2 ;
    printf ("builder phase 2: %g sec\n", t2) ;
    double t3 = GB_OPENMP_GET_WTIME ;
    #endif

    //--------------------------------------------------------------------------
    // phase3: look for duplicates (compare with phase1 of CUDA/select)
    //--------------------------------------------------------------------------

    // builder/phase3 determines which entries are the first in a sequence of
    // duplicates (output: Map), and which entries are the first in their
    // respective vectors (output: JDelta).  The output after builder/phase3 is
    // shown below, where LMap = Local_Map (a temporary shared array in each
    // threadblock of phase3).  Map is the cumsum of each chunk of LMap.  Note
    // the padding of LMap, Map, LJDelta, and JDelta.  Any given sequence of
    // duplicates can span across the chunks (the (1,5) entry does so).

    // LJDelta is 1 if the entry is first in its vector (a leading entry), held
    // in a temporary shared array in each threadblock.  It is computed from
    // Key_out.j only and is not affected by the presence of duplicates.
    // JDelta is the cumsum of chunk of LJDelta.

    // inputs:
    // Key_out.j:inf [ 0 0 0 0 | 1 1 1 1 | 1 2 2 4 | 4 4     ]
    // Key_out.i:inf [ 0 1 1 2 | 3 4 5 5 | 5 3 4 0 | 1 3     ]
    // output:
    // LMap:         [ 1 1 0 1 | 1 1 1 0 | 0 1 1 1 | 1 1 - - ]
    // Map:        0 [ 1 2 2 3 | 1 2 3 3 | 0 1 2 3 | 1 2 2 2 ]
    // LJDelta:    0 [ 1 0 0 0 | 1 0 0 0 | 0 1 0 1 | 0 0 0 0 ]
    // JDelta:     0 [ 1 1 1 1 | 1 1 1 1 | 0 1 1 2 | 0 0 0 0 ]
    // dupl:             ^ *         ^ *   *             <--1st entry in dupls
    // leading:        @         @           @   @       <--1st of vectors in C

    // ChunkSum [-1..nchunks] is the # of non-duplicates to be kept in each
    // chunk, with ChunkSum [-1] = 0 and ChunkSum [nchunks] = 0 for now.  This
    // matrix will have 11 total entries, which is the total sum of ChunkSum:
    //            0 [       3 |       3 |       3 |       2 ] 0

    // JDeltaSum [-1..nchunks] is the # of leading entries in each chunk,
    // with JDeltaSum [-1] = 0.  This matrix has 4 unique values of J (0, 1, 2,
    // and 4), which is the total sum of JDeltaSum:
    //            0 [       1 |       1 |       2 |       0 ] 0

    // allocate Map and ChunkSum: for cumsum of 1st entries in sequence of dupls
    #if !GB_KNOWN_NO_DUPLICATES
    if (!known_no_duplicates)
    {
        W_4 = GB_MALLOC_MEMORY (nvals+1 + CHUNKSIZE, sizeof (Int), &W_4_mem) ;
        W_5 = GB_MALLOC_MEMORY (nchunks+2, sizeof (GB_Tp_TYPE), &W_5_mem) ;
        if (W_4 == NULL || W_5 == NULL)
        {
            // out of memory
            GB_FREE_ALL ;
            return (GrB_OUT_OF_MEMORY) ;
        }
    }
    #endif

    // allocate JDelta, JDeltaSum: for cumsum of leading entries of vectors of C
    #if GB_MTX_BUILD
    W_6 = GB_MALLOC_MEMORY (nvals+1 + CHUNKSIZE, sizeof (Int), &W_6_mem) ;
    W_7 = GB_MALLOC_MEMORY (nchunks+2, sizeof (GB_Tp_TYPE), &W_7_mem) ;
    if (W_6 == NULL || W_7 == NULL)
    {
        // out of memory
        GB_FREE_ALL ;
        return (GrB_OUT_OF_MEMORY) ;
    }
    #endif

    #if !GB_KNOWN_NO_DUPLICATES
    // shift by one so Map [-1...nvals-1], etc can be used
    Int *Map              = ((Int        *) W_4) + 1 ;
    GB_Tp_TYPE *ChunkSum  = ((GB_Tp_TYPE *) W_5) + 1 ;
    #endif

    #if GB_MTX_BUILD
    Int *JDelta           = ((Int        *) W_6) + 1 ;
    GB_Tp_TYPE *JDeltaSum = ((GB_Tp_TYPE *) W_7) + 1 ;
    #endif

    #if GB_KNOWN_NO_DUPLICATES
    {
        // phase3 does not need to look for duplicates; this is known
        // at compile time of the JIT kernel
        #if GB_MTX_BUILD
        GB_cuda_builder_phase3_no_dupl <<<grid, block1, 0, stream>>>
            ( /* outputs: */
                JDelta, JDeltaSum,
              /* inputs: */ Key_out, nvals, nchunks) ;
        #endif
    }
    #else
    {
        if (known_no_duplicates)
        {
            // phase3 does not need to look for duplicates; this is known
            // only after checking for duplicates
            #if GB_MTX_BUILD
            GB_cuda_builder_phase3_no_dupl <<<grid, block1, 0, stream>>>
                ( /* outputs: */
                    JDelta, JDeltaSum,
                  /* inputs: */ Key_out, nvals, nchunks) ;
            #endif
        }
        else
        {
            GB_cuda_builder_phase3_with_dupl <<<grid, block1, 0, stream>>>
                ( /* outputs: */ Map, ChunkSum,
                    #if GB_MTX_BUILD
                    JDelta, JDeltaSum,
                    #endif
                  /* inputs: */ Key_out, nvals, nchunks) ;
        }
    }
    #endif

    CUDA_OK (cudaGetLastError ( )) ;
    CUDA_OK (cudaStreamSynchronize (stream)) ;

    //--------------------------------------------------------------------------
    // sanity check
    //--------------------------------------------------------------------------

    #if 0
    printf ("\nChecking phase3 output:\n") ;
    {
        int64_t alljdsum = 0 ;
        bool ok3 = true ;
        bool ok4 = true ;
        int64_t p = 0 ;
        for (int64_t chunk = 0 ; chunk < nchunks ; chunk++)
        {
            int64_t jdsum = 0 ;
            for (int c = 0 ; c < CHUNKSIZE ; c++, p++)
            {
                if (p >= nvals) break ;
                int64_t j, jprev, i, iprev ;
                if (p == 0)
                {
                    iprev = -1 ;
                    jprev = -1 ;
                }
                else
                {
                    GB_KEY_UNLOAD_I (Key_out, p-1, i0) ;
                    GB_KEY_UNLOAD_J (Key_out, p-1, j0) ;
                    iprev = i0 ;
                    jprev = j0 ;
                }
                GB_KEY_UNLOAD_I (Key_out, p, i1) ;
                GB_KEY_UNLOAD_J (Key_out, p, j1) ;
                i = i1 ;
                j = j1 ;
                jdsum += (j != jprev) ;
//              printf ("Key_out [%ld] : (i,j): (%ld, %ld)\n", p, i, j) ;
                bool outoforder = (jprev > j) || (jprev == j && iprev > i) ;
                if (outoforder)
                {
                    printf ("chunk %ld: p: %ld, iprev,jprev: (%ld,%ld), "
                        "i,j (%ld,%ld) out of order!\n",
                        chunk,p,iprev,jprev, i,j) ;
                    ok4 = false ;
                }
                bool bad_index = (j > vdim) || (i > vlen) ;
                if (bad_index)
                {
                    printf ("out of range!\n") ;
                    return (GrB_PANIC) ;
                }
            }
            if (jdsum != JDeltaSum [chunk])
            {
                printf ("chunk %ld jdsum %ld %ld\n", chunk, jdsum,
                    (int64_t) JDeltaSum [chunk]) ;
                ok3 = false ;
            }
            alljdsum += jdsum ;
        }
        printf ("alljdsum: %ld\n", alljdsum) ;
        if (!ok4) { printf ("out of order!\n") ; return (GrB_PANIC) ; }
        if (!ok3) { printf ("JDeltaSum is bad!\n") ; return (GrB_PANIC) ; }
    }
    #endif

    #ifdef GB_TIMING
    t3 = GB_OPENMP_GET_WTIME - t3 ;
    printf ("builder phase 3: %g sec\n", t3) ;
    double t4 = GB_OPENMP_GET_WTIME ;
    #endif

    //--------------------------------------------------------------------------
    // phase4: sum up the unique entries in each chunk (on the CPU)
    //--------------------------------------------------------------------------

    // compare with phase2 of CUDA/select

    // At the start of phase4, ChunkSum [-1..nchunks] holds the # of
    // unique entries in C from each chunk of (I,J,X), with tnz = 11 unique
    // entries in C for this example:

    // ChunkSum on input (x is 'dont care'):
    //            x [       3 |       3 |       3 |       2 ] 0
    // ChunkSum on output (an exclusive sum):
    //            0 [       0 |       3 |       6 |       9 ] 11

    // JDeltaSum on input:
    //            x [       1 |       1 |       2 |       0 ] 0
    // JDeltaSum on output (an exclusive sum):
    //            0 [       0 |       1 |       2 |       4 ] 4

    #if !GB_KNOWN_NO_DUPLICATES
    if (!known_no_duplicates)
    {
        ChunkSum [-1] = 0 ;         // sentinel value
    }
    #endif

    #if GB_MTX_BUILD
    JDeltaSum [-1] = 0 ;        // sentinel value
    #endif

    // overwrite ChunkSum [0..gridsz] with its cumulative sum
    for (int64_t chunk = 0 ; chunk < nchunks ; chunk++)
    {
        #if !GB_KNOWN_NO_DUPLICATES
        if (!known_no_duplicates)
        {
            // get the # of entries found in this chunk
            int64_t t = ChunkSum [chunk] ;
            // overwrite the entry with the cumulative sum, so that the new
            // ChunkSum [chunk] = original ChunkSum [0..chunk-1]
            ChunkSum [chunk] = tnz ;
            tnz += t ;
        }
        #endif

        // get the # of leading entries found in this chunk
        #if GB_MTX_BUILD
        int64_t s = JDeltaSum [chunk] ;
        // overwrite the entry with the cumulative sum, so that the new
        // JDeltaSum [chunk] = original JDeltaSum [0..chunk-1]
        JDeltaSum [chunk] = tnvec ;
        tnvec += s ;
        #endif
    }

    #if GB_KNOWN_NO_DUPLICATES
    {
        // tuples are known to have no duplicates on input
        tnz = nvals ;
    }
    #else
    {
        // tuples might have duplicates
        if (known_no_duplicates)
        {
            tnz = nvals ;
        }
        else
        {
            ChunkSum [nchunks] = tnz ;
        }
    }
    #endif

    #if GB_MTX_BUILD
    JDeltaSum [nchunks] = tnvec ;
    #else
    tnvec = 1 ;
    #endif

    #ifdef GB_TIMING
    t4 = GB_OPENMP_GET_WTIME - t4 ;
    printf ("builder phase 4: %g sec\n", t4) ;
    double t5 = GB_OPENMP_GET_WTIME ;
    #endif

    //--------------------------------------------------------------------------
    // phase5: allocate T and construct it
    //--------------------------------------------------------------------------

    // compare with phase3 of CUDA/select

    // builder/phase5 allocates the output T matrix (including T->p, T->h,
    // T->i, and T->x arrays), then moves the data from (Key_out,Sx) into
    // (Tp,Th,Ti,Tx), applying the dup operator to "sum" up the values of the
    // duplicates as it does so.  Each sequence of duplicates is a handled by a
    // single thread in a single threadblock.  This assumes there are not many
    // duplicates for each entry.

    // input:
    // Key_out.j:inf [ 0 0 0 0 | 1 1 1 1 | 1 2 2 4 | 4 4     ]
    // Key_out.i:inf [ 0 1 1 2 | 3 4 5 5 | 5 3 4 0 | 1 3     ]
    // Map:        0 [ 1 2 2 3 | 1 2 3 3 | 0 1 2 3 | 1 2 2 2 ]
    // JDelta:     0 [ 1 1 1 1 | 1 1 1 1 | 0 1 1 2 | 0 0 0 0 ]
    // dupl:             ^ *         ^ *   *             <--1st entry in dupls
    // leading:        @         @           @   @       <--1st of vectors in C
    // ChunkSum:   0 [       0 |       3 |       6 |       9 ] 11
    // JDeltaSum   0 [       0 |       1 |       2 |       4 ] 4

    // Assume for this example that Sx [...] = 1, and dup is "+", so the value
    // of Tx is simply the number of duplicates of that entry.

    // The position where the p-th entry in (I,J,X) appears in T is given by
    // pT = Map [p] + ChunkSum [chunk], where pT is a 1-based index.  If this
    // position differs from the position of the (p-1)st entry in (I,J,X),
    // then the entry is the first unique tuple in its set of duplicates.

    // output with "|" denoting the chunks:
    // Ti:           [ 0 1 2 | 3 4 5 | 3 4 0 | 1 3 ]
    // Tx:           [ 1 2 1 | 1 1 3 | 1 1 1 | 1 1 ]
    // Tp:           [ 0 3 6 8 11 ]
    // Th:           [ 0 1 2 4 ]

    // output repeated but with "|" denoting the vectors of T:
    // Ti:           [ 0 1 2 | 3 4 5 | 3 4 | 0 1 3 ]
    // Tx:           [ 1 2 1 | 1 1 3 | 1 1 | 1 1 1 ]
    // Tp:           [ 0       3       6     8       11 ] of length T->nvec+1
    // Th:           [ 0       1       2     4          ] of length T->nvec

    // Observation: when I,J are provided (not Key_in) then these are
    // always user-owned arrays and cannot be modified.  So do not try to
    // re-use them (as is done in the CPU builder).  However, if Sx was
    // allocated above, no duplicates were found, and no typecasting is needed,
    // then Sx can be tranplanted into T as T->x.

    #if GB_KNOWN_NO_DUPLICATES
    // the input tuples are known to have no duplicates
    #define no_duplicates true
    #else
    // the input tuples have been checked and no duplicates appear
    bool no_duplicates = (tnz == nvals) ;
    #endif

    #if GB_TRANSPLANT_IS_POSSIBLE
    // Sx can be transplanted into T->x if it is allocated workspace (W_2)
    // and if no duplicates exist.  However, if T is iso-valued, then Sx
    // does not exist and is not transplanted into T->x
    bool Sx_transplant = Sx_is_workspace && no_duplicates ;
    #else
    #define Sx_transplant false
    #endif

    // allocate the T matrix as hypersparse, with tnz entries and tnvec vectors,
    // or as sparse if T is a typecasted GrB_Vector.
    GB_OK (GB_new_bix (&T, ttype, vlen, vdim,
        /* Ap_option: */ (tnz == 0) ? GB_ph_calloc : GB_ph_malloc,
        is_csc,
        #if GB_MTX_BUILD
        /* sparsity: */ GxB_HYPERSPARSE,
        /* bitmap_calloc: */ false,
        /* hyper_switch: */ GB_ALWAYS_HYPER,
        #else
        /* sparsity: */ GxB_SPARSE,
        /* bitmap_calloc: */ false,
        /* hyper_switch: */ GB_NEVER_HYPER,
        #endif
        /* plen: */ tnvec,
        /* nzmax: */ tnz+2,
        /* numeric: */ !Sx_transplant, // don't allocate T->x if transplanting
        /* A_iso: */ GB_ISO_BUILD,
        /* p_is_32: */ (GB_Tp_BITS == 32),
        /* j_is_32: */ (GB_Tj_BITS == 32),
        /* i_is_32: */ (GB_Ti_BITS == 32),
        memlane)) ;

    T->nvals = tnz ;
    T->magic = GB_MAGIC ;
    T->nvec = tnvec ;
    T->nvec_nonempty = tnvec ;

    #if GB_TRANSPLANT_IS_POSSIBLE
    if (Sx_transplant)
    {
        // transplant Sx (aliased to W_2) into T->x; W_2 is not freed when done
        T->x = Sx ;
        T->x_mem = W_2_mem ;
        Sx = NULL ;
        W_2 = NULL ; W_2_mem = 0 ;
    }
    #endif

    #if GB_KNOWN_NO_DUPLICATES
    {

        //----------------------------------------------------------------------
        // construct Tp, Th, Ti, and Tx, no duplicates can appear
        //----------------------------------------------------------------------

        #if GB_TRANSPLANT_IS_POSSIBLE
        if (Sx_transplant)
        {
            // Sx has been transplanted into T->x
            GB_cuda_builder_phase5_transplant <<<grid, block1, 0, stream>>>
                (/* outputs: */ T,
                 /* inputs: */
                    #if GB_MTX_BUILD
                    JDelta, JDeltaSum,
                    #endif
                    Key_out, nvals, nchunks) ;
        }
        else
        #endif
        {
            // copy/cast Sx into T->x
            GB_cuda_builder_phase5_no_dupl <<<grid, block1, 0, stream>>>
                (/* outputs: */ T,
                 /* inputs: */
                    #if GB_MTX_BUILD
                    JDelta, JDeltaSum,
                    #endif
                    Key_out, Sx, nvals, nchunks) ;
        }

    }
    #else
    {

        //----------------------------------------------------------------------
        // duplicates are possible, but none might be found at run time
        //----------------------------------------------------------------------

        if (no_duplicates)
        {
            // construct Tp, Th, Ti, and Tx, no duplicates appear after
            // carefully checking for any possible duplicates
            #if GB_TRANSPLANT_IS_POSSIBLE
            if (Sx_transplant)
            {
                // Sx has been transplanted into T->x
                GB_cuda_builder_phase5_transplant <<<grid, block1, 0, stream>>>
                    (/* outputs: */ T,
                     /* inputs: */
                        #if GB_MTX_BUILD
                        JDelta, JDeltaSum,
                        #endif
                        Key_out, nvals, nchunks) ;
            }
            else
            #endif
            {
                // copy/cast Sx into T->x
                GB_cuda_builder_phase5_no_dupl <<<grid, block1, 0, stream>>>
                    (/* outputs: */ T,
                     /* inputs: */
                        #if GB_MTX_BUILD
                        JDelta, JDeltaSum,
                        #endif
                        Key_out, Sx, nvals, nchunks) ;
            }
        }
        else
        {
            // construct Tp, Th, Ti, and Tx, summing up duplicates
            // (at least one duplicate appears)
            GB_cuda_builder_phase5_with_dupl <<<grid, block1, 0, stream>>>
                (/* outputs: */ T,
                 /* inputs: */  Map, ChunkSum,
                    #if GB_MTX_BUILD
                    JDelta, JDeltaSum,
                    #endif
                    Key_out, Sx, nvals, nchunks) ;
        }
    }
    #endif

    CUDA_OK (cudaGetLastError ( )) ;
    CUDA_OK (cudaStreamSynchronize (stream)) ;

    #ifdef GB_TIMING
    t5 = GB_OPENMP_GET_WTIME - t5 ;
    printf ("builder phase 5: %g sec, tnz: %ld, duplicates: %ld\n", t5,
        tnz, nvals-tnz) ;
    printf ("builder all:     %g sec, no_casting: %d, iso: %d, "
        "transplant: %d\n",
        t1 + t2 + t3 + t4 + t5, GB_BLD_SXTYPE_IS_TXTYPE, GB_ISO_BUILD,
        Sx_transplant) ;
    #endif

    //--------------------------------------------------------------------------
    // free workspace and return result
    //--------------------------------------------------------------------------

    (*Thandle) = T ;
    GB_FREE_WORKSPACE ;
    return (GrB_SUCCESS) ;
}

