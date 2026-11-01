//------------------------------------------------------------------------------
// GraphBLAS/CUDA/builder/template/GB_cuda_construct_JDelta
//------------------------------------------------------------------------------

// SuiteSparse:GraphBLAS, Timothy A. Davis, (c) 2017-2026, All Rights Reserved.
// SPDX-License-Identifier: Apache-2.0

//------------------------------------------------------------------------------

// Construct Jdelta and JDeltaSum from the column indices in Cj [-1..nvals-1],
// to find the leading entries columns of C

template
<
    typename T_Cp,          // type of Cp and JDeltaSum
    typename T_Cj,          // type of Cj
    int chunksize,          // must be <= 65535 and a power of 2
    int log2_chunksize,     // log2 (chunksize)
    int blockdim,           // blockdim of kernel launch
    int items_per_thread,   // # of items handled by a single thread
    typename T_F            // type of unload_Cj function
>
__device__ void GB_cuda_construct_JDelta
(
    // outputs:
    uint16_t *JDelta,       // size nvals+1, in JDelta [-1..nvals-1]
    T_Cp *JDeltaSum,        // size nchunks+1
    // inputs, not modified, except for Cj [-1] sentinel value:
    T_Cj *Cj,               // size nvals+1, Cj [-1..nvals-1]
    int64_t nvals,          // # of entries in C
    int64_t nchunks,        // # of chunks of C
    T_F unload_Cj           // lamdba function to get j from Cj [p]
)
{

    //--------------------------------------------------------------------------
    // workspace for each threadblock
    //--------------------------------------------------------------------------

    __shared__ uint16_t Local_JDelta [chunksize] ;

    // cub::Block* workspace:
    GB_CUB_BLOCK_WORKSPACE (W, uint16_t, blockdim, items_per_thread) ;

    //--------------------------------------------------------------------------
    // the first thread of the threadblock fills in the sentinal values
    //--------------------------------------------------------------------------

    if (threadIdx.x == 0 && blockIdx.x == 0)
    {
        memset (&(Cj [-1]), 0xFF, sizeof (T_Cj)) ;
        JDelta [-1] = 0 ;
        JDeltaSum [-1] = 0 ;
    }

    // this_thread_block ( ).sync ( ) ; not needed since the thread that wrote
    // the Cj [-1] entry is the only thread that reads it.

    //--------------------------------------------------------------------------
    // compute each local chunk of Map
    //--------------------------------------------------------------------------

    for (int64_t chunk = blockIdx.x ;
                 chunk < nchunks ;
                 chunk += gridDim.x)        // grid-stride loop
    {

        //----------------------------------------------------------------------
        // determine the properties of this chunk
        //----------------------------------------------------------------------

        int64_t pfirst = chunk << log2_chunksize ;
        int64_t my_chunk_size ;
        // this computation is just the 2nd #if case of select/phase1:
        int64_t plast = pfirst + chunksize ;
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
            auto jprev = unload_Cj (Cj, p-1) ;  // jprev = Cj [p-1]
            auto j     = unload_Cj (Cj, p  ) ;  // j     = Cp [p  ]

            // leading is true if this is the first entry in vector j
            bool leading = (j != jprev) ;
            Local_JDelta [pdelta] = leading ;   // 1 if leading entry of vector
        }

        //----------------------------------------------------------------------
        // the remainder is similar to select/phase1:
        //----------------------------------------------------------------------

        // clear the unused part of the Local_Map and Local_JDelta
        for ( ; pdelta < chunksize ;
                pdelta += blockDim.x)
        {
            Local_JDelta [pdelta] = 0 ;
        }

        //----------------------------------------------------------------------
        // inclusive cumulative sum of Local_Map and Local_JDelta
        //----------------------------------------------------------------------

        // Map [pfirst..pfirst+chunksize-1] = inclusive cumsum of Local_Map,
        // where Local_Map [i] = sum (Local_Map [0:i]) is computed.

        // Similarly, JDelta [pfirst..pfirst+chunksize-1] = inclusive cumsum
        // of Local_JDelta, where Local_JDelta [i] = sum (Local_JDelta [0:i])
        // is computed.

        this_thread_block ( ).sync ( ) ;
        uint16_t s_block_aggregate ;

#if 0
        // This entire phase computes the following:
        if (threadIdx.x == blockDim.x - 1)
        {

            // construct JDelta and JDeltaSum
            for (int i = 1 ; i < chunksize ; i++)
            {
                Local_JDelta [i] += Local_JDelta [i-1] ;
            }
            for (int i = 0 ; i < chunksize ; i++)
            {
                JDelta [pfirst + i] = Local_JDelta [i] ;
            }
            s_block_aggregate = Local_JDelta [chunksize-1] ;
            JDeltaSum [chunk] = s_block_aggregate ;
        }

#else
        uint16_t t [items_per_thread] ;

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

