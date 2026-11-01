//------------------------------------------------------------------------------
// GraphBLAS/CUDA/extractTuples/template/GB_cuda_extractTuples_template.cuh
//------------------------------------------------------------------------------

// SuiteSparse:GraphBLAS, Timothy A. Davis, (c) 2017-2025, All Rights Reserved.
// SPDX-License-Identifier: Apache-2.0

//------------------------------------------------------------------------------

// Creates the column indices of a sparse or hypersparse matrix.

template
<   
    typename T_Ap,              // type of Ap
    typename T_Aj,              // type of Ah and Aj
    bool A_is_hyper,            // true if A is hypersparse, false if sparse
    int64_t chunksize,          // chunksize to use, always a power of 2
    int log2_chunksize          // log2 of chunksize
>
__device__ void GB_cuda_extractTuples_template
(
    // outputs:
    T_Aj *__restrict__ Aj,      // size anz; j = Aj [p] = col index of pth
                                // entry of A
    // inputs:
    const T_Ap *__restrict__ Ap,    // size anvec+1
    const T_Aj *__restrict__ Ah,    // size anvec if hypersparse, NULL if sparse
    const int64_t anvec,        // # of vectors in A
    const int64_t anz           // # entries in A
)
{

    //--------------------------------------------------------------------------
    // get inputs
    //--------------------------------------------------------------------------

    const int64_t anvec1 = anvec - 1 ;

    //--------------------------------------------------------------------------
    // each threadblock operates on one chunk of A at a time
    //--------------------------------------------------------------------------

    for (int64_t pfirst = blockIdx.x << log2_chunksize ;
                 pfirst < anz ;
                 pfirst += gridDim.x << log2_chunksize )
    {

        //----------------------------------------------------------------------
        // determine the chunk for this threadblock and its slope
        //----------------------------------------------------------------------

        int64_t my_chunk_size, kfirst ;
        float slope ;
        GB_cuda_ek_slice_setup<T_Ap> (Ap, anvec, anz, pfirst, chunksize,
            &kfirst, &my_chunk_size, &slope) ;

        //----------------------------------------------------------------------
        // contruct Aj using the threads in this threadblock
        //----------------------------------------------------------------------

        for (int64_t pdelta = threadIdx.x ;
                     pdelta < my_chunk_size ;
                     pdelta += blockDim.x)
        {

            //------------------------------------------------------------------
            // determine the kth vector that contains the pth entry
            //------------------------------------------------------------------

            int64_t p = pfirst + pdelta ;
            int64_t k = GB_cuda_ek_slice_entry<T_Ap> (p, pdelta, Ap, anvec1,
                kfirst, slope) ;

            //------------------------------------------------------------------
            // save the column index of the pth entry in Aj [p]
            //------------------------------------------------------------------

            if constexpr (A_is_hyper)
            {
                // A is hypersparse
                Aj [p] = Ah [k] ;
            }
            else
            {
                // A is sparse
                Aj [p] = k ;
            }
        }
    }
}

