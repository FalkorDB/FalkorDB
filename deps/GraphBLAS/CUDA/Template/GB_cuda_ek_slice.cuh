//------------------------------------------------------------------------------
// GraphBLAS/CUDA/Template/GB_cuda_ek_slice.cuh
//------------------------------------------------------------------------------

// SuiteSparse:GraphBLAS, Timothy A. Davis, (c) 2017-2024, All Rights Reserved.
// This file: Copyright (c) 2024, NVIDIA CORPORATION. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//------------------------------------------------------------------------------
// GB_cuda_ek_slice_setup
//------------------------------------------------------------------------------

static __device__ __inline__ int64_t GB_cuda_ek_slice_setup
(
    // inputs, not modified:
    const int64_t *Ap,          // array of size anvec+1
    const int64_t anvec,        // # of vectors in the matrix A
    const int64_t anz,          // # of entries in the sparse/hyper matrix A
    const int64_t pfirst,       // first entry in A to find k
    const int64_t max_pchunk,   // max # of entries in A to find k
    // output:
    int64_t *my_chunk_size,     // size of the chunk for this threadblock
    int64_t *anvec1,            // anvec-1
    float *slope                // slope of vectors from kfirst to klast
)
{

    //--------------------------------------------------------------------------
    // determine the range of entryes pfirst:plast-1 for this chunk
    //--------------------------------------------------------------------------

    // The slice for each threadblock contains entries pfirst:plast-1 of A.
    // The threadblock works on a chunk of entries in Ai/Ax [pfirst...plast-1].

    ASSERT (pfirst < anz) ;
    ASSERT (max_pchunk > 0) ;
    int64_t plast = pfirst + max_pchunk ;
    plast = GB_IMIN (plast, anz) ;
    (*my_chunk_size) = plast - pfirst ;
    ASSERT ((*my_chunk_size) > 0) ;

    //--------------------------------------------------------------------------
    // estimate the first and last vectors for this chunk
    //--------------------------------------------------------------------------

    // find kfirst, the first vector of the slice for this chunk.  kfirst is
    // the vector that owns the entry Ai [pfirst] and Ax [pfirst].  The search
    // does not need to be exact, so kfirst is an estimate.

    int64_t kfirst = 0 ;
    int64_t kright = anvec ;
    GB_TRIM_BINARY_SEARCH (pfirst, Ap, kfirst, kright) ;

    // find klast, the last vector of the slice for this chunk.  klast is the
    // vector that owns the entry Ai [plast-1] and Ax [plast-1].  The search
    // does not have to be exact, so klast is an estimate.

    int64_t klast = kfirst ;
    kright = anvec ;
    GB_TRIM_BINARY_SEARCH (plast, Ap, klast, kright) ;

    //--------------------------------------------------------------------------
    // find slope of vectors in this chunk, and return result
    //--------------------------------------------------------------------------

    // number of vectors in A for this chunk, where
    // Ap [kfirst:klast-1] will be searched.
    int64_t nk = klast - kfirst + 1 ;

    // slope is the estimated # of vectors in this chunk, divided by the
    // chunk size.
    (*slope) = ((float) nk) / ((float) (*my_chunk_size)) ;

    (*anvec1) = anvec - 1 ;
    return (kfirst) ;
}

//------------------------------------------------------------------------------
// GB_cuda_ek_slice_entry
//------------------------------------------------------------------------------

// Let p = kk + pfirst, where kk ranges from 0:my_chunk_size-1, and so p ranges
// from kk:(kk+my_chunk_size-1), and where my_chunk_size is normally of size
// max_pchunk, unless this is the last chunk in the entire matrix.
// GB_cuda_ek_slice_entry computes k for this entry, so that the kth vector
// contains the entry aij with row index i = Ai [p] and value aij = Ax [p]
// (assuming that A is a sparse or hypersparse matrix held by column).  That
// is, Ap [k] <= p < Ap [k+1] will hold.  If A is sparse and held by column,
// then aij is in column j = k.  If A is hypersparse, then aij is in column j =
// Ah [k].

// The method returns the index k of the vector in A that contains the pth
// entry in A, at position p = kk + pfirst.

static __device__ __inline__ int64_t GB_cuda_ek_slice_entry
(
    // inputs, not modified:
    const int64_t kk,           // find the k value of the kkth entry
    const int64_t pfirst,       // first entry in A to find k (for which kk=0)
    const int64_t *Ap,          // array of size anvec+1
    const int64_t anvec1,       // anvec-1
    const int64_t kfirst,       // estimate of first vector in the chunk
    const float slope           // estimate # vectors in chunk / my_chunk_size
)
{

    // get a rough estimate of k for the kkth entry
    int64_t k = kfirst + (int64_t) (slope * ((float) kk)) ;

    // The estimate of k cannot be smaller than kfirst, but it might be bigger
    // than anvec-1, so ensure it is in the valid range, kfirst to anvec-1.
    k = GB_IMIN (k, anvec1) ;

    // look for p in Ap, where p is in range pfirst:plast-1
    // where pfirst >= 0 and plast < anz
    int64_t p = kk + pfirst ;

    // linear-time search for the k value of the pth entry
    while (Ap [k+1] <= p) k++ ;
    while (Ap [k  ] >  p) k-- ;

    // the pth entry of A is contained in the kth vector of A
    ASSERT (Ap [k] <= p && p < Ap [k+1]) ;

    // return the result k
    return (k) ;
}

//------------------------------------------------------------------------------
// GB_cuda_ek_slice
//------------------------------------------------------------------------------

// GB_cuda_ek_slice finds the vector k that owns each entry in the sparse or
// hypersparse matrix A, in Ai/Ax [pfirst:plast-1], where plast = min (anz,
// pfirst+max_pchunk).  Returns my_chunk_size = plast - pfirst, which is the
// size of the chunk operated on by this threadblock.

// The function GB_cuda_ek_slice behaves somewhat like GB_ek_slice used on the
// CPU.  The latter is for OpenMP parallelism on the CPU only; it does not
// need to compute ks.

static __device__ __inline__ int64_t GB_cuda_ek_slice // returns my_chunk_size
(
    // inputs, not modified:
    const int64_t *Ap,          // array of size anvec+1
    const int64_t anvec,        // # of vectors in the matrix A
    const int64_t anz,          // # of entries in the sparse/hyper matrix A
    const int64_t pfirst,       // first entry in A to find k
    const int64_t max_pchunk,   // max # of entries in A to find k
    // output:
    int64_t *ks                 // k value for each pfirst:plast-1
)
{

    //--------------------------------------------------------------------------
    // determine the chunk for this threadblock and its slope
    //--------------------------------------------------------------------------

    int64_t my_chunk_size, anvec1 ;
    float slope ;
    int64_t kfirst = GB_cuda_ek_slice_setup (Ap, anvec, anz, pfirst,
        max_pchunk, &my_chunk_size, &anvec1, &slope) ;

    //--------------------------------------------------------------------------
    // find the kth vector that contains each entry p = pfirst:plast-1
    //--------------------------------------------------------------------------

    for (int64_t kk = threadIdx.x ; kk < my_chunk_size ; kk += blockDim.x)
    {

        //----------------------------------------------------------------------
        // determine the kth vector that contains the pth entry
        //----------------------------------------------------------------------

        int64_t k = GB_cuda_ek_slice_entry (kk, pfirst, Ap, anvec1, kfirst,
            slope) ;

        //----------------------------------------------------------------------
        // save the result in ks
        //----------------------------------------------------------------------

        ks [kk] = k ;
    }

    //--------------------------------------------------------------------------
    // sync all threads and return result
    //--------------------------------------------------------------------------

    this_thread_block().sync() ;
    return (my_chunk_size) ;
}

