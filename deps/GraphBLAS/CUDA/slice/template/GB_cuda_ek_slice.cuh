//------------------------------------------------------------------------------
// GraphBLAS/CUDA/template/GB_cuda_ek_slice.cuh
//------------------------------------------------------------------------------

// SuiteSparse:GraphBLAS, Timothy A. Davis, (c) 2017-2025, All Rights Reserved.
// This file: Copyright (c) 2024-2025, NVIDIA CORPORATION. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//------------------------------------------------------------------------------

// The GB_cuda_ek_slice* methods provide an efficient method for the
// threadblocks to work in parallel on a single sparse or hypersparse matrix A.
// Let Ap = A->p be an array of "pointers" of size anvec+1.
// Let Ai = A->i be the array of row indices of the sparse/hypersparse matrix.
// Let Ax = A->x be the array of values of the sparse/hypersparse matrix.
// Let anz be the # of entries in the Ai and Ax arrays.
//
// Then matrix of A can be traversed as follows (suppose it is stored by col):
//
//  for (k = 0 ; k < anvec ; k++)
//      j = k if A is sparse, or j = Ah [k] if A is hypersparse
//      for (p = Ap [k] ; p < Ap [k+1] ; p++)
//          i = Ai [p] ;
//          atype aij = Ax [p] ;
//          the entry A(i,j) has value aij, row index i, column index j, and
//              column j is the kth column in the data structure of A,
//          ...
//
// However, parallelizing that loop across the GPU threadblocks is difficult.
// Instead, consider a single loop:
//
//  for (p = 0 ; p < anz ; p++)
//      find k so that Ap [k] <= p < Ap [k+1]
//      j = k if A is sparse, or j = Ah [k] if A is hypersparse
//      i = Ai [p]
//      aij = Ax [p]
//      ...
//
// The above loop can be easily parallelized on the GPU, and the
// GB_cuda_ek_slice* methods provide a way for each thread to find k for each
// entry at position p.  The methods assume a single threadblock is given the
// task to compute iterations p = pfirst : plast=1 where plast = min (anz,
// pfirst + max_pchunk) for the loop above.
//
// First, all threads call GB_cuda_ek_slice_setup, and do two binary searches.
// This determines the slope, and gives a way to estimate the vector k that
// contains any entry p in the given range pfirst:plast-1.  This estimate is
// then used in GB_cuda_ek_slice_entry to determine the k for any given p in
// that range.
//
// If the thread that computes k for a given p is also the thread that uses k,
// then there is no need for the threadblock to share that computation.
// Otherwise, the set of k values can be computed in a shared array ks, using
// the single method GB_cuda_ek_slice.

// FIXME: discuss grid-stride loops, the pdelta loop, and the max chunk size
// here

// question: why chunks are necessary? why not just do ek_slice_setup across
// all entries in one go?  answer: the slope method is only useful for a small
// range of entries; non-uniform entry distributions can distort the usefulness
// of the slope (will require an exhaustive linear search) for a large range of
// entries

//------------------------------------------------------------------------------
// GB_cuda_ek_slice_setup
//------------------------------------------------------------------------------

template <typename T> __device__ void GB_cuda_ek_slice_setup
(
    // inputs, not modified:
    const T *Ap,                // array of size anvec+1
    const int64_t anvec,        // # of vectors in the matrix A
    const int64_t anz,          // # of entries in the sparse/hyper matrix A
    const int64_t pfirst,       // first entry in A to find k
    const int64_t max_pchunk,   // max # of entries in A to find k
    // output:
    int64_t *kfirst,            // first vector of the slice for this chunk
    int64_t *klast,             // last vector of the slice for this chunk
    int64_t *my_chunk_size,     // size of the chunk for this threadblock
    int64_t *anvec1,            // anvec-1
    float *slope                // slope of vectors from kfirst to klast
)
{

    //--------------------------------------------------------------------------
    // determine the range of entries pfirst:plast-1 for this chunk
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

    (*kfirst) = 0 ;
    int64_t kright = anvec ;
    if (sizeof (T) == sizeof (uint32_t))
    {
        GB_trim_binary_search_32 (pfirst, (const uint32_t *) Ap, kfirst, &kright) ;
    }
    else
    {
        GB_trim_binary_search_64 (pfirst, (const uint64_t *) Ap, kfirst, &kright) ;
    }

    // find klast, the last vector of the slice for this chunk.  klast is the
    // vector that owns the entry Ai [plast-1] and Ax [plast-1].  The search
    // does not have to be exact, so klast is an estimate.

    (*klast) = (*kfirst) ;
    kright = anvec ;
    if (sizeof (T) == sizeof (uint32_t))
    {
        GB_trim_binary_search_32 (plast, (const uint32_t *) Ap, klast, &kright) ;
    }
    else
    {
        GB_trim_binary_search_64 (plast, (const uint64_t *) Ap, klast, &kright) ;
    }
    //--------------------------------------------------------------------------
    // find slope of vectors in this chunk, and return result
    //--------------------------------------------------------------------------

    // number of vectors in A for this chunk, where
    // Ap [kfirst:klast-1] will be searched.
    int64_t nk = (*klast) - (*kfirst) + 1 ;

    // slope is the estimated # of vectors in this chunk, divided by the
    // chunk size.
    (*slope) = ((float) nk) / ((float) (*my_chunk_size)) ;
    (*anvec1) = anvec - 1 ;
}

//------------------------------------------------------------------------------
// GB_cuda_ek_slice_entry
//------------------------------------------------------------------------------

// Let p = pfirst + pdelta, where pdelta ranges from 0:my_chunk_size-1, and so
// p ranges from pdelta:(pdelta+my_chunk_size-1), and where my_chunk_size is
// normally of size max_pchunk, unless this is the last chunk in the entire
// matrix.  GB_cuda_ek_slice_entry computes k for this entry, so that the kth
// vector contains the entry aij with row index i = Ai [p] and value aij = Ax
// [p] (assuming that A is a sparse or hypersparse matrix held by column).
// That is, Ap [k] <= p < Ap [k+1] will hold.  If A is sparse and held by
// column, then aij is in column j = k.  If A is hypersparse, then aij is in
// column j = Ah [k].

// The method returns the index k of the vector in A that contains the pth
// entry in A, at position p = pfirst + pdelta.

template <typename T> __device__ int64_t GB_cuda_ek_slice_entry
(
    // inputs, not modified:
    const int64_t p,            // p = pfirst + pdelta
    const int64_t pdelta,       // find the k value of the pfirst+pdelta entry
    const T *Ap,                // array of size anvec+1
    const int64_t anvec1,       // anvec-1
    const int64_t kfirst,       // estimate of first vector in the chunk
    const float slope           // estimate # vectors in chunk / my_chunk_size
)
{

    // get a rough estimate of k for the pfirst + pdelta entry
    int64_t k = kfirst + (int64_t) (slope * ((float) pdelta)) ;

    // The estimate of k cannot be smaller than kfirst, but it might be bigger
    // than anvec-1, so ensure it is in the valid range, kfirst to anvec-1.
    k = GB_IMIN (k, anvec1) ;

    // look for p in Ap, where p is in range pfirst:plast-1
    // where pfirst >= 0 and plast < anz

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

template <typename T>__device__ int64_t GB_cuda_ek_slice // returns my_chunk_size
(
    // inputs, not modified:
    const T *Ap,                // array of size anvec+1
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

    int64_t my_chunk_size, anvec1, kfirst, klast ;
    float slope ;
    GB_cuda_ek_slice_setup<T> (Ap, anvec, anz, pfirst, max_pchunk,
        &kfirst, &klast, &my_chunk_size, &anvec1, &slope) ;

    //--------------------------------------------------------------------------
    // find the kth vector that contains each entry p = pfirst:plast-1
    //--------------------------------------------------------------------------

    for (int64_t pdelta = threadIdx.x ;
                 pdelta < my_chunk_size ;
                 pdelta += blockDim.x)
    {

        //----------------------------------------------------------------------
        // determine the kth vector that contains the pth entry
        //----------------------------------------------------------------------

        int64_t p = pfirst + pdelta ;
        int64_t k = GB_cuda_ek_slice_entry<T> (p, pdelta, Ap, anvec1, kfirst, slope) ;

        //----------------------------------------------------------------------
        // save the result in ks
        //----------------------------------------------------------------------

        // the pth entry of the matrix is in vector k
        ks [pdelta] = k ;
    }

    //--------------------------------------------------------------------------
    // sync all threads and return result
    //--------------------------------------------------------------------------

    this_thread_block().sync() ;
    return (my_chunk_size) ;
}

