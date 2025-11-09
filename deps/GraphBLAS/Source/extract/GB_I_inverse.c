//------------------------------------------------------------------------------
// GB_I_inverse: invert an index list
//------------------------------------------------------------------------------

// SuiteSparse:GraphBLAS, Timothy A. Davis, (c) 2017-2025, All Rights Reserved.
// SPDX-License-Identifier: Apache-2.0

//------------------------------------------------------------------------------

// I is a large list relative to the vector length, avlen, and it is not
// contiguous.  Scatter I into the I inverse buckets (Ihead and Inext) for quick
// lookup.

// FUTURE:: this code is sequential.  Constructing the I inverse buckets in
// parallel would require synchronization (a critical section for each bucket,
// or atomics).  A more parallel approach might use qsort first, to find
// duplicates in I, and then construct the buckets in parallel after the qsort.
// But the time complexity would be higher.

#include "extract/GB_subref.h"

GrB_Info GB_I_inverse           // invert the I list for C=A(I,:)
(
    const void *I,              // list of indices, duplicates OK
    const bool I_is_32,         // if true, I is 32-bit; else 64 bit
    int64_t nI,                 // length of I
    int64_t avlen,              // length of the vectors of A
    // outputs:
    void **p_Ihead,             // head pointers for buckets, size avlen
    size_t *p_Ihead_size,
    void **p_Inext,             // next pointers for buckets, size nI
    size_t *p_Inext_size,
    bool *p_Ihead_is_32,        // if true, Ihead and Inext are 32-bit; else 64
    int64_t *p_nduplicates,     // number of duplicate entries in I
    GB_Werk Werk
)
{

    //--------------------------------------------------------------------------
    // get inputs
    //--------------------------------------------------------------------------

    GB_MDECL (Ihead, , u) ; size_t Ihead_size = 0 ;
    GB_MDECL (Inext, , u) ; size_t Inext_size = 0 ;
    int64_t nduplicates = 0 ;

    (*p_Ihead) = NULL ; (*p_Ihead_size) = 0 ;
    (*p_Inext) = NULL ; (*p_Inext_size) = 0 ;
    (*p_nduplicates) = 0 ;

    GB_IDECL (I, const, u) ; GB_IPTR (I, I_is_32) ;

    //--------------------------------------------------------------------------
    // allocate workspace
    //--------------------------------------------------------------------------

    // Entries in Ihead and Inext range in value from 0 to nI.  Entries equal
    // to nI or larger are invalid indices, need to tag the end of each bucket.
    // Thus

    bool Ihead_is_32 = (nI < UINT32_MAX) ;
    size_t isize = (Ihead_is_32) ? sizeof (uint32_t) : sizeof (uint64_t) ;

    Ihead = GB_MALLOC_MEMORY (avlen, isize, &Ihead_size) ;
    Inext = GB_MALLOC_MEMORY (nI,    isize, &Inext_size) ;
    if (Inext == NULL || Ihead == NULL)
    { 
        // out of memory
        GB_FREE_MEMORY (&Ihead, Ihead_size) ;
        GB_FREE_MEMORY (&Inext, Inext_size) ;
        return (GrB_OUT_OF_MEMORY) ;
    }

    // set all entries of Ihead to UINT*_MAX (32-bit or 64-bit)
    int nthreads_max = GB_Context_nthreads_max ( ) ;
    GB_memset (Ihead, 0xFF, Ihead_size, nthreads_max) ;

    GB_IPTR (Ihead, Ihead_is_32) ;
    GB_IPTR (Inext, Ihead_is_32) ;

    //--------------------------------------------------------------------------
    // scatter the I indices into buckets
    //--------------------------------------------------------------------------

    // At this point, Ihead [0..avlen-1] >= nI = UINT64_MAX.

    // O(nI) time; not parallel
    for (int64_t inew = nI-1 ; inew >= 0 ; inew--)
    {
        int64_t i = GB_IGET (I, inew) ;
        ASSERT (i >= 0 && i < avlen) ;
        int64_t ihead = GB_IGET (Ihead, i) ;
        if (ihead < nI)
        { 
            // i has already been seen in the list I
            nduplicates++ ;
        }
        GB_ISET (Ihead, i, inew) ;      // Ihead [i] = inew ;
        GB_ISET (Inext, inew, ihead) ;  // Inext [inew] = ihead ;
    }

    // indices in I are now in buckets.  An index i might appear more than once
    // in the list I.  inew = Ihead [i] is the first position of i in I (i will
    // be I [inew]), Ihead [i] is the head of a link list of all places where i
    // appears in I.  inew = Inext [inew] traverses this list, until inew is >=
    // nI, which denotes the end of the bucket.

    // to traverse all entries in bucket i, do:
    // GB_for_each_index_in_bucket (inew,i,nI,Ihead,Inext) { ... }

    #define GB_for_each_index_in_bucket(inew,i,nI,Ihead,Inext)  \
        for (uint64_t inew = GB_IGET (Ihead, i) ;               \
                      inew < nI ;                               \
                      inew = GB_IGET (Inext, inew))

    // If Ihead [i] > nI, then the ith bucket is empty and i is not in I.
    // Otherise, the first index in bucket i is Ihead [i].

    #ifdef GB_DEBUG
    for (int64_t i = 0 ; i < avlen ; i++)
    {
        GB_for_each_index_in_bucket (inew, i, nI, Ihead, Inext)
        {
            // inew is the new index in C, and i is the index in A.
            // All entries in the ith bucket refer to the same row A(i,:),
            // but with different indices C (inew,:) in C.
            ASSERT (inew >= 0 && inew < nI) ;
            ASSERT (i == GB_IGET (I, inew)) ;
        }
    }
    #endif

    //--------------------------------------------------------------------------
    // return result
    //--------------------------------------------------------------------------

    (*p_Ihead) = Ihead ; (*p_Ihead_size) = Ihead_size ;
    (*p_Inext) = Inext ; (*p_Inext_size) = Inext_size ;
    (*p_Ihead_is_32) = Ihead_is_32 ;
    (*p_nduplicates) = nduplicates ;
    return (GrB_SUCCESS) ;
}

