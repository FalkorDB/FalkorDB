//------------------------------------------------------------------------------
// GB_I_inverse: invert an index list
//------------------------------------------------------------------------------

// SuiteSparse:GraphBLAS, Timothy A. Davis, (c) 2017-2023, All Rights Reserved.
// SPDX-License-Identifier: Apache-2.0

//------------------------------------------------------------------------------

// I is a large list relative to the vector length, avlen, and it is not
// contiguous.  Scatter I into the I inverse buckets (Mark and Inext) for quick
// lookup.

// FUTURE:: this code is sequential.  Constructing the I inverse buckets in
// parallel would require synchronization (a critical section for each bucket,
// or atomics).  A more parallel approach might use qsort first, to find
// duplicates in I, and then construct the buckets in parallel after the qsort.
// But the time complexity would be higher.

#include "GB_subref.h"

GrB_Info GB_I_inverse           // invert the I list for C=A(I,:)
(
    const GrB_Index *I,         // list of indices, duplicates OK
    int64_t nI,                 // length of I
    int64_t avlen,              // length of the vectors of A
    // outputs:
    int64_t *restrict *p_Mark,  // head pointers for buckets, size avlen
    size_t *p_Mark_size,
    int64_t *restrict *p_Inext, // next pointers for buckets, size nI
    size_t *p_Inext_size,
    int64_t *p_ndupl,           // number of duplicate entries in I
    GB_Werk Werk
)
{

    //--------------------------------------------------------------------------
    // get inputs
    //--------------------------------------------------------------------------

    int64_t *Mark  = NULL ; size_t Mark_size = 0 ;
    int64_t *Inext = NULL ; size_t Inext_size = 0 ;
    int64_t ndupl = 0 ;

    (*p_Mark ) = NULL ; (*p_Mark_size ) = 0 ;
    (*p_Inext) = NULL ; (*p_Inext_size) = 0 ;
    (*p_ndupl) = 0 ;

    //--------------------------------------------------------------------------
    // allocate workspace
    //--------------------------------------------------------------------------

    Mark  = GB_CALLOC_WORK (avlen, int64_t, &Mark_size) ;
    Inext = GB_MALLOC_WORK (nI,    int64_t, &Inext_size) ;
    if (Inext == NULL || Mark == NULL)
    { 
        // out of memory
        GB_FREE_WORK (&Mark, Mark_size) ;
        GB_FREE_WORK (&Inext, Inext_size) ;
        return (GrB_OUT_OF_MEMORY) ;
    }

    //--------------------------------------------------------------------------
    // scatter the I indices into buckets
    //--------------------------------------------------------------------------

    // at this point, Mark is all zero, so Mark [i] < 1 for all i in
    // the range 0 to avlen-1.

    // O(nI) time; not parallel
    for (int64_t inew = nI-1 ; inew >= 0 ; inew--)
    {
        int64_t i = I [inew] ;
        ASSERT (i >= 0 && i < avlen) ;
        int64_t ihead = (Mark [i] - 1) ;
        if (ihead < 0)
        { 
            // first time i has been seen in the list I
            ihead = -1 ;
        }
        else
        { 
            // i has already been seen in the list I
            ndupl++ ;
        }
        Mark [i] = inew + 1 ;       // (Mark [i] - 1) = inew
        Inext [inew] = ihead ;
    }

    // indices in I are now in buckets.  An index i might appear more than once
    // in the list I.  inew = (Mark [i] - 1) is the first position of i in I (i
    // will be I [inew]), (Mark [i] - 1) is the head of a link list of all
    // places where i appears in I.  inew = Inext [inew] traverses this list,
    // until inew is -1.

    // to traverse all entries in bucket i, do:
    // GB_for_each_index_in_bucket (inew,i)) { ... }

    #define GB_for_each_index_in_bucket(inew,i) \
        for (int64_t inew = Mark [i] - 1 ; inew >= 0 ; inew = Inext [inew])

    // If Mark [i] < 1, then the ith bucket is empty and i is not in I.
    // Otherise, the first index in bucket i is (Mark [i] - 1).

    #ifdef GB_DEBUG
    for (int64_t i = 0 ; i < avlen ; i++)
    {
        GB_for_each_index_in_bucket (inew, i)
        {
            ASSERT (inew >= 0 && inew < nI) ;
            ASSERT (i == I [inew]) ;
        }
    }
    #endif

    //--------------------------------------------------------------------------
    // return result
    //--------------------------------------------------------------------------

    (*p_Mark ) = Mark  ; (*p_Mark_size ) = Mark_size ;
    (*p_Inext) = Inext ; (*p_Inext_size) = Inext_size ;
    (*p_ndupl) = ndupl ;
    return (GrB_SUCCESS) ;
}

