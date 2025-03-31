//------------------------------------------------------------------------------
// GB_ijsort:  sort an index array I and remove duplicates
//------------------------------------------------------------------------------

// SuiteSparse:GraphBLAS, Timothy A. Davis, (c) 2017-2025, All Rights Reserved.
// SPDX-License-Identifier: Apache-2.0

//------------------------------------------------------------------------------

// Sort an index array and remove duplicates:

/*
    [I1 I1k] = sort (I) ;
    Iduplicate = [(I1 (1:end-1) == I1 (2:end)), false] ;
    I2  = I1  (~Iduplicate) ;
    I2k = I1k (~Iduplicate) ;
*/

#include "ij/GB_ij.h"
#include "sort/GB_sort.h"

#define GB_FREE_WORKSPACE               \
{                                       \
    GB_FREE_MEMORY (&I1, I1_size) ;       \
    GB_FREE_MEMORY (&I1k, I1k_size) ;     \
    GB_WERK_POP (W, uint64_t) ;         \
}

#define GB_FREE_ALL                     \
{                                       \
    GB_FREE_WORKSPACE ;                 \
    GB_FREE_MEMORY (&I2, I2_size) ;       \
    GB_FREE_MEMORY (&I2k, I2k_size) ;     \
}

GrB_Info GB_ijsort
(
    // input:
    const void *I,              // size ni, where ni > 1 always holds
    const bool I_is_32,
    const int64_t ni,           // length I
    const int64_t imax,         // maximum value in I 
    // output:
    int64_t *p_ni2,             // # of indices in I2 and I2k
    void **p_I2,                // size ni2, where I2 [0..ni2-1] contains the
                                // sorted indices with duplicates removed.
    bool *I2_is_32_handle,      // if I2_is_32 true, I2 is 32 bits; else 64 bits
    size_t *I2_size_handle,
    void **p_I2k,               // output array of size ni2
    bool *I2k_is_32_handle,     // if I2k_is_32 true, I2 is 32 bits; else 64
    size_t *I2k_size_handle,
    GB_Werk Werk
)
{

    //--------------------------------------------------------------------------
    // check inputs
    //--------------------------------------------------------------------------

    GrB_Info info ;
    ASSERT (I != NULL) ;
    ASSERT (p_ni2 != NULL) ;
    ASSERT (p_I2 != NULL) ;
    ASSERT (p_I2k != NULL) ;
    ASSERT (I2_is_32_handle != NULL) ;
    ASSERT (I2_size_handle != NULL) ;
    ASSERT (I2k_is_32_handle != NULL) ;
    ASSERT (I2k_size_handle != NULL) ;

    //--------------------------------------------------------------------------
    // declare workspace and get inputs
    //--------------------------------------------------------------------------

    GB_MDECL (I2 , , u) ; size_t I2_size  = 0 ;
    GB_MDECL (I2k, , u) ; size_t I2k_size = 0 ;
    GB_MDECL (I1 , , u) ; size_t I1_size = 0 ;
    GB_MDECL (I1k, , u) ; size_t I1k_size = 0 ;
    GB_WERK_DECLARE (W, uint64_t) ;

    ASSERT (ni > 1) ;
    int ntasks = 0 ;

    //--------------------------------------------------------------------------
    // determine the number of threads to use
    //--------------------------------------------------------------------------

    int nthreads_max = GB_Context_nthreads_max ( ) ;
    double chunk = GB_Context_chunk ( ) ;
    int nthreads = GB_nthreads (ni, chunk, nthreads_max) ;

    //--------------------------------------------------------------------------
    // determine number of tasks to create
    //--------------------------------------------------------------------------

    ntasks = (nthreads == 1) ? 1 : (32 * nthreads) ;
    ntasks = GB_IMIN (ntasks, ni) ;
    ntasks = GB_IMAX (ntasks, 1) ;

    //--------------------------------------------------------------------------
    // allocate workspace
    //--------------------------------------------------------------------------

    GB_WERK_PUSH (W, ntasks+1, uint64_t) ;

    bool I1_is_32  = (imax <= UINT32_MAX) ;
    bool I1k_is_32 = (ni <= UINT32_MAX) ;
    size_t i1size  = (I1_is_32 ) ? sizeof (uint32_t) : sizeof (uint64_t) ;
    size_t i1ksize = (I1k_is_32) ? sizeof (uint32_t) : sizeof (uint64_t) ;
    I1  = GB_MALLOC_MEMORY (ni, i1size , &I1_size) ;
    I1k = GB_MALLOC_MEMORY (ni, i1ksize, &I1k_size) ;
    GB_IPTR (I1 , I1_is_32) ;
    GB_IPTR (I1k, I1k_is_32) ;
    if (W == NULL || I1 == NULL || I1k == NULL)
    { 
        // out of memory
        GB_FREE_ALL ;
        return (GrB_OUT_OF_MEMORY) ;
    }

    //--------------------------------------------------------------------------
    // copy I into I1 and construct I1k
    //--------------------------------------------------------------------------

    GB_Type_code i1code = (I1_is_32) ? GB_UINT32_code : GB_UINT64_code ;
    GB_Type_code icode  = (I_is_32 ) ? GB_UINT32_code : GB_UINT64_code ;

    GB_cast_int (I1, i1code, I, icode, ni, nthreads_max) ;

    int64_t k ;
    #pragma omp parallel for num_threads(nthreads) schedule(static)
    for (k = 0 ; k < ni ; k++)
    { 
        // the key nik is selected so that the last duplicate entry comes first
        // in the sorted result.  It must be adjusted later, so that the kth
        // entry has a key equal to k.
        int64_t nik = ni - k ;
        GB_ISET (I1k, k, nik) ;     // I1k [k] = nik ;
    }

    //--------------------------------------------------------------------------
    // sort [I1 I1k]
    //--------------------------------------------------------------------------

    GB_OK (GB_msort_2 (I1, I1_is_32, I1k, I1k_is_32, ni, nthreads)) ;

    //--------------------------------------------------------------------------
    // count unique entries in I1
    //--------------------------------------------------------------------------

    int tid ;
    #pragma omp parallel for num_threads(nthreads) schedule(dynamic,1)
    for (tid = 0 ; tid < ntasks ; tid++)
    {
        int64_t kfirst, klast, my_count = 0 ;
        GB_PARTITION (kfirst, klast, ni, tid, ntasks) ;
        int64_t iprev = (kfirst == 0) ? (-1) : GB_IGET (I1, kfirst-1) ;
        for (int64_t k = kfirst ; k < klast ; k++)
        {
            int64_t i = GB_IGET (I1, k) ;
            if (iprev != i)
            { 
                my_count++ ;
            }
            iprev = i ;
        }
        W [tid] = my_count ;
    }

    GB_cumsum1_64 (W, ntasks) ;
    int64_t ni2 = W [ntasks] ;

    //--------------------------------------------------------------------------
    // allocate the result I2 and I2k
    //--------------------------------------------------------------------------

    const bool I2_is_32  = I1_is_32 ;
    const bool I2k_is_32 = I1k_is_32 ;
    I2  = GB_MALLOC_MEMORY (ni2, i1size , &I2_size) ;
    I2k = GB_MALLOC_MEMORY (ni2, i1ksize, &I2k_size) ;
    if (I2 == NULL || I2k == NULL)
    { 
        // out of memory
        GB_FREE_ALL ;
        return (GrB_OUT_OF_MEMORY) ;
    }

    GB_IPTR (I2,  I2_is_32 ) ;
    GB_IPTR (I2k, I2k_is_32) ;

    //--------------------------------------------------------------------------
    // construct the new list I2 from I1, removing duplicates
    //--------------------------------------------------------------------------

    #pragma omp parallel for num_threads(nthreads) schedule(dynamic,1)
    for (tid = 0 ; tid < ntasks ; tid++)
    {
        int64_t kfirst, klast, k2 = W [tid] ;
        GB_PARTITION (kfirst, klast, ni, tid, ntasks) ;
        int64_t iprev = (kfirst == 0) ? (-1) : GB_IGET (I1, kfirst-1) ;
        for (int64_t k = kfirst ; k < klast ; k++)
        {
            int64_t i = GB_IGET (I1, k) ;
            if (iprev != i)
            { 
                int64_t nik = ni - GB_IGET (I1k, k) ;
                GB_ISET (I2, k2, i) ;       // I2 [k2] = i
                GB_ISET (I2k, k2, nik) ;    // I2k [k2] = nik
                k2++ ;
            }
            iprev = i ;
        }
    }

    //--------------------------------------------------------------------------
    // check result: compare with single-pass, single-threaded algorithm
    //--------------------------------------------------------------------------

    #ifdef GB_DEBUG
    {
        // compute the result sequentally in-place, in I1 and I1k, and compare
        // with the output I2 and I2k.
        int64_t ni1 = 1 ;
        int64_t nik = ni - GB_IGET (I1k, 0) ;   // nik = ni - I1k [0]
        GB_ISET (I1k, 0, nik) ;                 // I1k [0] = nik
        for (int64_t k = 1 ; k < ni ; k++)
        {
            if (GB_IGET (I1, ni1-1) != GB_IGET (I1, k))
            {
                int64_t i = GB_IGET (I1, k) ;           // i = I1 [k]
                GB_ISET (I1, ni1, i) ;                  // I1 [ni1] = i
                int64_t nik = ni - GB_IGET (I1k, k) ;   // nik = ni - I1k [k]
                GB_ISET (I1k, ni1, nik) ;               // I1k [ni1] = nik
                ni1++ ;
            }
        }
        ASSERT (ni1 == ni2) ;
        for (int64_t k = 0 ; k < ni1 ; k++)
        {
            ASSERT (GB_IGET (I1 , k) == GB_IGET (I2 , k)) ;
            ASSERT (GB_IGET (I1k, k) == GB_IGET (I2k, k)) ;
        }
    }
    #endif

    //--------------------------------------------------------------------------
    // free workspace and return the new sorted lists
    //--------------------------------------------------------------------------

    GB_FREE_WORKSPACE ;
    (*p_ni2)            = ni2 ;
    (*p_I2 )            = I2  ;
    (*I2_size_handle )  = I2_size ;
    (*I2_is_32_handle)  = I2_is_32 ;
    (*p_I2k)            = I2k ;
    (*I2k_size_handle)  = I2k_size ;
    (*I2k_is_32_handle) = I2k_is_32 ;
    return (GrB_SUCCESS) ;
}

