//------------------------------------------------------------------------------
// GB_ijproperties: check I and determine its properties
//------------------------------------------------------------------------------

// SuiteSparse:GraphBLAS, Timothy A. Davis, (c) 2017-2023, All Rights Reserved.
// SPDX-License-Identifier: Apache-2.0

//------------------------------------------------------------------------------

// JIT: not needed.  Only one variant possible.

// check a list of indices I and determine its properties

#include "GB_ij.h"

// FUTURE:: if limit=0, print a different message.  see also setEl, extractEl.
#define GB_ICHECK(i,limit)                                                  \
{                                                                           \
    if ((i) < 0 || (i) >= (limit))                                          \
    {                                                                       \
        GB_ERROR (GrB_INDEX_OUT_OF_BOUNDS,                                  \
        "index " GBd " out of bounds, must be < " GBd , (i), (limit)) ;     \
    }                                                                       \
}

GrB_Info GB_ijproperties        // check I and determine its properties
(
    // input:
    const GrB_Index *I,         // list of indices, or special
    const int64_t ni,           // length I, or special
    const int64_t nI,           // actual length from GB_ijlength
    const int64_t limit,        // I must be in the range 0 to limit-1
    // input/output:
    int *Ikind,                 // kind of I, from GB_ijlength
    int64_t Icolon [3],         // begin:inc:end from GB_ijlength
    // output:
    bool *I_is_unsorted,        // true if I is out of order
    bool *I_has_dupl,           // true if I has a duplicate entry (undefined
                                // if I is unsorted)
    bool *I_is_contig,          // true if I is a contiguous list, imin:imax
    int64_t *imin_result,       // min (I)
    int64_t *imax_result,       // max (I)
    GB_Werk Werk
)
{

    //--------------------------------------------------------------------------
    // check inputs
    //--------------------------------------------------------------------------

    // inputs:
    // I: a list of indices if Ikind is GB_LIST
    // limit: the matrix dimension (# of rows or # of columns)
    // ni: only used if Ikind is GB_LIST: the length of the array I
    // nI: the length of the list I, either actual or implicit

    // input/output:  these can be modified
    // Ikind: GB_ALL, GB_RANGE, GB_STRIDE (both +/- inc), or GB_LIST
    // Icolon: begin:inc:end for all but GB_LIST

    // outputs:
    // I_is_unsorted: true if Ikind == GB_LIST and not in ascending order
    // I_is_contig: true if I has the form I = begin:end
    // imin, imax: min (I) and max (I), but only including actual indices
    //      in the sequence.  The end value of I=begin:inc:end may not be
    //      reached.  For example if I=1:2:10 then max(I)=9, not 10.

    ASSERT (I != NULL) ;
    ASSERT (limit >= 0) ;
    ASSERT (limit <= GB_NMAX) ;
    int64_t imin, imax ;

    //--------------------------------------------------------------------------
    // scan I
    //--------------------------------------------------------------------------

    // scan the list of indices: check if OK, determine if they are
    // unsorted, or contiguous, their min and max index, and actual length
    bool I_unsorted = false ;
    bool I_has_duplicates = false ;
    bool I_contig = true ;

    if ((*Ikind) == GB_ALL)
    { 

        //----------------------------------------------------------------------
        // I = 0:limit-1
        //----------------------------------------------------------------------

        imin = 0 ;
        imax = limit-1 ;

        ASSERT (Icolon [GxB_BEGIN] == imin) ;
        ASSERT (Icolon [GxB_INC  ] == 1) ;
        ASSERT (Icolon [GxB_END  ] == imax) ;

    }
    else if ((*Ikind) == GB_RANGE)
    {

        //----------------------------------------------------------------------
        // I = imin:imax
        //----------------------------------------------------------------------

        imin = Icolon [GxB_BEGIN] ;
        ASSERT (Icolon [GxB_INC] == 1) ;
        imax = Icolon [GxB_END  ] ;

        if (imin > imax)
        { 
            // imin > imax: list is empty
            imin = limit ;
            imax = -1 ;
        }
        else
        { 
            // check the limits
            GB_ICHECK (imin, limit) ;
            GB_ICHECK (imax, limit) ;
        }

    }
    else if ((*Ikind) == GB_STRIDE)
    {

        //----------------------------------------------------------------------
        // I = imin:iinc:imax
        //----------------------------------------------------------------------

        // int64_t ibegin = Icolon [GxB_BEGIN] ;
        int64_t iinc   = Icolon [GxB_INC  ] ;
        // int64_t iend   = Icolon [GxB_END  ] ;

        // if iinc == 1 on input, the kind has been changed to GB_RANGE
        ASSERT (iinc != 1) ;

        if (iinc == 0)
        { 
            // stride is zero: list is empty, contiguous, and sorted
            imin = limit ;
            imax = -1 ;
        }
        else if (iinc > 0)
        { 
            // stride is positive, get the first and last indices
            imin = GB_ijlist (I, 0,    GB_STRIDE, Icolon) ;
            imax = GB_ijlist (I, nI-1, GB_STRIDE, Icolon) ;
        }
        else
        { 
            // stride is negative, get the first and last indices
            imin = GB_ijlist (I, nI-1, GB_STRIDE, Icolon) ;
            imax = GB_ijlist (I, 0,    GB_STRIDE, Icolon) ;
        }

        if (imin > imax)
        { 
            // list is empty: so it is contiguous and sorted
            imin = limit ;
            imax = -1 ;

            // change this to an empty GB_RANGE
            (*Ikind) = GB_RANGE ;
            Icolon [GxB_BEGIN] = imin ;
            Icolon [GxB_INC  ] = 1 ;
            Icolon [GxB_END  ] = imax ;

        }
        else
        { 
            // list is contiguous if the stride is 1, not contiguous otherwise
            I_contig = false ;

            // check the limits
            GB_ICHECK (imin, limit) ;
            GB_ICHECK (imax, limit) ;
        }

    }
    else // (*Ikind) == GB_LIST
    {

        //----------------------------------------------------------------------
        // determine the number of threads to use
        //----------------------------------------------------------------------

        int nthreads_max = GB_Context_nthreads_max ( ) ;
        double chunk = GB_Context_chunk ( ) ;
        int nthreads = GB_nthreads (ni, chunk, nthreads_max) ;
        int ntasks = (nthreads == 1) ? 1 : (8 * nthreads) ;
        ntasks = GB_IMIN (ntasks, ni) ;
        ntasks = GB_IMAX (ntasks, 0) ;

        //----------------------------------------------------------------------
        // I is an array of indices
        //----------------------------------------------------------------------

        // scan I to find imin and imax, and validate the list. Also determine
        // if it is sorted or not, and contiguous or not.

        imin = limit ;
        imax = -1 ;

        // allocate workspace for imin and imax
        GB_WERK_DECLARE (Work_imin, int64_t) ;
        GB_WERK_DECLARE (Work_imax, int64_t) ;
        GB_WERK_PUSH (Work_imin, ntasks, int64_t) ;
        GB_WERK_PUSH (Work_imax, ntasks, int64_t) ;
        if (Work_imin == NULL || Work_imax == NULL)
        { 
            // out of memory
            GB_WERK_POP (Work_imax, int64_t) ;
            GB_WERK_POP (Work_imin, int64_t) ;
            return (GrB_OUT_OF_MEMORY) ;
        }

        int tid ;
        #pragma omp parallel for num_threads(nthreads) schedule(dynamic,1) \
            reduction(||:I_unsorted) reduction(&&:I_contig) \
            reduction(||:I_has_duplicates)
        for (tid = 0 ; tid < ntasks ; tid++)
        {
            int64_t my_imin = limit ;
            int64_t my_imax = -1 ;
            int64_t istart, iend ;
            GB_PARTITION (istart, iend, ni, tid, ntasks) ;
            int64_t ilast = (istart == 0) ? -1 : I [istart-1] ;
            for (int64_t inew = istart ; inew < iend ; inew++)
            {
                int64_t i = I [inew] ;
                if (inew > 0)
                {
                    if (i < ilast)
                    { 
                        // The list I of row indices is out of order, and
                        // C=A(I,J) will need to use qsort to sort each column.
                        // If C=A(I,J)' is computed, however, this flag will be
                        // set back to false, since qsort is not needed if the
                        // result is transposed.
                        I_unsorted = true ;
                    }
                    else if (i == ilast)
                    { 
                        // I has at least one duplicate entry.  If I is
                        // unsorted, then it is not known if I has duplicates
                        // or not.  But if I is sorted, but with duplicates,
                        // then this flag will be true.
                        I_has_duplicates = true ;
                    }
                    if (i != ilast + 1)
                    { 
                        I_contig = false ;
                    }
                }
                my_imin = GB_IMIN (my_imin, i) ;
                my_imax = GB_IMAX (my_imax, i) ;
                ilast = i ;
            }
            Work_imin [tid] = my_imin ;
            Work_imax [tid] = my_imax ;
        }

        // wrapup
        for (tid = 0 ; tid < ntasks ; tid++)
        { 
            imin = GB_IMIN (imin, Work_imin [tid]) ;
            imax = GB_IMAX (imax, Work_imax [tid]) ;
        }

        // free workspace
        GB_WERK_POP (Work_imax, int64_t) ;
        GB_WERK_POP (Work_imin, int64_t) ;

        #ifdef GB_DEBUG
        {
            // check result with one thread
            bool I_unsorted2 = false ;
            bool I_has_dupl2 = false ;
            bool I_contig2 = true ;
            int64_t imin2 = limit ;
            int64_t imax2 = -1 ;
            int64_t ilast = -1 ;
            for (int64_t inew = 0 ; inew < ni ; inew++)
            {
                int64_t i = I [inew] ;
                if (inew > 0)
                {
                    if (i < ilast) I_unsorted2 = true ;
                    else if (i == ilast) I_has_dupl2 = true ;
                    if (i != ilast + 1) I_contig2 = false ;
                }
                imin2 = GB_IMIN (imin2, i) ;
                imax2 = GB_IMAX (imax2, i) ;
                ilast = i ;
            }
            ASSERT (I_unsorted == I_unsorted2) ;
            ASSERT (I_has_duplicates == I_has_dupl2) ;
            ASSERT (I_contig   == I_contig2) ;
            ASSERT (imin       == imin2) ;
            ASSERT (imax       == imax2) ;
        }
        #endif

        if (ni > 0)
        { 
            // check the limits
            GB_ICHECK (imin, limit) ;
            GB_ICHECK (imax, limit) ;
        }

        if (ni == 1)
        { 
            // a single entry does not need to be sorted
            ASSERT (I [0] == imin) ;
            ASSERT (I [0] == imax) ;
            ASSERT (I_unsorted == false) ;
            ASSERT (I_contig   == true) ;
        }
        if (ni == 0)
        { 
            // the list is empty
            ASSERT (imin == limit && imax == -1) ;
        }

        //----------------------------------------------------------------------
        // change I if it is an explicit contiguous list of stride 1
        //----------------------------------------------------------------------

        if (I_contig)
        {
            // I is a contigous list of stride 1, imin:imax.
            // change Ikind to GB_ALL if 0:limit-1, or GB_RANGE otherwise
            if (imin == 0 && imax == limit-1)
            { 
                (*Ikind) = GB_ALL ;
            }
            else
            { 
                (*Ikind) = GB_RANGE ;
            }
            Icolon [GxB_BEGIN] = imin ;
            Icolon [GxB_INC  ] = 1 ;
            Icolon [GxB_END  ] = imax ;
        }
    }

    //--------------------------------------------------------------------------
    // return result
    //--------------------------------------------------------------------------

    ASSERT (GB_IMPLIES (I_contig, !I_unsorted)) ;
    ASSERT (((*Ikind) == GB_ALL || (*Ikind) == GB_RANGE) == I_contig) ;

    // I_is_contig is true if the list of row indices is a contiguous list,
    // imin:imax.  This is an important special case.

    // I_is_unsorted is true if I is an explicit list, the list is non-empty,
    // and the indices are not sorted in ascending order.

    (*I_is_contig) = I_contig ;
    (*I_is_unsorted) = I_unsorted ;
    (*I_has_dupl) = I_has_duplicates ;
    (*imin_result) = imin ;
    (*imax_result) = imax ;
    return (GrB_SUCCESS) ;
}

